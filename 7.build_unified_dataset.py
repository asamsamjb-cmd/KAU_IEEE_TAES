# 7. build_unified_dataset.py
# Unified observable dataset builder for the TAES revision comparisons.
# Revision: preserve the episode split policy; rebuild assignments from new episodes.
# Run beside the existing dataset/ folder. Only numpy and pandas are required.
# Default output: dataset/shared_nominal_v2/ (previous results are retained).
# Optional: --dataset-dir PATH --output-dir NEW_EMPTY_PATH
# This script neither regenerates data nor trains/evaluates any model.
# Scenario/episode labels are retained, including quiet/unobserved windows.
# Observable-equal rows are retained; no score-driven relabeling/filtering occurs.
# Each learning CSV needs its matching *_with_raw.csv from the revised maker.
# Counterfactual diagnostics are saved separately and never enter model inputs.
#
# One row in this experiment is a 1-second cross-layer observable window.
# TST sequence lengths L={1,2,3,5,10} therefore correspond to
# temporal contexts of {1,2,3,5,10} seconds.
#
# Purpose:
#   1) Load six learning-ready observable CSV files.
#   2) Validate that each file contains METADATA_FIELDS + 22 OBSERVABLE_FIELDS only.
#   3) Prevent raw-audit / leakage fields from entering the training-ready dataset.
#   4) Merge the six classes into one unified dataset.
#   5) Create episode-level train/validation/test splits.
#
# Important:
#   DistilBERT semantic input must be generated from OBSERVABLE_FIELDS only.
#   Metadata fields are retained only for splitting, ordering, analysis, and evaluation.
#   Do NOT serialize scenario_id, episode_id, episode_step, window_id, timestamp, or label.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd


# ============================================================
# 0. Basic settings
# ============================================================

DEFAULT_RANDOM_SEED = 20260610

# Ratio is applied at episode level, not at row level.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# If True, the script fails on unexpected schema problems.
# Keep this True for paper-grade experiment reproducibility.
STRICT_SCHEMA = True

# If True, row counts different from EXPECTED_ROWS_PER_CLASS produce an error.
# Keep this True for final paper-grade 1-second datasets.
STRICT_ROW_COUNT = True
EXPECTED_ROWS_PER_CLASS = 50000
EXPECTED_TOTAL_ROWS = EXPECTED_ROWS_PER_CLASS * 6

# 1-second observable aggregation setting.
# This is not a raw SDR I/Q sampling interval; it is a cross-layer RF/GNSS/TT&C
# observable aggregation window used before DistilBERT serialization and TST.
EXPECTED_WINDOW_SECONDS = 1
TST_SEQUENCE_LENGTHS = [1, 2, 3, 5, 10]
TST_TEMPORAL_CONTEXT_SECONDS = [length * EXPECTED_WINDOW_SECONDS for length in TST_SEQUENCE_LENGTHS]

# If duplicate episode_id values exist across classes, the script rewrites episode_id
# into a globally unique value in the output CSVs. This is metadata only and must not
# be used for DistilBERT text serialization.
REWRITE_DUPLICATE_EPISODE_IDS = True

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = DATASET_DIR / "shared_nominal_v2"
SPLIT_DIR = OUTPUT_DIR / "splits"
REPORT_DIR = OUTPUT_DIR / "reports"

UNIFIED_OUTPUT_PATH = OUTPUT_DIR / "unified_observable_dataset.csv"
TRAIN_OUTPUT_PATH = SPLIT_DIR / "train_observable.csv"
VAL_OUTPUT_PATH = SPLIT_DIR / "val_observable.csv"
TEST_OUTPUT_PATH = SPLIT_DIR / "test_observable.csv"

LABEL_MAPPING_PATH = SPLIT_DIR / "label_mapping.json"
SPLIT_MANIFEST_PATH = SPLIT_DIR / "episode_split_manifest.csv"
SPLIT_SUMMARY_PATH = SPLIT_DIR / "split_summary.csv"
LABEL_DISTRIBUTION_PATH = REPORT_DIR / "label_distribution.csv"
EPISODE_DISTRIBUTION_PATH = REPORT_DIR / "episode_distribution.csv"
FIELD_VALUE_SUMMARY_PATH = REPORT_DIR / "observable_field_value_summary.csv"
SCHEMA_REPORT_PATH = REPORT_DIR / "schema_validation_report.json"
TEMPORAL_CONTEXT_PLAN_PATH = REPORT_DIR / "tst_temporal_context_plan.json"


# ============================================================
# 1. Metadata and 22 observable fields
# ============================================================

METADATA_FIELDS: List[str] = [
    "scenario_id",
    "episode_id",
    "episode_type",
    "episode_step",
    "window_id",
    "timestamp",
    "label",
]

OBSERVABLE_FIELDS: List[str] = [
    # Observation Context
    "observation_stage",
    "path_type",
    "direction",
    "source_role",
    "destination_role",
    "protocol_context",
    "operational_phase",

    # Message and Control Semantics
    "message_type",
    "service_context",
    "command_intent",
    "payload_or_parameter_context",
    "sequence_freshness_status",
    "control_security_state",

    # RF/GNSS Receiver Evidence
    "signal_quality_state",
    "tracking_stability",
    "navigation_consistency",
    "timing_consistency",

    # Operational and Communication Trace
    "decoder_status",
    "command_execution_result",
    "retry_timeout_state",
    "session_state",
    "state_transition_consistency",
]

CANONICAL_COLUMNS: List[str] = METADATA_FIELDS + OBSERVABLE_FIELDS

AUDIT_SCHEMA_VERSION = "counterfactual_observable_change_v1"
AUDIT_FIELDS = [
    "audit_observable_effect_present", "audit_observable_difference_count",
    "audit_persistent_effect_present", "audit_attack_active",
]
AUDIT_SCOPE = (
    "Actual versus paired nominal 22-field projection differences; audit only. "
    "No current attack, no projection difference, and no class information are distinct. "
    "These annotations are not model inputs, an oracle bound, or a proof of identifiability."
)


def observability_audit_contract() -> Dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "split_files": {split: f"audit/{split}_window_observable_effects.csv"
                        for split in ("train", "val", "test")},
        "columns": METADATA_FIELDS + AUDIT_FIELDS,
        "scope": AUDIT_SCOPE,
    }

LABEL_ORDER: List[str] = [
    "NORMAL",
    "RF_LINK_JAMMING",
    "GNSS_SPOOFING",
    "UNAUTHORIZED_TELECOMMAND_INJECTION",
    "TELECOMMAND_REPLAY",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE",
]

LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}


# Some earlier files may use gns_spoofing instead of gnss_spoofing.
# Exactly one candidate must exist; ambiguity fails explicitly.
@dataclass(frozen=True)
class DatasetSpec:
    label: str
    candidates: Tuple[str, ...]


DATASET_SPECS: List[DatasetSpec] = [
    DatasetSpec(
        label="NORMAL",
        candidates=("normal_observable_dataset.csv",),
    ),
    DatasetSpec(
        label="RF_LINK_JAMMING",
        candidates=("rf_link_jamming_observable_dataset.csv",),
    ),
    DatasetSpec(
        label="GNSS_SPOOFING",
        candidates=(
            "gnss_spoofing_observable_dataset.csv",
            "gns_spoofing_observable_dataset.csv",
        ),
    ),
    DatasetSpec(
        label="UNAUTHORIZED_TELECOMMAND_INJECTION",
        candidates=("unauthorized_telecommand_injection_observable_dataset.csv",),
    ),
    DatasetSpec(
        label="TELECOMMAND_REPLAY",
        candidates=("telecommand_replay_observable_dataset.csv",),
    ),
    DatasetSpec(
        label="GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE",
        candidates=("ground_segment_ttc_control_path_compromise_observable_dataset.csv",),
    ),
]


# ============================================================
# 2. Utility functions
# ============================================================

def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_dataset_path(spec: DatasetSpec) -> Path:
    existing_paths = [DATASET_DIR / name for name in spec.candidates if (DATASET_DIR / name).exists()]
    if len(existing_paths) == 1:
        return existing_paths[0]

    if len(existing_paths) > 1:
        raise FileExistsError(
            f"Multiple candidate files found for {spec.label}: "
            f"{[p.name for p in existing_paths]}. "
            "Remove obsolete duplicate files before building the unified dataset."
        )
    candidate_text = "\n  - ".join(spec.candidates)
    raise FileNotFoundError(
        f"Could not find learning-ready CSV for label={spec.label} in {DATASET_DIR}.\n"
        f"Expected one of:\n  - {candidate_text}\n"
        f"Do not use *_with_raw.csv or *_sample_episode_timeline.csv here."
    )


def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def fail_or_warn(message: str, strict: bool = True) -> None:
    if strict:
        raise ValueError(message)
    print(f"[Warning] {message}")


def parse_timestamp_series(series: pd.Series, source_name: str) -> pd.Series:
    """Parse timestamp strings using pandas while producing a clear error message."""
    try:
        parsed = pd.to_datetime(series, errors="raise")
    except Exception as exc:
        raise ValueError(f"timestamp must be parseable as datetime in {source_name}.") from exc

    if parsed.isnull().any():
        raise ValueError(f"Null/invalid timestamp values found in {source_name}.")
    return parsed


def validate_one_second_temporal_consistency(
    df: pd.DataFrame,
    source_name: str,
    episode_id_col: str = "episode_id",
) -> Dict[str, Any]:
    """Validate that each episode is a 1-second window sequence.

    This check is intentionally strict because the downstream TST experiment
    interprets L={1,2,3,5,10} as {1,2,3,5,10} seconds. If any maker still
    produces 10-second timestamps, this function fails before unified/split
    artifacts are created.
    """
    required = [episode_id_col, "episode_step", "window_id", "timestamp"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing temporal validation columns in {source_name}: {missing}")

    work = df[[episode_id_col, "episode_step", "window_id", "timestamp"]].copy()
    work["__timestamp_dt"] = parse_timestamp_series(work["timestamp"], source_name)

    invalid_step_episodes: List[str] = []
    invalid_window_episodes: List[str] = []
    invalid_timestamp_episodes: List[str] = []
    observed_timestamp_diffs: List[float] = []
    episode_lengths: List[int] = []
    episode_timestamp_spans: List[float] = []

    for episode_id, group in work.groupby(episode_id_col, sort=False):
        group_sorted = group.sort_values(["episode_step", "window_id"]).copy()
        episode_lengths.append(int(len(group_sorted)))

        step_diffs = group_sorted["episode_step"].diff().dropna()
        if len(step_diffs) > 0 and not (step_diffs == 1).all():
            invalid_step_episodes.append(str(episode_id))

        window_diffs = group_sorted["window_id"].diff().dropna()
        if len(window_diffs) > 0 and not (window_diffs == 1).all():
            invalid_window_episodes.append(str(episode_id))

        timestamp_diffs = group_sorted["__timestamp_dt"].diff().dropna().dt.total_seconds()
        if len(timestamp_diffs) > 0:
            observed_timestamp_diffs.extend([float(x) for x in timestamp_diffs.tolist()])
            if not np.allclose(timestamp_diffs.to_numpy(dtype=float), EXPECTED_WINDOW_SECONDS, atol=1e-9, rtol=0):
                invalid_timestamp_episodes.append(str(episode_id))

        if len(group_sorted) >= 2:
            timestamp_span = (
                group_sorted["__timestamp_dt"].iloc[-1] - group_sorted["__timestamp_dt"].iloc[0]
            ).total_seconds()
            episode_timestamp_spans.append(float(timestamp_span))
        else:
            episode_timestamp_spans.append(0.0)

    if invalid_step_episodes:
        raise ValueError(
            f"episode_step must increase by 1 within each episode in {source_name}. "
            f"Examples: {invalid_step_episodes[:10]}"
        )

    if invalid_window_episodes:
        raise ValueError(
            f"window_id must increase by 1 within each episode in {source_name}. "
            f"Examples: {invalid_window_episodes[:10]}"
        )

    if invalid_timestamp_episodes:
        unique_diffs = sorted(set(round(x, 9) for x in observed_timestamp_diffs))[:20]
        raise ValueError(
            f"timestamp difference must be {EXPECTED_WINDOW_SECONDS} second(s) within each episode in {source_name}. "
            f"Examples: {invalid_timestamp_episodes[:10]}; observed diffs sample={unique_diffs}"
        )

    if not episode_lengths:
        raise ValueError(f"No episodes found for temporal validation in {source_name}.")

    timestamp_diff_summary: Dict[str, Any]
    if observed_timestamp_diffs:
        diffs = np.array(observed_timestamp_diffs, dtype=float)
        timestamp_diff_summary = {
            "min_timestamp_diff_seconds": float(np.min(diffs)),
            "median_timestamp_diff_seconds": float(np.median(diffs)),
            "max_timestamp_diff_seconds": float(np.max(diffs)),
            "unique_timestamp_diff_seconds_sample": sorted(set(float(round(x, 9)) for x in diffs.tolist()))[:20],
        }
    else:
        timestamp_diff_summary = {
            "min_timestamp_diff_seconds": None,
            "median_timestamp_diff_seconds": None,
            "max_timestamp_diff_seconds": None,
            "unique_timestamp_diff_seconds_sample": [],
        }

    lengths = np.array(episode_lengths, dtype=float)
    spans = np.array(episode_timestamp_spans, dtype=float)

    report = {
        "window_seconds": EXPECTED_WINDOW_SECONDS,
        "episodes": int(len(episode_lengths)),
        "min_episode_length_rows": int(np.min(lengths)),
        "mean_episode_length_rows": float(np.mean(lengths)),
        "max_episode_length_rows": int(np.max(lengths)),
        "min_episode_duration_seconds": int(np.min(lengths) * EXPECTED_WINDOW_SECONDS),
        "mean_episode_duration_seconds": float(np.mean(lengths) * EXPECTED_WINDOW_SECONDS),
        "max_episode_duration_seconds": int(np.max(lengths) * EXPECTED_WINDOW_SECONDS),
        "min_timestamp_span_seconds": float(np.min(spans)),
        "mean_timestamp_span_seconds": float(np.mean(spans)),
        "max_timestamp_span_seconds": float(np.max(spans)),
        "timestamp_spacing_check_passed": True,
        "episode_step_monotonic_check_passed": True,
        "window_id_consistency_check_passed": True,
    }
    report.update(timestamp_diff_summary)
    return report


def validate_expected_total_rows(df: pd.DataFrame) -> None:
    if len(df) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Unified dataset row count mismatch: got {len(df):,}, expected {EXPECTED_TOTAL_ROWS:,}. "
            f"Expected {EXPECTED_ROWS_PER_CLASS:,} rows × {len(LABEL_ORDER)} labels."
        )


def build_tst_temporal_context_plan() -> Dict[str, Any]:
    return {
        "observable_window_seconds": EXPECTED_WINDOW_SECONDS,
        "tst_sequence_lengths": TST_SEQUENCE_LENGTHS,
        "tst_temporal_context_seconds": TST_TEMPORAL_CONTEXT_SECONDS,
        "interpretation": (
            "Each row is a 1-second cross-layer observable window. "
            "A TST sequence of length L therefore corresponds to L seconds."
        ),
        "distilbert_input_rule": "Serialize OBSERVABLE_FIELDS only. Metadata is retained only for split/order/evaluation.",
        "tst_sequence_rule": "Build episode-bounded sequences only; never cross episode or split boundaries.",
    }


def validate_no_forbidden_columns(df: pd.DataFrame, source_name: str) -> None:
    forbidden_prefixes = ("raw_", "baseline_", "audit_")
    forbidden_exact = {
        "raw_attack_phase",
        "attack_phase",
        "phase",
        "severity",
        "case_type",
        "training_role",
        "window_training_role",
        "semantic_inconsistency_type",
        "temporal_recurrence_type",
        "control_path_behavior_type",
        "label_id",
        "text",
        "serialized_text",
    }

    forbidden_cols = [
        col for col in df.columns
        if col.startswith(forbidden_prefixes) or col in forbidden_exact
    ]

    if forbidden_cols:
        raise ValueError(
            f"Forbidden leakage/raw columns found in {source_name}: {forbidden_cols}\n"
            f"Use only learning-ready observable CSV files, not raw-audit CSV files."
        )


def validate_schema(
    df: pd.DataFrame,
    expected_label: str,
    source_path: Path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    source_name = source_path.name
    validate_no_forbidden_columns(df, source_name)

    missing = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in CANONICAL_COLUMNS]

    if missing:
        raise ValueError(f"Missing required columns in {source_name}: {missing}")

    if extra:
        msg = (
            f"Unexpected extra columns found in {source_name}: {extra}\n"
            f"Expected exactly METADATA_FIELDS + OBSERVABLE_FIELDS."
        )
        fail_or_warn(msg, strict=STRICT_SCHEMA)

    # Reorder and keep canonical columns only.
    df_canonical = df[CANONICAL_COLUMNS].copy()

    null_counts = df_canonical[CANONICAL_COLUMNS].isnull().sum()
    if int(null_counts.sum()) > 0:
        null_report = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Null values found in {source_name}: {null_report}")

    label_values = sorted(df_canonical["label"].astype(str).unique().tolist())
    if label_values != [expected_label]:
        raise ValueError(
            f"Label mismatch in {source_name}. Expected only {expected_label}, found {label_values}."
        )

    if len(df_canonical) != EXPECTED_ROWS_PER_CLASS:
        msg = (
            f"Unexpected row count in {source_name}: {len(df_canonical)}. "
            f"Expected {EXPECTED_ROWS_PER_CLASS}."
        )
        fail_or_warn(msg, strict=STRICT_ROW_COUNT)

    # Never truncate fractional IDs into valid-looking temporal indices.
    for field in ("episode_step", "window_id"):
        try:
            values = pd.to_numeric(df_canonical[field], errors="raise")
            if (not np.isfinite(values).all() or (values < 0).any()
                    or (values >= 2**63).any() or (values % 1 != 0).any()):
                raise ValueError("non-integral, negative, non-finite, or out-of-range index")
            df_canonical[field] = values.astype(np.int64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"{field} must contain nonnegative int64 indices in {source_name}.") from exc

    blank_fields = [field for field in CANONICAL_COLUMNS
                    if df_canonical[field].astype(str).str.strip().eq("").any()]
    if blank_fields:
        raise ValueError(f"Blank values in {source_name}: {blank_fields}")
    identities = df_canonical.groupby("episode_id")[["scenario_id", "episode_type"]].nunique()
    if (identities > 1).any().any():
        raise ValueError(f"An episode mixes scenario_id or episode_type in {source_name}.")

    # Detect duplicate rows by metadata identity within one source.
    duplicate_meta = df_canonical.duplicated(
        subset=["scenario_id", "episode_id", "episode_step", "window_id"],
        keep=False,
    ).sum()
    if duplicate_meta > 0:
        raise ValueError(f"Duplicated metadata rows found in {source_name}: {duplicate_meta}")

    temporal_report = validate_one_second_temporal_consistency(
        df_canonical,
        source_name=source_name,
        episode_id_col="episode_id",
    )

    return df_canonical, {
        "source_file": source_name,
        "label": expected_label,
        "rows": int(len(df_canonical)),
        "episodes": int(df_canonical["episode_id"].nunique()),
        "window_seconds": EXPECTED_WINDOW_SECONDS,
        "tst_sequence_lengths": TST_SEQUENCE_LENGTHS,
        "tst_temporal_context_seconds": TST_TEMPORAL_CONTEXT_SECONDS,
        "temporal_validation": temporal_report,
        "columns": list(df_canonical.columns),
        "extra_columns": extra,
        "missing_columns": missing,
    }


def raw_audit_path(source_path: Path) -> Path:
    path = source_path.with_name(source_path.stem + "_with_raw.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing companion raw audit: {path}. Run the corresponding revised maker "
            "and keep its learning CSV and *_with_raw.csv together in --dataset-dir. "
            "Do not substitute a sample timeline or an older raw file."
        )
    return path


def load_observability_audit(
    learning: pd.DataFrame, source_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Check the new maker contract without adding privileged data to learning.

    Read only the needed audit columns in chunks, not the entire raw-physics
    table. Validate row identity and recompute the difference of all 22 fields.
    Normalization/physics remain the maker's responsibility; this is a handoff
    and counterfactual-annotation check, not a detector or fingerprint test.
    """
    path = raw_audit_path(source_path)
    baseline = ["baseline_observable_" + f for f in OBSERVABLE_FIELDS]
    binary = ["raw_observable_effect_present", "raw_persistent_effect_present",
              "raw_attack_active", "raw_attacks_enabled", "raw_paired_audit_background"]
    additional = binary + ["raw_observable_difference_count", "raw_observable_difference_fields",
                           "raw_background_stream"]
    required = CANONICAL_COLUMNS + baseline + additional
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
    missing = [field for field in required if field not in header]
    if missing:
        raise ValueError(
            f"{path.name} does not have the shared-nominal audit schema: {missing}. "
            "Regenerate this class with the revised maker and shared_nominal.py."
        )
    offset, pieces, streams = 0, [], set()
    with pd.read_csv(path, encoding="utf-8-sig", usecols=required,
                     dtype=str, keep_default_na=False, chunksize=10000) as reader:
        for chunk in reader:
            chunk = chunk.reset_index(drop=True)
            expected = learning.iloc[offset:offset + len(chunk)].reset_index(drop=True)
            if len(expected) != len(chunk) or not chunk[CANONICAL_COLUMNS].equals(expected.astype(str)):
                raise ValueError(f"Learning and raw audit rows differ at offset {offset}: {path.name}")
            if chunk[required].apply(lambda s: s.str.strip().eq("")).any().any():
                raise ValueError(f"Blank audit values in {path.name} at offset {offset}.")
            numeric = {}
            for field in binary + ["raw_observable_difference_count"]:
                values = pd.to_numeric(chunk[field], errors="raise")
                allowed_max = 22 if field == "raw_observable_difference_count" else 1
                if (not np.isfinite(values).all() or (values % 1 != 0).any()
                        or not values.between(0, allowed_max).all()):
                    raise ValueError(f"Invalid integer audit field {field}: {path.name}")
                numeric[field] = values.to_numpy(dtype=np.int64)
            if numeric["raw_paired_audit_background"].any():
                raise ValueError(
                    f"Paired audit backgrounds are not permitted in the main corpus: {path.name}. "
                    "Use the maker's default independent background streams."
                )
            expected_enabled = int(expected["label"].iloc[0] != "NORMAL")
            if not (numeric["raw_attacks_enabled"] == expected_enabled).all():
                raise ValueError(f"Unexpected attacks_enabled setting for this class: {path.name}")
            changes = chunk[OBSERVABLE_FIELDS].to_numpy() != chunk[baseline].to_numpy()
            counts = changes.sum(axis=1).astype(np.int64)
            fields = np.asarray(["|".join(np.asarray(OBSERVABLE_FIELDS)[row]) or "NONE" for row in changes])
            if (not np.array_equal(counts, numeric["raw_observable_difference_count"])
                    or not np.array_equal((counts > 0).astype(np.int64), numeric["raw_observable_effect_present"])
                    or not np.array_equal(fields, chunk["raw_observable_difference_fields"].to_numpy())):
                raise ValueError(f"Counterfactual observable difference/effect annotations disagree: {path.name}")
            if not expected_enabled and (counts.any() or numeric["raw_attack_active"].any()
                                         or numeric["raw_persistent_effect_present"].any()):
                raise ValueError(f"NORMAL has an attack-related counterfactual change: {path.name}")
            streams.update(chunk["raw_background_stream"].unique().tolist())
            part = expected[METADATA_FIELDS].copy()
            for target, source in zip(AUDIT_FIELDS, ["raw_observable_effect_present", "raw_observable_difference_count",
                                                   "raw_persistent_effect_present", "raw_attack_active"]):
                part[target] = numeric[source]
            pieces.append(part)
            offset += len(chunk)
    if offset != len(learning) or not pieces:
        raise ValueError(f"Learning and raw audit row counts differ: {path.name}")
    if len(streams) != 1:
        raise ValueError(f"Expected one independent background stream per maker file: {path.name}")
    audit = pd.concat(pieces, ignore_index=True)
    return audit, {
        "raw_audit_file": path.name, "rows": len(audit), "background_stream": next(iter(streams)),
        "observable_effect_windows": int(audit["audit_observable_effect_present"].sum()),
        "persistent_effect_windows": int(audit["audit_persistent_effect_present"].sum()),
        "paired_background": False, "scope": AUDIT_SCOPE,
    }


def align_observability_audit(df_with_split: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Transfer only diagnostic fields after an exact key/metadata match.

    The original episode identity is used before any collision rewrite. The
    output uses the same rewritten identity and ordering as learning splits.
    """
    keys = ["label", "scenario_id", "episode_id", "episode_step", "window_id"]
    if list(audit.columns) != METADATA_FIELDS + AUDIT_FIELDS or audit.duplicated(keys).any():
        raise ValueError("Invalid or duplicated raw observability audit identities.")
    original = df_with_split[METADATA_FIELDS].copy()
    original["episode_id"] = df_with_split["__original_episode_id"].astype(str)
    if len(original) != len(audit) or original.duplicated(keys).any():
        raise ValueError("Learning and audit identities differ or are duplicated.")
    indexed = audit.set_index(keys)
    target_index = pd.MultiIndex.from_frame(original[keys])
    if not target_index.isin(indexed.index).all():
        raise ValueError("A learning row is missing from the raw observability audit.")
    aligned = indexed.reindex(target_index).reset_index()
    if not aligned[METADATA_FIELDS].astype(str).equals(original.reset_index(drop=True).astype(str)):
        raise ValueError("Learning and audit metadata do not align exactly.")
    output = df_with_split[METADATA_FIELDS + ["__split"]].reset_index(drop=True).copy()
    output[AUDIT_FIELDS] = aligned[AUDIT_FIELDS].to_numpy(dtype=np.int64)
    return output


def add_internal_global_episode_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__original_episode_id"] = out["episode_id"].astype(str)
    out["__global_episode_id"] = out["label"].astype(str) + "::" + out["episode_id"].astype(str)
    return out


def rewrite_duplicate_episode_ids_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure output episode_id is globally unique if original episode_id collides across labels."""
    out = df.copy()

    collision_table = (
        out[["episode_id", "label"]]
        .drop_duplicates()
        .groupby("episode_id")["label"]
        .nunique()
    )
    colliding_episode_ids = collision_table[collision_table > 1].index.tolist()

    if not colliding_episode_ids:
        return out

    message = (
        f"Detected {len(colliding_episode_ids)} episode_id values shared by multiple labels. "
        f"Example: {colliding_episode_ids[:5]}"
    )

    if not REWRITE_DUPLICATE_EPISODE_IDS:
        raise ValueError(
            message + "\nSet REWRITE_DUPLICATE_EPISODE_IDS=True or regenerate maker outputs with unique prefixes."
        )

    print(f"[Warning] {message}")
    print("[Action] Rewriting episode_id to label::episode_id in unified/split CSVs to prevent split/grouping leakage.")
    out["episode_id"] = out["label"].astype(str) + "::" + out["__original_episode_id"].astype(str)
    out["__global_episode_id"] = out["episode_id"].astype(str)
    return out


def validate_episode_temporal_order(df: pd.DataFrame) -> Dict[str, Any]:
    duplicate_steps: List[str] = []

    for episode_id, group in df.groupby("__global_episode_id", sort=False):
        steps = group["episode_step"].tolist()
        if len(steps) != len(set(steps)):
            duplicate_steps.append(str(episode_id))

    if duplicate_steps:
        raise ValueError(
            f"Found duplicate episode_step values within episodes. Examples: {duplicate_steps[:10]}"
        )

    return validate_one_second_temporal_consistency(
        df,
        source_name="unified in-memory dataset",
        episode_id_col="__global_episode_id",
    )


def summarize_distribution(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_distribution = (
        df.groupby("label", observed=True)
        .agg(
            rows=("label", "size"),
            episodes=("__global_episode_id", "nunique"),
        )
        .reindex(LABEL_ORDER)
        .reset_index()
    )
    label_distribution["row_ratio"] = label_distribution["rows"] / label_distribution["rows"].sum()
    label_distribution["episode_ratio"] = label_distribution["episodes"] / label_distribution["episodes"].sum()
    label_distribution["total_observable_seconds"] = label_distribution["rows"] * EXPECTED_WINDOW_SECONDS

    episode_distribution = (
        df.groupby(["label", "__global_episode_id"], observed=True)
        .agg(
            rows=("label", "size"),
            min_episode_step=("episode_step", "min"),
            max_episode_step=("episode_step", "max"),
            min_window_id=("window_id", "min"),
            max_window_id=("window_id", "max"),
            start_timestamp=("timestamp", "min"),
            end_timestamp=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"__global_episode_id": "global_episode_id"})
    )
    episode_distribution["episode_duration_seconds"] = episode_distribution["rows"] * EXPECTED_WINDOW_SECONDS

    return label_distribution, episode_distribution


def build_observable_field_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for field in OBSERVABLE_FIELDS:
        counts = df[field].value_counts(dropna=False)
        total = int(counts.sum())
        for value, count in counts.items():
            rows.append(
                {
                    "field": field,
                    "value": value,
                    "count": int(count),
                    "ratio": float(count) / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


# ============================================================
# 3. Episode-level split
# ============================================================

def allocate_episode_splits_for_label(
    episode_ids: List[str],
    rng: np.random.Generator,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Dict[str, str]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-8):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be 1.0")

    ids = list(episode_ids)
    rng.shuffle(ids)
    n = len(ids)

    if n < 3:
        raise ValueError(
            f"At least 3 episodes per label are required for train/val/test split. Found {n}."
        )

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_train = max(1, n_train - 1)
        n_test = n - n_train - n_val

    if n_test < 1:
        n_val = max(1, n_val - 1)
        n_test = n - n_train - n_val

    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(f"Could not allocate split for {n} episodes.")

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    split_map: Dict[str, str] = {}
    for eid in train_ids:
        split_map[eid] = "train"
    for eid in val_ids:
        split_map[eid] = "val"
    for eid in test_ids:
        split_map[eid] = "test"

    return split_map


def create_episode_level_split(df: pd.DataFrame, seed: int = DEFAULT_RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    split_map: Dict[str, str] = {}

    episode_table = (
        df.groupby(["label", "__global_episode_id"], observed=True)
        .agg(
            row_count=("label", "size"),
            original_episode_id=("__original_episode_id", "first"),
            min_episode_step=("episode_step", "min"),
            max_episode_step=("episode_step", "max"),
            min_window_id=("window_id", "min"),
            max_window_id=("window_id", "max"),
            start_timestamp=("timestamp", "min"),
            end_timestamp=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"__global_episode_id": "global_episode_id"})
    )

    for label in LABEL_ORDER:
        label_episode_ids = (
            episode_table.loc[episode_table["label"] == label, "global_episode_id"]
            .astype(str)
            .tolist()
        )
        if not label_episode_ids:
            raise ValueError(f"No episodes found for label={label}")
        label_split_map = allocate_episode_splits_for_label(label_episode_ids, rng)
        split_map.update(label_split_map)

    out = df.copy()
    out["__split"] = out["__global_episode_id"].map(split_map)

    if out["__split"].isnull().any():
        missing_count = int(out["__split"].isnull().sum())
        raise ValueError(f"Split assignment failed for {missing_count} rows.")

    episode_table["split"] = episode_table["global_episode_id"].map(split_map)
    if episode_table["split"].isnull().any():
        raise ValueError("Split assignment failed for one or more episodes.")

    episode_table["observable_window_seconds"] = EXPECTED_WINDOW_SECONDS
    episode_table["episode_duration_seconds"] = episode_table["row_count"] * EXPECTED_WINDOW_SECONDS
    episode_table["tst_context_lengths_available"] = episode_table["row_count"].map(
        lambda n: ",".join(str(length) for length in TST_SEQUENCE_LENGTHS if length <= n)
    )

    return out, episode_table


def validate_split_integrity(df_with_split: pd.DataFrame, episode_manifest: pd.DataFrame) -> None:
    valid_splits = {"train", "val", "test"}
    found_splits = set(df_with_split["__split"].unique().tolist())
    if found_splits != valid_splits:
        raise ValueError(f"Expected splits {valid_splits}, found {found_splits}")

    # No episode must appear in more than one split.
    episode_split_counts = (
        df_with_split[["__global_episode_id", "__split"]]
        .drop_duplicates()
        .groupby("__global_episode_id")["__split"]
        .nunique()
    )
    leaking_episodes = episode_split_counts[episode_split_counts > 1]
    if len(leaking_episodes) > 0:
        raise ValueError(
            f"Episode leakage detected across splits. Count={len(leaking_episodes)}. "
            f"Examples={leaking_episodes.index[:5].tolist()}"
        )

    # Every label must exist in every split.
    label_split_table = pd.crosstab(df_with_split["label"], df_with_split["__split"])
    for label in LABEL_ORDER:
        if label not in label_split_table.index:
            raise ValueError(f"Missing label in split table: {label}")
        for split in ["train", "val", "test"]:
            if label_split_table.loc[label, split] <= 0:
                raise ValueError(f"Label={label} has zero rows in split={split}")

    # Every split must have full canonical schema in output.
    for split in ["train", "val", "test"]:
        split_df = df_with_split.loc[df_with_split["__split"] == split, CANONICAL_COLUMNS]
        if list(split_df.columns) != CANONICAL_COLUMNS:
            raise ValueError(f"Canonical column mismatch in split={split}")

    # Manifest and dataframe must agree on episode count.
    manifest_episode_count = episode_manifest["global_episode_id"].nunique()
    df_episode_count = df_with_split["__global_episode_id"].nunique()
    if manifest_episode_count != df_episode_count:
        raise ValueError(
            f"Manifest episode count mismatch: manifest={manifest_episode_count}, df={df_episode_count}"
        )

    # Compare assignments and episode sizes, not only the number of unique IDs.
    if episode_manifest["global_episode_id"].duplicated().any():
        raise ValueError("Duplicate episode records in the split manifest.")
    actual = df_with_split.groupby("__global_episode_id").agg(
        split=("__split", "first"), label=("label", "first"), row_count=("label", "size")
    ).sort_index()
    recorded = episode_manifest.set_index("global_episode_id")[actual.columns].sort_index()
    if not actual.index.equals(recorded.index) or not actual.equals(recorded):
        raise ValueError("Split manifest differs from actual episode assignments/labels/row counts.")

    # Split files must preserve the 1-second temporal interpretation.
    for split in ["train", "val", "test"]:
        split_df = df_with_split.loc[df_with_split["__split"] == split].copy()
        validate_one_second_temporal_consistency(
            split_df,
            source_name=f"{split} split",
            episode_id_col="__global_episode_id",
        )


def build_split_summary(df_with_split: pd.DataFrame, episode_manifest: pd.DataFrame) -> pd.DataFrame:
    row_summary = (
        df_with_split.groupby(["__split", "label"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
        .rename(columns={"__split": "split"})
    )

    episode_summary = (
        episode_manifest.groupby(["split", "label"], observed=True)
        .size()
        .rename("episodes")
        .reset_index()
    )

    summary = pd.merge(row_summary, episode_summary, on=["split", "label"], how="outer")
    summary["rows"] = summary["rows"].fillna(0).astype(int)
    summary["episodes"] = summary["episodes"].fillna(0).astype(int)

    split_totals = summary.groupby("split")["rows"].transform("sum")
    summary["row_ratio_within_split"] = summary["rows"] / split_totals
    summary["observable_window_seconds"] = EXPECTED_WINDOW_SECONDS
    summary["total_observable_seconds"] = summary["rows"] * EXPECTED_WINDOW_SECONDS

    # Stable ordering.
    split_order = {"train": 0, "val": 1, "test": 2}
    label_order = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    summary["_split_order"] = summary["split"].map(split_order)
    summary["_label_order"] = summary["label"].map(label_order)
    summary = summary.sort_values(["_split_order", "_label_order"]).drop(columns=["_split_order", "_label_order"])
    return summary


# ============================================================
# 4. Save outputs
# ============================================================

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def save_learning_csv(path: Path, frame: pd.DataFrame) -> None:
    """Read back saved values: successful to_csv/hash alone does not prove completeness."""
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\r\n")
    offset = 0
    for chunk in pd.read_csv(path, encoding="utf-8-sig", dtype=str,
                             keep_default_na=False, chunksize=10000):
        expected = frame.iloc[offset:offset + len(chunk)].astype(str).reset_index(drop=True)
        if not chunk.reset_index(drop=True).equals(expected):
            raise ValueError(f"Saved CSV differs from in-memory values at row {offset}: {path}")
        offset += len(chunk)
    if offset != len(frame):
        raise ValueError(f"Incomplete saved CSV: {path}; got {offset} rows, expected {len(frame)}.")


def _write_outputs(
    df_with_split: pd.DataFrame,
    episode_manifest: pd.DataFrame,
    schema_reports: List[Dict[str, Any]],
    observability_audit: pd.DataFrame,
) -> None:
    # Canonical unified dataset only: METADATA_FIELDS + OBSERVABLE_FIELDS.
    unified_df = (
        df_with_split[CANONICAL_COLUMNS]
        .sort_values(["label", "episode_id", "episode_step", "window_id"])
        .reset_index(drop=True)
    )
    save_learning_csv(UNIFIED_OUTPUT_PATH, unified_df)

    for split_name, output_path in [
        ("train", TRAIN_OUTPUT_PATH),
        ("val", VAL_OUTPUT_PATH),
        ("test", TEST_OUTPUT_PATH),
    ]:
        split_df = (
            df_with_split.loc[df_with_split["__split"] == split_name, CANONICAL_COLUMNS]
            .sort_values(["label", "episode_id", "episode_step", "window_id"])
            .reset_index(drop=True)
        )
        save_learning_csv(output_path, split_df)
        audit = (observability_audit.loc[observability_audit["__split"] == split_name,
                                         METADATA_FIELDS + AUDIT_FIELDS]
                 .sort_values(["label", "episode_id", "episode_step", "window_id"])
                 .reset_index(drop=True))
        if not split_df[METADATA_FIELDS].equals(audit[METADATA_FIELDS]):
            raise ValueError(f"{split_name}: diagnostic and learning rows are not aligned.")
        audit_path = OUTPUT_DIR / observability_audit_contract()["split_files"][split_name]
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        save_learning_csv(audit_path, audit)

    episode_manifest = episode_manifest.sort_values(["split", "label", "global_episode_id"]).reset_index(drop=True)
    episode_manifest.to_csv(SPLIT_MANIFEST_PATH, index=False, encoding="utf-8-sig", lineterminator="\r\n")

    split_summary = build_split_summary(df_with_split, episode_manifest)
    split_summary.to_csv(SPLIT_SUMMARY_PATH, index=False, encoding="utf-8-sig", lineterminator="\r\n")

    label_distribution, episode_distribution = summarize_distribution(df_with_split)
    label_distribution.to_csv(LABEL_DISTRIBUTION_PATH, index=False, encoding="utf-8-sig", lineterminator="\r\n")
    episode_distribution.to_csv(EPISODE_DISTRIBUTION_PATH, index=False, encoding="utf-8-sig", lineterminator="\r\n")

    field_value_summary = build_observable_field_value_summary(df_with_split)
    field_value_summary.to_csv(FIELD_VALUE_SUMMARY_PATH, index=False, encoding="utf-8-sig", lineterminator="\r\n")

    label_mapping = {
        "label_order": LABEL_ORDER,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
        "metadata_fields": METADATA_FIELDS,
        "observable_fields": OBSERVABLE_FIELDS,
        "observable_window_seconds": EXPECTED_WINDOW_SECONDS,
        "tst_sequence_lengths": TST_SEQUENCE_LENGTHS,
        "tst_temporal_context_seconds": TST_TEMPORAL_CONTEXT_SECONDS,
        "distilbert_input_rule": "Serialize OBSERVABLE_FIELDS only. Never serialize metadata/raw/split columns.",
        "tst_sequence_rule": "Use episode-bounded sequence construction only. L maps directly to seconds because window_seconds=1.",
    }
    save_json(LABEL_MAPPING_PATH, label_mapping)
    save_json(TEMPORAL_CONTEXT_PLAN_PATH, build_tst_temporal_context_plan())

    schema_report = {
        "dataset_dir": str(DATASET_DIR),
        "unified_output_path": str(UNIFIED_OUTPUT_PATH),
        "train_output_path": str(TRAIN_OUTPUT_PATH),
        "val_output_path": str(VAL_OUTPUT_PATH),
        "test_output_path": str(TEST_OUTPUT_PATH),
        "random_seed": DEFAULT_RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "expected_rows_per_class": EXPECTED_ROWS_PER_CLASS,
        "expected_total_rows": EXPECTED_TOTAL_ROWS,
        "observable_window_seconds": EXPECTED_WINDOW_SECONDS,
        "tst_sequence_lengths": TST_SEQUENCE_LENGTHS,
        "tst_temporal_context_seconds": TST_TEMPORAL_CONTEXT_SECONDS,
        "canonical_columns": CANONICAL_COLUMNS,
        "metadata_fields": METADATA_FIELDS,
        "observable_fields": OBSERVABLE_FIELDS,
        "source_schema_reports": schema_reports,
        "observability_audit": observability_audit_contract(),
    }
    save_json(SCHEMA_REPORT_PATH, schema_report)


def configure_output_paths(root: Path) -> None:
    """Keep existing output names/layout within a separate dataset version."""
    global OUTPUT_DIR, SPLIT_DIR, REPORT_DIR
    OUTPUT_DIR = Path(root)
    SPLIT_DIR, REPORT_DIR = OUTPUT_DIR / "splits", OUTPUT_DIR / "reports"
    paths = {
        "UNIFIED_OUTPUT_PATH": "unified_observable_dataset.csv",
        "TRAIN_OUTPUT_PATH": "splits/train_observable.csv",
        "VAL_OUTPUT_PATH": "splits/val_observable.csv",
        "TEST_OUTPUT_PATH": "splits/test_observable.csv",
        "LABEL_MAPPING_PATH": "splits/label_mapping.json",
        "SPLIT_MANIFEST_PATH": "splits/episode_split_manifest.csv",
        "SPLIT_SUMMARY_PATH": "splits/split_summary.csv",
        "LABEL_DISTRIBUTION_PATH": "reports/label_distribution.csv",
        "EPISODE_DISTRIBUTION_PATH": "reports/episode_distribution.csv",
        "FIELD_VALUE_SUMMARY_PATH": "reports/observable_field_value_summary.csv",
        "SCHEMA_REPORT_PATH": "reports/schema_validation_report.json",
        "TEMPORAL_CONTEXT_PLAN_PATH": "reports/tst_temporal_context_plan.json",
    }
    for name, relative in paths.items():
        globals()[name] = OUTPUT_DIR / relative


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_contract(source_paths: List[Path]) -> Dict[str, Any]:
    """Bind learning and matching raw audit files; do not infer physical validity."""
    policy = {
        "version": "step7_episode_split_v2",
        "nominal_substrate": "shared_nominal_v2",
        "observability_audit_schema": AUDIT_SCHEMA_VERSION,
        "paired_background_policy": "excluded from the main corpus; independent defaults required",
        "seed": DEFAULT_RANDOM_SEED,
        "ratios": [TRAIN_RATIO, VAL_RATIO, TEST_RATIO],
        "stratification": "class; whole episodes; original NumPy shuffle and rounding",
        "columns": CANONICAL_COLUMNS,
        "label_order": LABEL_ORDER,
        "window_seconds": EXPECTED_WINDOW_SECONDS,
        "sequence_lengths": TST_SEQUENCE_LENGTHS,
        "expected_rows_per_class": EXPECTED_ROWS_PER_CLASS,
        "label_policy": "retain supplied scenario/episode label on every window",
        "filtering": "none; retain quiet/unobserved windows and repeated observable vectors",
    }
    if len(source_paths) != len(DATASET_SPECS):
        raise ValueError("Exactly six learning sources are required.")
    sources = []
    for spec, path in zip(DATASET_SPECS, source_paths):
        raw_path = raw_audit_path(path)
        sources.append({"label": spec.label, "file": path.name,
                        "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                        "raw_audit": {"file": raw_path.name, "sha256": sha256_file(raw_path),
                                      "size_bytes": raw_path.stat().st_size}})
    identity = {"policy": policy, "sources": sources}
    dataset_id = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                          allow_nan=False).encode("utf-8")).hexdigest()
    return {**identity, "dataset_id": dataset_id,
            "builder_sha256": sha256_file(Path(__file__)),
            "source_scope": "Learning/raw alignment and nominal counterfactual projection annotations; maker physics is not re-audited",
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}


def verify_existing_build(root: Path, contract: Dict[str, Any]) -> bool:
    """Reuse only a complete, byte-verified build; never overwrite other data."""
    if not root.exists():
        return False
    manifest_path = root / "build_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"Output already exists without a completed build: {root}. Use --output-dir NEW_PATH.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete"
            or manifest.get("dataset_id") != contract.get("dataset_id")
            or manifest.get("builder_sha256") != contract.get("builder_sha256")):
        raise FileExistsError(f"Output belongs to different inputs/policy/builder: {root}. Use --output-dir NEW_PATH.")
    artifacts = manifest.get("artifacts", {})
    required = {"unified_observable_dataset.csv", "splits/train_observable.csv",
                "splits/val_observable.csv", "splits/test_observable.csv",
                "splits/label_mapping.json", "splits/episode_split_manifest.csv",
                "splits/split_summary.csv", "reports/label_distribution.csv",
                "reports/episode_distribution.csv", "reports/observable_field_value_summary.csv",
                "reports/schema_validation_report.json", "reports/tst_temporal_context_plan.json"}
    required.update(observability_audit_contract()["split_files"].values())
    if set(artifacts) != required:
        raise ValueError("Completed build manifest has missing/unexpected artifacts.")
    if manifest.get("observability_audit") != observability_audit_contract():
        raise ValueError("Completed build has a different observability audit contract.")
    for name, expected in artifacts.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Completed build artifact is missing/changed: {path}. No overwrite performed.")
    return True


def save_outputs(
    df_with_split: pd.DataFrame,
    episode_manifest: pd.DataFrame,
    schema_reports: List[Dict[str, Any]],
    contract: Dict[str, Any] | None = None,
    observability_audit: pd.DataFrame | None = None,
) -> None:
    """Stage the complete bundle and publish a new directory only after success."""
    target = OUTPUT_DIR.resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {target}")
    if contract is None:
        contract = build_contract([resolve_dataset_path(spec) for spec in DATASET_SPECS])
    if observability_audit is None:
        raise ValueError("Matching shared-nominal audit rows are required before saving this dataset.")
    aligned_audit = align_observability_audit(df_with_split, observability_audit)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent))
    try:
        configure_output_paths(stage)
        ensure_directories()
        _write_outputs(df_with_split, episode_manifest, schema_reports, aligned_audit)
        # Reports describe the published paths, not temporary staging paths.
        report = json.loads(SCHEMA_REPORT_PATH.read_text(encoding="utf-8"))
        for key in ("unified_output_path", "train_output_path", "val_output_path", "test_output_path"):
            report[key] = str(target / Path(report[key]).relative_to(stage))
        report["dataset_id"] = contract["dataset_id"]
        save_json(SCHEMA_REPORT_PATH, report)
        # Fail if an input was replaced while it was being read/processed.
        for source in contract["sources"]:
            if sha256_file(DATASET_DIR / source["file"]) != source["sha256"]:
                raise ValueError(f"Input changed during build: {source['file']}")
            raw_source = source["raw_audit"]
            if sha256_file(DATASET_DIR / raw_source["file"]) != raw_source["sha256"]:
                raise ValueError(f"Raw audit changed during build: {raw_source['file']}")
        artifacts = {path.relative_to(stage).as_posix(): sha256_file(path)
                     for path in sorted(stage.rglob("*")) if path.is_file()}
        counts = {}
        for split in ("train", "val", "test"):
            lengths = episode_manifest.loc[episode_manifest["split"] == split, "row_count"]
            counts[split] = {"rows": int(lengths.sum()), "episodes": int(len(lengths)),
                             "sequences_stride1": {str(length): int((lengths - length + 1).clip(lower=0).sum())
                                                    for length in TST_SEQUENCE_LENGTHS}}
        save_json(stage / "build_manifest.json", {
            **contract, "status": "complete", "counts": counts, "artifacts": artifacts,
            "model_input_fields": OBSERVABLE_FIELDS,
            "metadata_not_model_features": METADATA_FIELDS,
            "observability_audit": observability_audit_contract(),
            "downstream_requirement": "Explicitly select this dataset_id and split files; old trained models retain their old data provenance.",
            "external_validity": "Not assessed by dataset merging or same-generator fresh realizations.",
        })
        if target.exists():
            raise FileExistsError(f"Output was created by another process: {target}")
        stage.rename(target)
    finally:
        configure_output_paths(target)
        if stage.exists():
            shutil.rmtree(stage)


def write_review_zip(root: Path, contract: Dict[str, Any]) -> Path:
    """Small review package; excludes learning CSVs, raw data, and model weights."""
    output = root.parent / f"07_UNIFIED_REVIEW_{contract['dataset_id'][:16]}.zip"
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip.tmp", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(Path(__file__), "source/7.build_unified_dataset.py")
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.name.endswith("_observable.csv") and path.name != "unified_observable_dataset.csv":
                    archive.write(path, path.relative_to(root).as_posix())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def print_summary(df_with_split: pd.DataFrame, episode_manifest: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("Unified observable dataset build completed")
    print("=" * 72)
    print(f"Rows: {len(df_with_split):,}")
    print(f"Expected rows: {EXPECTED_TOTAL_ROWS:,}")
    print(f"Observable window seconds: {EXPECTED_WINDOW_SECONDS}")
    print(f"TST sequence lengths: {TST_SEQUENCE_LENGTHS}")
    print(f"TST temporal contexts seconds: {TST_TEMPORAL_CONTEXT_SECONDS}")
    print(f"Episodes: {df_with_split['__global_episode_id'].nunique():,}")
    print(f"Observable fields: {len(OBSERVABLE_FIELDS)}")
    print(f"Canonical output columns: {len(CANONICAL_COLUMNS)}")

    print("\n[Label distribution: rows]")
    print(df_with_split["label"].value_counts().reindex(LABEL_ORDER))

    print("\n[Label distribution: episodes]")
    print(
        episode_manifest.groupby("label")["global_episode_id"]
        .nunique()
        .reindex(LABEL_ORDER)
    )

    print("\n[Split row distribution]")
    print(pd.crosstab(df_with_split["label"], df_with_split["__split"]).reindex(LABEL_ORDER))

    print("\n[Split episode distribution]")
    print(pd.crosstab(episode_manifest["label"], episode_manifest["split"]).reindex(LABEL_ORDER))

    print("\n[Saved files]")
    for path in [
        UNIFIED_OUTPUT_PATH,
        TRAIN_OUTPUT_PATH,
        VAL_OUTPUT_PATH,
        TEST_OUTPUT_PATH,
        LABEL_MAPPING_PATH,
        SPLIT_MANIFEST_PATH,
        SPLIT_SUMMARY_PATH,
        LABEL_DISTRIBUTION_PATH,
        EPISODE_DISTRIBUTION_PATH,
        FIELD_VALUE_SUMMARY_PATH,
        SCHEMA_REPORT_PATH,
        TEMPORAL_CONTEXT_PLAN_PATH,
    ]:
        print(f"- {path.resolve()}")

    print("\n[Next step]")
    print(f"New dataset root: {OUTPUT_DIR.resolve()}")
    print("Downstream training/evaluation must explicitly select this dataset version.")
    print("Run the revised Step 8 -> 9 -> 10 -> 11 on this new dataset.")
    print("Review the updated Step 41 audit before committing to long training runs.")
    print("Audit-only counterfactual annotations are stored separately under audit/.")
    print("Serialize OBSERVABLE_FIELDS only; metadata is not a model feature.")


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    global DATASET_DIR
    parser = argparse.ArgumentParser(description="Step 7: build a versioned observable dataset; no training.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR,
                        help="Folder containing the six already-generated learning CSV files")
    parser.add_argument("--output-dir", type=Path,
                        help="New output folder (default: INPUT/shared_nominal_v2)")
    args = parser.parse_args()
    DATASET_DIR = args.dataset_dir.resolve()
    configure_output_paths((args.output_dir or DATASET_DIR / "shared_nominal_v2").resolve())
    if OUTPUT_DIR == DATASET_DIR or OUTPUT_DIR in DATASET_DIR.parents:
        raise ValueError("Output must be a separate folder; it cannot replace the input dataset folder.")
    source_paths = [resolve_dataset_path(spec) for spec in DATASET_SPECS]
    contract = build_contract(source_paths)
    if verify_existing_build(OUTPUT_DIR, contract):
        print(f"[Reuse] Same inputs, split policy and builder; all output hashes verified: {OUTPUT_DIR}")
        print(f"[Review ZIP] {write_review_zip(OUTPUT_DIR, contract)}")
        return

    print("=" * 72)
    print("Step 7: Build unified observable dataset and episode-level splits")
    print("=" * 72)
    print(f"Base directory: {BASE_DIR.resolve()}")
    print(f"Dataset directory: {DATASET_DIR.resolve()}")
    print(f"Expected rows per class: {EXPECTED_ROWS_PER_CLASS:,}")
    print(f"Expected total rows: {EXPECTED_TOTAL_ROWS:,}")
    print(f"Observable window seconds: {EXPECTED_WINDOW_SECONDS}")
    print(f"TST sequence lengths: {TST_SEQUENCE_LENGTHS}")
    print(f"TST temporal contexts seconds: {TST_TEMPORAL_CONTEXT_SECONDS}")

    loaded_frames: List[pd.DataFrame] = []
    loaded_audits: List[pd.DataFrame] = []
    schema_reports: List[Dict[str, Any]] = []
    background_stream_labels: Dict[str, str] = {}

    for spec, path in zip(DATASET_SPECS, source_paths):
        print(f"\n[Load] {spec.label}: {path.name}")
        df = read_csv_robust(path)
        df_canonical, report = validate_schema(
            df,
            expected_label=spec.label,
            source_path=path,
        )
        audit, audit_report = load_observability_audit(df_canonical, path)
        stream = audit_report["background_stream"]
        if stream in background_stream_labels:
            raise ValueError(
                f"Repeated nominal background stream in {spec.label} and {background_stream_labels[stream]}. "
                "Paired/replicated audit backgrounds must not enter independently split main data."
            )
        background_stream_labels[stream] = spec.label
        loaded_audits.append(audit)
        report["observability_audit"] = audit_report
        schema_reports.append(report)
        loaded_frames.append(df_canonical)

        temporal = report["temporal_validation"]
        print(
            f"  rows={len(df_canonical):,}, episodes={df_canonical['episode_id'].nunique():,}, "
            f"mean_episode_duration_s={temporal['mean_episode_duration_seconds']:.2f}"
        )

    df_all = pd.concat(loaded_frames, axis=0, ignore_index=True)
    validate_expected_total_rows(df_all)
    df_all = add_internal_global_episode_id(df_all)
    df_all = rewrite_duplicate_episode_ids_if_needed(df_all)

    # add_internal_global_episode_id/rewrite_duplicate_episode_ids_if_needed
    # already maintain the global ID; never infer its format from a substring.
    unified_temporal_report = validate_episode_temporal_order(df_all)
    print("\n[Temporal validation] Unified 1-second window consistency passed.")
    print(f"  mean episode length rows: {unified_temporal_report['mean_episode_length_rows']:.2f}")
    print(f"  mean episode duration seconds: {unified_temporal_report['mean_episode_duration_seconds']:.2f}")

    # Verify all labels are known and ordered.
    unknown_labels = sorted(set(df_all["label"].unique()) - set(LABEL_ORDER))
    if unknown_labels:
        raise ValueError(f"Unknown labels found: {unknown_labels}")

    missing_labels = [label for label in LABEL_ORDER if label not in set(df_all["label"].unique())]
    if missing_labels:
        raise ValueError(f"Missing labels: {missing_labels}")

    df_with_split, episode_manifest = create_episode_level_split(df_all, seed=DEFAULT_RANDOM_SEED)
    validate_split_integrity(df_with_split, episode_manifest)
    save_outputs(df_with_split, episode_manifest, schema_reports, contract,
                 observability_audit=pd.concat(loaded_audits, ignore_index=True))
    print_summary(df_with_split, episode_manifest)
    print(f"[Dataset ID] {contract['dataset_id']}")
    print(f"[Review ZIP] {write_review_zip(OUTPUT_DIR, contract)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[ERROR] Unified dataset build failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
