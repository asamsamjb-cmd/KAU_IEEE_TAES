# 8. train_distilbert_window_baseline.py
# DistilBERT single-window baseline for the 1-second 22-field observable dataset.
#
# Pipeline position:
#   Step 7 produced 1-second episode-level train/val/test splits.
#   This script performs:
#     1) Load train/val/test observable CSVs.
#     2) Validate that only METADATA_FIELDS + 22 OBSERVABLE_FIELDS are present.
#     3) Serialize OBSERVABLE_FIELDS only into schema-preserving text.
#     4) Train a DistilBERT single-window 6-class baseline for 1-second observable records.
#     5) Save tokenizer/model/label mapping/reports for downstream Step 9 embedding extraction.
#
# Temporal interpretation:
#   One row = one 1-second cross-layer observable window.
#   This Step 8 baseline uses L=1 only. It does not perform TST learning.
#   Downstream TST experiments use L in {1,2,3,5,10}, corresponding to
#   temporal contexts of {1,2,3,5,10} seconds.
#
# Critical rule:
#   The DistilBERT text input must be generated from OBSERVABLE_FIELDS only.
#   Never serialize scenario_id, episode_id, episode_step, window_id, timestamp, label,
#   split information, raw fields, attack phase, severity, case type, or training role.

from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import platform
import tempfile
import zipfile
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
import datasets as hf_datasets
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# ============================================================
# 0. Basic settings
# ============================================================

DEFAULT_RANDOM_SEED = 20260610

# Step 7 must have built splits from 1-second observable records.
EXPECTED_OBSERVABLE_WINDOW_SECONDS = 1
TST_SEQUENCE_LENGTHS: List[int] = [1, 2, 3, 5, 10]
TST_TEMPORAL_CONTEXT_SECONDS: List[int] = [
    EXPECTED_OBSERVABLE_WINDOW_SECONDS * length for length in TST_SEQUENCE_LENGTHS
]
EXPECTED_TOTAL_ROWS = 300000
EXPECTED_ROWS_PER_CLASS = 50000
STRICT_TEMPORAL_CONFIG = True
STRICT_TOTAL_ROW_COUNT = True

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "shared_nominal_v2"
SPLIT_DIR = DATASET_DIR / "splits"

# Keep the immutable Step 7 bundle and earlier A/A-prime models separate.
OUTPUT_ROOT = BASE_DIR / "outputs" / "shared_nominal_v2" / "distilbert_window_baseline"
SERIALIZED_DIR = OUTPUT_ROOT / "serialized"
MODEL_OUTPUT_DIR = OUTPUT_ROOT / "trained_distilbert_model"
REPORT_DIR = OUTPUT_ROOT / "reports"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"

TRAIN_INPUT_PATH = SPLIT_DIR / "train_observable.csv"
VAL_INPUT_PATH = SPLIT_DIR / "val_observable.csv"
TEST_INPUT_PATH = SPLIT_DIR / "test_observable.csv"
LABEL_MAPPING_INPUT_PATH = SPLIT_DIR / "label_mapping.json"

TRAIN_SERIALIZED_PATH = SERIALIZED_DIR / "train_serialized.csv"
VAL_SERIALIZED_PATH = SERIALIZED_DIR / "val_serialized.csv"
TEST_SERIALIZED_PATH = SERIALIZED_DIR / "test_serialized.csv"
SERIALIZATION_SAMPLE_PATH = SERIALIZED_DIR / "serialization_samples.csv"
SERIALIZATION_REPORT_PATH = SERIALIZED_DIR / "serialization_report.json"
SERIALIZATION_MANIFEST_PATH = SERIALIZED_DIR / "serialization_manifest.json"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
TOKEN_LENGTH_SAMPLE_LIMIT = None  # every serialized row is checked; no silent truncation
SERIALIZER_VERSION = "observable22_fixed_order_is_v1"
ACTIVE_RUN: Dict[str, Any] | None = None

# Training defaults. Adjust batch size downward if GPU memory is limited.
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 2

# For quick pilot/debug runs, set these to small integers such as 3000/1000/1000.
# For final experiments, keep them as None.
MAX_TRAIN_SAMPLES: int | None = None #None
MAX_VAL_SAMPLES: int | None = None #None
MAX_TEST_SAMPLES: int | None = None #None

# Keep True for reproducibility and inspection of actual DistilBERT input text.
SAVE_SERIALIZED_INPUTS = True

# Existing serialized CSV reuse is deliberately disabled; source splits are verified each run.
USE_EXISTING_SERIALIZED_IF_AVAILABLE = False


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

DEFAULT_LABEL_ORDER: List[str] = [
    "NORMAL",
    "RF_LINK_JAMMING",
    "GNSS_SPOOFING",
    "UNAUTHORIZED_TELECOMMAND_INJECTION",
    "TELECOMMAND_REPLAY",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE",
]

FORBIDDEN_INPUT_COLUMNS = {
    "scenario_id",
    "episode_id",
    "episode_type",
    "episode_step",
    "window_id",
    "timestamp",
    "label",
    "label_id",
    "split",
    "__split",
    "text",
    "serialized_text",
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
}


# ============================================================
# 2. Reproducibility and IO utilities
# ============================================================

def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


def ensure_directories() -> None:
    SERIALIZED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}\n"
            f"Run Step 7 build_unified_dataset first."
        )
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)



def load_step7_manifest() -> Dict[str, Any]:
    if not LABEL_MAPPING_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Step 7 label mapping not found: {LABEL_MAPPING_INPUT_PATH}\n"
            "Run the revised 1-second Step 7 builder before Step 8."
        )

    with open(LABEL_MAPPING_INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def load_label_mapping() -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, Any]]:
    data = load_step7_manifest()

    label_order = data.get("label_order")
    expected_mapping = dict(zip(DEFAULT_LABEL_ORDER, range(len(DEFAULT_LABEL_ORDER))))
    label_to_id = data.get("label_to_id")
    if label_order != DEFAULT_LABEL_ORDER or label_to_id != expected_mapping:
        raise ValueError("Step 7 must provide the canonical six-class label order and bijective IDs 0..5.")
    if any(type(value) is not int for value in label_to_id.values()):
        raise ValueError("Label IDs must be integers, not bool/float/string values.")
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    return label_order, label_to_id, id_to_label, data


def validate_step7_temporal_config(step7_manifest: Dict[str, Any]) -> None:
    """Verify that Step 8 is consuming the revised 1-second Step 7 outputs."""
    observable_window_seconds = step7_manifest.get("observable_window_seconds")
    sequence_lengths = step7_manifest.get("tst_sequence_lengths")
    context_seconds = step7_manifest.get("tst_temporal_context_seconds")

    errors: List[str] = []

    if observable_window_seconds != EXPECTED_OBSERVABLE_WINDOW_SECONDS:
        errors.append(
            "Step 7 observable_window_seconds mismatch: "
            f"expected {EXPECTED_OBSERVABLE_WINDOW_SECONDS}, found {observable_window_seconds}."
        )

    if sequence_lengths is not None and list(sequence_lengths) != TST_SEQUENCE_LENGTHS:
        errors.append(
            f"Step 7 tst_sequence_lengths mismatch: expected {TST_SEQUENCE_LENGTHS}, found {sequence_lengths}."
        )

    if context_seconds is not None and list(context_seconds) != TST_TEMPORAL_CONTEXT_SECONDS:
        errors.append(
            "Step 7 tst_temporal_context_seconds mismatch: "
            f"expected {TST_TEMPORAL_CONTEXT_SECONDS}, found {context_seconds}."
        )

    if list(step7_manifest.get("metadata_fields", [])) != METADATA_FIELDS:
        errors.append("Step 7 metadata_fields do not match this Step 8 code.")

    if list(step7_manifest.get("observable_fields", [])) != OBSERVABLE_FIELDS:
        errors.append("Step 7 observable_fields do not match this Step 8 code.")

    if errors:
        message = "\n".join(errors)
        if STRICT_TEMPORAL_CONFIG:
            raise ValueError(
                "Step 7 temporal/schema configuration is not compatible with the 1-second TST experiment.\n"
                + message
            )
        print("[Warning] " + message)

    print("\n[Step 7 temporal configuration]")
    print(f"  observable_window_seconds: {observable_window_seconds}")
    print(f"  tst_sequence_lengths     : {sequence_lengths}")
    print(f"  tst_temporal_context_sec : {context_seconds}")


# ============================================================
# 3. Dataset validation
# ============================================================

def validate_observable_dataframe(df: pd.DataFrame, split_name: str, label_order: List[str]) -> pd.DataFrame:
    missing = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"[{split_name}] Missing required columns: {missing}")

    extra = [col for col in df.columns if col not in CANONICAL_COLUMNS]
    if extra:
        raise ValueError(
            f"[{split_name}] Unexpected extra columns found: {extra}\n"
            f"Expected exactly METADATA_FIELDS + OBSERVABLE_FIELDS from Step 7."
        )

    forbidden_raw_cols = [col for col in df.columns if col.startswith("raw_")]
    if forbidden_raw_cols:
        raise ValueError(f"[{split_name}] Raw-audit columns found: {forbidden_raw_cols}")

    null_counts = df[CANONICAL_COLUMNS].isnull().sum()
    if int(null_counts.sum()) > 0:
        raise ValueError(f"[{split_name}] Null values found: {null_counts[null_counts > 0].to_dict()}")

    unknown_labels = sorted(set(df["label"].astype(str).unique()) - set(label_order))
    if unknown_labels:
        raise ValueError(f"[{split_name}] Unknown labels found: {unknown_labels}")

    out = df[CANONICAL_COLUMNS].copy()
    for field in ("episode_step", "window_id"):
        values = pd.to_numeric(out[field], errors="raise")
        if (not np.isfinite(values).all() or (values < 0).any()
                or (values >= 2**63).any() or (values % 1 != 0).any()):
            raise ValueError(f"[{split_name}] {field} must be a nonnegative int64 index.")
        out[field] = values.astype(np.int64)
    if any(out[field].astype(str).str.strip().eq("").any() for field in CANONICAL_COLUMNS):
        raise ValueError(f"[{split_name}] Blank metadata/observable value.")
    if out.duplicated(["episode_id", "episode_step"]).any():
        raise ValueError(f"[{split_name}] Duplicate episode/step key.")
    if (out.groupby("episode_id")[["label", "scenario_id", "episode_type"]].nunique() > 1).any().any():
        raise ValueError(f"[{split_name}] Mixed identity/label within an episode.")

    return out


def validate_one_second_temporal_consistency(
    df: pd.DataFrame,
    split_name: str,
    expected_window_seconds: int = EXPECTED_OBSERVABLE_WINDOW_SECONDS,
) -> Dict[str, Any]:
    """Validate that each episode is internally ordered as 1-second windows."""
    invalid_step_examples: List[str] = []
    invalid_window_examples: List[str] = []
    invalid_timestamp_examples: List[str] = []
    short_episode_counts = {str(length): 0 for length in TST_SEQUENCE_LENGTHS}

    parsed = df.copy()
    parsed["__timestamp_dt"] = pd.to_datetime(parsed["timestamp"], errors="coerce")
    if parsed["__timestamp_dt"].isnull().any():
        bad_count = int(parsed["__timestamp_dt"].isnull().sum())
        raise ValueError(f"[{split_name}] Found {bad_count} unparsable timestamps.")

    episode_lengths: List[int] = []

    for episode_id, group in parsed.sort_values(["episode_id", "episode_step", "window_id"]).groupby("episode_id", sort=False):
        episode_lengths.append(len(group))

        if group["episode_step"].duplicated().any():
            invalid_step_examples.append(str(episode_id))
            continue

        if len(group) > 1:
            step_diffs = group["episode_step"].diff().dropna().to_numpy()
            if not np.all(step_diffs == 1):
                invalid_step_examples.append(str(episode_id))

            window_diffs = group["window_id"].diff().dropna().to_numpy()
            if not np.all(window_diffs == 1):
                invalid_window_examples.append(str(episode_id))

            timestamp_diffs = (
                group["__timestamp_dt"].diff().dropna().dt.total_seconds().to_numpy()
            )
            if not np.all(timestamp_diffs == expected_window_seconds):
                invalid_timestamp_examples.append(str(episode_id))

        for length in TST_SEQUENCE_LENGTHS:
            if len(group) < length:
                short_episode_counts[str(length)] += 1

    if invalid_step_examples:
        raise ValueError(f"[{split_name}] Invalid episode_step continuity. Examples: {invalid_step_examples[:5]}")

    if invalid_window_examples:
        raise ValueError(f"[{split_name}] Invalid window_id continuity. Examples: {invalid_window_examples[:5]}")

    if invalid_timestamp_examples:
        raise ValueError(
            f"[{split_name}] Timestamp diff is not {expected_window_seconds} second(s). "
            f"Examples: {invalid_timestamp_examples[:5]}"
        )

    arr = np.array(episode_lengths, dtype=np.int32)
    return {
        "split": split_name,
        "rows": int(len(df)),
        "episodes": int(len(arr)),
        "window_seconds": int(expected_window_seconds),
        "min_episode_length_rows": int(arr.min()) if len(arr) else 0,
        "mean_episode_length_rows": float(arr.mean()) if len(arr) else 0.0,
        "max_episode_length_rows": int(arr.max()) if len(arr) else 0,
        "mean_episode_duration_seconds": float(arr.mean() * expected_window_seconds) if len(arr) else 0.0,
        "short_episode_counts_by_L": short_episode_counts,
    }


def validate_total_rows_and_class_balance(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_order: List[str],
) -> Dict[str, Any]:
    combined = pd.concat([train_df.assign(__split="train"), val_df.assign(__split="val"), test_df.assign(__split="test")], ignore_index=True)
    total_rows = int(len(combined))

    if STRICT_TOTAL_ROW_COUNT and total_rows != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Unexpected total row count after Step 7 splits: {total_rows}, expected {EXPECTED_TOTAL_ROWS}.")

    label_counts = combined["label"].value_counts().reindex(label_order).fillna(0).astype(int).to_dict()
    bad_labels = {
        label: count for label, count in label_counts.items()
        if count != EXPECTED_ROWS_PER_CLASS
    }
    if STRICT_TOTAL_ROW_COUNT and bad_labels:
        raise ValueError(
            f"Unexpected class row count after Step 7 splits: {bad_labels}. "
            f"Expected {EXPECTED_ROWS_PER_CLASS} rows per class."
        )

    return {
        "total_rows": total_rows,
        "expected_total_rows": EXPECTED_TOTAL_ROWS,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
    }


def validate_no_episode_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_eps = set(train_df["episode_id"].astype(str).unique())
    val_eps = set(val_df["episode_id"].astype(str).unique())
    test_eps = set(test_df["episode_id"].astype(str).unique())

    overlaps = {
        "train_val": train_eps & val_eps,
        "train_test": train_eps & test_eps,
        "val_test": val_eps & test_eps,
    }
    bad = {name: eps for name, eps in overlaps.items() if eps}
    if bad:
        example = {name: list(eps)[:5] for name, eps in bad.items()}
        raise ValueError(f"Episode leakage detected across splits: {example}")


def maybe_subsample(df: pd.DataFrame, max_samples: int | None, split_name: str, seed: int) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df
    sampled = (
        df.sample(n=max_samples, random_state=seed)
        .sort_values(["label", "episode_id", "episode_step", "window_id"])
        .reset_index(drop=True)
    )
    print(f"[Pilot mode] {split_name}: sampled {len(sampled):,} / {len(df):,} rows")
    return sampled


# ============================================================
# 4. Schema-preserving serializer
# ============================================================

def normalize_value_for_text(value: Any) -> str:
    """Return a stable string representation of an observable value."""
    if pd.isna(value):
        raise ValueError("Observable value is NaN; serialization is not allowed.")
    text = str(value).strip()
    if not text:
        raise ValueError("Observable value is empty; serialization is not allowed.")
    return text


def serialize_observable_row(row: pd.Series) -> str:
    """
    Serialize one observable record into a fixed-order sentence-like string.

    Critical constraints:
      - Use OBSERVABLE_FIELDS only.
      - Preserve field names and canonical values.
      - Use fixed order for reproducibility.
      - Do not include label or any metadata.
    """
    parts = []
    for field in OBSERVABLE_FIELDS:
        value = normalize_value_for_text(row[field])
        parts.append(f"{field} is {value}")
    return "; ".join(parts) + "."


def validate_serialized_text(text: str) -> None:
    if not text.endswith("."):
        raise ValueError("Serialized record must end with a period.")
    clauses = text[:-1].split("; ")
    if len(clauses) != len(OBSERVABLE_FIELDS):
        raise ValueError("Serialized record must contain exactly 22 observable clauses.")
    for field, clause in zip(OBSERVABLE_FIELDS, clauses):
        prefix = f"{field} is "
        if not clause.startswith(prefix) or not clause[len(prefix):].strip():
            raise ValueError(f"Serialized field missing, reordered, or empty: {field}")


def add_serialized_text(df: pd.DataFrame, label_to_id: Dict[str, int], split_name: str) -> pd.DataFrame:
    out = df.copy()
    out["label_id"] = out["label"].map(label_to_id)
    if out["label_id"].isnull().any():
        bad_labels = sorted(out.loc[out["label_id"].isnull(), "label"].unique().tolist())
        raise ValueError(f"[{split_name}] Failed to map label_id for labels: {bad_labels}")
    out["label_id"] = out["label_id"].astype(int)

    out["text"] = out.apply(serialize_observable_row, axis=1)

    for text in out["text"].unique():
        validate_serialized_text(text)

    return out


def validate_existing_serialization_manifest() -> None:
    if not SERIALIZATION_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Existing serialized CSVs were requested, but manifest is missing: {SERIALIZATION_MANIFEST_PATH}"
        )

    with open(SERIALIZATION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("observable_window_seconds") != EXPECTED_OBSERVABLE_WINDOW_SECONDS:
        raise ValueError(
            "Existing serialized CSVs do not match the 1-second experiment. "
            f"Found observable_window_seconds={manifest.get('observable_window_seconds')}."
        )

    if list(manifest.get("observable_fields", [])) != OBSERVABLE_FIELDS:
        raise ValueError("Existing serialized CSVs were built with a different observable field schema.")


def build_or_load_serialized_inputs(
    label_order: List[str],
    label_to_id: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if USE_EXISTING_SERIALIZED_IF_AVAILABLE:
        raise ValueError("Unverified legacy serialization reuse is disabled; keep USE_EXISTING_SERIALIZED_IF_AVAILABLE=False.")

    print("[Load] Observable split CSV files from revised 1-second Step 7")
    train_raw = validate_observable_dataframe(read_csv_robust(TRAIN_INPUT_PATH), "train", label_order)
    val_raw = validate_observable_dataframe(read_csv_robust(VAL_INPUT_PATH), "val", label_order)
    test_raw = validate_observable_dataframe(read_csv_robust(TEST_INPUT_PATH), "test", label_order)

    validate_no_episode_overlap(train_raw, val_raw, test_raw)
    if ACTIVE_RUN is not None:
        validate_frames_against_build({"train": train_raw, "val": val_raw, "test": test_raw}, ACTIVE_RUN["step7_build"])

    row_report = validate_total_rows_and_class_balance(train_raw, val_raw, test_raw, label_order)
    temporal_report = {
        "row_count_check": row_report,
        "splits": {
            "train": validate_one_second_temporal_consistency(train_raw, "train"),
            "val": validate_one_second_temporal_consistency(val_raw, "val"),
            "test": validate_one_second_temporal_consistency(test_raw, "test"),
        },
        "loaded_existing_serialized_inputs": False,
    }

    print("\n[1-second temporal consistency]")
    for split, info in temporal_report["splits"].items():
        print(
            f"  {split}: rows={info['rows']:,}, episodes={info['episodes']:,}, "
            f"mean_episode_duration={info['mean_episode_duration_seconds']:.2f}s, "
            f"short_L10={info['short_episode_counts_by_L'].get('10', 0)}"
        )

    train_raw = maybe_subsample(train_raw, MAX_TRAIN_SAMPLES, "train", DEFAULT_RANDOM_SEED)
    val_raw = maybe_subsample(val_raw, MAX_VAL_SAMPLES, "val", DEFAULT_RANDOM_SEED)
    test_raw = maybe_subsample(test_raw, MAX_TEST_SAMPLES, "test", DEFAULT_RANDOM_SEED)

    print("[Serialize] Building schema-preserving text from OBSERVABLE_FIELDS only")
    train_df = add_serialized_text(train_raw, label_to_id, "train")
    val_df = add_serialized_text(val_raw, label_to_id, "val")
    test_df = add_serialized_text(test_raw, label_to_id, "test")

    if SAVE_SERIALIZED_INPUTS:
        serialized_columns = METADATA_FIELDS + ["label_id", "text"]
        save_verified_csv(TRAIN_SERIALIZED_PATH, train_df[serialized_columns])
        save_verified_csv(VAL_SERIALIZED_PATH, val_df[serialized_columns])
        save_verified_csv(TEST_SERIALIZED_PATH, test_df[serialized_columns])

        samples = pd.concat(
            [
                train_df[serialized_columns].head(3).assign(split="train"),
                val_df[serialized_columns].head(3).assign(split="val"),
                test_df[serialized_columns].head(3).assign(split="test"),
            ],
            axis=0,
            ignore_index=True,
        )
        save_verified_csv(SERIALIZATION_SAMPLE_PATH, samples)

        serialization_manifest = {
            "dataset_id": ACTIVE_RUN["dataset_id"] if ACTIVE_RUN else None,
            "serializer_version": SERIALIZER_VERSION,
            "observable_window_seconds": EXPECTED_OBSERVABLE_WINDOW_SECONDS,
            "tst_sequence_lengths": TST_SEQUENCE_LENGTHS,
            "tst_temporal_context_seconds": TST_TEMPORAL_CONTEXT_SECONDS,
            "metadata_fields": METADATA_FIELDS,
            "observable_fields": OBSERVABLE_FIELDS,
            "serialized_columns": serialized_columns,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "source_split_files": {
                "train": str(TRAIN_INPUT_PATH),
                "val": str(VAL_INPUT_PATH),
                "test": str(TEST_INPUT_PATH),
            },
            "use_existing_serialized_if_available": USE_EXISTING_SERIALIZED_IF_AVAILABLE,
        }
        save_json(SERIALIZATION_MANIFEST_PATH, serialization_manifest)

    return train_df, val_df, test_df, temporal_report


# ============================================================
# 5. HuggingFace dataset and tokenization
# ============================================================

def report_token_lengths(tokenizer: AutoTokenizer, dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"max_length": MAX_LENGTH, "sampled": False, "splits": {}}
    for split_name, df in dfs.items():
        lengths, unknown_rows = [], 0
        texts = df["text"].tolist()
        for start in range(0, len(texts), 1024):
            tokens = tokenizer(texts[start:start + 1024], truncation=False, padding=False,
                               add_special_tokens=True, return_attention_mask=False)
            lengths.extend(len(ids) for ids in tokens["input_ids"])
            unknown_rows += sum(tokenizer.unk_token_id in ids for ids in tokens["input_ids"])
        arr = np.asarray(lengths, dtype=np.int32)
        if not len(arr):
            raise ValueError(f"Empty {split_name} input.")
        over_max = int((arr > MAX_LENGTH).sum())
        report["splits"][split_name] = {
            "rows": len(df), "length_rows_evaluated": len(arr), "sampled": False,
            "min_tokens": int(arr.min()), "mean_tokens": float(arr.mean()),
            "p95_tokens": float(np.percentile(arr, 95)), "p99_tokens": float(np.percentile(arr, 99)),
            "max_tokens": int(arr.max()), "rows_over_max_length_in_evaluated_subset": over_max,
            "rows_with_tokenizer_unk": int(unknown_rows),
        }
    save_json(SERIALIZATION_REPORT_PATH, report)
    if any(info["rows_over_max_length_in_evaluated_subset"] or info["rows_with_tokenizer_unk"]
           for info in report["splits"].values()):
        raise ValueError("Tokenization loses information (overlength or tokenizer [UNK]); see serialization_report.json. No automatic truncation or parameter adjustment.")
    return report


def make_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}), preserve_index=False)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def tokenize_function(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        result = tokenizer(batch["text"], truncation=False, padding=False)
        if any(len(ids) > MAX_LENGTH for ids in result["input_ids"]):
            raise ValueError("Input exceeds MAX_LENGTH; refusing silent token truncation.")
        # DistilBERT does not consume token_type_ids, including when a local
        # compatible WordPiece tokenizer happens to emit them.
        result.pop("token_type_ids", None)
        return result

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized


# ============================================================
# 6. Training utilities
# ============================================================

class LossLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        pieces = []
        if "loss" in logs:
            pieces.append(f"loss={logs['loss']:.4f}")
        if "learning_rate" in logs:
            pieces.append(f"lr={logs['learning_rate']:.2e}")
        if pieces:
            print(f"[Train] step={state.global_step} | " + " | ".join(pieces))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[override]
        if not metrics:
            return
        acc = metrics.get("eval_accuracy")
        macro = metrics.get("eval_macro_f1")
        weighted = metrics.get("eval_weighted_f1")
        if acc is not None:
            # TrainerState.epoch is None during the initial pre-training evaluation.
            epoch = 0.0 if state.epoch is None else state.epoch
            print(
                f"[Eval] epoch={epoch:.2f} | "
                f"accuracy={acc:.4f} | macro_f1={macro:.4f} | weighted_f1={weighted:.4f}"
            )


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(DEFAULT_LABEL_ORDER))), average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(DEFAULT_LABEL_ORDER))), average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
    }


def build_training_arguments() -> TrainingArguments:
    kwargs: Dict[str, Any] = {
        "output_dir": str(OUTPUT_ROOT / "checkpoints"),
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "logging_dir": str(OUTPUT_ROOT / "logs"),
        "logging_steps": LOGGING_STEPS,
        "save_strategy": "epoch",
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": DEFAULT_RANDOM_SEED,
        "data_seed": DEFAULT_RANDOM_SEED,
        "fp16": torch.cuda.is_available(),
    }

    # transformers renamed evaluation_strategy -> eval_strategy in newer versions.
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    # New releases accept fractional warmup_steps; older ones expose warmup_ratio.
    if "warmup_ratio" not in signature.parameters:
        kwargs.pop("warmup_ratio")
        kwargs["warmup_steps"] = WARMUP_RATIO
    if "logging_dir" not in signature.parameters:
        kwargs.pop("logging_dir")
    return TrainingArguments(**kwargs)


def build_trainer(
    model: DistilBertForSequenceClassification,
    training_args: TrainingArguments,
    tokenized_train: Dataset,
    tokenized_val: Dataset,
    tokenizer: AutoTokenizer,
    data_collator: DataCollatorWithPadding,
) -> Trainer:
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_val,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": [LossLoggerCallback(), CheckpointIntegrityCallback()],
    }

    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def save_prediction_reports(
    split_name: str,
    trainer: Trainer,
    tokenized_dataset: Dataset,
    original_df: pd.DataFrame,
    label_order: List[str],
    id_to_label: Dict[int, str],
) -> Dict[str, Any]:
    predictions = trainer.predict(tokenized_dataset)
    logits = predictions.predictions
    if isinstance(logits, tuple): logits = logits[0]
    labels = predictions.label_ids
    if (logits.shape != (len(original_df), len(label_order)) or not np.isfinite(logits).all()
            or not np.array_equal(labels, original_df["label_id"].to_numpy())):
        raise ValueError(f"{split_name}: prediction shape/values/label order differ from source rows.")
    preds = np.argmax(logits, axis=1)

    target_names = [id_to_label[i] for i in range(len(label_order))]
    report_dict = classification_report(
        labels,
        preds,
        labels=list(range(len(label_order))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        labels,
        preds,
        labels=list(range(len(label_order))),
        target_names=target_names,
        zero_division=0,
    )

    cm = confusion_matrix(labels, preds, labels=list(range(len(label_order))))

    report_path = REPORT_DIR / f"{split_name}_classification_report.json"
    report_txt_path = REPORT_DIR / f"{split_name}_classification_report.txt"
    cm_path = REPORT_DIR / f"{split_name}_confusion_matrix.csv"
    pred_path = PREDICTION_DIR / f"{split_name}_predictions.csv"

    save_json(report_path, report_dict)
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    save_verified_csv(cm_path, cm_df.rename_axis("true_label").reset_index())

    pred_df = original_df[METADATA_FIELDS].copy()
    pred_df["true_label_id"] = labels
    pred_df["pred_label_id"] = preds
    pred_df["true_label"] = [id_to_label[int(i)] for i in labels]
    pred_df["pred_label"] = [id_to_label[int(i)] for i in preds]
    pred_df["correct"] = pred_df["true_label_id"] == pred_df["pred_label_id"]

    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    for idx, label in enumerate(label_order):
        pred_df[f"prob_{label}"] = probabilities[:, idx]

    save_verified_csv(pred_path, pred_df)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm)
        ax.set_title(f"{split_name} confusion matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(target_names)))
        ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(target_names, fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / f"{split_name}_confusion_matrix.png", dpi=300)
        plt.close(fig)

    return {
        "split": split_name,
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=list(range(len(label_order))), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, labels=list(range(len(label_order))), average="weighted", zero_division=0)),
        "rows": int(len(labels)),
        "report_path": str(report_path),
        "confusion_matrix_path": str(cm_path),
        "predictions_path": str(pred_path),
    }


# ============================================================
# 7. Dataset/run identity and checkpoint integrity
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def configure_paths(dataset_dir: Path, output_root: Path) -> None:
    global DATASET_DIR, SPLIT_DIR, OUTPUT_ROOT, SERIALIZED_DIR
    global MODEL_OUTPUT_DIR, REPORT_DIR, PREDICTION_DIR
    DATASET_DIR, OUTPUT_ROOT = dataset_dir.resolve(), output_root.resolve()
    SPLIT_DIR = DATASET_DIR / "splits"
    SERIALIZED_DIR = OUTPUT_ROOT / "serialized"
    MODEL_OUTPUT_DIR, REPORT_DIR, PREDICTION_DIR = (
        OUTPUT_ROOT / "trained_distilbert_model", OUTPUT_ROOT / "reports", OUTPUT_ROOT / "predictions")
    for split in ("train", "val", "test"):
        globals()[f"{split.upper()}_INPUT_PATH"] = SPLIT_DIR / f"{split}_observable.csv"
        globals()[f"{split.upper()}_SERIALIZED_PATH"] = SERIALIZED_DIR / f"{split}_serialized.csv"
    globals()["LABEL_MAPPING_INPUT_PATH"] = SPLIT_DIR / "label_mapping.json"
    for name, filename in (("SERIALIZATION_SAMPLE_PATH", "serialization_samples.csv"),
                           ("SERIALIZATION_REPORT_PATH", "serialization_report.json"),
                           ("SERIALIZATION_MANIFEST_PATH", "serialization_manifest.json")):
        globals()[name] = SERIALIZED_DIR / filename


def load_build_contract() -> Dict[str, Any]:
    path = DATASET_DIR / "build_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Completed revised Step 7 build required: {path}")
    build = json.loads(path.read_text(encoding="utf-8"))
    if build.get("status") != "complete":
        raise ValueError("Step 7 bundle is not complete.")
    if build.get("dataset_id") != object_hash({"policy": build["policy"], "sources": build["sources"]}):
        raise ValueError("Step 7 dataset identity does not match its declared sources/policy.")
    if build["policy"].get("nominal_substrate") != "shared_nominal_v2":
        raise ValueError("Step 7 must use the shared_nominal_v2 nominal substrate; rebuild the dataset with the revised makers.")
    if (build["policy"].get("columns") != CANONICAL_COLUMNS
            or build["policy"].get("label_order") != DEFAULT_LABEL_ORDER
            or build["policy"].get("window_seconds") != 1):
        raise ValueError("Step 7 schema, label mapping, or window duration differs.")
    required = [f"splits/{split}_observable.csv" for split in ("train", "val", "test")]
    required += ["splits/label_mapping.json", "splits/episode_split_manifest.csv"]
    hashes = {}
    for name in required:
        item = DATASET_DIR / name
        expected = build.get("artifacts", {}).get(name)
        if not expected or not item.is_file() or sha256_file(item) != expected:
            raise ValueError(f"Step 7 input is missing/changed: {item}")
        hashes[name] = expected
    return {"dataset_id": build["dataset_id"], "step7_build": build,
            "step7_build_manifest_sha256": sha256_file(path), "input_sha256": hashes}


def validate_frames_against_build(frames: Dict[str, pd.DataFrame], build: Dict[str, Any]) -> None:
    records = []
    for split, df in frames.items():
        expected = build["counts"][split]
        if len(df) != expected["rows"] or df.episode_id.nunique() != expected["episodes"]:
            raise ValueError(f"{split} row/episode count differs from its Step 7 build.")
        part = df.groupby("episode_id").agg(label=("label", "first"), row_count=("label", "size"))
        part["split"] = split
        records.append(part.reset_index())
    actual = pd.concat(records, ignore_index=True).sort_values("episode_id").reset_index(drop=True)
    manifest = read_csv_robust(SPLIT_DIR / "episode_split_manifest.csv")
    # Step 7 retains original IDs unless collisions require label::ID rewriting.
    known_ids = set(actual.episode_id.astype(str))
    manifest["episode_id"] = manifest.apply(
        lambda row: str(row.original_episode_id) if str(row.original_episode_id) in known_ids
        else str(row.global_episode_id), axis=1)
    expected = manifest[["episode_id", "label", "row_count", "split"]].sort_values("episode_id").reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Loaded split rows differ from the Step 7 episode split manifest.")


def save_verified_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".csv.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\r\n")
        offset = 0
        for chunk in pd.read_csv(temporary, encoding="utf-8-sig", dtype=str,
                                 keep_default_na=False, chunksize=10000):
            expected = frame.iloc[offset:offset + len(chunk)].astype(str).reset_index(drop=True)
            if not chunk.reset_index(drop=True).equals(expected):
                raise ValueError(f"Saved CSV differs from source at row {offset}: {path}")
            offset += len(chunk)
        if offset != len(frame):
            raise ValueError(f"Incomplete CSV: {path}; {offset} != {len(frame)} rows.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_hashes(root: Path) -> Dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def tokenizer_fingerprint(tokenizer) -> str:
    # Padding/truncation are per-call backend state, not tokenizer vocabulary identity.
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop("padding", None)
    backend.pop("truncation", None)
    return object_hash({"backend": backend, "special_tokens_map": tokenizer.special_tokens_map,
                        "model_input_names": tokenizer.model_input_names})


def verify_artifacts(root: Path, hashes: Dict[str, str]) -> None:
    if not hashes:
        raise ValueError(f"Empty artifact record: {root}")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Missing/changed artifact: {path}")


class CheckpointIntegrityCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if ACTIVE_RUN is None:
            return
        folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        hashes = {name: digest for name, digest in artifact_hashes(folder).items()
                  if name != "step8_checkpoint.json"}
        required = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json"}
        if not required <= set(hashes) or not ({"model.safetensors", "pytorch_model.bin"} & set(hashes)):
            raise ValueError(f"Incomplete training checkpoint: {folder}")
        save_json(folder / "step8_checkpoint.json", {"run_id": ACTIVE_RUN["run_id"], "artifacts": hashes})


def find_verified_checkpoint(root: Path, run_id: str) -> Path | None:
    candidates = [p for p in root.glob("checkpoint-*") if p.is_dir() and p.name[11:].isdigit()]
    for folder in sorted(candidates, key=lambda p: int(p.name[11:]), reverse=True):
        marker = folder / "step8_checkpoint.json"
        if not marker.is_file():
            print(f"[Skip incomplete checkpoint] {folder}")
            continue
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("run_id") != run_id:
            raise ValueError(f"Checkpoint belongs to different data/protocol: {folder}")
        verify_artifacts(folder, record.get("artifacts", {}))
        # load_best_model_at_end may also load an earlier validation-selected checkpoint.
        state = json.loads((folder / "trainer_state.json").read_text(encoding="utf-8"))
        best = state.get("best_model_checkpoint")
        if best and Path(best).resolve() != folder.resolve():
            best_path = Path(best)
            best_marker = best_path / "step8_checkpoint.json"
            if not best_path.resolve().is_relative_to(root.resolve()) or not best_marker.is_file():
                raise ValueError("Best checkpoint is missing or belongs to another output directory.")
            best_record = json.loads(best_marker.read_text(encoding="utf-8"))
            if best_record.get("run_id") != run_id:
                raise ValueError("Best checkpoint run identity mismatch.")
            verify_artifacts(best_path, best_record.get("artifacts", {}))
        return folder
    return None


def prepare_run(contract: Dict[str, Any]) -> bool:
    global ACTIVE_RUN
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run Step 8 as a single Python process; distributed launch is not supported.")
    path = OUTPUT_ROOT / "run_manifest.json"
    if OUTPUT_ROOT.exists() and not path.is_file():
        raise FileExistsError(f"Output already exists without this run's manifest: {OUTPUT_ROOT}. Select a new --output-root.")
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    # Resolve once and pin both model and tokenizer to the same immutable Hub revision.
    # Local initializer bundles are content-hashed, including weights.
    revision = previous.get("initializer_commit")
    initializer_config = AutoConfig.from_pretrained(MODEL_NAME, revision=revision)
    resolved_commit = getattr(initializer_config, "_commit_hash", None)
    if revision and resolved_commit != revision:
        raise ValueError("Initializer revision differs from the previous run.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, revision=resolved_commit)
    assets = {"initializer_commit": resolved_commit,
              "initializer_config_sha256": object_hash(initializer_config.to_dict()),
              "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
              "local_initializer_sha256": artifact_hashes(Path(MODEL_NAME)) if Path(MODEL_NAME).is_dir() else None}
    protocol = {"model_name": MODEL_NAME, "serializer_version": SERIALIZER_VERSION,
                "seed": DEFAULT_RANDOM_SEED, "max_length": MAX_LENGTH, "epochs": NUM_TRAIN_EPOCHS,
                "train_batch": PER_DEVICE_TRAIN_BATCH_SIZE, "eval_batch": PER_DEVICE_EVAL_BATCH_SIZE,
                "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
                "sample_limits": [MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES, MAX_TEST_SAMPLES],
                "selection": "validation macro-F1", "label_policy": "preserve supplied episode/scenario labels"}
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
               "torch": torch.__version__, "transformers": transformers.__version__,
               "datasets": hf_datasets.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
    identity = {"dataset_id": contract["dataset_id"], "input_sha256": contract["input_sha256"],
                "step7_build_manifest_sha256": contract["step7_build_manifest_sha256"],
                "script_sha256": sha256_file(Path(__file__)), "protocol": protocol, "runtime": runtime,
                **assets}
    run_id = object_hash(identity)
    if path.exists():
        if previous.get("run_id") != run_id:
            raise ValueError("Output belongs to different data/code/model/tokenizer/settings/runtime. Use a new --output-root; do not mix checkpoints.")
        if previous.get("status") == "COMPLETE":
            verify_artifacts(OUTPUT_ROOT, previous.get("artifacts", {}))
            ACTIVE_RUN = previous
            print(f"[Reuse completed Step 8] {OUTPUT_ROOT}")
            return False
    ACTIVE_RUN = {**identity, **contract, "run_id": run_id, "status": "RUNNING"}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_json(path, ACTIVE_RUN)
    return True


def finish_run() -> None:
    if sha256_file(DATASET_DIR / "build_manifest.json") != ACTIVE_RUN["step7_build_manifest_sha256"]:
        raise ValueError("Step 7 build manifest changed during training.")
    # Link the saved best encoder to its input dataset for Step 9 and later comparisons.
    model_hashes = {name: digest for name, digest in artifact_hashes(MODEL_OUTPUT_DIR).items()
                    if name != "training_provenance.json"}
    save_json(MODEL_OUTPUT_DIR / "training_provenance.json", {
        "dataset_id": ACTIVE_RUN["dataset_id"], "run_id": ACTIVE_RUN["run_id"],
        "step7_build_manifest_sha256": ACTIVE_RUN["step7_build_manifest_sha256"],
        "input_sha256": ACTIVE_RUN["input_sha256"], "serializer_version": SERIALIZER_VERSION,
        "max_length": MAX_LENGTH, "encoder_source": "finetuned",
        "initializer_commit": ACTIVE_RUN.get("initializer_commit"),
        "tokenizer_sha256": ACTIVE_RUN.get("tokenizer_sha256"), "model_artifacts": model_hashes,
    })
    hashes = {}
    for directory in (MODEL_OUTPUT_DIR, REPORT_DIR, PREDICTION_DIR, SERIALIZED_DIR):
        hashes.update({(directory.relative_to(OUTPUT_ROOT) / name).as_posix(): value
                       for name, value in artifact_hashes(directory).items()})
    # Inputs must still be the same at the end of a long training run.
    for name, digest in ACTIVE_RUN["input_sha256"].items():
        if sha256_file(DATASET_DIR / name) != digest:
            raise ValueError(f"Input changed during training: {name}")
    ACTIVE_RUN.update(status="COMPLETE", artifacts=hashes)
    save_json(OUTPUT_ROOT / "run_manifest.json", ACTIVE_RUN)


def write_review_zip() -> Path:
    output = OUTPUT_ROOT / "08_DISTILBERT_REVIEW.zip"
    with tempfile.NamedTemporaryFile(dir=OUTPUT_ROOT, suffix=".zip.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(Path(__file__), "source/8.train_distilbert_window_baseline.py")
            paths = [OUTPUT_ROOT / "run_manifest.json", MODEL_OUTPUT_DIR / "config.json",
                     MODEL_OUTPUT_DIR / "label_mapping.json", MODEL_OUTPUT_DIR / "training_provenance.json",
                     SERIALIZATION_SAMPLE_PATH, SERIALIZATION_REPORT_PATH, SERIALIZATION_MANIFEST_PATH]
            paths += [p for directory in (REPORT_DIR, PREDICTION_DIR) for p in directory.rglob("*") if p.is_file()]
            for path in paths:
                if path.is_file(): archive.write(path, path.relative_to(OUTPUT_ROOT).as_posix())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8: revised dataset, observable-only DistilBERT baseline")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--preflight-only", action="store_true", help="Validate/serialize/tokenize inputs, without training")
    args = parser.parse_args()
    configure_paths(args.dataset_dir, args.output_root)
    if OUTPUT_ROOT == DATASET_DIR or OUTPUT_ROOT.is_relative_to(DATASET_DIR):
        raise ValueError("Training outputs must be outside the immutable Step 7 dataset directory.")
    contract = load_build_contract()
    if prepare_run(contract):
        run_baseline(args.preflight_only)
    else:
        print(f"Review ZIP: {write_review_zip()}")


# ============================================================
# 8. Main training pipeline (original training protocol retained)
# ============================================================

def run_baseline(preflight_only: bool = False) -> None:
    set_all_seeds(DEFAULT_RANDOM_SEED)
    ensure_directories()

    print("=" * 78)
    print("Step 8: DistilBERT single-window baseline")
    print("=" * 78)
    print(f"Base directory: {BASE_DIR.resolve()}")
    print(f"Split directory: {SPLIT_DIR.resolve()}")
    print(f"Output directory: {OUTPUT_ROOT.resolve()}")
    print(f"Observable window seconds: {EXPECTED_OBSERVABLE_WINDOW_SECONDS}")
    print(f"Downstream TST L candidates: {TST_SEQUENCE_LENGTHS}")
    print(f"Downstream TST temporal contexts: {TST_TEMPORAL_CONTEXT_SECONDS} seconds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

    label_order, label_to_id, id_to_label, step7_manifest = load_label_mapping()
    validate_step7_temporal_config(step7_manifest)

    if label_order != DEFAULT_LABEL_ORDER:
        print(f"[Warning] Loaded label order differs from DEFAULT_LABEL_ORDER: {label_order}")
    print("Label mapping:")
    for label in label_order:
        print(f"  {label_to_id[label]} -> {label}")

    train_df, val_df, test_df, temporal_report = build_or_load_serialized_inputs(label_order, label_to_id)

    print("\n[Serialized split sizes]")
    print(f"  train: {len(train_df):,}")
    print(f"  val  : {len(val_df):,}")
    print(f"  test : {len(test_df):,}")

    print("\n[Label distribution: train]")
    print(train_df["label"].value_counts().reindex(label_order))
    print("\n[Label distribution: val]")
    print(val_df["label"].value_counts().reindex(label_order))
    # Test labels never choose a checkpoint or a training hyperparameter.

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=True, revision=ACTIVE_RUN["initializer_commit"])
    if tokenizer_fingerprint(tokenizer) != ACTIVE_RUN["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed after run preparation.")
    token_length_report = report_token_lengths(
        tokenizer,
        {"train": train_df, "val": val_df, "test": test_df},
    )
    print("\n[Token length report]")
    for split, info in token_length_report["splits"].items():
        print(
            f"  {split}: mean={info['mean_tokens']:.1f}, p95={info['p95_tokens']:.1f}, "
            f"max={info['max_tokens']}, evaluated={info['length_rows_evaluated']:,}, "
            f"over_{MAX_LENGTH}={info['rows_over_max_length_in_evaluated_subset']}"
        )

    train_dataset = make_hf_dataset(train_df)
    val_dataset = make_hf_dataset(val_df)
    test_dataset = make_hf_dataset(test_df)

    print("\n[Tokenize]")
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_val = tokenize_dataset(val_dataset, tokenizer)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer)

    if preflight_only:
        ACTIVE_RUN["status"] = "PREFLIGHT_COMPLETE"
        ACTIVE_RUN["token_length_report"] = token_length_report
        save_json(OUTPUT_ROOT / "run_manifest.json", ACTIVE_RUN)
        print("[Preflight complete] No model training or test prediction was performed.")
        return

    num_labels = len(label_order)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_order)},
        label2id=label_to_id,
        revision=ACTIVE_RUN["initializer_commit"],
    )
    model.config.step7_dataset_id = ACTIVE_RUN["dataset_id"]
    model.config.observable_serializer_version = SERIALIZER_VERSION
    model.to(device)
    if getattr(model.config, "_commit_hash", None) != ACTIVE_RUN["initializer_commit"]:
        raise ValueError("Loaded model revision differs from the recorded initializer.")
    if ACTIVE_RUN["local_initializer_sha256"] is not None:
        verify_artifacts(Path(MODEL_NAME), ACTIVE_RUN["local_initializer_sha256"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = build_training_arguments()

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        tokenized_train=tokenized_train,
        tokenized_val=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n[Initial validation before training]")
    initial_eval = trainer.evaluate(eval_dataset=tokenized_val)
    print(initial_eval)

    print("\n[Train]")
    checkpoint = find_verified_checkpoint(OUTPUT_ROOT / "checkpoints", ACTIVE_RUN["run_id"])
    print(f"  resume checkpoint: {checkpoint or 'none (new training)'}")
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.save_state()
    save_json(REPORT_DIR / "training_history.json", {"log_history": trainer.state.log_history,
              "best_model_checkpoint": trainer.state.best_model_checkpoint,
              "best_metric": trainer.state.best_metric, "global_step": trainer.state.global_step})

    print("\n[Save best model and tokenizer]")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    output_label_mapping = {
        "dataset_id": ACTIVE_RUN["dataset_id"],
        "step7_build_manifest_sha256": ACTIVE_RUN["step7_build_manifest_sha256"],
        "serializer_version": SERIALIZER_VERSION,
        "encoder_source": "finetuned",
        "label_order": label_order,
        "label_to_id": label_to_id,
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "metadata_fields": METADATA_FIELDS,
        "observable_fields": OBSERVABLE_FIELDS,
        "max_length": MAX_LENGTH,
        "model_name": MODEL_NAME,
        "observable_window_seconds": EXPECTED_OBSERVABLE_WINDOW_SECONDS,
        "tst_sequence_lengths": TST_SEQUENCE_LENGTHS,
        "tst_temporal_context_seconds": TST_TEMPORAL_CONTEXT_SECONDS,
        "distilbert_baseline_context": "single 1-second observable window; no temporal sequence is used in Step 8.",
        "distilbert_input_rule": "Serialize OBSERVABLE_FIELDS only. Metadata/raw/split/label columns are never serialized.",
    }
    save_json(MODEL_OUTPUT_DIR / "label_mapping.json", output_label_mapping)

    print("\n[Final evaluation]")
    val_summary = save_prediction_reports(
        "val",
        trainer,
        tokenized_val,
        val_df,
        label_order,
        id_to_label,
    )
    test_summary = save_prediction_reports(
        "test",
        trainer,
        tokenized_test,
        test_df,
        label_order,
        id_to_label,
    )

    summary = {
        "dataset_id": ACTIVE_RUN["dataset_id"],
        "run_id": ACTIVE_RUN["run_id"],
        "label_semantics": "supplied scenario/episode class; not instantaneous attack-presence annotation",
        "checkpoint_selection": "validation macro-F1 only; test evaluated after training",
        "token_length_report": token_length_report,
        "model_name": MODEL_NAME,
        "model_output_dir": str(MODEL_OUTPUT_DIR),
        "random_seed": DEFAULT_RANDOM_SEED,
        "observable_window_seconds": EXPECTED_OBSERVABLE_WINDOW_SECONDS,
        "tst_sequence_lengths": TST_SEQUENCE_LENGTHS,
        "tst_temporal_context_seconds": TST_TEMPORAL_CONTEXT_SECONDS,
        "temporal_report": temporal_report,
        "max_length": MAX_LENGTH,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "val_summary": val_summary,
        "test_summary": test_summary,
    }
    save_json(REPORT_DIR / "distilbert_window_baseline_summary.json", summary)

    finish_run()
    print("\n" + "=" * 78)
    print("DistilBERT single-window baseline completed")
    print("=" * 78)
    print(f"Validation accuracy: {val_summary['accuracy']:.4f}")
    print(f"Validation macro-F1: {val_summary['macro_f1']:.4f}")
    print(f"Test accuracy: {test_summary['accuracy']:.4f}")
    print(f"Test macro-F1: {test_summary['macro_f1']:.4f}")
    print("\nSaved artifacts:")
    print(f"- Model/tokenizer: {MODEL_OUTPUT_DIR.resolve()}")
    print(f"- Reports: {REPORT_DIR.resolve()}")
    print(f"- Predictions: {PREDICTION_DIR.resolve()}")
    print(f"- Serialized inputs: {SERIALIZED_DIR.resolve()}")
    print("\nNext step after checking this baseline:")
    print("Next: run the revised Step 9 to extract embeddings from the fine-tuned model saved above.")
    print(f"Review ZIP: {write_review_zip()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if ACTIVE_RUN is not None and ACTIVE_RUN.get("status") != "COMPLETE":
            ACTIVE_RUN["status"] = "FAILED"
            ACTIVE_RUN["error"] = f"{type(exc).__name__}: {exc}"
            save_json(OUTPUT_ROOT / "run_manifest.json", ACTIVE_RUN)
            print(f"Failure review ZIP: {write_review_zip()}", file=sys.stderr)
        print("\n[ERROR] DistilBERT baseline failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
