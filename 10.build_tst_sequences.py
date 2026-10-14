# 10. build_tst_sequences.py
# Build episode-bounded TST sequence datasets from DistilBERT window embeddings.
#
# Pipeline position:
#   Step 9 extracted one DistilBERT [CLS] embedding e_t per 1-second observation window.
#   This script performs:
#     1) Load split-specific window embeddings and window metadata from Step 9.
#     2) Validate row alignment between embeddings, label_ids, row_indices, and metadata.
#     3) Group windows by episode_id and sort each episode by episode_step/window_id.
#     4) Build sliding temporal sequences strictly inside each episode boundary.
#     5) Save memory-efficient sequence-index datasets for Step 11 TST training.
#     6) Optionally materialize full X arrays with shape [num_sequences, L, embedding_dim].
#
# Revised temporal design:
#   - One row is one 1-second cross-layer observable window.
#   - TST sequence lengths are L={1,2,3,5,10}.
#   - Therefore, the temporal contexts are {1,2,3,5,10} seconds.
#
# Critical rules:
#   - Never build a sequence across episode boundaries.
#   - Never mix train/val/test splits.
#   - The sequence label is the label of the last window in the sequence.
#   - Metadata is used only for grouping, sorting, validation, and interpretation.
#   - No DistilBERT text, raw fields, attack phase, or shortcut/leakage fields are used as TST input.
#
# Recommended downstream use:
#   Step 11 should load:
#     - {split}_window_embeddings.npz from outputs/shared_nominal_v2/tst_embeddings_finetuned_main/
#     - {split}_tst_sequences_L{L}.npz from outputs/shared_nominal_v2/tst_sequences_finetuned_main/
#   and construct X dynamically as embeddings[sequence_indices] inside a PyTorch Dataset.
#   This avoids duplicating large [N_seq, L, 768] arrays on disk.

from __future__ import annotations

import argparse
import hashlib
import platform
import tempfile
import zipfile
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# ============================================================
# 0. Basic settings
# ============================================================

DEFAULT_RANDOM_SEED = 20260610

BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_DIR = BASE_DIR / "outputs" / "shared_nominal_v2" / "tst_embeddings_finetuned_main"
OUTPUT_DIR = BASE_DIR / "outputs" / "shared_nominal_v2" / "tst_sequences_finetuned_main"

EMBEDDING_REPORT_PATH = EMBEDDING_DIR / "embedding_extraction_report.json"
SEQUENCE_BUILD_REPORT_TEMPLATE = "sequence_build_report_L{L}.json"

# One observable row is one 1-second window. These values must match Step 7/8/9.
EXPECTED_OBSERVABLE_WINDOW_SECONDS = 1
SEQUENCE_LENGTHS: List[int] = [1, 2, 3, 5, 10]
TST_TEMPORAL_CONTEXT_SECONDS: List[int] = [
    length * EXPECTED_OBSERVABLE_WINDOW_SECONDS for length in SEQUENCE_LENGTHS
]
TST_CONTEXT_MAPPING: Dict[int, int] = {
    length: length * EXPECTED_OBSERVABLE_WINDOW_SECONDS for length in SEQUENCE_LENGTHS
}
STRIDE = 1

# Sorting inside each episode. episode_step is the primary temporal index.
EPISODE_SORT_COLUMNS: List[str] = ["episode_step", "window_id"]

# By default, save memory-efficient sequence_indices rather than full X.
# Full X has size approximately num_sequences * L * embedding_dim * 4 bytes.
MATERIALIZE_X = False
MATERIALIZE_CHUNK_SIZE = 50_000
MATERIALIZED_X_DTYPE = "float32"

# Sequence metadata is useful for timeline visualization and error analysis.
SAVE_SEQUENCE_METADATA = True

# Existing outputs are never deleted; source/run identity controls reuse or recovery.
CLEAN_OUTPUT_DIR_BEFORE_BUILD = False

# Use compressed NPZ for smaller files. Compression can be slower but is safer for disk usage.
SAVE_COMPRESSED_NPZ = True

# Expected fields from Step 9 metadata CSV.
# embedding_row_index is produced by the revised Step 9 script and is the safest
# way to map sequence metadata rows back to the embedding matrix.
MINIMUM_REQUIRED_METADATA_FIELDS: List[str] = [
    "scenario_id",
    "episode_id",
    "episode_type",
    "episode_step",
    "window_id",
    "timestamp",
    "label",
]

LABEL_ID_FIELD = "label_id"
EMBEDDING_ROW_INDEX_FIELD = "embedding_row_index"

SPLITS: List[str] = ["train", "val", "test"]
SERIALIZER_VERSION = "observable22_fixed_order_is_v1"
ACTIVE_RUN: Dict[str, Any] | None = None
SOURCE_CONTRACT: Dict[str, Any] | None = None

# Counterfactual diagnostics stay in separate files and never enter model inputs.
OBSERVABILITY_AUDIT_SCHEMA = "counterfactual_observable_change_v1"
WINDOW_AUDIT_FIELDS = [
    "audit_observable_effect_present", "audit_observable_difference_count",
    "audit_persistent_effect_present", "audit_attack_active",
]
SEQUENCE_AUDIT_FIELDS = [
    "audit_current_observable_effect", "audit_history_effect_window_count",
    "audit_context_effect_window_count", "audit_effect_stratum",
]


# ============================================================
# 1. Reproducibility and IO utilities
# ============================================================

def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_previous_sequence_outputs() -> Dict[str, Any]:
    """Keep prior results; a different source/configuration requires another output directory."""
    if CLEAN_OUTPUT_DIR_BEFORE_BUILD:
        raise ValueError("Automatic deletion is disabled. Keep CLEAN_OUTPUT_DIR_BEFORE_BUILD=False.")
    return {"enabled": False, "removed_files": [], "removed_count": 0}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def verify_artifacts(root: Path, hashes: Dict[str, str]) -> None:
    if not hashes:
        raise ValueError(f"Empty artifact record: {root}")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Missing/changed artifact: {path}")


DEFAULT_LABEL_ORDER: List[str] = [
    "NORMAL",
    "RF_LINK_JAMMING",
    "GNSS_SPOOFING",
    "UNAUTHORIZED_TELECOMMAND_INJECTION",
    "TELECOMMAND_REPLAY",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE",
]


DEFAULT_OBSERVABLE_FIELDS: List[str] = [
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


def progress_bar(iterable: Iterable[Any], total: int | None, desc: str) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def load_json_required(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON report not found: {path}\n"
            f"Run Step 9 extract_distilbert_embeddings.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            writer = np.savez_compressed if SAVE_COMPRESSED_NPZ else np.savez
            writer(stream, **arrays)
        with np.load(temporary, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays): raise ValueError("Saved NPZ keys differ.")
            for name, expected in arrays.items():
                actual = saved[name]
                if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                    raise ValueError(f"Saved sequence NPZ differs: {name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_int_list(values: Any, name: str) -> List[int]:
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError(f"{name} must be a list of integers, without coercion.")
    return values


# ============================================================
# 2. Path helpers
# ============================================================

def embedding_npz_path(split_name: str) -> Path:
    return EMBEDDING_DIR / f"{split_name}_window_embeddings.npz"


def window_metadata_path(split_name: str) -> Path:
    return EMBEDDING_DIR / f"{split_name}_window_metadata.csv"


def sequence_npz_path(split_name: str, sequence_length: int) -> Path:
    return OUTPUT_DIR / f"{split_name}_tst_sequences_L{sequence_length}.npz"


def sequence_metadata_path(split_name: str, sequence_length: int) -> Path:
    return OUTPUT_DIR / f"{split_name}_tst_sequence_metadata_L{sequence_length}.csv"


def sequence_build_report_path(sequence_length: int) -> Path:
    return OUTPUT_DIR / SEQUENCE_BUILD_REPORT_TEMPLATE.format(L=sequence_length)


def sequence_observable_effects_path(split_name: str, sequence_length: int) -> Path:
    return OUTPUT_DIR / f"{split_name}_tst_sequence_observable_effects_L{sequence_length}.csv"


# ============================================================
# 3. Step 9 report validation
# ============================================================

def validate_step9_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure Step 9 artifacts were generated under the revised 1-second TST design."""
    if report.get("status") != "COMPLETE":
        raise ValueError("Step 9 report is not COMPLETE. Finish Step 9 before running Step 10.")
    for key in ("dataset_id", "step8_run_id", "run_id", "input_sha256", "step8_model_sha256"):
        if not report.get(key): raise ValueError(f"Revised Step 9 report is missing {key}.")
    if (report.get("serializer_version") != SERIALIZER_VERSION
            or report.get("observable_fields") != DEFAULT_OBSERVABLE_FIELDS):
        raise ValueError("Step 9 observable schema/serializer does not match the revised pipeline.")
    if not report:
        raise ValueError("Step 9 embedding extraction report is empty.")

    observable_window_seconds = report.get("observable_window_seconds")
    if type(observable_window_seconds) is not int:
        raise ValueError("Step 9 window duration must be an integer.")
    if observable_window_seconds != EXPECTED_OBSERVABLE_WINDOW_SECONDS:
        raise ValueError(
            f"Step 9 observable_window_seconds mismatch: expected "
            f"{EXPECTED_OBSERVABLE_WINDOW_SECONDS}, got {observable_window_seconds}."
        )

    step9_lengths = normalize_int_list(report.get("tst_sequence_lengths"), "Step 9 tst_sequence_lengths")
    if step9_lengths != SEQUENCE_LENGTHS:
        raise ValueError(
            f"Step 9 TST sequence lengths mismatch: expected {SEQUENCE_LENGTHS}, got {step9_lengths}."
        )

    step9_contexts = normalize_int_list(
        report.get("tst_temporal_context_seconds"),
        "Step 9 tst_temporal_context_seconds",
    )
    if step9_contexts != TST_TEMPORAL_CONTEXT_SECONDS:
        raise ValueError(
            f"Step 9 TST temporal contexts mismatch: expected "
            f"{TST_TEMPORAL_CONTEXT_SECONDS}, got {step9_contexts}."
        )

    # Main revision provenance guard: Step 10 must consume supervised fine-tuned
    # DistilBERT embeddings, never the legacy base-pretrained embedding artifact.
    embedding_encoder_source = str(report.get("embedding_encoder_source", "")).strip().lower()
    if embedding_encoder_source not in {"finetuned", "fine-tuned", "fine_tuned"}:
        raise ValueError(
            "Main revision Step 10 requires Step 9 embeddings with "
            "embedding_encoder_source='finetuned'. "
            f"Found: {report.get('embedding_encoder_source')!r}."
        )

    encoder_source_summary = report.get("encoder_source_summary", {})
    representation_type = str(encoder_source_summary.get("representation_type", "")).strip()
    if representation_type != "supervised_finetuned_distilbert_cls":
        raise ValueError(
            "Unexpected Step 9 representation_type for the main revision: "
            f"{representation_type!r}."
        )

    embedding_dim = report.get("embedding_dim")
    if type(embedding_dim) is not int or embedding_dim <= 0:
        raise ValueError("Step 9 report does not contain a valid embedding_dim.")

    label_order = report.get("label_order", [])
    label_to_id = report.get("label_to_id", {})
    if (label_order != DEFAULT_LABEL_ORDER
            or label_to_id != dict(zip(DEFAULT_LABEL_ORDER, range(len(DEFAULT_LABEL_ORDER))))
            or any(type(value) is not int for value in label_to_id.values())):
        raise ValueError("Step 9 label order/IDs must match the canonical six-class mapping.")

    id_to_label = {int(label_to_id[str(label)]): str(label) for label in label_order}

    metadata_fields = report.get("metadata_fields")
    if metadata_fields != MINIMUM_REQUIRED_METADATA_FIELDS:
        raise ValueError("Step 9 must declare exactly the seven canonical metadata fields.")

    return {
        "dataset_id": report["dataset_id"], "step8_run_id": report["step8_run_id"],
        "step9_run_id": report["run_id"], "serializer_version": SERIALIZER_VERSION,
        "observable_window_seconds": observable_window_seconds,
        "tst_sequence_lengths": step9_lengths,
        "tst_temporal_context_seconds": step9_contexts,
        "tst_context_mapping": {str(k): int(v) for k, v in TST_CONTEXT_MAPPING.items()},
        "embedding_dim": embedding_dim,
        "embedding_encoder_source": embedding_encoder_source,
        "representation_type": representation_type,
        "label_order": [str(x) for x in label_order],
        "label_to_id": {str(k): int(v) for k, v in label_to_id.items()},
        "id_to_label": id_to_label,
        "metadata_fields": metadata_fields,
    }


# ============================================================
# 4. Loading and validation of Step 9 artifacts
# ============================================================

def load_embedding_npz(path: Path, split_name: str) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"[{split_name}] Required embedding NPZ not found: {path}\n"
            f"Run Step 9 extract_distilbert_embeddings.py first."
        )

    with np.load(path, allow_pickle=False) as data:
        required = ["embeddings", "label_ids"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"[{split_name}] Missing keys in {path.name}: {missing}")

        for name in ("label_ids", "row_indices", "embedding_row_indices"):
            if name in data.files:
                values = data[name]
                if values.ndim != 1 or values.dtype != np.int64 or (values < 0).any():
                    raise ValueError(f"[{split_name}] {name} must be a nonnegative int64 vector.")
        embeddings = data["embeddings"]
        if embeddings.ndim != 2 or embeddings.dtype != np.float32:
            raise ValueError(f"[{split_name}] Main Step 9 embeddings must be a float32 matrix.")
        if "row_indices" in data.files:
            row_indices = data["row_indices"].astype(np.int64)
        elif "embedding_row_indices" in data.files:
            row_indices = data["embedding_row_indices"].astype(np.int64)
        else:
            raise ValueError(
                f"[{split_name}] Missing row index key in {path.name}. "
                f"Expected 'row_indices' or 'embedding_row_indices'."
            )

        if "embedding_row_indices" in data.files:
            alias_indices = data["embedding_row_indices"].astype(np.int64)
            if not np.array_equal(row_indices, alias_indices):
                raise ValueError(f"[{split_name}] row_indices and embedding_row_indices mismatch in {path.name}.")

        return {
            "embeddings": embeddings,
            "label_ids": data["label_ids"].astype(np.int64),
            "row_indices": row_indices,
        }


def validate_window_metadata(
    metadata_df: pd.DataFrame,
    split_name: str,
    metadata_fields: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if metadata_fields != MINIMUM_REQUIRED_METADATA_FIELDS:
        raise ValueError("Unexpected Step 9 metadata schema.")
    if metadata_df.empty or metadata_df.columns.duplicated().any():
        raise ValueError(f"[{split_name}] Empty metadata or duplicate columns.")
    metadata_fields = [str(col) for col in metadata_fields]

    missing_declared = [col for col in metadata_fields if col not in metadata_df.columns]
    if missing_declared:
        raise ValueError(
            f"[{split_name}] Metadata CSV is missing fields declared by Step 9 report: "
            f"{missing_declared}"
        )

    missing_minimum = [
        col for col in MINIMUM_REQUIRED_METADATA_FIELDS
        if col not in metadata_df.columns
    ]
    if missing_minimum:
        raise ValueError(
            f"[{split_name}] Metadata CSV is missing fields required for sequence construction: "
            f"{missing_minimum}"
        )

    if LABEL_ID_FIELD not in metadata_df.columns:
        raise ValueError(f"[{split_name}] Metadata CSV is missing required column: {LABEL_ID_FIELD}")

    has_embedding_row_index = EMBEDDING_ROW_INDEX_FIELD in metadata_df.columns
    if not has_embedding_row_index:
        raise ValueError("Step 9 embedding_row_index is required; no legacy row-identity fallback.")

    required_base_fields = list(dict.fromkeys(metadata_fields + [LABEL_ID_FIELD]))
    allowed_fields = set(required_base_fields)
    if has_embedding_row_index:
        allowed_fields.add(EMBEDDING_ROW_INDEX_FIELD)

    extra = [col for col in metadata_df.columns if col not in allowed_fields]
    if extra:
        raise ValueError(
            f"[{split_name}] Unexpected metadata columns found: {extra}\n"
            f"Allowed fields are Step 9 metadata_fields + "
            f"['{LABEL_ID_FIELD}', '{EMBEDDING_ROW_INDEX_FIELD}']."
        )

    out = metadata_df[required_base_fields].copy().reset_index(drop=True)

    out[EMBEDDING_ROW_INDEX_FIELD] = metadata_df[EMBEDDING_ROW_INDEX_FIELD].to_numpy()
    embedding_row_index_source = "provided_by_step9_metadata"

    required_fields_for_null_check = required_base_fields + [EMBEDDING_ROW_INDEX_FIELD]
    null_counts = out[required_fields_for_null_check].isnull().sum()
    if int(null_counts.sum()) > 0:
        raise ValueError(f"[{split_name}] Null values in metadata: {null_counts[null_counts > 0].to_dict()}")

    # Cast core fields required by sequence construction.
    out["scenario_id"] = out["scenario_id"].astype(str)
    out["episode_id"] = out["episode_id"].astype(str)
    out["episode_type"] = out["episode_type"].astype(str)
    out["timestamp"] = out["timestamp"].astype(str)
    out["label"] = out["label"].astype(str)
    for field in ("episode_step", "window_id", LABEL_ID_FIELD, EMBEDDING_ROW_INDEX_FIELD):
        values = pd.to_numeric(out[field], errors="raise")
        if (not np.isfinite(values).all() or (values < 0).any()
                or (values >= 2**63).any() or (values % 1 != 0).any()):
            raise ValueError(f"[{split_name}] {field} must be a nonnegative int64 index.")
        out[field] = values.astype(np.int64)
    if any(out[field].astype(str).str.strip().eq("").any() for field in out.columns):
        raise ValueError(f"[{split_name}] Blank metadata value.")
    mapping = dict(zip(DEFAULT_LABEL_ORDER, range(len(DEFAULT_LABEL_ORDER))))
    if not np.array_equal(out["label"].map(mapping).to_numpy(), out[LABEL_ID_FIELD].to_numpy()):
        raise ValueError(f"[{split_name}] Label names and IDs disagree.")
    if (out.groupby("episode_id")[["scenario_id", "episode_type", "label", "label_id"]].nunique() > 1).any().any():
        raise ValueError(f"[{split_name}] Episode identity/scenario label changes inside an episode.")

    duplicate_keys = out.duplicated(subset=["episode_id", "episode_step"], keep=False)
    if duplicate_keys.any():
        examples = out.loc[
            duplicate_keys,
            ["episode_id", "episode_step", "window_id"],
        ].head(10).to_dict("records")
        raise ValueError(f"[{split_name}] Duplicate episode-step-window keys found: {examples}")

    if out[EMBEDDING_ROW_INDEX_FIELD].duplicated().any():
        examples = out.loc[
            out[EMBEDDING_ROW_INDEX_FIELD].duplicated(keep=False),
            EMBEDDING_ROW_INDEX_FIELD,
        ].head(10).tolist()
        raise ValueError(f"[{split_name}] Duplicate embedding_row_index values found: {examples}")

    if (out[EMBEDDING_ROW_INDEX_FIELD] < 0).any():
        bad = out.loc[
            out[EMBEDDING_ROW_INDEX_FIELD] < 0,
            EMBEDDING_ROW_INDEX_FIELD,
        ].head(10).tolist()
        raise ValueError(f"[{split_name}] Negative embedding_row_index values found: {bad}")

    extra_metadata_fields = [
        col for col in metadata_fields
        if col not in MINIMUM_REQUIRED_METADATA_FIELDS
    ]

    return out, {
        "embedding_row_index_present": bool(has_embedding_row_index),
        "embedding_row_index_source": embedding_row_index_source,
        "metadata_fields_from_step9_report": metadata_fields,
        "extra_metadata_fields_preserved": extra_metadata_fields,
    }

def validate_embeddings_and_metadata(
    split_name: str,
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    row_indices: np.ndarray,
    metadata_df: pd.DataFrame,
    expected_embedding_dim: int,
) -> Dict[str, Any]:
    if (embeddings.dtype != np.float32 or label_ids.dtype != np.int64 or row_indices.dtype != np.int64
            or label_ids.ndim != 1 or row_indices.ndim != 1):
        raise ValueError(f"[{split_name}] Invalid embedding or index dtype/rank.")
    if embeddings.ndim != 2:
        raise ValueError(f"[{split_name}] embeddings must be 2-D, got shape={embeddings.shape}")

    n_rows, embedding_dim = embeddings.shape
    if n_rows == 0:
        raise ValueError(f"[{split_name}] embeddings are empty")
    if embedding_dim != expected_embedding_dim:
        raise ValueError(
            f"[{split_name}] embedding_dim mismatch: Step9 report={expected_embedding_dim}, "
            f"NPZ={embedding_dim}"
        )

    if len(label_ids) != n_rows:
        raise ValueError(f"[{split_name}] label_ids length mismatch: {len(label_ids)} vs {n_rows}")
    if len(row_indices) != n_rows:
        raise ValueError(f"[{split_name}] row_indices length mismatch: {len(row_indices)} vs {n_rows}")
    if len(metadata_df) != n_rows:
        raise ValueError(f"[{split_name}] metadata rows mismatch: {len(metadata_df)} vs {n_rows}")

    expected_row_indices = np.arange(n_rows, dtype=np.int64)
    if not np.array_equal(row_indices.astype(np.int64), expected_row_indices):
        bad_positions = np.where(row_indices.astype(np.int64) != expected_row_indices)[0][:10].tolist()
        raise ValueError(f"[{split_name}] row_indices do not match embedding matrix order. Examples: {bad_positions}")

    metadata_row_indices = metadata_df[EMBEDDING_ROW_INDEX_FIELD].to_numpy(dtype=np.int64)
    if not np.array_equal(metadata_row_indices, expected_row_indices):
        bad_positions = np.where(metadata_row_indices != expected_row_indices)[0][:10].tolist()
        raise ValueError(
            f"[{split_name}] metadata embedding_row_index must match embedding row order. "
            f"Examples: {bad_positions}"
        )

    metadata_label_ids = metadata_df["label_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(label_ids.astype(np.int64), metadata_label_ids):
        bad_positions = np.where(label_ids.astype(np.int64) != metadata_label_ids)[0][:10].tolist()
        raise ValueError(f"[{split_name}] label_ids do not match metadata label_id. Examples: {bad_positions}")

    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())
    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"[{split_name}] Invalid embedding values: nan_count={nan_count}, inf_count={inf_count}")

    return {
        "input_windows": int(n_rows),
        "embedding_dim": int(embedding_dim),
        "embedding_dtype": str(embeddings.dtype),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "row_indices_alignment_check_passed": True,
        "embedding_row_index_alignment_check_passed": True,
        "label_alignment_check_passed": True,
    }


def validate_temporal_integrity(
    metadata_df: pd.DataFrame,
    split_name: str,
    expected_window_seconds: int,
) -> Dict[str, Any]:
    parsed = pd.to_datetime(metadata_df["timestamp"], errors="coerce")
    if parsed.isnull().any():
        bad_count = int(parsed.isnull().sum())
        raise ValueError(f"[{split_name}] Found {bad_count} unparsable timestamp values in Step 9 metadata.")

    work = metadata_df.copy()
    work["timestamp_dt"] = parsed

    invalid_step_episodes: List[str] = []
    invalid_window_episodes: List[str] = []
    invalid_timestamp_episodes: List[str] = []
    episode_lengths: List[int] = []
    episodes_with_label_transitions = 0

    for episode_id, group in work.groupby("episode_id", sort=False):
        ordered = group.sort_values(EPISODE_SORT_COLUMNS, kind="mergesort").reset_index(drop=True)
        episode_lengths.append(int(len(ordered)))

        step_diff = ordered["episode_step"].diff().dropna().to_numpy(dtype=np.int64)
        if len(step_diff) and not np.all(step_diff == 1):
            invalid_step_episodes.append(str(episode_id))

        window_diff = ordered["window_id"].diff().dropna().to_numpy(dtype=np.int64)
        if len(window_diff) and not np.all(window_diff == 1):
            invalid_window_episodes.append(str(episode_id))

        timestamp_diff = ordered["timestamp_dt"].diff().dropna().dt.total_seconds().to_numpy(dtype=float)
        if len(timestamp_diff) and not np.all(timestamp_diff == float(expected_window_seconds)):
            invalid_timestamp_episodes.append(str(episode_id))

        if ordered["label_id"].nunique() > 1:
            episodes_with_label_transitions += 1

    if invalid_step_episodes:
        raise ValueError(f"[{split_name}] Non-consecutive episode_step values. Examples={invalid_step_episodes[:10]}")
    if invalid_window_episodes:
        raise ValueError(f"[{split_name}] Non-consecutive window_id values. Examples={invalid_window_episodes[:10]}")
    if invalid_timestamp_episodes:
        raise ValueError(
            f"[{split_name}] Timestamp spacing is not {expected_window_seconds}s within episodes. "
            f"Examples={invalid_timestamp_episodes[:10]}"
        )

    arr = np.asarray(episode_lengths, dtype=np.int64)
    return {
        "temporal_integrity_check_passed": True,
        "timestamp_spacing_seconds": int(expected_window_seconds),
        "episode_count": int(len(arr)),
        "min_episode_length_rows": int(arr.min()) if len(arr) else 0,
        "mean_episode_length_rows": float(arr.mean()) if len(arr) else 0.0,
        "median_episode_length_rows": float(np.median(arr)) if len(arr) else 0.0,
        "max_episode_length_rows": int(arr.max()) if len(arr) else 0,
        "episodes_with_label_transitions": int(episodes_with_label_transitions),
    }


def validate_no_episode_overlap(split_metadata: Dict[str, pd.DataFrame]) -> None:
    split_eps = {
        split: set(df["episode_id"].astype(str).unique())
        for split, df in split_metadata.items()
    }

    names = list(split_eps.keys())
    overlaps: Dict[str, set[str]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a}_{b}"] = split_eps[a] & split_eps[b]

    bad = {key: eps for key, eps in overlaps.items() if eps}
    if bad:
        examples = {key: sorted(list(eps))[:5] for key, eps in bad.items()}
        raise ValueError(f"Episode leakage detected across splits: {examples}")


# ============================================================
# 5. Sequence construction utilities
# ============================================================

def validate_window_observable_effects(
    audit: pd.DataFrame, metadata: pd.DataFrame, split_name: str,
) -> pd.DataFrame:
    """Require exact ordered row identity before using any diagnostic flags."""
    expected_columns = MINIMUM_REQUIRED_METADATA_FIELDS + WINDOW_AUDIT_FIELDS
    if (audit.columns.tolist() != expected_columns or audit.empty
            or len(audit) != len(metadata) or audit.isna().any().any()):
        raise ValueError(f"[{split_name}] Invalid observability sidecar schema, row count, or missing values.")
    identity = audit[MINIMUM_REQUIRED_METADATA_FIELDS].copy()
    identity[LABEL_ID_FIELD] = identity["label"].map(dict(zip(DEFAULT_LABEL_ORDER, range(len(DEFAULT_LABEL_ORDER)))))
    identity[EMBEDDING_ROW_INDEX_FIELD] = np.arange(len(identity), dtype=np.int64)
    identity, _ = validate_window_metadata(identity, split_name, MINIMUM_REQUIRED_METADATA_FIELDS)
    expected = metadata[MINIMUM_REQUIRED_METADATA_FIELDS].reset_index(drop=True).astype(str)
    if not identity[MINIMUM_REQUIRED_METADATA_FIELDS].astype(str).equals(expected):
        raise ValueError(f"[{split_name}] Observability sidecar keys, labels, or row order differ from Step 9 metadata.")
    result = identity[MINIMUM_REQUIRED_METADATA_FIELDS].copy()
    for field in WINDOW_AUDIT_FIELDS:
        values = pd.to_numeric(audit[field], errors="raise")
        maximum = len(DEFAULT_OBSERVABLE_FIELDS) if field == "audit_observable_difference_count" else 1
        if (not np.isfinite(values).all() or (values % 1 != 0).any()
                or (values < 0).any() or (values > maximum).any()):
            raise ValueError(f"[{split_name}] Invalid diagnostic values in {field}.")
        result[field] = values.to_numpy(dtype=np.int64)
    if not np.array_equal(result["audit_observable_effect_present"].to_numpy(),
                          (result["audit_observable_difference_count"].to_numpy() > 0).astype(np.int64)):
        raise ValueError(f"[{split_name}] Observable-effect flag and changed-field count disagree.")
    return result


def load_window_observable_effects(split_name: str, metadata: pd.DataFrame) -> pd.DataFrame:
    contract = SOURCE_CONTRACT["observability_audit"]
    root = Path(contract["source_dataset_dir"])
    name = contract["split_files"][split_name]
    verify_artifacts(root, {name: contract["artifacts"][name]})
    return validate_window_observable_effects(read_csv_robust(root / name), metadata, split_name)


def build_sequence_observable_effects(
    sequence_metadata: pd.DataFrame, sequence_indices: np.ndarray,
    window_effects: pd.DataFrame,
) -> pd.DataFrame:
    """Describe changes within the causal context; this is not a detectability bound."""
    if (sequence_indices.ndim != 2 or sequence_indices.dtype != np.int64
            or sequence_indices.shape[1] < 1 or len(sequence_indices) != len(sequence_metadata)
            or not len(sequence_indices) or sequence_indices.min() < 0
            or sequence_indices.max() >= len(window_effects)):
        raise ValueError("Invalid sequence indices for counterfactual diagnostics.")
    flags = window_effects["audit_observable_effect_present"].to_numpy()
    if not np.isin(flags, [0, 1]).all():
        raise ValueError("Observable-effect diagnostics must be binary.")
    selected = flags[sequence_indices].astype(np.int64)
    current = selected[:, -1]
    history = selected[:, :-1].sum(axis=1, dtype=np.int64)
    result = sequence_metadata.copy()
    result[SEQUENCE_AUDIT_FIELDS[0]] = current
    result[SEQUENCE_AUDIT_FIELDS[1]] = history
    result[SEQUENCE_AUDIT_FIELDS[2]] = history + current
    result[SEQUENCE_AUDIT_FIELDS[3]] = np.where(
        current == 1, "current_changed", np.where(history > 0, "history_only_changed", "no_observable_change"))
    return result


def label_distribution_from_ids(y: np.ndarray, id_to_label: Dict[int, str]) -> Dict[str, int]:
    counts = pd.Series(y.astype(np.int64)).value_counts().sort_index()
    return {
        id_to_label.get(int(label_id), str(int(label_id))): int(count)
        for label_id, count in counts.items()
    }


def compute_episode_length_stats(lengths: List[int]) -> Dict[str, Any]:
    if not lengths:
        return {
            "num_episodes": 0,
            "min_episode_length": 0,
            "mean_episode_length": 0.0,
            "median_episode_length": 0.0,
            "max_episode_length": 0,
        }
    arr = np.asarray(lengths, dtype=np.int64)
    return {
        "num_episodes": int(len(arr)),
        "min_episode_length": int(arr.min()),
        "mean_episode_length": float(arr.mean()),
        "median_episode_length": float(np.median(arr)),
        "max_episode_length": int(arr.max()),
    }


def materialize_sequences(
    embeddings: np.ndarray,
    sequence_indices: np.ndarray,
    dtype: str,
    chunk_size: int,
) -> np.ndarray:
    """Optionally create full X with shape [num_sequences, L, embedding_dim]."""
    if dtype not in {"float32", "float16"}:
        raise ValueError(f"Unsupported MATERIALIZED_X_DTYPE: {dtype}")

    out_dtype = np.float32 if dtype == "float32" else np.float16
    num_sequences, sequence_length = sequence_indices.shape
    embedding_dim = embeddings.shape[1]
    X = np.empty((num_sequences, sequence_length, embedding_dim), dtype=out_dtype)

    for start in range(0, num_sequences, chunk_size):
        end = min(start + chunk_size, num_sequences)
        X[start:end] = embeddings[sequence_indices[start:end]].astype(out_dtype, copy=False)

    return X


def build_sequences_for_split(
    split_name: str,
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    metadata_df: pd.DataFrame,
    sequence_length: int,
    stride: int,
    observable_window_seconds: int,
    id_to_label: Dict[int, str],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, Any]]:
    if type(sequence_length) is not int or type(stride) is not int:
        raise ValueError("Sequence length and stride must be integers.")
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    temporal_context_seconds = int(sequence_length * observable_window_seconds)

    sequence_index_blocks: List[np.ndarray] = []
    start_index_blocks: List[np.ndarray] = []
    end_index_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    metadata_blocks: List[pd.DataFrame] = []

    episode_lengths: List[int] = []
    skipped_short_episodes = 0
    skipped_short_windows = 0
    episodes_with_label_transitions = 0

    grouped = metadata_df.groupby("episode_id", sort=False)
    iterator = progress_bar(
        grouped,
        total=metadata_df["episode_id"].nunique(),
        desc=f"Build {split_name} L={sequence_length}",
    )

    for episode_id, group in iterator:
        group_sorted = group.sort_values(EPISODE_SORT_COLUMNS, kind="mergesort").reset_index(drop=False)
        # 'index' is the metadata row position before sorting. embedding_row_index is the actual embedding row.
        embedding_row_idx = group_sorted[EMBEDDING_ROW_INDEX_FIELD].to_numpy(dtype=np.int64)
        n = len(embedding_row_idx)
        episode_lengths.append(int(n))

        if n < sequence_length:
            skipped_short_episodes += 1
            skipped_short_windows += int(n)
            continue

        labels_in_episode = group_sorted["label_id"].to_numpy(dtype=np.int64)
        if len(np.unique(labels_in_episode)) > 1:
            episodes_with_label_transitions += 1

        starts = np.arange(0, n - sequence_length + 1, stride, dtype=np.int64)
        offsets = np.arange(sequence_length, dtype=np.int64)
        seq_indices = embedding_row_idx[starts[:, None] + offsets[None, :]].astype(np.int64)

        start_indices = seq_indices[:, 0].astype(np.int64)
        end_indices = seq_indices[:, -1].astype(np.int64)
        y = label_ids[end_indices].astype(np.int64)

        # Validate that the sequence label follows the last window in the sequence.
        expected_y = metadata_df.iloc[end_indices]["label_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(y, expected_y):
            raise ValueError(f"[{split_name}] Last-window label mismatch in episode_id={episode_id}")

        # Validate that the selected indices are contiguous in time inside the same episode.
        seq_steps = metadata_df.iloc[seq_indices.reshape(-1)]["episode_step"].to_numpy(dtype=np.int64).reshape(seq_indices.shape)
        if sequence_length > 1:
            if not np.all(np.diff(seq_steps, axis=1) == 1):
                raise ValueError(f"[{split_name}] Non-consecutive episode_step inside generated sequence. episode_id={episode_id}")

        start_meta = metadata_df.iloc[start_indices]
        end_meta = metadata_df.iloc[end_indices]

        label_matrix = label_ids[seq_indices]
        if sequence_length > 1:
            num_label_changes = np.sum(label_matrix[:, 1:] != label_matrix[:, :-1], axis=1).astype(np.int64)
        else:
            num_label_changes = np.zeros(seq_indices.shape[0], dtype=np.int64)
        has_label_transition = (num_label_changes > 0).astype(np.int64)

        seq_meta = pd.DataFrame({
            "source_split": split_name,
            "episode_id": end_meta["episode_id"].to_numpy(dtype=str),
            "episode_type": end_meta["episode_type"].to_numpy(dtype=str),
            "scenario_id": end_meta["scenario_id"].to_numpy(dtype=str),
            "sequence_length": np.full(len(y), sequence_length, dtype=np.int64),
            "observable_window_seconds": np.full(len(y), observable_window_seconds, dtype=np.int64),
            "temporal_context_seconds": np.full(len(y), temporal_context_seconds, dtype=np.int64),
            "sequence_start_row": start_indices,
            "sequence_end_row": end_indices,
            "sequence_start_step": start_meta["episode_step"].to_numpy(dtype=np.int64),
            "sequence_end_step": end_meta["episode_step"].to_numpy(dtype=np.int64),
            "sequence_start_window_id": start_meta["window_id"].to_numpy(dtype=np.int64),
            "sequence_end_window_id": end_meta["window_id"].to_numpy(dtype=np.int64),
            "sequence_start_timestamp": start_meta["timestamp"].to_numpy(dtype=str),
            "sequence_end_timestamp": end_meta["timestamp"].to_numpy(dtype=str),
            "start_label": start_meta["label"].to_numpy(dtype=str),
            "label": end_meta["label"].to_numpy(dtype=str),
            "label_id": y,
            "sequence_num_label_changes": num_label_changes,
            "sequence_has_label_transition": has_label_transition,
        })

        # Preserve additional Step 9 metadata fields at the sequence level.
        # These fields are not model inputs. They are saved only for analysis,
        # variant-holdout evaluation, and error interpretation.
        reserved_metadata_cols = {
            "scenario_id",
            "episode_id",
            "episode_type",
            "episode_step",
            "window_id",
            "timestamp",
            "label",
            LABEL_ID_FIELD,
            EMBEDDING_ROW_INDEX_FIELD,
        }
        extra_metadata_cols = [
            col for col in metadata_df.columns
            if col not in reserved_metadata_cols
        ]

        for col in extra_metadata_cols:
            seq_meta[col] = end_meta[col].to_numpy()

        sequence_index_blocks.append(seq_indices)
        start_index_blocks.append(start_indices)
        end_index_blocks.append(end_indices)
        y_blocks.append(y)
        metadata_blocks.append(seq_meta)

    if not sequence_index_blocks:
        raise ValueError(
            f"[{split_name}] No sequences were generated. "
            f"Check sequence_length={sequence_length}, stride={stride}, and episode lengths."
        )

    sequence_indices = np.concatenate(sequence_index_blocks, axis=0).astype(np.int64)
    start_row_indices = np.concatenate(start_index_blocks, axis=0).astype(np.int64)
    end_row_indices = np.concatenate(end_index_blocks, axis=0).astype(np.int64)
    y = np.concatenate(y_blocks, axis=0).astype(np.int64)
    sequence_metadata = pd.concat(metadata_blocks, axis=0, ignore_index=True)
    sequence_metadata.insert(0, "sequence_id", np.arange(len(sequence_metadata), dtype=np.int64))

    expected_count = sum(max(0, (length - sequence_length) // stride + 1) for length in episode_lengths)
    if len(sequence_indices) != expected_count or len(np.unique(end_row_indices)) != expected_count:
        raise ValueError(f"[{split_name}] Sequence count or unique endpoint coverage differs from the episode lengths.")

    # Final alignment checks.
    if sequence_indices.ndim != 2 or sequence_indices.shape[1] != sequence_length:
        raise ValueError(f"[{split_name}] sequence_indices has invalid shape: {sequence_indices.shape}")
    if int(sequence_indices.min()) < 0 or int(sequence_indices.max()) >= len(metadata_df):
        raise ValueError(f"[{split_name}] sequence_indices out of range")
    if not np.array_equal(sequence_indices[:, 0], start_row_indices):
        raise ValueError(f"[{split_name}] start_row_indices mismatch")
    if not np.array_equal(sequence_indices[:, -1], end_row_indices):
        raise ValueError(f"[{split_name}] end_row_indices mismatch")
    if not np.array_equal(y, metadata_df.iloc[end_row_indices]["label_id"].to_numpy(dtype=np.int64)):
        raise ValueError(f"[{split_name}] y is not equal to the last-window label_id")
    if not np.array_equal(y, sequence_metadata["label_id"].to_numpy(dtype=np.int64)):
        raise ValueError(f"[{split_name}] y is not equal to sequence metadata label_id")

    arrays: Dict[str, np.ndarray] = {
        "sequence_indices": sequence_indices,
        "y": y,
        "label_ids": y,  # compatibility alias for downstream loaders
        "start_row_indices": start_row_indices,
        "end_row_indices": end_row_indices,
        "sequence_length": np.asarray([sequence_length], dtype=np.int64),
        "stride": np.asarray([stride], dtype=np.int64),
        "observable_window_seconds": np.asarray([observable_window_seconds], dtype=np.int64),
        "temporal_context_seconds": np.asarray([temporal_context_seconds], dtype=np.int64),
        "embedding_dim": np.asarray([embeddings.shape[1]], dtype=np.int64),
    }

    if MATERIALIZE_X:
        print(f"[{split_name}] MATERIALIZE_X=True; building full X array. This may require large RAM/disk.")
        arrays["X"] = materialize_sequences(
            embeddings=embeddings,
            sequence_indices=sequence_indices,
            dtype=MATERIALIZED_X_DTYPE,
            chunk_size=MATERIALIZE_CHUNK_SIZE,
        )

    transition_count = int(sequence_metadata["sequence_has_label_transition"].sum())

    summary = {
        "split": split_name,
        "sequence_length": int(sequence_length),
        "stride": int(stride),
        "observable_window_seconds": int(observable_window_seconds),
        "temporal_context_seconds": int(temporal_context_seconds),
        "input_windows": int(len(metadata_df)),
        "embedding_dim": int(embeddings.shape[1]),
        "num_episodes": int(metadata_df["episode_id"].nunique()),
        "num_sequences": int(len(sequence_metadata)),
        "expected_num_sequences": int(expected_count),
        "end_row_indices_sha256": hashlib.sha256(end_row_indices.astype("<i8", copy=False).tobytes()).hexdigest(),
        "short_episode_rule": "For each L, an episode shorter than L yields no sequence; no padding or relabeling.",
        "skipped_short_episodes": int(skipped_short_episodes),
        "skipped_short_windows": int(skipped_short_windows),
        "episodes_with_label_transitions": int(episodes_with_label_transitions),
        "sequences_with_label_transition": transition_count,
        "sequences_with_label_transition_ratio": float(transition_count) / len(sequence_metadata),
        "window_label_distribution": label_distribution_from_ids(label_ids, id_to_label),
        "sequence_label_distribution": label_distribution_from_ids(y, id_to_label),
        "episode_length_stats": compute_episode_length_stats(episode_lengths),
        "sequence_indices_shape": [int(x) for x in sequence_indices.shape],
        "materialized_X_saved": bool(MATERIALIZE_X),
        "materialized_X_shape": [int(x) for x in arrays["X"].shape] if MATERIALIZE_X else None,
    }

    return arrays, sequence_metadata, summary


# ============================================================
# 6. Split-level processing
# ============================================================

def load_step9_split(
    split_name: str,
    expected_embedding_dim: int,
    metadata_fields: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    npz = load_embedding_npz(embedding_npz_path(split_name), split_name=split_name)
    metadata_df, metadata_validation = validate_window_metadata(
        read_csv_robust(window_metadata_path(split_name)),
        split_name=split_name,
        metadata_fields=metadata_fields,
    )

    alignment_validation = validate_embeddings_and_metadata(
        split_name=split_name,
        embeddings=npz["embeddings"],
        label_ids=npz["label_ids"],
        row_indices=npz["row_indices"],
        metadata_df=metadata_df,
        expected_embedding_dim=expected_embedding_dim,
    )

    temporal_validation = validate_temporal_integrity(
        metadata_df=metadata_df,
        split_name=split_name,
        expected_window_seconds=EXPECTED_OBSERVABLE_WINDOW_SECONDS,
    )

    validation = {
        **metadata_validation,
        **alignment_validation,
        **temporal_validation,
    }

    return (
        npz["embeddings"],
        npz["label_ids"].astype(np.int64),
        npz["row_indices"].astype(np.int64),
        metadata_df,
        validation,
    )


def process_split_for_length(
    split_name: str,
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    metadata_df: pd.DataFrame,
    sequence_length: int,
    id_to_label: Dict[int, str],
    observable_effects: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    arrays, sequence_metadata, summary = build_sequences_for_split(
        split_name=split_name,
        embeddings=embeddings,
        label_ids=label_ids,
        metadata_df=metadata_df,
        sequence_length=sequence_length,
        stride=STRIDE,
        observable_window_seconds=EXPECTED_OBSERVABLE_WINDOW_SECONDS,
        id_to_label=id_to_label,
    )

    out_npz = sequence_npz_path(split_name, sequence_length)
    save_npz(out_npz, **arrays)

    out_meta = sequence_metadata_path(split_name, sequence_length)
    if SAVE_SEQUENCE_METADATA:
        save_verified_csv(out_meta, sequence_metadata)

    if observable_effects is None:
        observable_effects = load_window_observable_effects(split_name, metadata_df)
    effect_metadata = build_sequence_observable_effects(
        sequence_metadata, arrays["sequence_indices"], observable_effects)
    out_effects = sequence_observable_effects_path(split_name, sequence_length)
    save_verified_csv(out_effects, effect_metadata)

    audit_sample = pd.concat([sequence_metadata.groupby("label", sort=False).head(1),
                              sequence_metadata.groupby("label", sort=False).tail(1)], ignore_index=True)
    audit_sample = audit_sample.drop_duplicates("sequence_id").sort_values("sequence_id").reset_index(drop=True)
    save_verified_csv(OUTPUT_DIR / "review_samples" / f"{split_name}_L{sequence_length}.csv", audit_sample)

    summary.update({
        "sequence_npz_path": str(out_npz),
        "sequence_metadata_csv_path": str(out_meta) if SAVE_SEQUENCE_METADATA else None,
        "sequence_observable_effects_csv_path": str(out_effects),
        "sequence_observable_effects_sha256": sha256_file(out_effects),
        "observable_effect_stratum_counts": {
            name: int(count) for name, count in effect_metadata["audit_effect_stratum"].value_counts().items()
        },
        "save_compressed_npz": bool(SAVE_COMPRESSED_NPZ),
        "save_sequence_metadata": bool(SAVE_SEQUENCE_METADATA),
    })

    print(
        f"[{split_name}] L={sequence_length} saved | "
        f"context={summary['temporal_context_seconds']}s | "
        f"sequences={summary['num_sequences']:,} | "
        f"indices_shape={summary['sequence_indices_shape']} | "
        f"materialized_X={summary['materialized_X_saved']}"
    )

    return summary


# ============================================================
# 7. Main
# ============================================================

def apply_runtime_args() -> None:
    global EMBEDDING_DIR, OUTPUT_DIR, EMBEDDING_REPORT_PATH
    parser = argparse.ArgumentParser(description="Step 10: episode-bounded sequences from completed revised Step 9")
    parser.add_argument("--embedding-dir", type=Path, default=EMBEDDING_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    def resolve(path):
        return path.resolve()
    EMBEDDING_DIR, OUTPUT_DIR = resolve(args.embedding_dir), resolve(args.output_dir)
    if (OUTPUT_DIR == EMBEDDING_DIR or OUTPUT_DIR.is_relative_to(EMBEDDING_DIR)
            or EMBEDDING_DIR.is_relative_to(OUTPUT_DIR)):
        raise ValueError("Step 10 outputs must be separate from Step 9 inputs.")
    EMBEDDING_REPORT_PATH = EMBEDDING_DIR / "embedding_extraction_report.json"


def load_observability_audit_contract(source: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    """Authenticate the Step 7 audit referenced by the completed Step 9 build."""
    source_dir = source.get("source_dataset_dir")
    if not isinstance(source_dir, str) or not source_dir or not Path(source_dir).is_absolute():
        raise ValueError("Step 9 must record its absolute source_dataset_dir for diagnostic lineage.")
    root = Path(source_dir).resolve()
    path = root / "build_manifest.json"
    if not path.is_file() or sha256_file(path) != source.get("step7_build_manifest_sha256"):
        raise ValueError("Step 7 manifest is missing or differs from the completed Step 9 input.")
    build = load_json_required(path)
    if (build != source.get("step7_build") or build.get("status") != "complete"
            or build.get("dataset_id") != dataset_id):
        raise ValueError("Step 7 manifest payload or dataset identity differs from completed Step 9.")
    policy = build.get("policy", {})
    if (policy.get("nominal_substrate") != "shared_nominal_v2"
            or policy.get("columns") != MINIMUM_REQUIRED_METADATA_FIELDS + DEFAULT_OBSERVABLE_FIELDS
            or policy.get("label_order") != DEFAULT_LABEL_ORDER
            or policy.get("window_seconds") != EXPECTED_OBSERVABLE_WINDOW_SECONDS
            or dataset_id != object_hash({"policy": policy, "sources": build.get("sources")})):
        raise ValueError("Step 7 schema, shared nominal policy, or dataset identity is invalid.")
    audit = build.get("observability_audit", {})
    files = {split: f"audit/{split}_window_observable_effects.csv" for split in SPLITS}
    if (audit.get("schema_version") != OBSERVABILITY_AUDIT_SCHEMA
            or audit.get("split_files") != files or not audit.get("scope")):
        raise ValueError("Step 7 must declare all three counterfactual observable-change sidecars.")
    hashes = {name: build.get("artifacts", {}).get(name) for name in files.values()}
    if not all(hashes.values()):
        raise ValueError("Step 7 completion record lacks observability sidecar hashes.")
    verify_artifacts(root, hashes)
    return {"schema_version": OBSERVABILITY_AUDIT_SCHEMA, "dataset_id": dataset_id,
            "source_dataset_dir": str(root), "step7_build_manifest_sha256": sha256_file(path),
            "split_files": files, "artifacts": hashes, "scope": audit["scope"]}


def load_source_contract() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest_path = EMBEDDING_DIR / "run_manifest.json"
    manifest = load_json_required(manifest_path)
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Step 9 is not COMPLETE. Finish embedding extraction before running Step 10.")
    names = [EMBEDDING_REPORT_PATH.name]
    for split in SPLITS: names += [f"{split}_window_embeddings.npz", f"{split}_window_metadata.csv"]
    hashes = {name: manifest.get("artifacts", {}).get(name) for name in names}
    if not all(hashes.values()): raise ValueError("Step 9 completion record lacks required embedding/metadata artifacts.")
    verify_artifacts(EMBEDDING_DIR, hashes)
    report = load_json_required(EMBEDDING_REPORT_PATH)
    config = validate_step9_report(report)
    for key in ("dataset_id", "step8_run_id", "run_id"):
        if report[key] != manifest.get(key): raise ValueError(f"Step 9 report/manifest mismatch: {key}")
    source = manifest.get("source_contract", {})
    model_hashes = {name: value for name,value in source.get("step8_artifacts", {}).items()
                    if name.startswith("trained_distilbert_model/")}
    if (source.get("dataset_id") != report["dataset_id"]
            or source.get("step8_run_id") != report["step8_run_id"]
            or source.get("input_sha256") != report["input_sha256"]
            or model_hashes != report["step8_model_sha256"]
            or manifest.get("encoder_source") != "finetuned"
            or manifest.get("encoder") != report["encoder_source_summary"]):
        raise ValueError("Step 9 input/model provenance is inconsistent.")
    completed = manifest.get("completed_splits", {})
    if set(completed) != set(SPLITS) or set(report.get("splits", {})) != set(SPLITS):
        raise ValueError("Step 9 does not contain all three completed splits.")
    for split in SPLITS:
        for suffix in ("window_embeddings.npz", "window_metadata.csv"):
            name = f"{split}_{suffix}"
            if completed[split].get("artifacts", {}).get(name) != hashes[name]:
                raise ValueError(f"{split}: Step 9 split and completion hashes disagree.")
        if completed[split].get("summary") != report["splits"][split]:
            raise ValueError(f"{split}: Step 9 split summary differs from its completion record.")
    contract = {"dataset_id": report["dataset_id"], "step8_run_id": report["step8_run_id"],
                "step9_run_id": report["run_id"], "step9_manifest_sha256": sha256_file(manifest_path),
                "artifacts": hashes, "step9_config": config,
                "observability_audit": load_observability_audit_contract(source, report["dataset_id"])}
    return report, contract


def prepare_run() -> bool:
    global ACTIVE_RUN
    identity = {"source": SOURCE_CONTRACT, "source_embedding_dir": str(EMBEDDING_DIR),
                "script_sha256": sha256_file(Path(__file__)),
                "protocol": {"sequence_lengths": SEQUENCE_LENGTHS, "stride": STRIDE,
                             "sort_columns": EPISODE_SORT_COLUMNS, "seed": DEFAULT_RANDOM_SEED,
                             "window_seconds": EXPECTED_OBSERVABLE_WINDOW_SECONDS,
                             "materialize_X": MATERIALIZE_X, "X_dtype": MATERIALIZED_X_DTYPE,
                             "save_metadata": SAVE_SEQUENCE_METADATA, "compressed_npz": SAVE_COMPRESSED_NPZ},
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    run_id = object_hash(identity)
    path = OUTPUT_DIR / "run_manifest.json"
    if OUTPUT_DIR.exists() and not path.is_file():
        raise FileExistsError(f"Output exists without this Step 10 manifest: {OUTPUT_DIR}. Select a new --output-dir.")
    previous = load_json_required(path) if path.exists() else {}
    if previous and previous.get("run_id") != run_id:
        raise ValueError("Output belongs to different data/encoder/code/settings/runtime. Use a new --output-dir.")
    if previous.get("status") == "COMPLETE":
        diagnostic_files = {sequence_observable_effects_path(split, length).name
                            for split in SPLITS for length in SEQUENCE_LENGTHS}
        if not diagnostic_files.issubset(previous.get("artifacts", {})):
            raise ValueError("Completed Step 10 record lacks required diagnostic sidecars.")
        verify_artifacts(OUTPUT_DIR, previous.get("artifacts", {}))
        ACTIVE_RUN = previous
        return False
    # Construction is CPU-only: a partial matching build is safely rebuilt with atomic writers.
    ACTIVE_RUN = {**identity, "run_id": run_id, "dataset_id": SOURCE_CONTRACT["dataset_id"],
                  "step8_run_id": SOURCE_CONTRACT["step8_run_id"], "step9_run_id": SOURCE_CONTRACT["step9_run_id"],
                  "status": "RUNNING"}
    ensure_directories()
    save_json(path, ACTIVE_RUN)
    return True


def verify_split_counts(split: str, metadata: pd.DataFrame, validation: Dict[str, Any], report: Dict[str, Any]) -> None:
    expected = report["splits"][split]
    temporal = report["split_temporal_summaries"][split]
    if (len(metadata) != expected["rows"] or validation["episode_count"] != temporal["episode_count"]
            or expected["embedding_shape"] != [len(metadata), validation["embedding_dim"]]
            or metadata["label"].value_counts().reindex(DEFAULT_LABEL_ORDER, fill_value=0).to_dict() != expected["label_distribution"]):
        raise ValueError(f"{split}: Loaded rows/episodes/labels differ from completed Step 9.")


def common_endpoint_report() -> Dict[str, Any]:
    """Record matched prediction times without removing any training or evaluation sequence."""
    result = {"reference_length": max(SEQUENCE_LENGTHS),
              "rule": "Intersection of end_row_indices across L within each split; no rows filtered by Step 10.",
              "downstream_rule": "Compare context lengths on these same endpoints; report all-available-window results separately.",
              "splits": {}}
    for split in SPLITS:
        endpoints = {}
        for length in SEQUENCE_LENGTHS:
            with np.load(sequence_npz_path(split, length), allow_pickle=False) as arrays:
                endpoints[length] = np.sort(arrays["end_row_indices"])
        common = endpoints[SEQUENCE_LENGTHS[0]]
        for length in SEQUENCE_LENGTHS[1:]: common = np.intersect1d(common, endpoints[length])
        if STRIDE == 1 and not np.array_equal(common, endpoints[max(SEQUENCE_LENGTHS)]):
            raise ValueError(f"{split}: Common endpoints differ from the longest-context endpoints.")
        result["splits"][split] = {"common_endpoint_count": int(len(common)),
            "sorted_end_row_indices_sha256": hashlib.sha256(common.astype("<i8", copy=False).tobytes()).hexdigest(),
            "available_sequences_by_L": {str(length): int(len(values)) for length,values in endpoints.items()}}
    return result


def finish_run() -> None:
    verify_artifacts(EMBEDDING_DIR, SOURCE_CONTRACT["artifacts"])
    if sha256_file(EMBEDDING_DIR / "run_manifest.json") != SOURCE_CONTRACT["step9_manifest_sha256"]:
        raise ValueError("Step 9 manifest changed during sequence construction.")
    audit = SOURCE_CONTRACT["observability_audit"]
    audit_root = Path(audit["source_dataset_dir"])
    verify_artifacts(audit_root, audit["artifacts"])
    if sha256_file(audit_root / "build_manifest.json") != audit["step7_build_manifest_sha256"]:
        raise ValueError("Step 7 diagnostic source manifest changed during sequence construction.")
    index_path = OUTPUT_DIR / "sequence_build_report_index.json"
    index = load_json_required(index_path)
    index["status"] = "COMPLETE"
    save_json(index_path, index)
    paths = [index_path]
    for length in SEQUENCE_LENGTHS:
        paths.append(sequence_build_report_path(length))
        for split in SPLITS:
            paths.append(sequence_npz_path(split, length))
            if SAVE_SEQUENCE_METADATA: paths.append(sequence_metadata_path(split, length))
            paths.append(sequence_observable_effects_path(split, length))
            paths.append(OUTPUT_DIR / "review_samples" / f"{split}_L{length}.csv")
    hashes = {path.relative_to(OUTPUT_DIR).as_posix(): sha256_file(path) for path in paths}
    ACTIVE_RUN.update(status="COMPLETE", artifacts=hashes)
    save_json(OUTPUT_DIR / "run_manifest.json", ACTIVE_RUN)


def write_review_zip() -> Path:
    output = OUTPUT_DIR / "10_SEQUENCES_REVIEW.zip"
    with tempfile.NamedTemporaryFile(dir=OUTPUT_DIR, suffix=".zip.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(Path(__file__), "source/10.build_tst_sequences.py")
            paths = [OUTPUT_DIR / "run_manifest.json", OUTPUT_DIR / "sequence_build_report_index.json"]
            paths += list(OUTPUT_DIR.glob("sequence_build_report_L*.json"))
            paths += list((OUTPUT_DIR / "review_samples").glob("*.csv"))
            for path in paths:
                if path.is_file(): archive.write(path, path.relative_to(OUTPUT_DIR).as_posix())
            archive.write(EMBEDDING_REPORT_PATH, "step9/embedding_extraction_report.json")
            archive.write(EMBEDDING_DIR / "run_manifest.json", "step9/run_manifest.json")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    global SOURCE_CONTRACT
    apply_runtime_args()
    set_all_seeds(DEFAULT_RANDOM_SEED)

    print("=" * 78)
    print("Step 10: Build episode-bounded TST sequences")
    print("=" * 78)
    print(f"Base directory: {BASE_DIR.resolve()}")
    print(f"Embedding directory: {EMBEDDING_DIR.resolve()}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Observable window seconds: {EXPECTED_OBSERVABLE_WINDOW_SECONDS}")
    print(f"Sequence lengths: {SEQUENCE_LENGTHS}")
    print(f"Temporal contexts seconds: {TST_TEMPORAL_CONTEXT_SECONDS}")
    print(f"Stride: {STRIDE}")
    print(f"MATERIALIZE_X: {MATERIALIZE_X}")

    if not EMBEDDING_DIR.exists():
        raise FileNotFoundError(
            f"Embedding directory not found: {EMBEDDING_DIR}\n"
            f"Run Step 9 extract_distilbert_embeddings.py first."
        )

    extraction_report, SOURCE_CONTRACT = load_source_contract()
    step9_config = SOURCE_CONTRACT["step9_config"]
    if not prepare_run():
        print(f"[Reuse completed Step 10] {OUTPUT_DIR}")
        print(f"Review ZIP: {write_review_zip()}")
        return
    id_to_label: Dict[int, str] = step9_config["id_to_label"]
    expected_embedding_dim = int(step9_config["embedding_dim"])
    metadata_fields: List[str] = step9_config["metadata_fields"]

    cleanup_summary = clean_previous_sequence_outputs()
    if cleanup_summary.get("enabled"):
        print(f"\n[Cleanup] Removed stale Step 10 files: {cleanup_summary.get('removed_count', 0)}")

    print("\n[Step 9 design check]")
    print(f"  observable window: {step9_config['observable_window_seconds']} second")
    print(f"  TST sequence lengths: {step9_config['tst_sequence_lengths']}")
    print(f"  TST temporal contexts: {step9_config['tst_temporal_context_seconds']} seconds")
    print(f"  embedding_dim: {expected_embedding_dim}")

    print("\n[Load Step 9 artifacts]")
    split_embeddings: Dict[str, np.ndarray] = {}
    split_label_ids: Dict[str, np.ndarray] = {}
    split_metadata: Dict[str, pd.DataFrame] = {}
    split_observable_effects: Dict[str, pd.DataFrame] = {}
    split_window_validation: Dict[str, Any] = {}

    embedding_dims: List[int] = []
    for split_name in SPLITS:
        embeddings, label_ids, row_indices, metadata_df, validation = load_step9_split(
            split_name=split_name,
            expected_embedding_dim=expected_embedding_dim,
            metadata_fields=metadata_fields,
        )
        verify_split_counts(split_name, metadata_df, validation, extraction_report)
        split_embeddings[split_name] = embeddings
        split_label_ids[split_name] = label_ids
        split_metadata[split_name] = metadata_df
        split_observable_effects[split_name] = load_window_observable_effects(split_name, metadata_df)
        split_window_validation[split_name] = validation
        embedding_dims.append(int(embeddings.shape[1]))
        print(
            f"  {split_name}: windows={len(metadata_df):,}, "
            f"embeddings={embeddings.shape}, dtype={embeddings.dtype}, "
            f"episodes={validation['episode_count']:,}"
        )

    validate_no_episode_overlap(split_metadata)
    print("  Episode overlap check: passed")

    if len(set(embedding_dims)) != 1:
        raise ValueError(f"Embedding dimension mismatch across splits: {embedding_dims}")
    embedding_dim = embedding_dims[0]

    all_reports: Dict[str, Any] = {}

    for sequence_length in SEQUENCE_LENGTHS:
        print("\n" + "-" * 78)
        print(
            f"Build TST sequence datasets for L={sequence_length} "
            f"({sequence_length * EXPECTED_OBSERVABLE_WINDOW_SECONDS}s context)"
        )
        print("-" * 78)

        split_summaries: Dict[str, Any] = {}
        for split_name in SPLITS:
            split_summaries[split_name] = process_split_for_length(
                split_name=split_name,
                embeddings=split_embeddings[split_name],
                label_ids=split_label_ids[split_name],
                metadata_df=split_metadata[split_name],
                sequence_length=sequence_length,
                id_to_label=id_to_label,
                observable_effects=split_observable_effects[split_name],
            )

        report = {
            "step": "10_build_tst_sequences",
            "dataset_id": SOURCE_CONTRACT["dataset_id"], "step8_run_id": SOURCE_CONTRACT["step8_run_id"],
            "step9_run_id": SOURCE_CONTRACT["step9_run_id"], "run_id": ACTIVE_RUN["run_id"],
            "source_artifact_sha256": SOURCE_CONTRACT["artifacts"],
            "label_semantics": "supplied episode/scenario class; no instantaneous-presence relabeling",
            "observability_audit": SOURCE_CONTRACT["observability_audit"],
            "diagnostic_rule": "Separate posthoc counterfactual-change sidecars; no diagnostic fields enter embeddings, sequence arrays, labels, fitting, or selection. No observable change is not a detectability guarantee or Bayes error bound.",
            "source_embedding_dir": str(EMBEDDING_DIR),
            "source_embedding_report": str(EMBEDDING_REPORT_PATH),
            "output_dir": str(OUTPUT_DIR),
            "random_seed": DEFAULT_RANDOM_SEED,
            "observable_window_seconds": EXPECTED_OBSERVABLE_WINDOW_SECONDS,
            "sequence_length": int(sequence_length),
            "temporal_context_seconds": int(sequence_length * EXPECTED_OBSERVABLE_WINDOW_SECONDS),
            "all_sequence_lengths": [int(x) for x in SEQUENCE_LENGTHS],
            "all_temporal_context_seconds": [int(x) for x in TST_TEMPORAL_CONTEXT_SECONDS],
            "tst_context_mapping": {str(k): int(v) for k, v in TST_CONTEXT_MAPPING.items()},
            "stride": int(STRIDE),
            "embedding_dim": int(embedding_dim),
            "embedding_encoder_source": step9_config["embedding_encoder_source"],
            "representation_type": step9_config["representation_type"],
            "sort_columns_inside_episode": EPISODE_SORT_COLUMNS,
            "label_rule": "The TST sequence label is the label_id of the last 1-second window in the sequence.",
            "boundary_rule": "Sliding sequences are generated only within the same split and same episode_id; sequence construction never crosses episode boundaries.",
            "temporal_integrity_rule": "Within each episode, episode_step, window_id, and timestamp must advance by one 1-second window.",
            "input_rule": "TST input is embeddings[sequence_indices]. Metadata is not used as model input.",
            "storage_rule": "Default output stores sequence_indices and labels. Full materialized X is saved only when MATERIALIZE_X=True.",
            "materialize_X": bool(MATERIALIZE_X),
            "materialized_X_dtype": MATERIALIZED_X_DTYPE if MATERIALIZE_X else None,
            "save_compressed_npz": bool(SAVE_COMPRESSED_NPZ),
            "save_sequence_metadata": bool(SAVE_SEQUENCE_METADATA),
            "cleanup_summary": cleanup_summary,
            "step9_config": {
                key: value for key, value in step9_config.items() if key != "id_to_label"
            },
            "window_validation": split_window_validation,
            "splits": split_summaries,
        }

        report_path = sequence_build_report_path(sequence_length)
        save_json(report_path, report)
        all_reports[f"L{sequence_length}"] = report

        print(f"\n[L={sequence_length}] Report saved: {report_path.resolve()}")

    combined_report_path = OUTPUT_DIR / "sequence_build_report_index.json"
    save_json(
        combined_report_path,
        {
            "step": "10_build_tst_sequences_index",
            "status": "BUILT",
            "dataset_id": SOURCE_CONTRACT["dataset_id"], "step8_run_id": SOURCE_CONTRACT["step8_run_id"],
            "step9_run_id": SOURCE_CONTRACT["step9_run_id"], "run_id": ACTIVE_RUN["run_id"],
            "source_artifact_sha256": SOURCE_CONTRACT["artifacts"],
            "common_endpoint_report": common_endpoint_report(),
            "observability_audit": SOURCE_CONTRACT["observability_audit"],
            "source_embedding_dir": str(EMBEDDING_DIR),
            "output_dir": str(OUTPUT_DIR),
            "observable_window_seconds": EXPECTED_OBSERVABLE_WINDOW_SECONDS,
            "sequence_lengths": [int(x) for x in SEQUENCE_LENGTHS],
            "temporal_context_seconds": [int(x) for x in TST_TEMPORAL_CONTEXT_SECONDS],
            "tst_context_mapping": {str(k): int(v) for k, v in TST_CONTEXT_MAPPING.items()},
            "stride": int(STRIDE),
            "embedding_encoder_source": step9_config["embedding_encoder_source"],
            "representation_type": step9_config["representation_type"],
            "materialize_X": bool(MATERIALIZE_X),
            "reports": {
                key: str(sequence_build_report_path(int(key[1:])))
                for key in all_reports.keys()
            },
            "required_step11_loading_rule": (
                f"For each L and split, load embeddings from {EMBEDDING_DIR}/{{split}}_window_embeddings.npz "
                f"and sequence_indices from {OUTPUT_DIR}/{{split}}_tst_sequences_L{{L}}.npz; "
                "construct X_batch as embeddings[sequence_indices_batch]."
            ),
        },
    )

    finish_run()

    print("\n" + "=" * 78)
    print("TST sequence building completed")
    print("=" * 78)
    for sequence_length in SEQUENCE_LENGTHS:
        print(f"L={sequence_length} ({sequence_length * EXPECTED_OBSERVABLE_WINDOW_SECONDS}s):")
        report = all_reports[f"L{sequence_length}"]
        for split_name in SPLITS:
            summary = report["splits"][split_name]
            print(
                f"  {split_name}: sequences={summary['num_sequences']:,}, "
                f"indices_shape={summary['sequence_indices_shape']}, "
                f"transition_seq_ratio={summary['sequences_with_label_transition_ratio']:.4f}"
            )
    print("\nSaved artifacts:")
    print(f"- TST sequences: {OUTPUT_DIR.resolve()}")
    print(f"- Report index: {combined_report_path.resolve()}")
    print("\nNext step:")
    print("Next: update Step 11 input paths and source identity checks, retaining the agreed training protocol.")
    print(f"Review ZIP: {write_review_zip()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if ACTIVE_RUN is not None and ACTIVE_RUN.get("status") != "COMPLETE":
            ACTIVE_RUN["status"] = "FAILED"
            ACTIVE_RUN["error"] = f"{type(exc).__name__}: {exc}"
            save_json(OUTPUT_DIR / "run_manifest.json", ACTIVE_RUN)
            print(f"Failure review ZIP: {write_review_zip()}", file=sys.stderr)
        print("\n[ERROR] TST sequence building failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
