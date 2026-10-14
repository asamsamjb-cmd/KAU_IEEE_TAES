#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
14.lstm_finetuned_distilbert_l1_baseline.py

Standalone L=1 companion to the original Step14 L=10 LSTM baseline.
Place this file beside the original script in the project root and run it.
It reads the same completed shared_nominal_v2 Steps 8/9/10 artifacts.
No encoder fine-tuning or embedding extraction is repeated.

The LSTM receives only the current fixed 768-dimensional DistilBERT embedding.
Architecture, ten head seeds and training settings match the supplied L10 script.
Train on all eligible L1 training windows; choose checkpoints using full L1
validation macro-F1 (validation loss breaks ties), as in the TST L1 protocol.
Evaluate all L1 test windows and separately report the verified common
L10-eligible endpoints. L10 indices are used ONLY to identify that reporting
cohort; no historical embeddings or audit fields enter the L1 model.

Expected existing inputs:
  dataset/shared_nominal_v2/splits/
  outputs/shared_nominal_v2/distilbert_window_baseline/trained_distilbert_model/
  outputs/shared_nominal_v2/tst_embeddings_finetuned_main/
  outputs/shared_nominal_v2/tst_sequences_finetuned_main/
The original completed Step10 L=1,2,3,5,10 artifacts remain authenticated.

Outputs (separate from the original L10 run):
  outputs/shared_nominal_v2/lstm_finetuned_distilbert_l1/
    lstm_finetuned_distilbert_l1_results_by_seed.csv
    lstm_finetuned_distilbert_l1_summary.csv
    lstm_finetuned_distilbert_l1_paper_table.csv  <-- COMMON endpoint results
    lstm_finetuned_distilbert_l1_latex_rows.txt  <-- L1 F1 and FA_N cells
    reports/, predictions/, models/, run_manifest.json
Rates in machine-readable metrics use [0,1]; paper-table/LaTeX FA_N uses %.
Summary SD is the sample SD over ten independently trained heads, conditional
on the fixed encoder and split; predictions are not ensembled.

Run: python 14.lstm_finetuned_distilbert_l1_baseline.py
Check inputs without training: append --check-inputs-only
After a stopped/failed run: append --retry-failed (completed seeds are reused).
An incomplete seed restarts at epoch 1, matching the original resume behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import platform
import sys
import time
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# ============================================================
# USER CONFIG: Alt+F10 실행용. 보통 수정하지 않아도 됩니다.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SPLIT_DIR = BASE_DIR / "dataset" / "shared_nominal_v2" / "splits"
EMBEDDING_DIR = BASE_DIR / "outputs" / "shared_nominal_v2" / "tst_embeddings_finetuned_main"
SEQUENCE_DIR = BASE_DIR / "outputs" / "shared_nominal_v2" / "tst_sequences_finetuned_main"
OUTPUT_ROOT = BASE_DIR / "outputs" / "shared_nominal_v2" / "lstm_finetuned_distilbert_l1"
REPORT_DIR = OUTPUT_ROOT / "reports"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"
MODEL_DIR = OUTPUT_ROOT / "models"

TRAIN_INPUT_PATH = SPLIT_DIR / "train_observable.csv"
VAL_INPUT_PATH = SPLIT_DIR / "val_observable.csv"
TEST_INPUT_PATH = SPLIT_DIR / "test_observable.csv"
LABEL_MAPPING_INPUT_PATH = SPLIT_DIR / "label_mapping.json"

EMBEDDING_REPORT_PATH = EMBEDDING_DIR / "embedding_extraction_report.json"
SEQUENCE_REPORT_INDEX_PATH = SEQUENCE_DIR / "sequence_build_report_index.json"
EXPECTED_FINETUNED_MODEL_DIR = BASE_DIR / "outputs" / "shared_nominal_v2" / "distilbert_window_baseline" / "trained_distilbert_model"
EXPECTED_SOURCE_CHECKPOINT_DIRNAME = "trained_distilbert_model"
EXPECTED_REPRESENTATION_TYPE = "supervised_finetuned_distilbert_cls"
EXPECTED_ENCODER_MODEL_NAME = "distilbert-base-uncased"

# Single-window control; L10 defines reporting endpoints only, never model input.
SEQUENCE_LENGTH = 1
TEMPORAL_CONTEXT_SECONDS = 1
COMMON_EVALUATION_REFERENCE_LENGTH = 10
ORIGINAL_L10_OUTPUT_ROOT = BASE_DIR / "outputs" / "shared_nominal_v2" / "lstm_finetuned_distilbert_l10"
COMMON_ENDPOINTS: Dict[str, np.ndarray] = {}

# Same ten temporal-head seeds as the revised TST and pooling controls.
SEEDS: Tuple[int, ...] = tuple(range(20260610, 20260620))

# LSTM architecture. Keep close to the TST temporal-head scale.
INPUT_EMBEDDING_DIM_EXPECTED = 768  # DistilBERT-base [CLS]
PROJECT_TO_DIM = 128
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.25
BIDIRECTIONAL = False

# Training settings. Kept close to TST temporal training setup.
MAX_EPOCHS = 30
PATIENCE = 7
BATCH_SIZE = 256
LEARNING_RATE = 7e-5
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 1.0
USE_AMP = torch.cuda.is_available()
USE_CLASS_WEIGHTS = False
LABEL_SMOOTHING = 0.03
NUM_WORKERS = 0  # Windows/PyCharm safe

# Output behavior.
CLEAN_OUTPUT_DIR_BEFORE_RUN = False
SAVE_MODEL_CHECKPOINTS = True
SAVE_PREDICTIONS = True
SAVE_CONFUSION_MATRIX_PNG = True

# Attribution/provenance guards for the revised main pipeline.
# The authoritative provenance is recorded by the Step 9 and Step 10 JSON reports.
EXPECTED_ENCODER_SOURCE = "finetuned"
SPLITS = ("train", "val", "test")
SEQUENCE_LENGTHS = [1, 2, 3, 5, 10]

OBSERVABLE_EFFECT_AUDIT_COLUMNS = [
    "audit_current_observable_effect", "audit_history_effect_window_count",
    "audit_context_effect_window_count", "audit_effect_stratum",
]
OBSERVABLE_EFFECT_STRATA = ["current_changed", "history_only_changed", "no_observable_change"]
# Separate diagnostic tables; never consumed by a model, loader or checkpoint selector.
OBSERVABLE_EFFECTS: Dict[str, pd.DataFrame] = {}


def sequence_observable_effects_path(split_name: str, sequence_length: int) -> Path:
    return SEQUENCE_DIR / f"{split_name}_tst_sequence_observable_effects_L{sequence_length}.csv"


def validate_sequence_observable_effects(
    effects: pd.DataFrame, metadata: pd.DataFrame, sequence_length: int, split_name: str,
) -> None:
    """Authenticate row alignment and the causal current/history count contract."""
    context = f"{split_name}/L{sequence_length} observable-effect audit"
    if list(effects.columns) != list(metadata.columns) + OBSERVABLE_EFFECT_AUDIT_COLUMNS:
        raise ValueError(f"{context}: sidecar columns must be exact metadata plus diagnostic columns.")
    try:
        pd.testing.assert_frame_equal(effects[list(metadata.columns)].reset_index(drop=True),
                                      metadata.reset_index(drop=True), check_exact=True)
    except AssertionError as exc:
        raise ValueError(f"{context}: sidecar metadata/sequence row alignment mismatch.") from exc
    current, history, whole = [integer_array(effects[column], f"{context}: {column}")
                               for column in OBSERVABLE_EFFECT_AUDIT_COLUMNS[:3]]
    if (not np.isin(current, [0, 1]).all() or np.any((history < 0) | (history >= sequence_length))
            or not np.array_equal(whole, history + current)):
        raise ValueError(f"{context}: invalid current/history/context counts.")
    expected = np.where(current == 1, "current_changed",
                        np.where(history > 0, "history_only_changed", "no_observable_change"))
    if not np.array_equal(effects["audit_effect_stratum"].to_numpy(), expected):
        raise ValueError(f"{context}: inconsistent observable-effect stratum.")


def load_sequence_observable_effects(
    split_name: str, sequence_length: int, metadata: pd.DataFrame,
) -> pd.DataFrame:
    effects = read_csv_robust(sequence_observable_effects_path(split_name, sequence_length))
    validate_sequence_observable_effects(effects, metadata, sequence_length, split_name)
    return effects


def validate_observability_source_contract(
    audit: Dict[str, Any], source9: Dict[str, Any], run8: Dict[str, Any], dataset_id: str,
) -> None:
    """Bind Step 10 diagnostics to the same Step 7 build used by the encoder."""
    build = source9.get("step7_build", {})
    spec = build.get("observability_audit", {})
    split_files = {split: f"audit/{split}_window_observable_effects.csv" for split in SPLITS}
    manifest_hash = source9.get("step7_build_manifest_sha256")
    artifacts = {name: build.get("artifacts", {}).get(name) for name in split_files.values()}
    expected = {
        "schema_version": "counterfactual_observable_change_v1", "dataset_id": dataset_id,
        "source_dataset_dir": source9.get("source_dataset_dir"),
        "step7_build_manifest_sha256": manifest_hash, "split_files": split_files,
        "artifacts": artifacts, "scope": spec.get("scope"),
    }
    if (build.get("dataset_id") != dataset_id or spec.get("schema_version") != expected["schema_version"]
            or spec.get("split_files") != split_files or not spec.get("scope")
            or not expected["source_dataset_dir"] or not manifest_hash
            or run8.get("step7_build_manifest_sha256") != manifest_hash
            or any(not isinstance(value, str) or len(value) != 64 for value in artifacts.values())
            or audit != expected):
        raise ValueError("Step 10 observable-effect diagnostics are not bound to the same Step 7 dataset/build.")


def compute_observable_effect_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray, metadata: pd.DataFrame,
    effects: pd.DataFrame, id_to_label: Dict[int, str], split_name: str,
) -> Dict[str, Any]:
    """Post-hoc diagnostics only; never select training rows or checkpoints."""
    validate_sequence_observable_effects(effects, metadata, SEQUENCE_LENGTH, split_name)
    truth = integer_array(y_true, "diagnostic true labels")
    predicted = integer_array(y_pred, "diagnostic predicted labels")
    if (not np.array_equal(truth, metadata["label_id"].to_numpy()) or len(predicted) != len(truth)):
        raise ValueError("Diagnostic predictions do not match L1 metadata.")
    labels = [id_to_label[i] for i in range(len(id_to_label))]
    populations = {}
    for name, population_mask in {"full": np.ones(len(truth), dtype=bool),
                                  "common_population": verified_common_mask(metadata, split_name)}.items():
        rows = {}
        for stratum in OBSERVABLE_EFFECT_STRATA:
            mask = population_mask & effects["audit_effect_stratum"].eq(stratum).to_numpy()
            summary, per_class, confusion = compute_metrics(truth[mask], predicted[mask], len(labels), id_to_label)
            n = int(mask.sum()); present = per_class["support"].gt(0)
            rows[stratum] = {
                "num_sequences": n, "fraction_of_population": float(n / population_mask.sum()),
                "support_by_label": dict(zip(labels, per_class["support"].astype(int).tolist())),
                "present_true_labels": per_class.loc[present, "label"].tolist(),
                "absent_true_labels": per_class.loc[~present, "label"].tolist(),
                "accuracy": summary["accuracy"] if n else None,
                "normal_prediction_fraction": float(np.mean(predicted[mask] == labels.index("NORMAL"))) if n else None,
                "macro_f1_present_true_classes": float(per_class.loc[present, "f1"].mean()) if n else None,
                "confusion_matrix": confusion.to_numpy(dtype=np.int64).tolist(),
            }
        populations[name] = {"total_sequences": int(population_mask.sum()), "strata": rows}
    return {
        "split": split_name, "sequence_length": SEQUENCE_LENGTH,
        "diagnostic_basis": "Actual versus same-window nominal-counterfactual 22-field projection differences.",
        "interpretation": "Descriptive fixed-checkpoint analysis; no filtering of fitting/selection data. "
                          "Zero observable change is not a no-information proof or bound. "
                          "An L1 input has no history-only stratum, even at L10-eligible endpoints.",
        "metric_basis": "Stratum macro-F1 uses positive-support classes; it does not replace six-class macro-F1. "
                        "Full L1 and common endpoint populations are reported separately.",
        "confusion_matrix_labels": labels, "populations": populations,
    }


SOURCE_CONTRACT: Dict[str, Any] = {}
INPUT_AUDIT: Dict[str, Any] = {}
ACTIVE_RUN: Dict[str, Any] = {}


# ============================================================
# Schema / labels
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
    "observation_stage",
    "path_type",
    "direction",
    "source_role",
    "destination_role",
    "protocol_context",
    "operational_phase",
    "message_type",
    "service_context",
    "command_intent",
    "payload_or_parameter_context",
    "sequence_freshness_status",
    "control_security_state",
    "signal_quality_state",
    "tracking_stability",
    "navigation_consistency",
    "timing_consistency",
    "decoder_status",
    "command_execution_result",
    "retry_timeout_state",
    "session_state",
    "state_transition_consistency",
]

DEFAULT_LABEL_ORDER: List[str] = [
    "NORMAL",
    "RF_LINK_JAMMING",
    "GNSS_SPOOFING",
    "UNAUTHORIZED_TELECOMMAND_INJECTION",
    "TELECOMMAND_REPLAY",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE",
]


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class ModelConfig:
    input_dim: int
    projected_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    bidirectional: bool
    sequence_length: int
    num_classes: int


@dataclass
class TrainConfig:
    max_epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    use_amp: bool
    use_class_weights: bool
    label_smoothing: float


# ============================================================
# Utilities
# ============================================================

def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility/throughput compromise. Deterministic algorithms can be slow and brittle on Windows.
    torch.backends.cudnn.benchmark = True


def ensure_output_directories() -> None:
    if CLEAN_OUTPUT_DIR_BEFORE_RUN:
        raise ValueError("Automatic output deletion is disabled. Use a separate --output-root.")
    for path in [OUTPUT_ROOT, REPORT_DIR, PREDICTION_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def progress_bar(iterable: Iterable[Any], total: Optional[int], desc: str) -> Iterable[Any]:
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
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    atomic_text(path, "\ufeff" + df.to_csv(index=False))


def expected_test_sequence_count_from_report(sequence_length: int = SEQUENCE_LENGTH) -> int:
    """Resolve the complete test count for a specific input length."""
    index = load_json_required(SEQUENCE_REPORT_INDEX_PATH)
    reports = index.get("reports", {})
    report_path_text = reports.get(f"L{sequence_length}")
    if not report_path_text:
        raise ValueError(f"Step 10 report index has no report for L={sequence_length}.")

    report_path = SEQUENCE_DIR / f"sequence_build_report_L{sequence_length}.json"
    if Path(report_path_text).resolve() != report_path.resolve():
        raise ValueError(f"Step 10 report index points to a different L{sequence_length} report.")
    report = load_json_required(report_path)

    splits = report.get("splits", {})
    test_summary = splits.get("test", {})
    count = int(test_summary.get("num_sequences", -1))
    if count <= 0:
        raise ValueError(
            f"Invalid/missing Step 10 L{sequence_length} test num_sequences in {report_path}: {count}"
        )
    return count


def safe_name(text: str) -> str:
    return (
        str(text)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("&", "and")
        .replace("-", "_")
        .replace("+", "plus")
        .replace(".", "")
    )


def autocast_context(device: torch.device) -> Any:
    if device.type == "cuda" and USE_AMP:
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            try:
                return torch.amp.autocast(device_type="cuda")
            except TypeError:
                pass
        return torch.cuda.amp.autocast()
    return nullcontext()


def make_grad_scaler(device: torch.device) -> Optional[Any]:
    if device.type != "cuda" or not USE_AMP:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            try:
                return torch.amp.GradScaler(enabled=True)
            except TypeError:
                pass
    return torch.cuda.amp.GradScaler()


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ============================================================
# Label / split loading
# ============================================================

def load_label_config() -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, Any]]:
    data = load_json_required(LABEL_MAPPING_INPUT_PATH)
    if not data.get("label_order") or not data.get("label_to_id"):
        raise ValueError("The current Step 7 label order and mapping are required.")
    label_order = [str(x) for x in data.get("label_order", DEFAULT_LABEL_ORDER)]
    label_to_id = {str(k): int(v) for k, v in data.get("label_to_id", {}).items()}
    if not label_to_id:
        label_to_id = {label: idx for idx, label in enumerate(label_order)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    ids = sorted(label_to_id.values())
    expected = list(range(len(label_order)))
    if ids != expected:
        raise ValueError(f"label_to_id must be contiguous {expected}, got {ids}")
    missing = [label for label in label_order if label not in label_to_id]
    if missing:
        raise ValueError(f"label_to_id is missing labels: {missing}")
    return label_order, label_to_id, id_to_label, data


def add_label_id(df: pd.DataFrame, label_to_id: Dict[str, int], split_name: str) -> pd.DataFrame:
    if "label" not in df.columns:
        raise ValueError(f"[{split_name}] missing label column")
    out = df.copy().reset_index(drop=True)
    out["label"] = out["label"].astype(str)
    out["label_id"] = out["label"].map(label_to_id)
    if out["label_id"].isnull().any():
        bad = sorted(out.loc[out["label_id"].isnull(), "label"].unique().tolist())
        raise ValueError(f"[{split_name}] failed to map label_id for labels: {bad}")
    out["label_id"] = out["label_id"].astype(np.int64)
    return out


def load_observable_splits(label_to_id: Dict[str, int]) -> Dict[str, pd.DataFrame]:
    paths = {
        "train": TRAIN_INPUT_PATH,
        "val": VAL_INPUT_PATH,
        "test": TEST_INPUT_PATH,
    }
    splits: Dict[str, pd.DataFrame] = {}
    for split_name, path in paths.items():
        df = read_csv_robust(path)
        df = add_label_id(df, label_to_id, split_name)
        splits[split_name] = df
    return splits


# ============================================================
# Revised pipeline provenance validation
# ============================================================

def validate_revised_pipeline_provenance() -> Dict[str, Any]:
    """Verify the same completed encoder and sequence artifacts used by Step 11."""
    global SOURCE_CONTRACT
    embedding_report = load_json_required(EMBEDDING_REPORT_PATH)
    sequence_report = load_json_required(SEQUENCE_REPORT_INDEX_PATH)

    encoder_source = str(embedding_report.get("embedding_encoder_source", "")).strip().lower().replace("-", "_")
    if encoder_source not in {"finetuned", "fine_tuned"}:
        raise ValueError(
            f"Step 9 embedding_encoder_source must be finetuned, got "
            f"{embedding_report.get('embedding_encoder_source')!r}"
        )

    # Step 9 stores representation_type inside encoder_source_summary in the
    # current revised extractor. Accept a top-level copy as well for backward/
    # forward compatibility, but require the resolved value to match exactly.
    encoder_source_summary = embedding_report.get("encoder_source_summary", {})
    if not isinstance(encoder_source_summary, dict):
        encoder_source_summary = {}
    representation_type = str(
        embedding_report.get("representation_type")
        or encoder_source_summary.get("representation_type")
        or ""
    ).strip()
    if representation_type != EXPECTED_REPRESENTATION_TYPE:
        raise ValueError(
            f"Step 9 representation_type mismatch: expected "
            f"{EXPECTED_REPRESENTATION_TYPE!r}, got {representation_type!r}. "
            "Checked both the top-level field and encoder_source_summary.representation_type."
        )

    embedding_dim = int(embedding_report.get("embedding_dim", -1))
    if embedding_dim != INPUT_EMBEDDING_DIM_EXPECTED:
        raise ValueError(
            f"Step 9 embedding_dim mismatch: expected {INPUT_EMBEDDING_DIM_EXPECTED}, got {embedding_dim}"
        )

    source_model_dir = str(embedding_report.get("source_model_dir", ""))
    if EXPECTED_SOURCE_CHECKPOINT_DIRNAME.lower() not in source_model_dir.lower():
        raise ValueError(
            "Step 9 source_model_dir does not point to the verified Step 8 fine-tuned checkpoint: "
            f"{source_model_dir!r}"
        )

    seq_encoder_source = str(sequence_report.get("embedding_encoder_source", "")).strip().lower().replace("-", "_")
    if seq_encoder_source not in {"finetuned", "fine_tuned"}:
        raise ValueError(
            f"Step 10 embedding_encoder_source must be finetuned, got "
            f"{sequence_report.get('embedding_encoder_source')!r}"
        )

    seq_representation = str(sequence_report.get("representation_type", "")).strip()
    if seq_representation != EXPECTED_REPRESENTATION_TYPE:
        raise ValueError(
            f"Step 10 representation_type mismatch: expected "
            f"{EXPECTED_REPRESENTATION_TYPE!r}, got {seq_representation!r}"
        )

    lengths = [int(x) for x in sequence_report.get("sequence_lengths", [])]
    if SEQUENCE_LENGTH not in lengths:
        raise ValueError(
            f"Step 10 report does not contain required L={SEQUENCE_LENGTH}; lengths={lengths}"
        )

    # Resolve paths only after existence has already been established by load_json_required.
    report_embedding_dir = Path(str(sequence_report.get("source_embedding_dir", ""))).resolve()
    report_sequence_dir = Path(str(sequence_report.get("output_dir", ""))).resolve()
    if report_embedding_dir != EMBEDDING_DIR.resolve():
        raise ValueError(
            f"Step 10 source_embedding_dir mismatch: report={report_embedding_dir}, "
            f"expected={EMBEDDING_DIR.resolve()}"
        )
    if report_sequence_dir != SEQUENCE_DIR.resolve():
        raise ValueError(
            f"Step 10 output_dir mismatch: report={report_sequence_dir}, "
            f"expected={SEQUENCE_DIR.resolve()}"
        )

    SOURCE_CONTRACT = load_source_contract(embedding_report, sequence_report)
    required_inputs = {f"splits/{split}_observable.csv" for split in SPLITS} | {
        "splits/label_mapping.json", "splits/episode_split_manifest.csv"}
    if not required_inputs.issubset(SOURCE_CONTRACT["input_sha256"]):
        raise ValueError("Step 8 must record every current Step 7 split and mapping.")
    verify_artifacts(SPLIT_DIR.parent, SOURCE_CONTRACT["input_sha256"])
    if Path(embedding_report["source_model_dir"]).resolve() != EXPECTED_FINETUNED_MODEL_DIR.resolve():
        raise ValueError("Step 9 encoder directory differs from the verified Step 8 model.")
    return {
        **{key: SOURCE_CONTRACT[key] for key in ("dataset_id", "step8_run_id", "step9_run_id", "step10_run_id")},
        "embedding_report_path": str(EMBEDDING_REPORT_PATH),
        "sequence_report_index_path": str(SEQUENCE_REPORT_INDEX_PATH),
        "embedding_encoder_source": embedding_report.get("embedding_encoder_source"),
        "representation_type": representation_type,
        "source_model_dir": source_model_dir,
        "embedding_dim": embedding_dim,
        "step10_source_embedding_dir": sequence_report.get("source_embedding_dir"),
        "step10_output_dir": sequence_report.get("output_dir"),
        "step10_sequence_lengths": lengths,
        "status": "verified_finetuned_main",
    }


# ============================================================
# Artifact loading: same embeddings; L1 inputs and separate L10 reporting endpoints
# ============================================================

def embedding_npz_path(split_name: str) -> Path:
    return EMBEDDING_DIR / f"{split_name}_window_embeddings.npz"


def sequence_npz_path(split_name: str) -> Path:
    return SEQUENCE_DIR / f"{split_name}_tst_sequences_L{SEQUENCE_LENGTH}.npz"


def sequence_metadata_path(split_name: str) -> Path:
    return SEQUENCE_DIR / f"{split_name}_tst_sequence_metadata_L{SEQUENCE_LENGTH}.csv"


def _load_npz_array(data: np.lib.npyio.NpzFile, keys: List[str], path: Path) -> np.ndarray:
    for key in keys:
        if key in data.files:
            return data[key]
    raise ValueError(f"No expected keys {keys} in {path}. Available keys: {data.files}")


def _npz_scalar_to_string(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
    elif arr.size == 1:
        item = arr.reshape(-1)[0].item()
    else:
        item = arr.tolist()
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def _get_npz_string(data: np.lib.npyio.NpzFile, keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in data.files:
            return _npz_scalar_to_string(data[key]).strip()
    return None


def _get_npz_bool(data: np.lib.npyio.NpzFile, keys: List[str]) -> Optional[bool]:
    value = _get_npz_string(data, keys)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "frozen"}:
        return True
    if normalized in {"0", "false", "no", "n", "finetuned", "fine_tuned"}:
        return False
    return None


def validate_embedding_encoder_metadata(data: np.lib.npyio.NpzFile, path: Path, split_name: str) -> Dict[str, Any]:
    """Validate NPZ metadata when present; Step 9 report is authoritative when absent."""
    encoder_source = _get_npz_string(data, [
        "embedding_encoder_source", "encoder_source", "distilbert_encoder_source", "encoder_checkpoint_source"
    ])
    encoder_model = _get_npz_string(data, [
        "encoder_model_name", "model_name", "distilbert_model_name", "pretrained_model_name"
    ])
    frozen_flag = _get_npz_bool(data, [
        "frozen_encoder", "is_frozen_encoder", "encoder_frozen", "requires_grad_false"
    ])

    meta = {
        "embedding_path": str(path),
        "embedding_encoder_source": encoder_source,
        "encoder_model_name": encoder_model,
        "frozen_encoder": frozen_flag,
        "available_npz_keys": list(data.files),
        "metadata_verification_status": "unchecked",
    }

    if encoder_source is None:
        meta["metadata_verification_status"] = "npz_metadata_absent_report_verified"
        return meta

    source_norm = encoder_source.strip().lower().replace("-", "_")
    if source_norm not in {"finetuned", "fine_tuned", "task_tuned", "task_adapted"}:
        raise ValueError(
            f"[{split_name}] embedding_encoder_source must indicate the fine-tuned encoder, got {encoder_source!r}"
        )
    if encoder_model is not None and EXPECTED_ENCODER_MODEL_NAME not in encoder_model:
        raise ValueError(
            f"[{split_name}] expected encoder model containing {EXPECTED_ENCODER_MODEL_NAME!r}, got {encoder_model!r}"
        )
    meta["metadata_verification_status"] = "verified_finetuned_if_present"
    return meta


def load_embedding_split(split_name: str, split_df: pd.DataFrame) -> Dict[str, Any]:
    path = embedding_npz_path(split_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding NPZ for {split_name}: {path}")
    data = np.load(path, allow_pickle=False)
    encoder_meta = validate_embedding_encoder_metadata(data, path, split_name)

    embeddings = _load_npz_array(data, ["embeddings", "window_embeddings", "E", "X", "arr_0"], path).astype(np.float32, copy=False)
    label_ids = integer_array(_load_npz_array(data, ["label_ids", "labels", "y"], path), f"{split_name}: labels")
    row_indices = integer_array(_load_npz_array(data, ["row_indices", "indices"], path), f"{split_name}: rows")
    data.close()

    if embeddings.ndim != 2:
        raise ValueError(f"[{split_name}] embeddings must be [N,D], got {embeddings.shape}")
    if embeddings.shape[0] != len(split_df):
        raise ValueError(f"[{split_name}] embeddings rows {embeddings.shape[0]} != split CSV rows {len(split_df)}")
    if embeddings.shape[1] != INPUT_EMBEDDING_DIM_EXPECTED:
        raise ValueError(f"[{split_name}] embeddings dim {embeddings.shape[1]} != expected {INPUT_EMBEDDING_DIM_EXPECTED}")
    if len(label_ids) != len(split_df):
        raise ValueError(f"[{split_name}] label_ids length mismatch")
    if not np.array_equal(row_indices, np.arange(len(row_indices), dtype=np.int64)):
        raise ValueError(f"[{split_name}] row_indices are not contiguous 0..N-1")
    if not np.array_equal(label_ids, split_df["label_id"].to_numpy(dtype=np.int64)):
        raise ValueError(f"[{split_name}] embedding label_ids do not match split CSV labels")
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError(f"[{split_name}] embeddings contain NaN or Inf")

    return {"embeddings": embeddings, "label_ids": label_ids, "row_indices": row_indices, "encoder_meta": encoder_meta}


def load_sequence_split(split_name: str, split_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    path = sequence_npz_path(split_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing L{SEQUENCE_LENGTH} sequence NPZ for {split_name}: {path}")
    data = np.load(path, allow_pickle=False)

    sequence_indices = integer_array(_load_npz_array(data, ["sequence_indices", "indices", "X_indices"], path), f"{split_name}: indices", 2)
    y = integer_array(_load_npz_array(data, ["y", "label_ids", "labels"], path), f"{split_name}: y")
    start_row_indices = integer_array(data["start_row_indices"], "start_row_indices")
    end_row_indices = integer_array(data["end_row_indices"], "end_row_indices")
    data.close()

    if sequence_indices.ndim != 2 or sequence_indices.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(f"[{split_name}] sequence_indices must be [N,{SEQUENCE_LENGTH}], got {sequence_indices.shape}")
    if len(y) != len(sequence_indices):
        raise ValueError(f"[{split_name}] y length mismatch")
    if not np.array_equal(sequence_indices[:, 0], start_row_indices):
        raise ValueError(f"[{split_name}] start_row_indices mismatch")
    if not np.array_equal(sequence_indices[:, -1], end_row_indices):
        raise ValueError(f"[{split_name}] end_row_indices mismatch")
    if sequence_indices.size and (int(sequence_indices.min()) < 0 or int(sequence_indices.max()) >= len(split_df)):
        raise ValueError(f"[{split_name}] sequence indices outside split row range")

    expected_y = split_df["label_id"].to_numpy(dtype=np.int64)[end_row_indices]
    if not np.array_equal(y, expected_y):
        raise ValueError(f"[{split_name}] sequence labels are not terminal-window labels")

    if split_name == "test":
        actual = int(len(y))
        expected = expected_test_sequence_count_from_report()
        if actual != expected:
            raise ValueError(
                f"[test] Step 10 report expects {expected:,} L1 sequences, got {actual:,}"
            )

    return {
        "sequence_indices": sequence_indices,
        "y": y,
        "label_ids": y,
        "start_row_indices": start_row_indices,
        "end_row_indices": end_row_indices,
    }


def validate_sequence_count(split_name: str, y: np.ndarray) -> None:
    if split_name == "test":
        actual = int(len(y))
        expected = expected_test_sequence_count_from_report()
        if actual != expected:
            raise ValueError(
                f"[test] L1 full evaluation population mismatch: Step 10 report expects "
                f"{expected:,} sequences, got {actual:,}. "
                "Full L1 inference must precede the separate common-endpoint summary."
            )


def load_sequence_metadata_if_exists(split_name: str) -> Optional[pd.DataFrame]:
    path = sequence_metadata_path(split_name)
    if not path.exists():
        raise FileNotFoundError(f"Required sequence metadata missing: {path}")
    return read_csv_robust(path).reset_index(drop=True)


def validate_lstm_input_artifacts(
    embedding_split: Dict[str, Any],
    sequence_split: Dict[str, np.ndarray],
    split_name: str,
) -> Dict[str, Any]:
    """Validate lazy LSTM input construction without materializing [N,L,D]."""
    E = embedding_split["embeddings"]
    indices = sequence_split["sequence_indices"]
    y = sequence_split["y"].astype(np.int64, copy=False)
    validate_sequence_count(split_name, y)
    if E.ndim != 2 or E.shape[1] != INPUT_EMBEDDING_DIM_EXPECTED:
        raise ValueError(f"[{split_name}] embeddings must be [N,{INPUT_EMBEDDING_DIM_EXPECTED}], got {E.shape}")
    if indices.ndim != 2 or indices.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(f"[{split_name}] sequence_indices must be [N,{SEQUENCE_LENGTH}], got {indices.shape}")
    if indices.shape[0] != len(y):
        raise ValueError(f"[{split_name}] sequence_indices/y length mismatch")
    return {
        "embedding_shape": list(E.shape),
        "sequence_indices_shape": list(indices.shape),
        "label_shape": list(y.shape),
        "lazy_sequence_construction": True,
    }


# ============================================================
# Dataset / model
# ============================================================

class SequenceEmbeddingDataset(Dataset):
    """Lazy dataset: builds each [L,D] sequence from fixed window embeddings on demand."""
    def __init__(self, embeddings: np.ndarray, sequence_indices: np.ndarray, y: np.ndarray):
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [N,D], got {embeddings.shape}")
        if sequence_indices.ndim != 2 or sequence_indices.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(f"sequence_indices must be [N,{SEQUENCE_LENGTH}], got {sequence_indices.shape}")
        if len(sequence_indices) != len(y):
            raise ValueError("sequence_indices/y length mismatch")
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.sequence_indices = sequence_indices.astype(np.int64, copy=False)
        self.y = y.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row_ids = self.sequence_indices[idx]
        x = self.embeddings[row_ids]
        return {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "labels": torch.tensor(int(self.y[idx]), dtype=torch.long),
            "sequence_id": torch.tensor(int(idx), dtype=torch.long),
        }


class FixedFineTunedEmbeddingLSTMClassifier(nn.Module):
    """LSTM classifier over fixed [CLS] embeddings from the fine-tuned DistilBERT encoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_projection = nn.Linear(config.input_dim, config.projected_dim)
        self.input_dropout = nn.Dropout(config.dropout)

        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.projected_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        direction_factor = 2 if config.bidirectional else 1
        self.output_norm = nn.LayerNorm(config.hidden_dim * direction_factor)
        self.output_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim * direction_factor, config.num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,L,D], got {tuple(x.shape)}")
        if x.shape[1] != self.config.sequence_length:
            raise ValueError(f"Expected L={self.config.sequence_length}, got {x.shape[1]}")
        if x.shape[2] != self.config.input_dim:
            raise ValueError(f"Expected D={self.config.input_dim}, got {x.shape[2]}")
        h = self.input_norm(x.float())
        h = self.input_projection(h)
        h = self.input_dropout(h)
        out, _ = self.lstm(h)
        last = out[:, -1, :]
        last = self.output_norm(last)
        last = self.output_dropout(last)
        return self.classifier(last)


def make_loader(
    embeddings: np.ndarray,
    sequence_indices: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        SequenceEmbeddingDataset(embeddings, sequence_indices, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tensor values in a DataLoader batch to the selected device."""
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_class_weights(y_train: np.ndarray, num_classes: int, device: torch.device) -> Optional[torch.Tensor]:
    if not USE_CLASS_WEIGHTS:
        return None
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Cannot use class weights with zero-count class: {counts}")
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ============================================================
# Metrics / evaluation
# ============================================================

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    id_to_label: Dict[int, str],
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if (y_true.ndim != 1 or y_pred.shape != y_true.shape
            or y_true.dtype.kind not in "iu" or y_pred.dtype.kind not in "iu"
            or set(id_to_label) != set(range(num_classes))
            or np.any((y_true < 0) | (y_true >= num_classes))
            or np.any((y_pred < 0) | (y_pred >= num_classes))):
        raise ValueError("Invalid aligned six-class truth/prediction arrays.")
    normal_ids = [i for i, label in id_to_label.items() if label == "NORMAL"]
    if len(normal_ids) != 1:
        raise ValueError("Exactly one NORMAL label is required.")
    normal_mask = y_true == normal_ids[0]
    normal_count = int(normal_mask.sum())
    false_alarms = int(np.count_nonzero(y_pred[normal_mask] != normal_ids[0]))
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    accuracy = float(correct / total) if total else 0.0

    rows: List[Dict[str, Any]] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    supports: List[int] = []

    for class_id in range(num_classes):
        tp = int(cm[class_id, class_id])
        fp = int(cm[:, class_id].sum() - tp)
        fn = int(cm[class_id, :].sum() - tp)
        support = int(cm[class_id, :].sum())
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append({
            "class_id": int(class_id),
            "label": id_to_label.get(class_id, str(class_id)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    support_arr = np.asarray(supports, dtype=np.float64)
    weights = support_arr / support_arr.sum() if support_arr.sum() else np.zeros_like(support_arr)
    summary = {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "weighted_precision": float(np.sum(np.asarray(precisions) * weights)) if len(weights) else 0.0,
        "weighted_recall": float(np.sum(np.asarray(recalls) * weights)) if len(weights) else 0.0,
        "weighted_f1": float(np.sum(np.asarray(f1s) * weights)) if len(weights) else 0.0,
        "total": total,
        "correct": correct,
        "normal_count": normal_count,
        "normal_false_alarm_count": false_alarms,
        "normal_false_alarm_rate": float(false_alarms / normal_count) if normal_count else None,
    }
    per_class_df = pd.DataFrame(rows)
    labels = [id_to_label.get(i, str(i)) for i in range(num_classes)]
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    return summary, per_class_df, cm_df


def save_confusion_matrix_png(path: Path, cm_df: pd.DataFrame, title: str) -> None:
    if plt is None or not SAVE_CONFUSION_MATRIX_PNG:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    matrix = cm_df.to_numpy(dtype=np.float64)
    im = ax.imshow(matrix)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(cm_df.columns)))
    ax.set_yticks(np.arange(len(cm_df.index)))
    ax.set_xticklabels([str(c).replace("pred_", "") for c in cm_df.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([str(i).replace("true_", "") for i in cm_df.index], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    id_to_label: Dict[int, str],
    desc: str,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_logits: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_sequence_ids: List[np.ndarray] = []

    iterator = progress_bar(loader, total=len(loader), desc=desc)
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        with autocast_context(device):
            logits = model(batch["x"])
            loss = criterion(logits, batch["labels"])
        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        all_logits.append(logits.detach().float().cpu().numpy())
        all_y.append(batch["labels"].detach().cpu().numpy())
        all_sequence_ids.append(batch["sequence_id"].detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0).astype(np.int64)
    sequence_ids = np.concatenate(all_sequence_ids, axis=0).astype(np.int64)
    y_pred = logits_np.argmax(axis=1).astype(np.int64)
    probs = softmax_numpy(logits_np)
    summary, per_class_df, cm_df = compute_metrics(y_true, y_pred, num_classes, id_to_label)
    summary["loss"] = total_loss / total_count if total_count else 0.0
    return {
        "summary": summary,
        "per_class": per_class_df,
        "confusion_matrix": cm_df,
        "sequence_ids": sequence_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }


def save_eval_outputs(
    seed: int,
    split_name: str,
    eval_result: Dict[str, Any],
    id_to_label: Dict[int, str],
    sequence_metadata: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    if sequence_metadata is None or len(sequence_metadata) != len(eval_result["y_true"]):
        raise ValueError("Verified full L1 endpoint metadata is required for every prediction.")
    ids = integer_array(eval_result["sequence_ids"], "prediction sequence_ids")
    if (not np.array_equal(ids, np.arange(len(sequence_metadata)))
            or not np.array_equal(eval_result["y_true"], sequence_metadata["label_id"].to_numpy())):
        raise ValueError("Prediction sequence IDs/labels differ from the verified metadata.")
    probs = eval_result["probs"]
    if (probs.shape != (len(ids), len(id_to_label)) or not np.isfinite(probs).all()
            or np.any((probs < 0) | (probs > 1)) or not np.allclose(probs.sum(axis=1), 1, atol=2e-6, rtol=0)
            or not np.array_equal(probs.argmax(axis=1), eval_result["y_pred"])):
        raise ValueError("Invalid prediction probabilities; no completed evaluation can be recorded.")
    common_mask = verified_common_mask(sequence_metadata, split_name)
    common_summary, common_per_class, common_confusion = compute_metrics(
        eval_result["y_true"][common_mask], eval_result["y_pred"][common_mask], len(id_to_label), id_to_label)
    if common_summary["normal_count"] <= 0:
        raise ValueError("No NORMAL endpoints in the common evaluation cohort.")
    summary = dict(eval_result["summary"])
    summary["common_population"] = common_summary
    summary.update(result_provenance())
    summary.update(seed=int(seed), split=split_name,
                   evaluation_population="all eligible L1 endpoints; common_population is the verified L10 intersection",
                   common_endpoint_count=INPUT_AUDIT[split_name]["common_endpoint_count"],
                   sorted_end_row_indices_sha256=INPUT_AUDIT[split_name]["sorted_end_row_indices_sha256"])
    prefix = f"seed{seed}_{split_name}"
    save_json(REPORT_DIR / f"{prefix}_common_metrics.json", {
        **common_summary, **result_provenance(), "seed": int(seed), "split": split_name,
        "sequence_length": SEQUENCE_LENGTH, "evaluation_population": "common_L10_eligible_endpoints",
        "sorted_end_row_indices_sha256": INPUT_AUDIT[split_name]["sorted_end_row_indices_sha256"]})
    save_dataframe(REPORT_DIR / f"{prefix}_common_per_class.csv", common_per_class)
    save_dataframe(REPORT_DIR / f"{prefix}_common_confusion_matrix.csv", common_confusion.reset_index())
    effects = OBSERVABLE_EFFECTS[split_name]
    diagnostic = compute_observable_effect_diagnostics(
        eval_result["y_true"], eval_result["y_pred"], sequence_metadata, effects, id_to_label, split_name)
    diagnostic.update(provenance=result_provenance(), source_audit=SOURCE_CONTRACT["observability_audit"], seed=int(seed))
    diagnostic_path = REPORT_DIR / f"{prefix}_observable_effect_diagnostics.json"
    summary["observable_effect_diagnostics"] = str(diagnostic_path.resolve())
    save_json(diagnostic_path, diagnostic)
    save_json(REPORT_DIR / f"{prefix}_metrics.json", summary)
    save_dataframe(REPORT_DIR / f"{prefix}_per_class.csv", eval_result["per_class"])
    save_dataframe(REPORT_DIR / f"{prefix}_confusion_matrix.csv", eval_result["confusion_matrix"].reset_index())
    save_confusion_matrix_png(
        REPORT_DIR / f"{prefix}_confusion_matrix.png",
        eval_result["confusion_matrix"],
        title=f"Fixed fine-tuned DistilBERT L1 LSTM {split_name} seed={seed}",
    )

    if not SAVE_PREDICTIONS:
        return common_summary
    y_true = eval_result["y_true"].astype(np.int64)
    y_pred = eval_result["y_pred"].astype(np.int64)
    probs = eval_result["probs"]
    pred_df = pd.DataFrame({
        "sequence_id": eval_result["sequence_ids"].astype(np.int64),
        "true_label_id": y_true,
        "pred_label_id": y_pred,
        "true_label": [id_to_label[int(x)] for x in y_true],
        "pred_label": [id_to_label[int(x)] for x in y_pred],
        "correct": (y_true == y_pred).astype(np.int64),
        "confidence": probs.max(axis=1),
    })
    for class_id in range(probs.shape[1]):
        pred_df[f"prob_{safe_name(id_to_label[class_id])}"] = probs[:, class_id]
    for key, value in result_provenance().items():
        pred_df[key] = value
    pred_df["seed"] = int(seed)
    pred_df["common_population"] = common_mask.astype(np.int64)

    if sequence_metadata is not None and len(sequence_metadata) == len(pred_df):
        useful_cols = [
            col for col in [
                "episode_id", "episode_type", "scenario_id", "sequence_start_row",
                "sequence_end_row", "sequence_start_step", "sequence_end_step",
                "sequence_start_timestamp", "sequence_end_timestamp", "start_label", "label",
                "sequence_length", "temporal_context_seconds",
            ] if col in sequence_metadata.columns
        ]
        if useful_cols:
            pred_df = pd.concat([pred_df, sequence_metadata[useful_cols].reset_index(drop=True)], axis=1)

    pred_df["source_split"] = split_name
    pred_df = pd.concat([pred_df, effects[OBSERVABLE_EFFECT_AUDIT_COLUMNS].reset_index(drop=True)], axis=1)
    save_dataframe(PREDICTION_DIR / f"{prefix}_predictions.csv", pred_df)
    return common_summary


# ============================================================
# Training loop
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[Any],
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    correct = 0
    iterator = progress_bar(loader, total=len(loader), desc="train")
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda" and scaler is not None:
            with autocast_context(device):
                logits = model(batch["x"])
                loss = criterion(logits, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch["x"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        correct += int((torch.argmax(logits.detach(), dim=1) == batch["labels"]).sum().item())

    return {
        "loss": total_loss / total_count if total_count else 0.0,
        "accuracy": correct / total_count if total_count else 0.0,
    }


def train_for_seed(
    seed: int,
    E_train: np.ndarray,
    idx_train: np.ndarray,
    y_train: np.ndarray,
    E_val: np.ndarray,
    idx_val: np.ndarray,
    y_val: np.ndarray,
    E_test: np.ndarray,
    idx_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    id_to_label: Dict[int, str],
    metadata_splits: Dict[str, Optional[pd.DataFrame]],
    device: torch.device,
) -> Dict[str, Any]:
    print("\n" + "-" * 78)
    print(f"Train Fixed fine-tuned DistilBERT L1 LSTM baseline | seed={seed}")
    print("-" * 78)
    set_all_seeds(seed)

    model_config = ModelConfig(
        input_dim=int(E_train.shape[1]),
        projected_dim=PROJECT_TO_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        bidirectional=BIDIRECTIONAL,
        sequence_length=SEQUENCE_LENGTH,
        num_classes=num_classes,
    )
    train_config = TrainConfig(
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM,
        use_amp=USE_AMP,
        use_class_weights=USE_CLASS_WEIGHTS,
        label_smoothing=LABEL_SMOOTHING,
    )

    train_loader = make_loader(E_train, idx_train, y_train, BATCH_SIZE, shuffle=True, device=device)
    val_loader = make_loader(E_val, idx_val, y_val, BATCH_SIZE, shuffle=False, device=device)
    test_loader = make_loader(E_test, idx_test, y_test, BATCH_SIZE, shuffle=False, device=device)

    model = FixedFineTunedEmbeddingLSTMClassifier(model_config).to(device)
    class_weights = compute_class_weights(y_train, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    scaler = make_grad_scaler(device)

    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history_rows: List[Dict[str, Any]] = []
    best_ckpt_path = MODEL_DIR / f"best_lstm_finetuned_distilbert_L{SEQUENCE_LENGTH}_seed{seed}.pt"
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n[seed={seed}] Epoch {epoch}/{MAX_EPOCHS}")
        train_summary = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_eval = evaluate_model(model, val_loader, criterion, device, num_classes, id_to_label, desc="val")
        val_summary = val_eval["summary"]
        scheduler.step(float(val_summary["macro_f1"]))
        lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "seed": int(seed),
            "epoch": int(epoch),
            "learning_rate": lr,
            "train_loss": float(train_summary["loss"]),
            "train_accuracy": float(train_summary["accuracy"]),
            "val_loss": float(val_summary["loss"]),
            "val_accuracy": float(val_summary["accuracy"]),
            "val_macro_precision": float(val_summary["macro_precision"]),
            "val_macro_recall": float(val_summary["macro_recall"]),
            "val_macro_f1": float(val_summary["macro_f1"]),
            "val_weighted_f1": float(val_summary["weighted_f1"]),
        }
        history_rows.append(row)
        print(
            f"[seed={seed}] train_loss={row['train_loss']:.4f}, "
            f"train_acc={row['train_accuracy']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, "
            f"val_acc={row['val_accuracy']:.4f}, "
            f"val_macro_f1={row['val_macro_f1']:.4f}, lr={lr:.2e}"
        )

        current_f1 = float(val_summary["macro_f1"])
        current_loss = float(val_summary["loss"])
        is_better = (
            current_f1 > best_val_macro_f1 + 1e-12
            or (abs(current_f1 - best_val_macro_f1) <= 1e-12 and current_loss < best_val_loss)
        )
        if is_better:
            best_val_macro_f1 = current_f1
            best_val_loss = current_loss
            best_epoch = epoch
            bad_epochs = 0
            if SAVE_MODEL_CHECKPOINTS:
                torch.save({
                    "provenance": result_provenance(),
                    "source_contract": SOURCE_CONTRACT,
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "train_config": asdict(train_config),
                    "best_epoch": int(best_epoch),
                    "best_val_summary": val_summary,
                    "seed": int(seed),
                    "input_representation": "fixed fine-tuned DistilBERT [CLS] embedding sequence",
                    "sequence_length": int(SEQUENCE_LENGTH),
                    "temporal_context_seconds": int(TEMPORAL_CONTEXT_SECONDS),
                }, best_ckpt_path)
            print(f"[seed={seed}] Best checkpoint updated: epoch={best_epoch}, val_macro_f1={best_val_macro_f1:.5f}")
        else:
            bad_epochs += 1
            print(f"[seed={seed}] No improvement: {bad_epochs}/{PATIENCE}")
            if bad_epochs >= PATIENCE:
                print(f"[seed={seed}] Early stopping at epoch {epoch}")
                break

    elapsed_seconds = float(time.time() - start_time)
    save_dataframe(REPORT_DIR / f"seed{seed}_training_history.csv", pd.DataFrame(history_rows))

    if SAVE_MODEL_CHECKPOINTS and best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        if checkpoint.get("provenance") != result_provenance():
            raise ValueError("Best checkpoint belongs to a different data/model run.")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise FileNotFoundError(f"Validation-selected checkpoint missing: {best_ckpt_path}")

    val_eval = evaluate_model(model, val_loader, criterion, device, num_classes, id_to_label, desc="val_best")
    test_eval = evaluate_model(model, test_loader, criterion, device, num_classes, id_to_label, desc="test_best")

    val_common = save_eval_outputs(seed, "val", val_eval, id_to_label, metadata_splits.get("val"))
    test_common = save_eval_outputs(seed, "test", test_eval, id_to_label, metadata_splits.get("test"))

    result = {
        **result_provenance(),
        "seed": int(seed),
        "model": "Fixed fine-tuned DistilBERT L1 LSTM",
        "input_representation": "Fixed fine-tuned DistilBERT [CLS] embedding sequence",
        "temporal_modeling": "LSTM head with one timestep; no cross-window history",
        "sequence_length": int(SEQUENCE_LENGTH),
        "temporal_context_seconds": int(TEMPORAL_CONTEXT_SECONDS),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "elapsed_seconds": elapsed_seconds,
        "accuracy": float(test_eval["summary"]["accuracy"]),
        "macro_precision": float(test_eval["summary"]["macro_precision"]),
        "macro_recall": float(test_eval["summary"]["macro_recall"]),
        "macro_f1": float(test_eval["summary"]["macro_f1"]),
        "weighted_f1": float(test_eval["summary"]["weighted_f1"]),
        "total": int(test_eval["summary"]["total"]),
        "test_common_count": INPUT_AUDIT["test"]["common_endpoint_count"],
        "normal_count": int(test_eval["summary"]["normal_count"]),
        "normal_false_alarm_count": int(test_eval["summary"]["normal_false_alarm_count"]),
        "normal_false_alarm_rate": float(test_eval["summary"]["normal_false_alarm_rate"]),
        "test_common_accuracy": float(test_common["accuracy"]),
        "test_common_macro_f1": float(test_common["macro_f1"]),
        "test_common_weighted_f1": float(test_common["weighted_f1"]),
        "test_common_normal_count": int(test_common["normal_count"]),
        "test_common_normal_false_alarm_count": int(test_common["normal_false_alarm_count"]),
        "test_common_normal_false_alarm_rate": float(test_common["normal_false_alarm_rate"]),
        "val_common_macro_f1": float(val_common["macro_f1"]),
        "val_common_normal_false_alarm_rate": float(val_common["normal_false_alarm_rate"]),
        "checkpoint_path": str(best_ckpt_path) if SAVE_MODEL_CHECKPOINTS else "",
    }
    save_json(REPORT_DIR / f"seed{seed}_result_summary.json", result)

    print(
        f"[seed={seed}][test] acc={result['accuracy']:.4f}, "
        f"full_macro_f1={result['macro_f1']:.4f}, full_total={result['total']}; "
        f"COMMON F1={result['test_common_macro_f1']:.6f}, "
        f"FA_N={100 * result['test_common_normal_false_alarm_rate']:.3f}%, "
        f"n={result['test_common_count']}"
    )
    return result


# ============================================================
# Summary helpers
# ============================================================

def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    if ("seed" not in results_df or len(results_df) != len(SEEDS)
            or results_df["seed"].duplicated().any() or set(results_df["seed"]) != set(SEEDS)):
        raise ValueError("Incomplete/duplicate requested seed grid; no summary can be generated.")
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "normal_false_alarm_rate", "test_common_accuracy", "test_common_macro_f1", "test_common_weighted_f1", "test_common_normal_false_alarm_rate"]
    values = results_df[metrics].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Nonfinite/out-of-range scores cannot be averaged.")
    if results_df["total"].nunique() != 1 or results_df["total"].iloc[0] <= 0:
        raise ValueError("All runs must use the same nonempty evaluation population.")
    for column in ("test_common_count", "test_common_normal_count"):
        if results_df[column].nunique() != 1 or results_df[column].iloc[0] <= 0:
            raise ValueError(f"Inconsistent common-population denominator: {column}")
    rows: List[Dict[str, Any]] = []
    row: Dict[str, Any] = {
        "test_common_count": int(results_df["test_common_count"].iloc[0]),
        "test_common_normal_count": int(results_df["test_common_normal_count"].iloc[0]),
        "model": "Fixed fine-tuned DistilBERT L1 LSTM",
        "input_representation": "Fixed fine-tuned DistilBERT [CLS] embedding sequence",
        "temporal_modeling": "LSTM head with one timestep; no cross-window history",
        "sequence_length": int(SEQUENCE_LENGTH),
        "n_seeds": int(len(results_df)),
        "total_mean": float(results_df["total"].mean()) if len(results_df) else np.nan,
    }
    for metric in metrics:
        row[f"{metric}_mean"] = float(results_df[metric].mean())
        row[f"{metric}_std"] = float(results_df[metric].std(ddof=1)) if len(results_df) > 1 else None
        row[f"{metric}_mean_std"] = (f"{row[f'{metric}_mean']:.4f} $\\pm$ {row[f'{metric}_std']:.4f}"
            if len(results_df) > 1 else f"{row[f'{metric}_mean']:.4f} (SD n/a)")
    rows.append(row)
    return pd.DataFrame(rows)


def make_paper_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Only the common endpoint result belongs in the manuscript comparison."""
    if summary_df.empty:
        return pd.DataFrame()
    s = summary_df.iloc[0]
    return pd.DataFrame([{
        "Detector": "DistilBERT + LSTM", "L": SEQUENCE_LENGTH,
        "Evaluation population": "common_L10_eligible_test_endpoints",
        "Endpoints": int(s["test_common_count"]),
        "Normal endpoints": int(s["test_common_normal_count"]),
        "Head seeds": int(s["n_seeds"]),
        "Macro-F1": float(s["test_common_macro_f1_mean"]),
        "Macro-F1 SD": s["test_common_macro_f1_std"],
        "FA_N (%)": 100 * float(s["test_common_normal_false_alarm_rate_mean"]),
        "FA_N SD (percentage points)": 100 * float(s["test_common_normal_false_alarm_rate_std"])
            if s["test_common_normal_false_alarm_rate_std"] is not None else None,
    }])


def write_latex_rows(path: Path, summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        atomic_text(path, "")
        return
    s = summary_df.iloc[0]
    cells = f"{s['test_common_macro_f1_mean']:.3f} & {100 * s['test_common_normal_false_alarm_rate_mean']:.1f}\\%"
    atomic_text(path, "% L=1 cells only: replace the two LSTM L=1 dashes; preserve the verified L=10 cells.\n"
                      "% Common L10-eligible endpoints; means over head seeds.\n" + cells + "\n")


# ============================================================
# Main
# ============================================================

def run_experiment(retry_failed: bool = False, check_inputs_only: bool = False) -> None:
    global INPUT_AUDIT
    print("=" * 78)
    print("Step 14: Fixed fine-tuned DistilBERT L1 LSTM baseline")
    print("=" * 78)
    print(f"Base directory     : {BASE_DIR.resolve()}")
    print(f"Split directory    : {SPLIT_DIR.resolve()}")
    print(f"Embedding directory: {EMBEDDING_DIR.resolve()}")
    print(f"Sequence directory : {SEQUENCE_DIR.resolve()}")
    print(f"Output root        : {OUTPUT_ROOT.resolve()}")
    print(f"Seeds              : {SEEDS}")
    print(f"Training           : lr={LEARNING_RATE}, wd={WEIGHT_DECAY}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}, label_smoothing={LABEL_SMOOTHING}")

    print("\n[Revised fine-tuned pipeline provenance check]")
    provenance = validate_revised_pipeline_provenance()
    for key, value in provenance.items():
        print(f"  {key}: {value}")

    label_order, label_to_id, id_to_label, label_manifest = load_label_config()
    print("\n[Label mapping]")
    for label in label_order:
        print(f"  {label_to_id[label]} -> {label}")

    print("\n[Load observable splits]")
    splits = load_observable_splits(label_to_id)
    for split_name, df in splits.items():
        print(f"  {split_name}: rows={len(df):,}")
        print(df["label"].value_counts().reindex(label_order).fillna(0).astype(int))

    print("\n[Load fixed fine-tuned DistilBERT embeddings and L1 sequence artifacts]")
    embedding_splits: Dict[str, Dict[str, Any]] = {}
    sequence_splits: Dict[str, Dict[str, np.ndarray]] = {}
    metadata_splits: Dict[str, Optional[pd.DataFrame]] = {}
    input_validation: Dict[str, Dict[str, Any]] = {}

    for split_name in ["train", "val", "test"]:
        embedding_splits[split_name] = load_embedding_split(split_name, splits[split_name])
        sequence_splits[split_name] = load_sequence_split(split_name, splits[split_name])
        metadata_splits[split_name] = load_sequence_metadata_if_exists(split_name)
        input_validation[split_name] = validate_lstm_input_artifacts(
            embedding_splits[split_name], sequence_splits[split_name], split_name
        )
        print(
            f"  {split_name}: embeddings={embedding_splits[split_name]['embeddings'].shape}, "
            f"sequence_indices={sequence_splits[split_name]['sequence_indices'].shape}, "
            f"y={sequence_splits[split_name]['y'].shape} | lazy=True"
        )

    INPUT_AUDIT = validate_input_alignment(splits, embedding_splits, sequence_splits, metadata_splits)
    for split, info in INPUT_AUDIT.items():
        print(f"[Verified {split}] full L1={info['full_endpoint_count']:,}; common L10 endpoints={info['common_endpoint_count']:,}")
    if check_inputs_only:
        print("Input validation passed. No training or evaluation results were created.")
        return
    input_dim = int(embedding_splits["train"]["embeddings"].shape[1])
    num_classes = len(id_to_label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if torch.cuda.is_available():
        print(f"  CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"  AMP     : {USE_AMP}")

    manifest = {
        "experiment": "finetuned_distilbert_l1_lstm_baseline",
        "purpose": "L1 LSTM companion; full L1 fitting/selection, common L10-eligible endpoint reporting",
        "input_representation": "Fixed fine-tuned DistilBERT [CLS] embedding sequence",
        "sequence_length": int(SEQUENCE_LENGTH),
        "temporal_context_seconds": int(TEMPORAL_CONTEXT_SECONDS),
        "same_as_tst": {
            "fixed_finetuned_distilbert_embeddings": True,
            "episode_bounded_l1_sequence_indices": True,
            "terminal_window_labels": True,
            "train_val_test_splits": True,
        },
        "not_used_as_features": [
            "scenario_id", "episode_id", "episode_type", "episode_step", "window_id", "timestamp", "label", "split", *OBSERVABLE_EFFECT_AUDIT_COLUMNS
        ],
        "label_order": label_order,
        "label_to_id": label_to_id,
        "input_shapes": input_validation,
        "revised_pipeline_provenance": provenance,
        "embedding_encoder_metadata": {
            split: embedding_splits[split].get("encoder_meta", {}) for split in ["train", "val", "test"]
        },
        "expected_test_full_l1_sequence_count": int(expected_test_sequence_count_from_report()),
        "expected_test_common_endpoint_count": INPUT_AUDIT["test"]["common_endpoint_count"],
        "embedding_metadata_policy": {
            "authoritative_provenance": "Step 9/Step 10 JSON reports",
            "npz_metadata": "validated when present; report provenance used when absent",
        },
        "model_config": asdict(ModelConfig(
            input_dim=input_dim,
            projected_dim=PROJECT_TO_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT,
            bidirectional=BIDIRECTIONAL,
            sequence_length=SEQUENCE_LENGTH,
            num_classes=num_classes,
        )),
        "train_config": asdict(TrainConfig(
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            grad_clip_norm=GRAD_CLIP_NORM,
            use_amp=USE_AMP,
            use_class_weights=USE_CLASS_WEIGHTS,
            label_smoothing=LABEL_SMOOTHING,
        )),
        "source_artifacts": {
            "embedding_dir": str(EMBEDDING_DIR),
            "sequence_dir": str(SEQUENCE_DIR),
            "split_dir": str(SPLIT_DIR),
        },
    }
    if not prepare_run(manifest, device, retry_failed):
        return
    save_json(OUTPUT_ROOT / "input_validation_report.json", INPUT_AUDIT)
    save_json(OUTPUT_ROOT / "lstm_finetuned_distilbert_l1_manifest.json", manifest)

    results: List[Dict[str, Any]] = []
    for seed in SEEDS:
        completed = ACTIVE_RUN["completed_runs"].get(str(seed))
        if completed is not None:
            results.append(completed["result"])
            print(f"[Verified completed seed reused] {seed}")
            continue
        ACTIVE_RUN["active_trial"] = {"seed": int(seed)}
        save_json(OUTPUT_ROOT / "run_manifest.json", ACTIVE_RUN)
        result = train_for_seed(
            seed=seed,
            E_train=embedding_splits["train"]["embeddings"],
            idx_train=sequence_splits["train"]["sequence_indices"],
            y_train=sequence_splits["train"]["y"],
            E_val=embedding_splits["val"]["embeddings"],
            idx_val=sequence_splits["val"]["sequence_indices"],
            y_val=sequence_splits["val"]["y"],
            E_test=embedding_splits["test"]["embeddings"],
            idx_test=sequence_splits["test"]["sequence_indices"],
            y_test=sequence_splits["test"]["y"],
            num_classes=num_classes,
            id_to_label=id_to_label,
            metadata_splits=metadata_splits,
            device=device,
        )
        record_completed_trial(seed, result)
        results.append(result)

    results_df = pd.DataFrame(results)
    summary_df = summarize_results(results_df)
    paper_table_df = make_paper_table(summary_df)

    results_csv = OUTPUT_ROOT / "lstm_finetuned_distilbert_l1_results_by_seed.csv"
    summary_csv = OUTPUT_ROOT / "lstm_finetuned_distilbert_l1_summary.csv"
    paper_csv = OUTPUT_ROOT / "lstm_finetuned_distilbert_l1_paper_table.csv"
    latex_path = OUTPUT_ROOT / "lstm_finetuned_distilbert_l1_latex_rows.txt"
    summary_json = OUTPUT_ROOT / "lstm_finetuned_distilbert_l1_summary.json"

    save_dataframe(results_csv, results_df)
    save_dataframe(summary_csv, summary_df)
    save_dataframe(paper_csv, paper_table_df)
    write_latex_rows(latex_path, summary_df)
    save_json(summary_json, {
        "manifest": manifest,
        "results_by_seed": results,
        "summary": summary_df.to_dict(orient="records"),
    })

    print("\n" + "=" * 78)
    finish_run()
    print("Step 14 completed")
    print("=" * 78)
    print("\n[Results by seed]")
    display_cols = ["seed", "test_common_count", "test_common_macro_f1", "test_common_normal_false_alarm_rate", "test_common_normal_false_alarm_count", "test_common_normal_count"]
    print(results_df[display_cols].to_string(index=False))

    print("\n[Summary]")
    print(summary_df[[
        "model", "test_common_macro_f1_mean_std", "test_common_normal_false_alarm_rate_mean_std"
    ]].to_string(index=False))

    print("\nSaved artifacts:")
    print(f"- Results by seed : {results_csv.resolve()}")
    print(f"- Summary CSV     : {summary_csv.resolve()}")
    print(f"- Paper table CSV : {paper_csv.resolve()}")
    print(f"- LaTeX row       : {latex_path.resolve()}")
    print(f"- Reports         : {REPORT_DIR.resolve()}")
    print(f"- Predictions     : {PREDICTION_DIR.resolve()}")
    print(f"- Models          : {MODEL_DIR.resolve()}")



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
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
        if (not path.is_relative_to(root.resolve()) or not path.is_file()
                or not isinstance(expected, str) or sha256_file(path) != expected):
            raise ValueError(f"Missing/changed artifact: {path}")


def integer_array(value: Any, context: str, ndim: int = 1) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != ndim or arr.dtype.kind not in "iu" or not arr.size:
        raise ValueError(f"{context}: expected a nonempty {ndim}-D integer array; got {arr.shape}/{arr.dtype}")
    if arr.dtype.kind == "u" and arr.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{context}: integer overflow")
    return arr.astype(np.int64, copy=False)


def endpoint_digest(endpoints: np.ndarray) -> str:
    return hashlib.sha256(np.sort(endpoints).astype("<i8", copy=False).tobytes()).hexdigest()


def load_source_contract(extraction: Dict[str, Any], index: Dict[str, Any]) -> Dict[str, Any]:
    """Authenticate the completed artifacts actually consumed, before creating outputs."""
    root8 = EXPECTED_FINETUNED_MODEL_DIR.parent
    roots = {"step8": root8, "step9": EMBEDDING_DIR, "step10": SEQUENCE_DIR}
    runs = {key: load_json_required(root / "run_manifest.json") for key, root in roots.items()}
    for key, run in runs.items():
        if run.get("status") != "COMPLETE" or not run.get("run_id") or not run.get("dataset_id"):
            raise ValueError(f"{key} is not a completed revised run.")
    r8, r9, r10 = runs["step8"], runs["step9"], runs["step10"]
    if extraction.get("status") != "COMPLETE" or index.get("status") != "COMPLETE":
        raise ValueError("Step 9/10 reports must both be COMPLETE.")
    dataset_id = r8["dataset_id"]
    if any(record.get("dataset_id") != dataset_id for record in (r9, r10, extraction, index)):
        raise ValueError("Mixed dataset_id across Steps 8/9/10.")
    for record in (r9, r10, extraction, index):
        if record.get("step8_run_id") != r8["run_id"]:
            raise ValueError("Mixed Step 8 encoder run IDs.")
    if (extraction.get("run_id") != r9["run_id"] or index.get("run_id") != r10["run_id"]
            or r10.get("step9_run_id") != r9["run_id"] or index.get("step9_run_id") != r9["run_id"]):
        raise ValueError("Mixed Step 9/10 run IDs.")
    source9, source10 = r9.get("source_contract", {}), r10.get("source", {})
    model_hashes = {name: digest for name, digest in r8.get("artifacts", {}).items()
                    if name.startswith("trained_distilbert_model/")}
    if not any(name.endswith((".safetensors", ".bin")) and "model" in Path(name).name
               for name in model_hashes):
        raise ValueError("Step 8 completion record has no model weights.")
    present = {path.relative_to(root8).as_posix() for path in EXPECTED_FINETUNED_MODEL_DIR.rglob("*")
               if path.is_file()}
    if present != set(model_hashes):
        raise ValueError("Step 8 model directory differs from its completed inventory.")
    recorded9_models = {name: value for name, value in source9.get("step8_artifacts", {}).items()
                        if name.startswith("trained_distilbert_model/")}
    if (model_hashes != recorded9_models or model_hashes != extraction.get("step8_model_sha256")
            or source9.get("step8_manifest_sha256") != sha256_file(root8 / "run_manifest.json")
            or source9.get("dataset_id") != dataset_id or source9.get("step8_run_id") != r8["run_id"]
            or not r8.get("input_sha256") or source9.get("input_sha256") != r8["input_sha256"]
            or extraction.get("input_sha256") != r8["input_sha256"]):
        raise ValueError("Step 8 -> 9 model/input provenance mismatch.")
    verify_artifacts(root8, model_hashes)
    names9 = ["embedding_extraction_report.json"]
    for split in SPLITS:
        names9 += [f"{split}_window_embeddings.npz", f"{split}_window_metadata.csv"]
    hashes9 = {name: r9.get("artifacts", {}).get(name) for name in names9}
    verify_artifacts(EMBEDDING_DIR, hashes9)
    if (r9.get("encoder_source") != "finetuned" or r9.get("encoder") != extraction.get("encoder_source_summary")
            or extraction.get("encoder_source_summary", {}).get("representation_type") != "supervised_finetuned_distilbert_cls"
            or extraction.get("encoder_source_summary", {}).get("pooling") != "last_hidden_state[:, 0, :]"
            or extraction.get("encoder_source_summary", {}).get("normalization") != "none"):
        raise ValueError("Step 9 must contain unnormalized fine-tuned CLS embeddings.")
    if set(r9.get("completed_splits", {})) != set(SPLITS) or set(extraction.get("splits", {})) != set(SPLITS):
        raise ValueError("Step 9 is missing completed splits.")
    for split in SPLITS:
        completed = r9["completed_splits"][split]
        if completed.get("summary") != extraction["splits"][split]:
            raise ValueError(f"{split}: Step 9 summary mismatch.")
        for suffix in ("window_embeddings.npz", "window_metadata.csv"):
            name = f"{split}_{suffix}"
            if completed.get("artifacts", {}).get(name) != hashes9[name]:
                raise ValueError(f"{split}: Step 9 split hash mismatch.")
    if (source10.get("dataset_id") != dataset_id or source10.get("step8_run_id") != r8["run_id"]
            or source10.get("step9_run_id") != r9["run_id"] or source10.get("artifacts") != hashes9
            or source10.get("step9_manifest_sha256") != sha256_file(EMBEDDING_DIR / "run_manifest.json")
            or index.get("source_artifact_sha256") != hashes9):
        raise ValueError("Step 9 -> 10 input provenance mismatch.")
    protocol10 = r10.get("protocol", {})
    if (protocol10.get("stride") != 1 or protocol10.get("sequence_lengths") != SEQUENCE_LENGTHS
            or protocol10.get("save_metadata") is not True or index.get("stride") != 1):
        raise ValueError("Step 14 requires the complete stride-1 Step 10 sequence/metadata protocol.")
    build = source9.get("step7_build", {})
    policy = build.get("policy", {})
    dataset_root = SPLIT_DIR.parent.resolve()
    if (policy.get("nominal_substrate") != "shared_nominal_v2"
            or policy.get("columns") != METADATA_FIELDS + OBSERVABLE_FIELDS
            or Path(source9.get("source_dataset_dir", "")).resolve() != dataset_root
            or sha256_file(dataset_root / "build_manifest.json") != source9.get("step7_build_manifest_sha256")
            or load_json_required(dataset_root / "build_manifest.json") != build):
        raise ValueError("Use the same completed shared_nominal_v2 Step 7 dataset consumed by Steps 8–10.")
    observability_audit = source10.get("observability_audit", {})
    validate_observability_source_contract(observability_audit, source9, r8, dataset_id)
    if index.get("observability_audit") != observability_audit:
        raise ValueError("Step 10 index and manifest observable-effect audit contracts differ.")
    names10 = ["sequence_build_report_index.json"]
    for length in SEQUENCE_LENGTHS:
        names10 += [f"sequence_build_report_L{length}.json"]
        for split in SPLITS:
            names10 += [f"{split}_tst_sequences_L{length}.npz", f"{split}_tst_sequence_metadata_L{length}.csv",
                        sequence_observable_effects_path(split, length).name]
    hashes10 = {name: r10.get("artifacts", {}).get(name) for name in names10}
    verify_artifacts(SEQUENCE_DIR, hashes10)
    for length in SEQUENCE_LENGTHS:
        report = load_json_required(SEQUENCE_DIR / f"sequence_build_report_L{length}.json")
        for key in ("dataset_id", "step8_run_id", "step9_run_id", "run_id", "source_artifact_sha256", "observability_audit"):
            if report.get(key) != index.get(key):
                raise ValueError(f"L{length}: Step 10 report mismatch: {key}")
        for split in SPLITS:
            path = sequence_observable_effects_path(split, length)
            summary = report.get("splits", {}).get(split, {})
            if (summary.get("sequence_observable_effects_sha256") != hashes10[path.name]
                    or Path(summary.get("sequence_observable_effects_csv_path", "")).resolve() != path.resolve()):
                raise ValueError(f"{split}/L{length}: Step 10 diagnostic file/hash mismatch.")
    return {"dataset_id": dataset_id, "step8_run_id": r8["run_id"], "step9_run_id": r9["run_id"],
            "step10_run_id": r10["run_id"], "input_sha256": r8["input_sha256"],
            "step8_model_sha256": model_hashes, "step9_artifacts": hashes9, "step10_artifacts": hashes10,
            "manifest_sha256": {key: sha256_file(root / "run_manifest.json") for key, root in roots.items()},
            "common_endpoint_report": index.get("common_endpoint_report", {}),
            "observability_audit": observability_audit}


def validate_sequence_alignment(split: str, length: int, arrays: Dict[str, np.ndarray],
                                meta: pd.DataFrame, windows: pd.DataFrame) -> None:
    """Check every selected window, not only the sequence's terminal label."""
    indices, ends, starts = arrays["sequence_indices"], arrays["end_row_indices"], arrays["start_row_indices"]
    if len(meta) != len(indices) or len(np.unique(ends)) != len(ends):
        raise ValueError(f"{split}/L{length}: duplicate endpoints or metadata row count mismatch.")
    expected_numeric = {"sequence_id": np.arange(len(meta)), "sequence_start_row": starts,
        "sequence_end_row": ends, "label_id": arrays["y"],
        "sequence_start_step": windows["episode_step"].to_numpy()[starts],
        "sequence_end_step": windows["episode_step"].to_numpy()[ends],
        "sequence_start_window_id": windows["window_id"].to_numpy()[starts],
        "sequence_end_window_id": windows["window_id"].to_numpy()[ends],
        "sequence_length": np.full(len(meta), length), "observable_window_seconds": np.ones(len(meta), dtype=int),
        "temporal_context_seconds": np.full(len(meta), length)}
    for column, expected in expected_numeric.items():
        if column not in meta or not np.array_equal(integer_array(meta[column], column), expected):
            raise ValueError(f"{split}/L{length}: misaligned sequence metadata {column}.")
    for col, source, rows in [("episode_id", "episode_id", ends), ("episode_type", "episode_type", ends),
                             ("scenario_id", "scenario_id", ends), ("label", "label", ends),
                             ("start_label", "label", starts), ("sequence_start_timestamp", "timestamp", starts),
                             ("sequence_end_timestamp", "timestamp", ends)]:
        if col not in meta or not np.array_equal(meta[col].astype(str).to_numpy(), windows[source].astype(str).to_numpy()[rows]):
            raise ValueError(f"{split}/L{length}: misaligned sequence metadata {col}.")
    if "source_split" not in meta or not meta["source_split"].eq(split).all():
        raise ValueError(f"{split}/L{length}: incorrect source_split.")
    episode_codes = pd.factorize(windows["episode_id"])[0]
    if not np.all(episode_codes[indices] == episode_codes[ends, None]):
        raise ValueError(f"{split}/L{length}: sequence crosses episode boundary.")
    for column in ("episode_step", "window_id"):
        if length > 1 and not np.all(np.diff(windows[column].to_numpy()[indices], axis=1) == 1):
            raise ValueError(f"{split}/L{length}: nonconsecutive {column}.")
    expected_ends = []
    for _, group in windows.groupby("episode_id", sort=False):
        ordered = group.sort_values(["episode_step", "window_id"], kind="mergesort")
        expected_ends.extend(ordered.index.to_numpy()[length - 1:].tolist())
    if not np.array_equal(np.sort(ends), np.sort(expected_ends)):
        raise ValueError(f"{split}/L{length}: incomplete causal endpoint coverage.")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_input_alignment(splits, embedding_splits, sequence_splits, metadata_splits):
    OBSERVABLE_EFFECTS.clear()
    COMMON_ENDPOINTS.clear()
    extraction = load_json_required(EMBEDDING_REPORT_PATH)
    index = load_json_required(SEQUENCE_REPORT_INDEX_PATH)
    common = index.get("common_endpoint_report", {})
    if common.get("reference_length") != COMMON_EVALUATION_REFERENCE_LENGTH:
        raise ValueError("The Step 10 common endpoint reference must be L10.")
    mapping = load_json_required(LABEL_MAPPING_INPUT_PATH)
    if (mapping["label_to_id"] != extraction["label_to_id"]
            or mapping["label_order"] != extraction["label_order"]):
        raise ValueError("Step 7 and Step 9 label mappings differ.")
    if SEQUENCE_LENGTH != 1 or not SAVE_PREDICTIONS or not SAVE_MODEL_CHECKPOINTS:
        raise ValueError("The L1 control requires L1 inputs, predictions and best checkpoints.")
    audit, episode_sets = {}, {}
    for split in SPLITS:
        windows = read_csv_robust(EMBEDDING_DIR / f"{split}_window_metadata.csv")
        required = METADATA_FIELDS + ["label_id", "embedding_row_index"]
        if not set(required).issubset(windows.columns) or windows[required].isna().any().any():
            raise ValueError(f"{split}: incomplete window metadata.")
        frame, emb, seq = splits[split], embedding_splits[split], sequence_splits[split]
        if len(windows) != len(frame):
            raise ValueError(f"{split}: raw split and embedding metadata row counts differ.")
        for column in METADATA_FIELDS + ["label_id"]:
            if not np.array_equal(windows[column].astype(str), frame[column].astype(str)):
                raise ValueError(f"{split}: raw split/window metadata mismatch: {column}")
        if not np.array_equal(integer_array(windows["embedding_row_index"], "embedding_row_index"), emb["row_indices"]):
            raise ValueError(f"{split}: incorrect embedding row mapping.")
        for column in ("episode_step", "window_id"):
            integer_array(windows[column], column)
        timestamps = pd.to_datetime(windows["timestamp"], errors="raise", utc=True)
        for _, group in windows.groupby("episode_id", sort=False):
            ordered = group.sort_values(["episode_step", "window_id"], kind="mergesort")
            if (not np.all(np.diff(ordered["episode_step"]) == 1)
                    or not np.all(np.diff(ordered["window_id"]) == 1)
                    or not timestamps.loc[ordered.index].diff().dropna().dt.total_seconds().eq(1).all()):
                raise ValueError(f"{split}: nonconsecutive episode timeline.")
        validate_sequence_alignment(split, SEQUENCE_LENGTH, seq, metadata_splits[split], windows)
        OBSERVABLE_EFFECTS[split] = load_sequence_observable_effects(split, SEQUENCE_LENGTH, metadata_splits[split])
        ends = seq["end_row_indices"]
        endpoints_by_length, counts_by_length = {}, {}
        for length in SEQUENCE_LENGTHS:
            if length == SEQUENCE_LENGTH:
                arrays, meta = seq, metadata_splits[split]
            else:
                with np.load(SEQUENCE_DIR / f"{split}_tst_sequences_L{length}.npz", allow_pickle=False) as data:
                    arrays = {key: integer_array(data[key], f"{split}/L{length}/{key}",
                              2 if key == "sequence_indices" else 1)
                              for key in ("sequence_indices", "y", "start_row_indices", "end_row_indices")}
                ix = arrays["sequence_indices"]
                if (ix.shape != (len(arrays["y"]), length) or ix.min() < 0 or ix.max() >= len(windows)
                        or not np.array_equal(ix[:, 0], arrays["start_row_indices"])
                        or not np.array_equal(ix[:, -1], arrays["end_row_indices"])
                        or not np.array_equal(arrays["y"], windows["label_id"].to_numpy()[ix[:, -1]])):
                    raise ValueError(f"{split}/L{length}: invalid common-cohort source arrays.")
                meta = read_csv_robust(SEQUENCE_DIR / f"{split}_tst_sequence_metadata_L{length}.csv")
                validate_sequence_alignment(split, length, arrays, meta, windows)
                load_sequence_observable_effects(split, length, meta)
            report = load_json_required(SEQUENCE_DIR / f"sequence_build_report_L{length}.json")["splits"][split]
            length_ends = arrays["end_row_indices"]
            if (report["num_sequences"] != len(length_ends)
                    or report["num_episodes"] != windows["episode_id"].nunique()
                    or report["end_row_indices_sha256"] != hashlib.sha256(length_ends.astype("<i8").tobytes()).hexdigest()):
                raise ValueError(f"{split}/L{length}: Step 10 sequence report mismatch.")
            endpoints_by_length[length] = np.sort(length_ends)
            counts_by_length[str(length)] = len(length_ends)
        common_ends = endpoints_by_length[SEQUENCE_LENGTHS[0]]
        for length in SEQUENCE_LENGTHS[1:]:
            common_ends = np.intersect1d(common_ends, endpoints_by_length[length])
        recorded = common.get("splits", {}).get(split, {})
        if (not len(common_ends)
                or not np.array_equal(common_ends, endpoints_by_length[COMMON_EVALUATION_REFERENCE_LENGTH])
                or recorded.get("common_endpoint_count") != len(common_ends)
                or recorded.get("sorted_end_row_indices_sha256") != endpoint_digest(common_ends)
                or recorded.get("available_sequences_by_L") != counts_by_length):
            raise ValueError(f"{split}: actual common endpoints differ from Step 10.")
        COMMON_ENDPOINTS[split] = common_ends
        episodes = set(windows["episode_id"].astype(str))
        summary9 = extraction["splits"][split]
        if (summary9.get("rows") != len(windows)
                or summary9.get("embedding_shape") != list(emb["embeddings"].shape)
                or extraction["split_temporal_summaries"][split]["episode_count"] != len(episodes)
                or windows["label"].value_counts().reindex(extraction["label_order"], fill_value=0).to_dict() != summary9["label_distribution"]):
            raise ValueError(f"{split}: completed Step 9 population differs.")
        if set(seq["y"].tolist()) != set(range(len(mapping["label_order"]))):
            raise ValueError(f"{split}: missing class in full L1 population.")
        episode_sets[split] = episodes
        audit[split] = {"windows": len(windows), "episodes": len(episodes),
                        "full_endpoint_count": len(ends),
                        "common_endpoint_count": len(common_ends),
                        "sequence_counts": counts_by_length,
                        "full_end_row_indices_sha256": endpoint_digest(ends),
                        "sorted_end_row_indices_sha256": endpoint_digest(common_ends)}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            if episode_sets[left] & episode_sets[right]:
                raise ValueError(f"Episode split leakage: {left}/{right}")
    return audit


def verified_common_mask(metadata: pd.DataFrame, split_name: str) -> np.ndarray:
    """Select the complete, authenticated intersection without using labels/predictions."""
    if split_name not in COMMON_ENDPOINTS:
        raise ValueError(f"{split_name}: common endpoints have not been authenticated.")
    ends = integer_array(metadata["sequence_end_row"], "evaluation endpoints")
    common = COMMON_ENDPOINTS[split_name]
    mask = np.isin(ends, common)
    if (len(np.unique(ends)) != len(ends)
            or not np.array_equal(np.sort(ends[mask]), common)):
        raise ValueError(f"{split_name}: incomplete/duplicate common evaluation endpoints.")
    return mask


def result_provenance() -> Dict[str, Any]:
    return {key: SOURCE_CONTRACT[key] for key in ("dataset_id", "step8_run_id", "step9_run_id", "step10_run_id")} | {"step14_run_id": ACTIVE_RUN["run_id"]}


@contextmanager
def output_lock():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / ".step14.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started": time.time()}, stream)
    except FileExistsError as exc:
        raise RuntimeError(f"Step 14 lock exists: {path}. After a crash, confirm the old process stopped before removing this lock.") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def trial_files(seed: int) -> List[Path]:
    prefix = f"seed{seed}_"
    files = list(REPORT_DIR.glob(prefix + "*")) + list(PREDICTION_DIR.glob(prefix + "*"))
    checkpoint = MODEL_DIR / f"best_lstm_finetuned_distilbert_L{SEQUENCE_LENGTH}_seed{seed}.pt"
    return [path for path in [*files, checkpoint] if path.is_file()]


def prepare_run(manifest: Dict[str, Any], device: torch.device, retry_failed: bool) -> bool:
    global ACTIVE_RUN
    if len(SEEDS) != len(set(SEEDS)) or not SEEDS:
        raise ValueError("A nonempty list of unique head seeds is required.")
    settings = {}
    for name, value in globals().items():
        if name.isupper() and name not in {"SOURCE_CONTRACT", "INPUT_AUDIT", "ACTIVE_RUN", "OBSERVABLE_EFFECTS", "COMMON_ENDPOINTS"}:
            try: json.dumps(value, allow_nan=False)
            except (TypeError, ValueError): continue
            settings[name] = value
    identity = {"source_contract": SOURCE_CONTRACT, "input_audit": INPUT_AUDIT,
        "script_sha256": sha256_file(Path(__file__)), "configuration": manifest, "settings": settings,
        "output_root": str(OUTPUT_ROOT.resolve()),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": np.__version__, "pandas": pd.__version__, "device": str(device),
                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                    "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.machine()},
        "comparison_policy": "Full L1 training and validation; validation-only checkpoint selection. Report full and verified common L10-eligible endpoints separately. No historical input to L1.",
        "uncertainty": "Sample SD across head seeds conditional on one encoder/split; not a bootstrap CI."}
    identity = json.loads(json.dumps(identity, allow_nan=False))
    run_id = object_hash(identity)
    path = OUTPUT_ROOT / "run_manifest.json"
    previous = load_json_required(path) if path.exists() else None
    if previous is not None:
        if previous.get("run_id") != run_id or previous.get("identity") != identity:
            raise ValueError("Existing Step 14 output uses different inputs/code/settings. Use a separate --output-root; existing results are preserved.")
        expected_trials = {str(seed) for seed in SEEDS}
        for key, item in previous.get("completed_runs", {}).items():
            if key not in expected_trials:
                raise ValueError("Unexpected completed seed trial in the run record.")
            verify_artifacts(OUTPUT_ROOT, item["artifacts"])
            seed_text = key
            summary_name = f"reports/seed{int(seed_text)}_result_summary.json"
            if summary_name not in item["artifacts"] or load_json_required(OUTPUT_ROOT / summary_name) != item["result"]:
                raise ValueError("Saved result summary differs from its completed trial record.")
        if previous.get("status") == "COMPLETE":
            verify_artifacts(OUTPUT_ROOT, previous["artifacts"])
            expected = {str(seed) for seed in SEEDS}
            if set(previous.get("completed_runs", {})) != expected:
                raise ValueError("Completed Step 14 grid is incomplete.")
            print("[Verified COMPLETE Step 14 reused] No training or output overwrite.")
            return False
        if not retry_failed:
            raise ValueError("Incomplete Step 14 run preserved. Once the old process has stopped, rerun with --retry-failed to reuse completed trials and restart the incomplete trial.")
    elif any(p.name != ".step14.lock" for p in OUTPUT_ROOT.iterdir()):
        raise FileExistsError("Output exists without a verified Step 14 run record. Use a separate --output-root.")
    ACTIVE_RUN = previous or {"run_id": run_id, "dataset_id": SOURCE_CONTRACT["dataset_id"],
        "identity": identity, "completed_runs": {}, "active_trial": None, "archived_incomplete": []}
    active = ACTIVE_RUN.get("active_trial")
    if active:
        seed = int(active["seed"])
        archive = OUTPUT_ROOT / "_incomplete" / f"seed{seed}_{time.time_ns()}"
        archive.mkdir(parents=True, exist_ok=False)
        for file in trial_files(seed):
            target = archive / file.relative_to(OUTPUT_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            file.rename(target)
        ACTIVE_RUN["archived_incomplete"].append(str(archive))
    ACTIVE_RUN.update(status="RUNNING", active_trial=None, last_error=None)
    ensure_output_directories()
    save_json(path, ACTIVE_RUN)
    return True


def record_completed_trial(seed: int, result: Dict[str, Any]) -> None:
    required = [MODEL_DIR / f"best_lstm_finetuned_distilbert_L{SEQUENCE_LENGTH}_seed{seed}.pt",
                REPORT_DIR / f"seed{seed}_result_summary.json"]
    for split in ("val", "test"):
        required += [REPORT_DIR / f"seed{seed}_{split}_metrics.json",
                     PREDICTION_DIR / f"seed{seed}_{split}_predictions.csv",
                     REPORT_DIR / f"seed{seed}_{split}_observable_effect_diagnostics.json",
                     REPORT_DIR / f"seed{seed}_{split}_common_metrics.json"]
    if (any(not file.is_file() for file in required)
            or result["total"] != INPUT_AUDIT["test"]["full_endpoint_count"]
            or result["test_common_count"] != INPUT_AUDIT["test"]["common_endpoint_count"]):
        raise ValueError(f"Incomplete artifacts/population for seed={seed}")
    ACTIVE_RUN["completed_runs"][str(seed)] = {
        "result": result, "artifacts": {p.relative_to(OUTPUT_ROOT).as_posix(): sha256_file(p) for p in trial_files(seed)}}
    ACTIVE_RUN["active_trial"] = None
    save_json(OUTPUT_ROOT / "run_manifest.json", ACTIVE_RUN)


def finish_run() -> None:
    if sha256_file(Path(__file__)) != ACTIVE_RUN["identity"]["script_sha256"]:
        raise ValueError("Step 14 source changed during training; completion was not recorded.")
    before = object_hash(SOURCE_CONTRACT)
    validate_revised_pipeline_provenance()
    if object_hash(SOURCE_CONTRACT) != before:
        raise ValueError("Upstream inputs changed during training; completion was not recorded.")
    artifacts = {p.relative_to(OUTPUT_ROOT).as_posix(): sha256_file(p)
                 for p in OUTPUT_ROOT.rglob("*") if p.is_file()
                 and p.name not in {"run_manifest.json", ".step14.lock"}
                 and "_incomplete" not in p.relative_to(OUTPUT_ROOT).parts}
    ACTIVE_RUN.update(status="COMPLETE", active_trial=None, artifacts=artifacts)
    save_json(OUTPUT_ROOT / "run_manifest.json", ACTIVE_RUN)


def main() -> None:
    global OUTPUT_ROOT, REPORT_DIR, PREDICTION_DIR, MODEL_DIR, ACTIVE_RUN
    parser = argparse.ArgumentParser(description="Verified fine-tuned DistilBERT L1 LSTM head-seed comparison.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT,
                        help="Separate output folder. Explicit relative paths use the working directory.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Only after the old process stopped: reuse completed trials, archive/restart the incomplete trial.")
    parser.add_argument("--check-inputs-only", action="store_true",
                        help="Validate the existing dataset/encoder/sequence/cohort artifacts without training.")
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_root.expanduser().resolve()
    for source in (SPLIT_DIR.parent, EMBEDDING_DIR, SEQUENCE_DIR, EXPECTED_FINETUNED_MODEL_DIR.parent, ORIGINAL_L10_OUTPUT_ROOT):
        if OUTPUT_ROOT.is_relative_to(source.resolve()) or source.resolve().is_relative_to(OUTPUT_ROOT):
            raise ValueError("L1 output must be separate from upstream inputs and the existing L10 output tree.")
    if Path(__file__).resolve().is_relative_to(OUTPUT_ROOT):
        raise ValueError("Step 14 output cannot contain the source/project.")
    REPORT_DIR, PREDICTION_DIR, MODEL_DIR = (OUTPUT_ROOT / name for name in ("reports", "predictions", "models"))
    ACTIVE_RUN = {}
    if args.check_inputs_only:
        run_experiment(check_inputs_only=True)
        return
    with output_lock():
        try:
            run_experiment(args.retry_failed)
        except BaseException as exc:
            if ACTIVE_RUN and ACTIVE_RUN.get("status") != "COMPLETE":
                ACTIVE_RUN.update(status="FAILED", last_error=f"{type(exc).__name__}: {exc}")
                save_json(OUTPUT_ROOT / "run_manifest.json", ACTIVE_RUN)
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[ERROR] Step 14 Fixed fine-tuned DistilBERT L1 LSTM baseline failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
