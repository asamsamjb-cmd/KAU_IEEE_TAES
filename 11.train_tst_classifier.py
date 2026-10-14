# 11.train_tst_classifier.py
# Train and evaluate Time-Series Transformer (TST) classifiers using
# DistilBERT window embeddings and episode-bounded sequence indices.
#
# Pipeline position:
#   Step 9 extracts one DistilBERT [CLS] embedding e_t per 1-second observable window.
#   Step 10 builds episode-bounded sequence_indices for L={1,2,3,5,10}.
#   This script trains one TST classifier per sequence length L and evaluates it.
#
# Core experimental meaning:
#   - DistilBERT baseline: single-window cross-field semantic classifier.
#   - TST-L1: temporal ablation using one 1-second embedding.
#   - TST-L2/L3/L5/L10: temporal classifiers using 2/3/5/10 seconds of context.
# Revision goal: test semantic-observable representation and cross-window evidence
# aggregation. Architecture superiority is not assumed; competitive lightweight
# baselines are valid outcomes. TST-L1 is a different head from the Step 8 classifier.
#
# Critical rules:
#   - TST input is embeddings[sequence_indices] only.
#   - Metadata is never used as model input.
#   - Sequence label is the label_id of the last window in the sequence.
#   - Train/val/test splits are never mixed.
#   - One model is trained per L to make temporal-context comparison defensible.
#
# Expected inputs:
#   outputs/shared_nominal_v2/tst_embeddings_finetuned_main/
#     train_window_embeddings.npz
#     val_window_embeddings.npz
#     test_window_embeddings.npz
#     embedding_extraction_report.json
#   outputs/shared_nominal_v2/tst_sequences_finetuned_main/
#     train_tst_sequences_L{L}.npz
#     val_tst_sequences_L{L}.npz
#     test_tst_sequences_L{L}.npz
#     train_tst_sequence_metadata_L{L}.csv  (required)
#     val_tst_sequence_metadata_L{L}.csv    (required)
#     test_tst_sequence_metadata_L{L}.csv   (required)
#     {split}_tst_sequence_observable_effects_L{L}.csv (required audit-only sidecars)
#
# Main outputs:
#   outputs/shared_nominal_v2/tst_training_finetuned_main/
#     L1/, L2/, L3/, L5/, L10/
#       checkpoints/best_tst_model_L{L}.pt
#       reports/training_history_L{L}.csv
#       reports/evaluation_metrics_L{L}.json
#       reports/confusion_matrix_test_L{L}.csv
#       predictions/test_predictions_L{L}.csv
#     tst_overall_comparison.csv
#     tst_overall_comparison.json

from __future__ import annotations

import argparse
import hashlib
import platform
import tempfile
import zipfile
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
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
# 0. Basic settings
# ============================================================

DEFAULT_RANDOM_SEED = 20260610

BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_DIR = BASE_DIR / "outputs" / "shared_nominal_v2" / "tst_embeddings_finetuned_main"
SEQUENCE_DIR = BASE_DIR / "outputs" / "shared_nominal_v2" / "tst_sequences_finetuned_main"
OUTPUT_ROOT = BASE_DIR / "outputs" / "shared_nominal_v2" / "tst_training_finetuned_main"
EXPECTED_FINETUNED_MODEL_DIR = BASE_DIR / "outputs" / "shared_nominal_v2" / "distilbert_window_baseline" / "trained_distilbert_model"

EMBEDDING_REPORT_PATH = EMBEDDING_DIR / "embedding_extraction_report.json"
SEQUENCE_REPORT_INDEX_PATH = SEQUENCE_DIR / "sequence_build_report_index.json"

SPLITS: List[str] = ["train", "val", "test"]
SEQUENCE_LENGTHS: List[int] = [1, 2, 3, 5, 10]
EXPECTED_OBSERVABLE_WINDOW_SECONDS = 1
TST_TEMPORAL_CONTEXT_SECONDS: List[int] = [
    length * EXPECTED_OBSERVABLE_WINDOW_SECONDS for length in SEQUENCE_LENGTHS
]

# Training controls.
MAX_EPOCHS = 30
PATIENCE = 7
BATCH_SIZE = 256
NUM_WORKERS = 0  # Keep 0 for Windows compatibility.
LEARNING_RATE = 7e-5
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 1.0
USE_CLASS_WEIGHTS = False
USE_AMP = torch.cuda.is_available()

# Model controls. Input dimension is read from Step 9/10 artifacts, usually 768.
D_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 1
DIM_FEEDFORWARD = 256
DROPOUT = 0.25
POOLING = "last"  # Must be "last" for current-state classification.

# Evaluation controls.
SAVE_VAL_PREDICTIONS = True
SAVE_TEST_PREDICTIONS = True
SAVE_TRAINING_CURVE_PNG = True
PRIMARY_SELECTION_METRIC = "val_macro_f1"

# Evaluation-audit controls.
# The full evaluation is kept unchanged. Additional subset metrics make L={1,2,3,5,10}
# comparisons defensible when sequence lengths produce different evaluable populations.
COMMON_EVALUATION_REFERENCE_LENGTH = max(SEQUENCE_LENGTHS)
COMMON_EVALUATION_MIN_END_POSITION = COMMON_EVALUATION_REFERENCE_LENGTH - 1
REPORT_COMMON_POPULATION_METRICS = True
REPORT_NON_OVERLAP_METRICS = True
STRICT_CLASS_COVERAGE_CHECK = True

# Strict sequence-independence controls.
# Keep all available sequences for training and validation selection.
# Non-overlap subsets are descriptive audits, not independent samples or a CI method.
# The primary test result uses the full test population to avoid a small non-overlap
# subset accidentally reporting perfect scores while full-population errors exist.
STRICT_SEQUENCE_INDEPENDENCE_MODE = True
SEQUENCE_FIT_SUBSET = "full"       # used for train and validation loaders
PRIMARY_TEST_SUBSET = "full"              # used for the primary test result; non_overlap is reported as an audit metric
RAISE_ON_EPISODE_SPLIT_OVERLAP = True

# Optional training regularization controls, disabled in the clean comparison.
# They perturb training embeddings only. Their historical "shortcut-resistant"
# names are retained for compatibility and are not claims about the data's
# information content, desired error rate, or superiority of longer contexts.
USE_SHORTCUT_RESISTANT_TRAINING = False
LABEL_SMOOTHING = 0.03
TRAIN_GAUSSIAN_NOISE_STD = 0.015
TRAIN_HISTORY_TOKEN_DROPOUT_PROB = 0.10
TRAIN_LAST_WINDOW_REPLACEMENT_PROB = 0.15
TRAIN_LAST_WINDOW_REPLACEMENT = "history_mean"  # choices: "history_mean" or "zero"

# Length-aware shortcut-resistant regularization.
# This is applied only during training. Validation/test inputs remain unchanged.
# The goal is to reduce near-deterministic shortcut learning in longer sequences
# without changing labels, validation inputs, or test inputs.
USE_LENGTH_AWARE_REGULARIZATION = False

LENGTH_AWARE_REGULARIZATION = {
    1: {
        "gaussian_noise_std": 0.000,
        "history_token_dropout_prob": 0.00,
        "last_window_replacement_prob": 0.00,
    },
    2: {
        "gaussian_noise_std": 0.015,
        "history_token_dropout_prob": 0.06,
        "last_window_replacement_prob": 0.05,
    },
    3: {
        "gaussian_noise_std": 0.015,
        "history_token_dropout_prob": 0.07,
        "last_window_replacement_prob": 0.06,
    },
    5: {
        "gaussian_noise_std": 0.018,
        "history_token_dropout_prob": 0.08,
        "last_window_replacement_prob": 0.07,
    },
    10: {
        "gaussian_noise_std": 0.020,
        "history_token_dropout_prob": 0.10,
        "last_window_replacement_prob": 0.08,
    },
}

# Optional embedding-space augmentation controls (legacy "missing-aware" name).
# Embedding dimensions do not have a verified one-to-one field mapping. Dimension
# dropout, noise and token replacement are generic regularizers, not literal
# missing-observable interventions or evidence of field-level robustness.
# Only training batches are modified. Actual field-loss experiments require
# reserializing the 22-field input and re-encoding it outside this script.
# Model input dimension and checkpoint format remain unchanged.
USE_MISSING_AWARE_TRAINING = False
MISSING_AWARE_REPLACEMENT = "sequence_mean"  # choices: "sequence_mean", "history_mean", "zero"
MISSING_AWARE_REGULARIZATION = {
    # Clean-performance safe defaults. Missing-aware augmentation is OFF by default
    # and can be enabled explicitly with --missing-aware. L1/L2 are kept clean so
    # they remain valid short-context baselines. Stronger robustness pressure is
    # applied only to longer contexts where missing-field robustness is evaluated.
    1: {
        "gaussian_noise_std": 0.000,
        "sequence_dim_dropout_prob": 0.000,
        "window_dim_dropout_prob": 0.000,
        "token_replacement_prob": 0.000,
        "last_token_replacement_prob": 0.000,
    },
    2: {
        "gaussian_noise_std": 0.000,
        "sequence_dim_dropout_prob": 0.000,
        "window_dim_dropout_prob": 0.000,
        "token_replacement_prob": 0.000,
        "last_token_replacement_prob": 0.000,
    },
    3: {
        "gaussian_noise_std": 0.003,
        "sequence_dim_dropout_prob": 0.010,
        "window_dim_dropout_prob": 0.010,
        "token_replacement_prob": 0.000,
        "last_token_replacement_prob": 0.000,
    },
    5: {
        "gaussian_noise_std": 0.006,
        "sequence_dim_dropout_prob": 0.020,
        "window_dim_dropout_prob": 0.020,
        "token_replacement_prob": 0.004,
        "last_token_replacement_prob": 0.001,
    },
    10: {
        "gaussian_noise_std": 0.008,
        "sequence_dim_dropout_prob": 0.030,
        "window_dim_dropout_prob": 0.030,
        "token_replacement_prob": 0.006,
        "last_token_replacement_prob": 0.002,
    },
}


# Legacy dormant embedding-stress helper, retained for compatibility.
# Both flags remain False in the agreed clean protocol; prepare_run rejects
# robust-as-primary. This helper is not the counterfactual observable-effect audit.
# Known legacy limitation: "last_window_keep_prob" controls a selection-priority
# boost, not actual retention probability; p=0 does not exclude the last token.
# Do not interpret its configured values as calibrated field/window missingness.
RUN_ROBUST_EVIDENCE_BUDGET_EVAL = False
USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS = False
ROBUST_PRIMARY_MIN_SEQUENCE_LENGTH = 5
ROBUST_MC_REPEATS = 5
ROBUST_KEEP_WINDOWS_BY_LENGTH = {
    1: 1,
    2: 2,
    3: 3,
    5: 1,
    10: 1,
}
ROBUST_LAST_WINDOW_KEEP_PROB_BY_LENGTH = {
    1: 1.00,
    2: 1.00,
    3: 1.00,
    5: 0.00,
    10: 0.00,
}
ROBUST_DIM_DROPOUT_PROB_BY_LENGTH = {
    1: 0.00,
    2: 0.00,
    3: 0.00,
    5: 0.30,
    10: 0.40,
}
ROBUST_GAUSSIAN_NOISE_STD_BY_LENGTH = {
    1: 0.00,
    2: 0.00,
    3: 0.00,
    5: 0.070,
    10: 0.090,
}

# Temporal ablation controls.
# These audits test whether the TST actually uses historical temporal context,
# or whether most performance comes from the last-window embedding alone.
RUN_TEMPORAL_ABLATION_EVAL = True
TEMPORAL_ABLATION_MODES = [
    "history_zero",    # keep the last window, zero out all previous windows
    "repeat_last",     # replace the entire sequence with the last window
    "prefix_reverse",  # reverse only the historical prefix and keep the last window fixed
]

# Reproducibility controls.
DETERMINISTIC_CUDNN = False  # True is more reproducible but can be slower.


# ============================================================
# 1. Dataclasses
# ============================================================

@dataclass
class ModelConfig:
    input_dim: int
    num_classes: int
    sequence_length: int
    temporal_context_seconds: int
    d_model: int = D_MODEL
    num_heads: int = NUM_HEADS
    num_encoder_layers: int = NUM_ENCODER_LAYERS
    dim_feedforward: int = DIM_FEEDFORWARD
    dropout: float = DROPOUT
    pooling: str = POOLING


@dataclass
class TrainConfig:
    random_seed: int = DEFAULT_RANDOM_SEED
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    grad_clip_norm: float = GRAD_CLIP_NORM
    use_class_weights: bool = USE_CLASS_WEIGHTS
    use_amp: bool = USE_AMP
    primary_selection_metric: str = PRIMARY_SELECTION_METRIC
    use_shortcut_resistant_training: bool = USE_SHORTCUT_RESISTANT_TRAINING
    label_smoothing: float = LABEL_SMOOTHING
    train_gaussian_noise_std: float = TRAIN_GAUSSIAN_NOISE_STD
    train_history_token_dropout_prob: float = TRAIN_HISTORY_TOKEN_DROPOUT_PROB
    train_last_window_replacement_prob: float = TRAIN_LAST_WINDOW_REPLACEMENT_PROB
    train_last_window_replacement: str = TRAIN_LAST_WINDOW_REPLACEMENT
    use_length_aware_regularization: bool = USE_LENGTH_AWARE_REGULARIZATION
    use_missing_aware_training: bool = USE_MISSING_AWARE_TRAINING
    missing_aware_replacement: str = MISSING_AWARE_REPLACEMENT
    run_robust_evidence_budget_eval: bool = RUN_ROBUST_EVIDENCE_BUDGET_EVAL
    use_robust_eval_as_primary_for_long_contexts: bool = USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS
    robust_primary_min_sequence_length: int = ROBUST_PRIMARY_MIN_SEQUENCE_LENGTH
    robust_mc_repeats: int = ROBUST_MC_REPEATS


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
    torch.backends.cudnn.deterministic = bool(DETERMINISTIC_CUDNN)
    torch.backends.cudnn.benchmark = not bool(DETERMINISTIC_CUDNN)


def ensure_directories() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def progress_bar(iterable: Iterable[Any], total: Optional[int], desc: str) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8", keep_default_na=False)


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".csv.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        df.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint_atomic(checkpoint: Dict[str, Any], path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sequence_npz_path(split_name: str, sequence_length: int) -> Path:
    return SEQUENCE_DIR / f"{split_name}_tst_sequences_L{sequence_length}.npz"


def sequence_metadata_path(split_name: str, sequence_length: int) -> Path:
    return SEQUENCE_DIR / f"{split_name}_tst_sequence_metadata_L{sequence_length}.csv"


def embedding_npz_path(split_name: str) -> Path:
    return EMBEDDING_DIR / f"{split_name}_window_embeddings.npz"


def length_output_dir(sequence_length: int) -> Path:
    return OUTPUT_ROOT / f"L{sequence_length}"


def resolve_cli_path(value: str) -> Path:
    """Resolve an explicitly supplied path from the caller's working directory."""
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train selected TST sequence lengths. "
            "Examples: --lengths 1, --lengths 2 5 10, --lengths 2,3,5,10, --lengths all"
        )
    )
    parser.add_argument(
        "--lengths",
        nargs="*",
        default=None,
        help=(
            "Sequence lengths to train. "
            "Use one or more values from {1,2,3,5,10}, comma-separated values, or 'all'. "
            "Examples: --lengths 1 / --lengths 2 5 10 / --lengths 2,3,5,10 / --lengths all"
        ),
    )
    parser.add_argument(
        "--missing-aware",
        action="store_true",
        help=(
            "Enable missing-aware train-time augmentation. Keep this OFF for the clean "
            "TST comparison run; enable it only for a separate robustness-tuned run."
        ),
    )
    parser.add_argument(
        "--no-missing-aware",
        action="store_true",
        help="Explicitly disable missing-aware train-time augmentation.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Optional output directory. Relative paths use the current working directory. "
            "Example: --output-root outputs/tst_training_clean_check"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Training seed; recorded in run identity.")
    args = parser.parse_args()
    if args.missing_aware and args.no_missing_aware:
        raise ValueError("Use only one of --missing-aware or --no-missing-aware.")
    return args


def resolve_selected_sequence_lengths(args: argparse.Namespace) -> List[int]:
    allowed = set(SEQUENCE_LENGTHS)

    if args.lengths is None or len(args.lengths) == 0:
        return list(SEQUENCE_LENGTHS)

    raw_items: List[str] = []
    for item in args.lengths:
        if item is None:
            continue
        for part in str(item).split(","):
            part = part.strip()
            if part:
                raw_items.append(part)

    if not raw_items:
        return list(SEQUENCE_LENGTHS)

    if any(item.lower() == "all" for item in raw_items):
        if len(raw_items) != 1:
            raise ValueError("Use 'all' alone, or provide explicit supported sequence lengths.")
        return list(SEQUENCE_LENGTHS)

    selected: List[int] = []
    for item in raw_items:
        try:
            length = int(item)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --lengths value: {item}. "
                f"Use values from {SEQUENCE_LENGTHS} or 'all'."
            ) from exc

        if length not in allowed:
            raise ValueError(
                f"Unsupported sequence length: {length}. "
                f"Allowed values are {SEQUENCE_LENGTHS}."
            )

        if length not in selected:
            selected.append(length)

    return selected

# ============================================================
# 3. Label and artifact metadata utilities
# ============================================================


def build_label_maps(extraction_report: Dict[str, Any]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    label_order = extraction_report.get("label_order")
    label_to_id = extraction_report.get("label_to_id")

    if not isinstance(label_order, list) or not label_order:
        raise ValueError(
            f"Step 9 report is missing label_order. Expected: {EMBEDDING_REPORT_PATH}"
        )
    if not isinstance(label_to_id, dict) or not label_to_id:
        raise ValueError(
            f"Step 9 report is missing label_to_id. Expected: {EMBEDDING_REPORT_PATH}"
        )

    label_order = [str(label) for label in label_order]
    label_to_id = {str(label): int(idx) for label, idx in label_to_id.items()}

    missing = [label for label in label_order if label not in label_to_id]
    if missing:
        raise ValueError(f"label_to_id is missing label_order entries: {missing}")

    ids = sorted(label_to_id.values())
    expected = list(range(len(label_order)))
    if ids != expected:
        raise ValueError(f"label_to_id IDs must be contiguous {expected}, got {ids}")

    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_order, label_to_id, id_to_label


def validate_temporal_design_from_reports(extraction_report: Dict[str, Any]) -> Dict[str, Any]:
    if not extraction_report:
        raise FileNotFoundError(
            f"Step 9 extraction report not found: {EMBEDDING_REPORT_PATH}\n"
            f"Run Step 9 before Step 11."
        )

    observable_window_seconds = int(extraction_report.get("observable_window_seconds", -1))
    if observable_window_seconds != EXPECTED_OBSERVABLE_WINDOW_SECONDS:
        raise ValueError(
            f"Step 9 observable_window_seconds mismatch: expected "
            f"{EXPECTED_OBSERVABLE_WINDOW_SECONDS}, got {observable_window_seconds}"
        )

    report_lengths = [int(x) for x in extraction_report.get("tst_sequence_lengths", [])]
    if report_lengths != SEQUENCE_LENGTHS:
        raise ValueError(
            f"Step 9 TST sequence lengths mismatch: expected {SEQUENCE_LENGTHS}, got {report_lengths}"
        )

    report_contexts = [int(x) for x in extraction_report.get("tst_temporal_context_seconds", [])]
    if report_contexts != TST_TEMPORAL_CONTEXT_SECONDS:
        raise ValueError(
            f"Step 9 temporal context mismatch: expected {TST_TEMPORAL_CONTEXT_SECONDS}, got {report_contexts}"
        )

    embedding_dim = int(extraction_report.get("embedding_dim", -1))
    if embedding_dim <= 0:
        raise ValueError("Step 9 report has invalid embedding_dim.")

    return {
        "observable_window_seconds": observable_window_seconds,
        "tst_sequence_lengths": report_lengths,
        "tst_temporal_context_seconds": report_contexts,
        "embedding_dim": embedding_dim,
    }


def validate_step10_report_index() -> Dict[str, Any]:
    data = load_json_if_exists(SEQUENCE_REPORT_INDEX_PATH)
    if not data:
        raise FileNotFoundError(
            f"Step 10 report index not found: {SEQUENCE_REPORT_INDEX_PATH}\n"
            f"Run Step 10 before Step 11."
        )

    lengths = [int(x) for x in data.get("sequence_lengths", [])]
    if lengths != SEQUENCE_LENGTHS:
        raise ValueError(
            f"Step 10 sequence lengths mismatch: expected {SEQUENCE_LENGTHS}, got {lengths}"
        )

    contexts = data.get("temporal_context_seconds")
    if contexts is not None:
        contexts = [int(x) for x in contexts]
        if contexts != TST_TEMPORAL_CONTEXT_SECONDS:
            raise ValueError(
                f"Step 10 temporal contexts mismatch: expected "
                f"{TST_TEMPORAL_CONTEXT_SECONDS}, got {contexts}"
            )

    reported_encoder_source = str(data.get("embedding_encoder_source", "")).strip().lower()
    if reported_encoder_source not in {"finetuned", "fine-tuned", "fine_tuned"}:
        raise ValueError(
            "Step 10 sequence index is not marked as originating from fine-tuned embeddings: "
            f"{data.get('embedding_encoder_source')!r}"
        )

    reported_source_dir = str(data.get("source_embedding_dir", "")).strip()
    if reported_source_dir:
        expected_source_dir = str(EMBEDDING_DIR.resolve())
        if os.path.normcase(os.path.normpath(reported_source_dir)) != os.path.normcase(os.path.normpath(expected_source_dir)):
            raise ValueError(
                "Step 10 source_embedding_dir does not match the main revision embedding directory. "
                f"Expected {expected_source_dir!r}, found {reported_source_dir!r}."
            )

    return data


# ============================================================
# 4. Data loading and validation
# ============================================================


def load_embedding_split(split_name: str, expected_embedding_dim: int) -> Dict[str, np.ndarray]:
    path = embedding_npz_path(split_name)
    if not path.exists():
        raise FileNotFoundError(
            f"[{split_name}] Missing embedding NPZ: {path}\n"
            f"Run Step 9 first."
        )
    data = np.load(path, allow_pickle=False)
    required = ["embeddings", "label_ids", "row_indices"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"[{split_name}] Missing keys in {path.name}: {missing}")

    embeddings = data["embeddings"]
    label_ids = integer_array(data["label_ids"], f"{split_name}: label_ids")
    row_indices = integer_array(data["row_indices"], f"{split_name}: row_indices")

    if embeddings.ndim != 2:
        raise ValueError(f"[{split_name}] embeddings must be 2-D, got {embeddings.shape}")
    if embeddings.shape[1] != expected_embedding_dim:
        raise ValueError(
            f"[{split_name}] embedding_dim mismatch: expected {expected_embedding_dim}, got {embeddings.shape[1]}"
        )
    if len(label_ids) != len(embeddings):
        raise ValueError(f"[{split_name}] label_ids length mismatch with embeddings")
    if len(row_indices) != len(embeddings):
        raise ValueError(f"[{split_name}] row_indices length mismatch with embeddings")
    if not np.array_equal(row_indices, np.arange(len(row_indices), dtype=np.int64)):
        bad = np.where(row_indices != np.arange(len(row_indices), dtype=np.int64))[0][:10].tolist()
        raise ValueError(f"[{split_name}] row_indices are not contiguous embedding rows. Examples={bad}")
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError(f"[{split_name}] embeddings contain NaN or Inf")

    if embeddings.dtype != np.float32:
        raise ValueError(f"{split_name}: completed Step 9 embeddings must be float32.")
    data.close()

    return {
        "embeddings": embeddings,
        "label_ids": label_ids,
        "row_indices": row_indices,
    }


def load_sequence_split(
    split_name: str,
    sequence_length: int,
    embedding_split: Dict[str, np.ndarray],
    expected_embedding_dim: int,
) -> Dict[str, np.ndarray]:
    path = sequence_npz_path(split_name, sequence_length)
    if not path.exists():
        raise FileNotFoundError(
            f"[{split_name}] Missing sequence NPZ for L={sequence_length}: {path}\n"
            f"Run Step 10 first."
        )

    data = np.load(path, allow_pickle=False)
    required = ["sequence_indices", "y", "label_ids", "start_row_indices", "end_row_indices"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"[{split_name}] Missing keys in {path.name}: {missing}")

    sequence_indices = integer_array(data["sequence_indices"], "sequence_indices", ndim=2)
    y = integer_array(data["y"], "y", ndim=1)
    alias_label_ids = integer_array(data["label_ids"], "label_ids", ndim=1)
    start_row_indices = integer_array(data["start_row_indices"], "start_row_indices", ndim=1)
    end_row_indices = integer_array(data["end_row_indices"], "end_row_indices", ndim=1)

    if sequence_indices.ndim != 2:
        raise ValueError(f"[{split_name}] sequence_indices must be 2-D, got {sequence_indices.shape}")
    if sequence_indices.shape[1] != sequence_length:
        raise ValueError(
            f"[{split_name}] sequence length mismatch: expected L={sequence_length}, "
            f"got shape={sequence_indices.shape}"
        )
    if len(y) != len(sequence_indices):
        raise ValueError(f"[{split_name}] y length mismatch with sequence_indices")
    if not np.array_equal(y, alias_label_ids):
        raise ValueError(f"[{split_name}] y and label_ids alias do not match")
    if not np.array_equal(start_row_indices, sequence_indices[:, 0]):
        raise ValueError(f"[{split_name}] start_row_indices do not match sequence_indices[:,0]")
    if not np.array_equal(end_row_indices, sequence_indices[:, -1]):
        raise ValueError(f"[{split_name}] end_row_indices do not match sequence_indices[:,-1]")

    n_windows = len(embedding_split["embeddings"])
    if int(sequence_indices.min()) < 0 or int(sequence_indices.max()) >= n_windows:
        raise ValueError(f"[{split_name}] sequence_indices out of embedding range")

    embedding_label_ids = embedding_split["label_ids"]
    expected_y = embedding_label_ids[sequence_indices[:, -1]]
    if not np.array_equal(y, expected_y):
        bad = np.where(y != expected_y)[0][:10].tolist()
        raise ValueError(
            f"[{split_name}] sequence y does not equal last-window label_id. Examples={bad}"
        )

    if "embedding_dim" in data.files:
        seq_embedding_dim = int(np.asarray(data["embedding_dim"]).reshape(-1)[0])
        if seq_embedding_dim != expected_embedding_dim:
            raise ValueError(
                f"[{split_name}] Step 10 embedding_dim mismatch: expected "
                f"{expected_embedding_dim}, got {seq_embedding_dim}"
            )

    if "sequence_length" in data.files:
        seq_len_recorded = int(np.asarray(data["sequence_length"]).reshape(-1)[0])
        if seq_len_recorded != sequence_length:
            raise ValueError(
                f"[{split_name}] Recorded sequence_length mismatch: expected "
                f"{sequence_length}, got {seq_len_recorded}"
            )

    data.close()
    return {
        "sequence_indices": sequence_indices,
        "y": y,
        "start_row_indices": start_row_indices,
        "end_row_indices": end_row_indices,
    }


def load_sequence_metadata_if_exists(split_name: str, sequence_length: int) -> Optional[pd.DataFrame]:
    path = sequence_metadata_path(split_name, sequence_length)
    if not path.exists():
        raise FileNotFoundError(f"Required Step 10 sequence metadata missing: {path}")
    df = read_csv_robust(path)
    return df.reset_index(drop=True)


# Diagnostic-only sidecars. These columns never enter SequenceIndexDataset,
# make_tst_collate_fn, fitting, validation checkpoint selection or model inputs.
OBSERVABLE_EFFECT_AUDIT_COLUMNS = [
    "audit_current_observable_effect",
    "audit_history_effect_window_count",
    "audit_context_effect_window_count",
    "audit_effect_stratum",
]
OBSERVABLE_EFFECT_STRATA = ["current_changed", "history_only_changed", "no_observable_change"]


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


# ============================================================
# 5. Dataset and collator
# ============================================================


class SequenceIndexDataset(Dataset):
    """Dataset that returns only integer sequence IDs.

    The collate function performs vectorized lookup:
      X_batch = embeddings[sequence_indices[batch_ids]]
    This avoids storing materialized X on disk and reduces Python overhead.
    """

    def __init__(self, num_sequences: int):
        if num_sequences <= 0:
            raise ValueError(f"num_sequences must be positive, got {num_sequences}")
        self.num_sequences = int(num_sequences)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> int:
        return int(idx)


def make_tst_collate_fn(
    embeddings: np.ndarray,
    sequence_indices: np.ndarray,
    y: np.ndarray,
) -> Any:
    def collate_fn(batch_sequence_ids: List[int]) -> Dict[str, torch.Tensor]:
        batch_ids = np.asarray(batch_sequence_ids, dtype=np.int64)
        batch_sequence_indices = sequence_indices[batch_ids]
        X = embeddings[batch_sequence_indices].astype(np.float32, copy=False)
        labels = y[batch_ids].astype(np.int64, copy=False)

        return {
            "x": torch.from_numpy(X),
            "labels": torch.from_numpy(labels),
            "sequence_ids": torch.from_numpy(batch_ids),
        }

    return collate_fn


def make_loader(
    embeddings: np.ndarray,
    sequence_data: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = SequenceIndexDataset(num_sequences=len(sequence_data["y"]))
    collate_fn = make_tst_collate_fn(
        embeddings=embeddings,
        sequence_indices=sequence_data["sequence_indices"],
        y=sequence_data["y"],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )


# ============================================================
# 6. TST model
# ============================================================


class TimeSeriesTransformerClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.pooling != "last":
            raise ValueError("Only pooling='last' is supported for last-window state classification.")
        if config.d_model % config.num_heads != 0:
            raise ValueError(
                f"d_model={config.d_model} must be divisible by num_heads={config.num_heads}"
            )

        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.sequence_length, config.d_model))
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_encoder_layers,
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        final_linear = self.classifier[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.xavier_uniform_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, L, input_dim]
        if x.ndim != 3:
            raise ValueError(f"TST input must be 3-D [B,L,D], got {tuple(x.shape)}")
        if x.shape[1] != self.config.sequence_length:
            raise ValueError(
                f"Sequence length mismatch: model L={self.config.sequence_length}, input L={x.shape[1]}"
            )
        h = self.input_norm(x)
        h = self.input_projection(h)
        h = h + self.position_embedding[:, : h.shape[1], :]
        h = self.dropout(h)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        h_last = self.output_norm(h_last)
        logits = self.classifier(h_last)
        return logits


# ============================================================
# 7. Metrics
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
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
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
    weight = support_arr / support_arr.sum() if support_arr.sum() else np.zeros_like(support_arr)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_precision = float(np.sum(np.asarray(precisions) * weight)) if len(weight) else 0.0
    weighted_recall = float(np.sum(np.asarray(recalls) * weight)) if len(weight) else 0.0
    weighted_f1 = float(np.sum(np.asarray(f1s) * weight)) if len(weight) else 0.0

    summary = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "total": total,
        "correct": correct,
    }

    per_class_df = pd.DataFrame(rows)
    labels = [id_to_label.get(i, str(i)) for i in range(num_classes)]
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    return summary, per_class_df, cm_df


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def normalize_selection_metric_name(metric_name: str) -> str:
    """Map config-style metric names to keys stored in evaluation summaries."""
    name = str(metric_name or "macro_f1").strip().lower()
    if name.startswith("val_"):
        name = name[4:]
    if name.startswith("test_"):
        name = name[5:]
    aliases = {
        "acc": "accuracy",
        "f1": "macro_f1",
        "macro": "macro_f1",
        "weighted": "weighted_f1",
    }
    return aliases.get(name, name)


def validate_class_coverage(
    split_name: str,
    sequence_length: int,
    y: np.ndarray,
    num_classes: int,
    id_to_label: Dict[int, str],
) -> Dict[str, Any]:
    counts = np.bincount(y.astype(np.int64), minlength=num_classes)
    missing_ids = [int(i) for i, count in enumerate(counts) if int(count) == 0]
    summary = {
        "split": split_name,
        "sequence_length": int(sequence_length),
        "num_sequences": int(len(y)),
        "num_classes": int(num_classes),
        "support_by_class_id": {str(i): int(counts[i]) for i in range(num_classes)},
        "support_by_label": {id_to_label.get(i, str(i)): int(counts[i]) for i in range(num_classes)},
        "missing_class_ids": missing_ids,
        "missing_labels": [id_to_label.get(i, str(i)) for i in missing_ids],
    }
    if STRICT_CLASS_COVERAGE_CHECK and missing_ids:
        raise ValueError(
            f"[L={sequence_length}][{split_name}] Missing class(es) in sequence labels: "
            f"{summary['missing_labels']}. Check Step 7 class episode lengths and Step 10 "
            f"sequence_label_distribution/skipped_short_episodes before comparing models."
        )
    return summary


def align_metadata_to_sequence_ids(
    metadata_df: Optional[pd.DataFrame],
    sequence_ids: np.ndarray,
    context: str,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Align metadata rows to evaluation order.

    In the current Step 11 dataset, sequence_ids are split-local row positions because
    SequenceIndexDataset returns batch indices. This function keeps that behavior but
    also supports future artifacts where sequence_ids are stored as explicit metadata
    values.
    """
    if metadata_df is None:
        return None, {"available": False, "reason": "metadata_df is None", "context": context}

    meta = metadata_df.reset_index(drop=True).copy()
    seq_ids = np.asarray(sequence_ids, dtype=np.int64)
    if len(seq_ids) == 0:
        return meta.iloc[[]].copy(), {
            "available": True,
            "context": context,
            "alignment_method": "empty",
            "num_rows": 0,
        }

    min_id = int(seq_ids.min())
    max_id = int(seq_ids.max())

    # Current and preferred path for this script: split-local row positions.
    if min_id >= 0 and max_id < len(meta):
        aligned = meta.iloc[seq_ids].reset_index(drop=True)
        info = {
            "available": True,
            "context": context,
            "alignment_method": "row_position_iloc",
            "num_metadata_rows": int(len(meta)),
            "num_aligned_rows": int(len(aligned)),
            "sequence_id_min": min_id,
            "sequence_id_max": max_id,
        }
        if "sequence_id" in meta.columns:
            numeric_sequence_id = pd.to_numeric(aligned["sequence_id"], errors="coerce")
            if numeric_sequence_id.notna().all():
                info["metadata_sequence_id_matches_eval_sequence_ids"] = bool(
                    np.array_equal(numeric_sequence_id.to_numpy(dtype=np.int64), seq_ids)
                )
        return aligned, info

    # Fallback: align by explicit sequence_id column when eval ids are not row positions.
    if "sequence_id" in meta.columns:
        meta_numeric = meta.copy()
        meta_numeric["sequence_id"] = pd.to_numeric(meta_numeric["sequence_id"], errors="coerce")
        if meta_numeric["sequence_id"].isna().any():
            raise ValueError(f"[{context}] metadata sequence_id contains non-numeric values.")
        if meta_numeric["sequence_id"].duplicated().any():
            duplicated = meta_numeric.loc[meta_numeric["sequence_id"].duplicated(), "sequence_id"].head(10).tolist()
            raise ValueError(f"[{context}] metadata sequence_id has duplicated values: {duplicated}")
        indexed = meta_numeric.set_index("sequence_id", drop=False)
        missing = [int(x) for x in seq_ids if int(x) not in indexed.index]
        if missing:
            raise ValueError(f"[{context}] sequence_ids not found in metadata sequence_id column: {missing[:10]}")
        aligned = indexed.loc[seq_ids].reset_index(drop=True)
        return aligned, {
            "available": True,
            "context": context,
            "alignment_method": "sequence_id_loc",
            "num_metadata_rows": int(len(meta)),
            "num_aligned_rows": int(len(aligned)),
            "sequence_id_min": min_id,
            "sequence_id_max": max_id,
        }

    raise ValueError(
        f"[{context}] Cannot align metadata. Eval sequence_ids are outside metadata row-position range "
        f"and metadata has no sequence_id column. sequence_id range=({min_id}, {max_id}), "
        f"metadata rows={len(meta)}"
    )


def build_evaluation_subset_masks(
    eval_result: Dict[str, Any],
    metadata_df: Optional[pd.DataFrame],
    sequence_length: int,
    split_name: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build fair-comparison and independence-audit evaluation masks.

    - common_population: removes early episode positions that are unavailable to the
      largest L, allowing L={1,2,3,5,10} to be compared on a matched population.
    - non_overlap: keeps one sequence every L windows within each episode, reducing
      dependence from overlapping sliding windows.
    - common_non_overlap: intersection of the two masks.
    """
    seq_ids = eval_result["sequence_ids"].astype(np.int64)
    total = int(len(seq_ids))
    masks: Dict[str, np.ndarray] = {"full": np.ones(total, dtype=bool)}
    info: Dict[str, Any] = {
        "split": split_name,
        "sequence_length": int(sequence_length),
        "total_sequences": total,
        "common_evaluation_reference_length": int(COMMON_EVALUATION_REFERENCE_LENGTH),
        "common_evaluation_min_end_position": int(COMMON_EVALUATION_MIN_END_POSITION),
    }

    aligned_meta, align_info = align_metadata_to_sequence_ids(
        metadata_df=metadata_df,
        sequence_ids=seq_ids,
        context=f"{split_name}_L{sequence_length}_subset_masks",
    )
    info["metadata_alignment"] = align_info

    required_cols = {"episode_id", "sequence_start_step", "sequence_end_step"}
    if aligned_meta is None:
        info["subset_metrics_available"] = False
        info["reason"] = "sequence metadata is unavailable"
        return masks, info

    missing_cols = sorted(required_cols.difference(aligned_meta.columns))
    if missing_cols:
        info["subset_metrics_available"] = False
        info["reason"] = f"metadata missing required columns: {missing_cols}"
        return masks, info

    meta = aligned_meta.reset_index(drop=True).copy()
    for col in ["sequence_start_step", "sequence_end_step"]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    if meta[["sequence_start_step", "sequence_end_step"]].isna().any().any():
        info["subset_metrics_available"] = False
        info["reason"] = "sequence_start_step or sequence_end_step contains non-numeric values"
        return masks, info

    origins = metadata_df.groupby("episode_id")["sequence_start_step"].min()
    first_step = meta["episode_id"].map(origins).to_numpy(dtype=np.int64)
    end_pos = meta["sequence_end_step"].to_numpy(dtype=np.int64) - first_step

    common_mask = verified_common_mask(meta, split_name)
    if sequence_length <= 1:
        non_overlap_mask = np.ones(total, dtype=bool)
    else:
        # Sequence ending positions L-1, 2L-1, 3L-1, ... correspond to non-overlapping
        # start positions 0, L, 2L, ... under normal Step 10 sliding construction.
        non_overlap_mask = (end_pos % int(sequence_length)) == (int(sequence_length) - 1)

    masks["common_population"] = common_mask.astype(bool)
    masks["non_overlap"] = non_overlap_mask.astype(bool)
    masks["common_non_overlap"] = (common_mask & non_overlap_mask).astype(bool)

    info["subset_metrics_available"] = True
    info["end_position_min"] = int(end_pos.min()) if total else None
    info["end_position_max"] = int(end_pos.max()) if total else None
    info["counts"] = {name: int(mask.sum()) for name, mask in masks.items()}
    info["ratios"] = {name: float(mask.sum() / total) if total else 0.0 for name, mask in masks.items()}
    return masks, info



def build_sequence_position_masks_from_metadata(
    metadata_df: Optional[pd.DataFrame],
    sequence_length: int,
    split_name: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build row-aligned sequence masks directly from Step 10 metadata.

    The non_overlap mask keeps episode-local sequence ending positions
    L-1, 2L-1, 3L-1, ... . This reduces dependence caused by stride-1
    sliding sequences and makes per-L comparison more conservative.
    """
    if metadata_df is None:
        return {}, {
            "available": False,
            "split": split_name,
            "sequence_length": int(sequence_length),
            "reason": "sequence metadata is unavailable",
        }

    meta = metadata_df.reset_index(drop=True).copy()
    total = int(len(meta))
    masks: Dict[str, np.ndarray] = {"full": np.ones(total, dtype=bool)}
    info: Dict[str, Any] = {
        "available": True,
        "split": split_name,
        "sequence_length": int(sequence_length),
        "total_sequences": total,
        "common_evaluation_reference_length": int(COMMON_EVALUATION_REFERENCE_LENGTH),
        "common_evaluation_min_end_position": int(COMMON_EVALUATION_MIN_END_POSITION),
    }

    required_cols = {"episode_id", "sequence_start_step", "sequence_end_step"}
    missing_cols = sorted(required_cols.difference(meta.columns))
    if missing_cols:
        info.update({
            "available": False,
            "reason": f"metadata missing required columns: {missing_cols}",
        })
        return masks, info

    for col in ["sequence_start_step", "sequence_end_step"]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    if meta[["sequence_start_step", "sequence_end_step"]].isna().any().any():
        info.update({
            "available": False,
            "reason": "sequence_start_step or sequence_end_step contains non-numeric values",
        })
        return masks, info

    origins = metadata_df.groupby("episode_id")["sequence_start_step"].min()
    first_step = meta["episode_id"].map(origins).to_numpy(dtype=np.int64)
    end_pos = meta["sequence_end_step"].to_numpy(dtype=np.int64) - first_step

    common_mask = verified_common_mask(meta, split_name)
    if sequence_length <= 1:
        non_overlap_mask = np.ones(total, dtype=bool)
    else:
        non_overlap_mask = (end_pos % int(sequence_length)) == (int(sequence_length) - 1)

    masks["common_population"] = common_mask.astype(bool)
    masks["non_overlap"] = non_overlap_mask.astype(bool)
    masks["common_non_overlap"] = (common_mask & non_overlap_mask).astype(bool)

    info["available"] = True
    info["end_position_min"] = int(end_pos.min()) if total else None
    info["end_position_max"] = int(end_pos.max()) if total else None
    info["counts"] = {name: int(mask.sum()) for name, mask in masks.items()}
    info["ratios"] = {name: float(mask.sum() / total) if total else 0.0 for name, mask in masks.items()}
    return masks, info


def filter_sequence_data_by_mask(
    sequence_data: Dict[str, np.ndarray],
    mask: np.ndarray,
    split_name: str,
    sequence_length: int,
    subset_name: str,
) -> Dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    n = len(sequence_data["y"])
    if mask.ndim != 1 or len(mask) != n:
        raise ValueError(
            f"[L={sequence_length}][{split_name}] {subset_name} mask length mismatch: "
            f"mask={mask.shape}, sequences={n}"
        )
    if int(mask.sum()) <= 0:
        raise ValueError(
            f"[L={sequence_length}][{split_name}] {subset_name} mask selected zero sequences."
        )

    filtered: Dict[str, np.ndarray] = {}
    for key, value in sequence_data.items():
        arr = np.asarray(value)
        if len(arr) == n:
            filtered[key] = arr[mask]
        else:
            filtered[key] = arr
    return filtered


def filter_metadata_by_mask(metadata_df: Optional[pd.DataFrame], mask: np.ndarray) -> Optional[pd.DataFrame]:
    if metadata_df is None:
        return None
    return metadata_df.reset_index(drop=True).loc[np.asarray(mask, dtype=bool)].reset_index(drop=True)


def build_split_episode_leakage_audit(
    metadata_splits: Dict[str, Optional[pd.DataFrame]],
    sequence_length: int,
) -> Dict[str, Any]:
    """Check whether the same episode_id appears across train/val/test splits."""
    audit: Dict[str, Any] = {
        "sequence_length": int(sequence_length),
        "episode_id_available": True,
        "overlaps": {},
    }
    episode_sets: Dict[str, set] = {}

    for split_name in SPLITS:
        df = metadata_splits.get(split_name)
        if df is None or "episode_id" not in df.columns:
            audit["episode_id_available"] = False
            audit["reason"] = f"episode_id unavailable for split={split_name}"
            return audit
        episode_sets[split_name] = set(df["episode_id"].astype(str).unique().tolist())
        audit[f"{split_name}_num_episodes"] = int(len(episode_sets[split_name]))

    has_overlap = False
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            overlap = sorted(episode_sets[left].intersection(episode_sets[right]))
            audit["overlaps"][f"{left}_vs_{right}"] = {
                "count": int(len(overlap)),
                "examples": overlap[:20],
            }
            if overlap:
                has_overlap = True

    audit["has_episode_overlap"] = bool(has_overlap)
    if has_overlap and RAISE_ON_EPISODE_SPLIT_OVERLAP:
        raise ValueError(
            f"[L={sequence_length}] Episode-level split leakage detected: {audit['overlaps']}"
        )
    return audit


def prepare_strict_sequence_views(
    sequence_splits: Dict[str, Dict[str, np.ndarray]],
    metadata_splits: Dict[str, Optional[pd.DataFrame]],
    sequence_length: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Optional[pd.DataFrame]], Dict[str, Any]]:
    """Prepare train/val/test sequence views for conservative TST evaluation.

    The raw sequence_splits are preserved for full-audit metrics. The returned
    views are used for model fitting, validation selection, and primary testing.
    """
    prepared_sequences = {split: data for split, data in sequence_splits.items()}
    prepared_metadata = {split: metadata_splits.get(split) for split in SPLITS}
    report: Dict[str, Any] = {
        "enabled": bool(STRICT_SEQUENCE_INDEPENDENCE_MODE),
        "fit_subset": SEQUENCE_FIT_SUBSET,
        "primary_test_subset": PRIMARY_TEST_SUBSET,
        "split_masks": {},
        "selected_counts": {},
    }

    report["episode_leakage_audit"] = build_split_episode_leakage_audit(
        metadata_splits=metadata_splits,
        sequence_length=sequence_length,
    )

    if not STRICT_SEQUENCE_INDEPENDENCE_MODE:
        for split_name in SPLITS:
            report["selected_counts"][split_name] = int(len(sequence_splits[split_name]["y"]))
        return prepared_sequences, prepared_metadata, report

    for split_name in SPLITS:
        masks, info = build_sequence_position_masks_from_metadata(
            metadata_df=metadata_splits.get(split_name),
            sequence_length=sequence_length,
            split_name=split_name,
        )
        report["split_masks"][split_name] = info

        if not info.get("available", False):
            # Keep full data if metadata is not available. This is intentionally
            # explicit in the report rather than silently changing the protocol.
            report.setdefault("warnings", []).append(
                f"[{split_name}] strict subset unavailable; using full sequences. Reason: {info.get('reason')}"
            )
            report["selected_counts"][split_name] = int(len(sequence_splits[split_name]["y"]))
            continue

        subset_name = SEQUENCE_FIT_SUBSET if split_name in {"train", "val"} else PRIMARY_TEST_SUBSET
        if subset_name not in masks:
            raise ValueError(
                f"[L={sequence_length}][{split_name}] Unknown strict subset '{subset_name}'. "
                f"Available subsets: {sorted(masks.keys())}"
            )

        mask = masks[subset_name]
        prepared_sequences[split_name] = filter_sequence_data_by_mask(
            sequence_data=sequence_splits[split_name],
            mask=mask,
            split_name=split_name,
            sequence_length=sequence_length,
            subset_name=subset_name,
        )
        prepared_metadata[split_name] = filter_metadata_by_mask(metadata_splits.get(split_name), mask)
        report["selected_counts"][split_name] = int(mask.sum())
        report.setdefault("selected_subset_by_split", {})[split_name] = subset_name

    return prepared_sequences, prepared_metadata, report


def evaluate_temporal_ablations(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    id_to_label: Dict[int, str],
    sequence_length: int,
    split_name: str,
    report_dir: Path,
    metadata_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    if not RUN_TEMPORAL_ABLATION_EVAL or sequence_length <= 1:
        return {
            "enabled": bool(RUN_TEMPORAL_ABLATION_EVAL),
            "skipped": True,
            "reason": "sequence_length <= 1 or temporal ablation disabled",
        }

    results: Dict[str, Any] = {
        "enabled": True, "split": split_name, "sequence_length": int(sequence_length),
        "population": "all available sequences for this L, with verified common-endpoint results saved separately",
        "interpretation": (
            "Fixed-checkpoint controls. history_zero/repeat_last perturb the embedding distribution; "
            "they are not retrained baselines or literal field deletion. prefix_reverse tests historical "
            "order sensitivity while keeping the current window fixed (it is unchanged at L=2). "
            "Full-population scores do not share endpoints across L; common_population scores do. "
            "These probes alone do not establish architectural superiority or information bounds."
        ),
    }
    for mode in TEMPORAL_ABLATION_MODES:
        eval_result = evaluate_model(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            id_to_label=id_to_label,
            desc=f"{split_name}_{mode}",
            input_transform=mode,
        )
        results[mode] = eval_result["summary"]

        per_class_path = report_dir / f"temporal_ablation_per_class_{split_name}_{mode}_L{sequence_length}.csv"
        cm_path = report_dir / f"temporal_ablation_confusion_matrix_{split_name}_{mode}_L{sequence_length}.csv"
        save_dataframe(per_class_path, eval_result["per_class"])
        save_dataframe(cm_path, eval_result["confusion_matrix"].reset_index())
        results[mode]["per_class_csv"] = str(per_class_path)
        results[mode]["confusion_matrix_csv"] = str(cm_path)
        if metadata_df is not None:
            aligned, _ = align_metadata_to_sequence_ids(metadata_df, eval_result["sequence_ids"],
                                                        f"{split_name}_{mode}_common")
            common_result = compute_masked_eval_result(
                eval_result, verified_common_mask(aligned, split_name), num_classes, id_to_label)
            results[mode]["common_population"] = common_result["summary"] if common_result is not None else None
            if common_result is not None:
                common_per_class = report_dir / f"temporal_ablation_per_class_{split_name}_{mode}_common_population_L{sequence_length}.csv"
                common_confusion = report_dir / f"temporal_ablation_confusion_matrix_{split_name}_{mode}_common_population_L{sequence_length}.csv"
                save_dataframe(common_per_class, common_result["per_class"])
                save_dataframe(common_confusion, common_result["confusion_matrix"].reset_index())
                results[mode]["common_population"]["per_class_csv"] = str(common_per_class)
                results[mode]["common_population"]["confusion_matrix_csv"] = str(common_confusion)

    return results



def compute_masked_eval_result(
    eval_result: Dict[str, Any],
    mask: np.ndarray,
    num_classes: int,
    id_to_label: Dict[int, str],
) -> Optional[Dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != len(eval_result["y_true"]):
        raise ValueError("Evaluation mask shape does not match eval_result length.")
    if int(mask.sum()) == 0:
        return None

    summary, per_class_df, cm_df = compute_metrics(
        y_true=eval_result["y_true"][mask],
        y_pred=eval_result["y_pred"][mask],
        num_classes=num_classes,
        id_to_label=id_to_label,
    )
    summary["evaluated_sequences"] = int(mask.sum())
    summary["selection_ratio"] = float(mask.sum() / len(mask)) if len(mask) else 0.0
    return {
        "summary": summary,
        "per_class": per_class_df,
        "confusion_matrix": cm_df,
    }


def save_subset_eval_artifacts(
    subset_results: Dict[str, Optional[Dict[str, Any]]],
    report_dir: Path,
    split_name: str,
    sequence_length: int,
) -> Dict[str, Dict[str, Optional[str]]]:
    artifacts: Dict[str, Dict[str, Optional[str]]] = {}
    for subset_name, result in subset_results.items():
        if subset_name == "full":
            continue
        if result is None:
            artifacts[subset_name] = {"per_class_csv": None, "confusion_matrix_csv": None}
            continue
        per_class_path = report_dir / f"per_class_metrics_{split_name}_{subset_name}_L{sequence_length}.csv"
        cm_path = report_dir / f"confusion_matrix_{split_name}_{subset_name}_L{sequence_length}.csv"
        save_dataframe(per_class_path, result["per_class"])
        save_dataframe(cm_path, result["confusion_matrix"].reset_index())
        artifacts[subset_name] = {
            "per_class_csv": str(per_class_path),
            "confusion_matrix_csv": str(cm_path),
        }
    return artifacts


def build_subset_evaluation_report(
    eval_result: Dict[str, Any],
    metadata_df: Optional[pd.DataFrame],
    sequence_length: int,
    split_name: str,
    num_classes: int,
    id_to_label: Dict[int, str],
    report_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Optional[str]]]]:
    masks, mask_info = build_evaluation_subset_masks(
        eval_result=eval_result,
        metadata_df=metadata_df,
        sequence_length=sequence_length,
        split_name=split_name,
    )

    subset_results: Dict[str, Optional[Dict[str, Any]]] = {"full": eval_result}
    if mask_info.get("subset_metrics_available", False):
        requested_subsets = []
        if REPORT_COMMON_POPULATION_METRICS:
            requested_subsets.append("common_population")
        if REPORT_NON_OVERLAP_METRICS:
            requested_subsets.extend(["non_overlap", "common_non_overlap"])
        for subset_name in requested_subsets:
            subset_results[subset_name] = compute_masked_eval_result(
                eval_result=eval_result,
                mask=masks[subset_name],
                num_classes=num_classes,
                id_to_label=id_to_label,
            )

    subset_artifacts = save_subset_eval_artifacts(
        subset_results=subset_results,
        report_dir=report_dir,
        split_name=split_name,
        sequence_length=sequence_length,
    )

    summaries: Dict[str, Any] = {
        "full": eval_result["summary"],
        "mask_info": mask_info,
    }
    for subset_name, result in subset_results.items():
        if subset_name == "full":
            continue
        summaries[subset_name] = result["summary"] if result is not None else None

    return summaries, subset_artifacts


def get_subset_metric(
    subset_summaries: Dict[str, Any],
    subset_name: str,
    metric_name: str,
) -> Optional[float]:
    summary = subset_summaries.get(subset_name)
    if not isinstance(summary, dict) or metric_name not in summary:
        return None
    value = summary.get(metric_name)
    return float(value) if isinstance(value, (int, float, np.integer, np.floating)) else None


def compute_observable_effect_diagnostics(
    eval_result: Dict[str, Any], effects: pd.DataFrame, sequence_length: int,
    split_name: str, num_classes: int, id_to_label: Dict[int, str],
) -> Dict[str, Any]:
    """Describe existing predictions; never rerun, select or alter the model/data."""
    aligned, _ = align_metadata_to_sequence_ids(effects, eval_result["sequence_ids"],
                                               f"{split_name}_L{sequence_length}_observable_effects")
    y_true, y_pred = eval_result["y_true"], eval_result["y_pred"]
    if (aligned is None or not np.array_equal(integer_array(aligned["label_id"], "audit label_id"), y_true)
            or len(y_pred) != len(y_true)
            or np.any((y_true < 0) | (y_true >= num_classes) | (y_pred < 0) | (y_pred >= num_classes))):
        raise ValueError("Observable-effect diagnostic labels do not match evaluated predictions.")
    normal_ids = [key for key, label in id_to_label.items() if label == "NORMAL"]
    if len(normal_ids) != 1:
        raise ValueError("Observable-effect diagnostics require one NORMAL label.")
    population_masks = {"full": np.ones(len(y_true), dtype=bool),
                        "common_population": verified_common_mask(aligned, split_name)}
    labels = [id_to_label[key] for key in range(num_classes)]
    report: Dict[str, Any] = {
        "split": split_name, "sequence_length": int(sequence_length),
        "diagnostic_basis": "Actual versus same-window nominal-counterfactual 22-field projection differences.",
        "interpretation": (
            "Descriptive post-hoc conditioning of fixed-checkpoint predictions. A zero difference is not a "
            "no-information proof, Bayes oracle/error bound or statement about class identifiability. "
            "Conditioning can change class/context composition and introduce selection bias. "
            "No labels, predictions, fitting samples or checkpoint-selection criteria are changed."
        ),
        "metric_basis": (
            "Accuracy and NORMAL prediction fraction condition on each population/stratum. "
            "macro_f1_present_true_classes averages only classes with positive true support in that stratum; "
            "absent classes are listed, so this changing-class-basis metric must not replace or be directly "
            "compared with the main fixed-six-class macro-F1. Empty strata have null scalar metrics."
        ),
        "confusion_matrix_labels": labels,
        "populations": {},
    }
    for population, population_mask in population_masks.items():
        total = int(population_mask.sum())
        rows = {}
        for stratum in OBSERVABLE_EFFECT_STRATA:
            mask = population_mask & aligned["audit_effect_stratum"].eq(stratum).to_numpy()
            summary, per_class, confusion = compute_metrics(y_true[mask], y_pred[mask], num_classes, id_to_label)
            n = int(mask.sum())
            present = per_class["support"].gt(0)
            rows[stratum] = {
                "num_sequences": n,
                "fraction_of_population": float(n / total) if total else None,
                "support_by_label": dict(zip(labels, per_class["support"].astype(int).tolist())),
                "present_true_labels": per_class.loc[present, "label"].tolist(),
                "absent_true_labels": per_class.loc[~present, "label"].tolist(),
                "accuracy": summary["accuracy"] if n else None,
                "normal_prediction_fraction": float(np.mean(y_pred[mask] == normal_ids[0])) if n else None,
                "macro_f1_present_true_classes": float(per_class.loc[present, "f1"].mean()) if n else None,
                "confusion_matrix": confusion.to_numpy(dtype=np.int64).tolist(),
            }
        report["populations"][population] = {"total_sequences": total, "strata": rows}
    return report


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


def autocast_context(device: torch.device) -> Any:
    if device.type != "cuda" or not USE_AMP:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type="cuda")
        except TypeError:
            pass
    return torch.cuda.amp.autocast()


def load_checkpoint_safely(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(path, map_location=device)
    except Exception as exc:
        print(
            f"[WARN] torch.load(..., weights_only=True) failed for {path.name}: {exc}. "
            f"Falling back to weights_only=False for this locally generated checkpoint.",
            file=sys.stderr,
        )
        return torch.load(path, map_location=device, weights_only=False)


# ============================================================
# 8. Train/evaluate functions
# ============================================================


def get_missing_aware_regularization_params(sequence_length: int) -> Dict[str, float]:
    """Return train-time missing-aware augmentation strengths for a sequence length.

    This augmentation is applied only to training batches. It is a generic
    embedding-space regularizer; it does not recreate UNK/MISSING_FIELD edits
    to the serialized 22-field record.
    """
    default = {
        "gaussian_noise_std": 0.0,
        "sequence_dim_dropout_prob": 0.0,
        "window_dim_dropout_prob": 0.0,
        "token_replacement_prob": 0.0,
        "last_token_replacement_prob": 0.0,
    }
    if not USE_MISSING_AWARE_TRAINING:
        return default

    params = MISSING_AWARE_REGULARIZATION.get(int(sequence_length), default)
    return {
        key: float(params.get(key, default[key]))
        for key in default.keys()
    }


def _apply_inverted_dropout_mask(x: torch.Tensor, keep_mask: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """Apply a manually created dropout mask with inverted scaling."""
    if drop_prob <= 0:
        return x
    keep_prob = max(1.0 - float(drop_prob), 1e-6)
    return torch.where(keep_mask, x / keep_prob, torch.zeros_like(x))


def _build_missing_replacement(x: torch.Tensor, replacement_mode: str) -> torch.Tensor:
    """Build a replacement vector for heavily damaged window embeddings."""
    mode = str(replacement_mode or "sequence_mean").strip().lower()
    if mode == "sequence_mean":
        return x.mean(dim=1, keepdim=True).expand_as(x)
    if mode == "history_mean":
        if x.shape[1] <= 1:
            return torch.zeros_like(x)
        hist_mean = x[:, :-1, :].mean(dim=1, keepdim=True)
        return hist_mean.expand_as(x)
    if mode == "zero":
        return torch.zeros_like(x)
    raise ValueError(
        f"Unsupported MISSING_AWARE_REPLACEMENT={replacement_mode!r}. "
        "Use 'sequence_mean', 'history_mean', or 'zero'."
    )


def apply_missing_aware_training_augmentation(
    x: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    """Apply missing-aware augmentation to DistilBERT embedding sequences.

    The uploaded pipeline uses only one DistilBERT [CLS] embedding per
    1-second observable record. Because this script no longer has access to
    field-level tokens, field-level dropout cannot be reconstructed exactly here.
    This function applies generic embedding perturbations while preserving
    the model interface. No correspondence between embedding dimensions and
    particular fields or evidence sources is assumed:

    - sequence_dim_dropout_prob drops the same dimensions across the sequence;
    - window_dim_dropout_prob drops dimensions independently per window;
    - token_replacement_prob replaces whole embeddings with a configured vector.
      That vector is not guaranteed to represent a neutral or missing record.

    Validation and test batches are never augmented here.
    """
    if not USE_MISSING_AWARE_TRAINING:
        return x
    if x.ndim != 3:
        return x

    params = get_missing_aware_regularization_params(sequence_length)
    gaussian_noise_std = params["gaussian_noise_std"]
    sequence_dim_dropout_prob = params["sequence_dim_dropout_prob"]
    window_dim_dropout_prob = params["window_dim_dropout_prob"]
    token_replacement_prob = params["token_replacement_prob"]
    last_token_replacement_prob = params["last_token_replacement_prob"]

    if (
        gaussian_noise_std <= 0
        and sequence_dim_dropout_prob <= 0
        and window_dim_dropout_prob <= 0
        and token_replacement_prob <= 0
        and last_token_replacement_prob <= 0
    ):
        return x

    batch_size, seq_len, embed_dim = x.shape
    if batch_size == 0 or seq_len == 0 or embed_dim == 0:
        return x

    x_aug = x.clone()

    # Generic embedding-space noise; not calibrated to any field-value edit.
    if gaussian_noise_std > 0:
        x_aug = x_aug + torch.randn_like(x_aug) * float(gaussian_noise_std)

    # Sequence-consistent embedding-dimension dropout, with no field mapping.
    if sequence_dim_dropout_prob > 0:
        keep_mask = (
            torch.rand((batch_size, 1, embed_dim), device=x_aug.device, dtype=x_aug.dtype)
            >= float(sequence_dim_dropout_prob)
        )
        x_aug = _apply_inverted_dropout_mask(x_aug, keep_mask, sequence_dim_dropout_prob)

    # Window-specific embedding-dimension dropout.
    if window_dim_dropout_prob > 0:
        keep_mask = (
            torch.rand((batch_size, seq_len, embed_dim), device=x_aug.device, dtype=x_aug.dtype)
            >= float(window_dim_dropout_prob)
        )
        x_aug = _apply_inverted_dropout_mask(x_aug, keep_mask, window_dim_dropout_prob)

    # Whole-embedding replacement. The final embedding has a separate rate;
    # neither replacement rate denotes a probability of literal field deletion.
    if token_replacement_prob > 0 or last_token_replacement_prob > 0:
        replacement = _build_missing_replacement(x_aug, MISSING_AWARE_REPLACEMENT)
        token_probs = torch.full(
            (batch_size, seq_len, 1),
            fill_value=float(token_replacement_prob),
            device=x_aug.device,
            dtype=x_aug.dtype,
        )
        token_probs[:, -1:, :] = float(last_token_replacement_prob)
        token_mask = torch.rand((batch_size, seq_len, 1), device=x_aug.device, dtype=x_aug.dtype) < token_probs
        x_aug = torch.where(token_mask, replacement, x_aug)

    return x_aug


def get_length_aware_regularization_params(sequence_length: int) -> Dict[str, float]:
    """Return train-time regularization strengths for a sequence length.

    Validation and test inputs are never modified by these parameters. The purpose is
    to apply stronger shortcut-resistant regularization only for longer sequences,
    where repeated/persistent evidence can otherwise make the classifier nearly
    deterministic.
    """
    fallback = {
        "gaussian_noise_std": float(TRAIN_GAUSSIAN_NOISE_STD),
        "history_token_dropout_prob": float(TRAIN_HISTORY_TOKEN_DROPOUT_PROB),
        "last_window_replacement_prob": float(TRAIN_LAST_WINDOW_REPLACEMENT_PROB),
    }

    if not USE_LENGTH_AWARE_REGULARIZATION:
        return fallback

    params = LENGTH_AWARE_REGULARIZATION.get(int(sequence_length))
    if params is None:
        return fallback

    return {
        "gaussian_noise_std": float(params.get("gaussian_noise_std", fallback["gaussian_noise_std"])),
        "history_token_dropout_prob": float(params.get("history_token_dropout_prob", fallback["history_token_dropout_prob"])),
        "last_window_replacement_prob": float(params.get("last_window_replacement_prob", fallback["last_window_replacement_prob"])),
    }


def apply_shortcut_resistant_training_regularization(
    x: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    """Apply train-time input regularization to reduce last-window shortcut reliance.

    This function is used only during training. It does not change labels, validation
    inputs, or test inputs. For longer sequences, length-aware parameters apply
    stronger noise/dropout/replacement to discourage near-deterministic shortcut
    learning from persistent sequence-level cues.
    """
    if not USE_SHORTCUT_RESISTANT_TRAINING:
        return x

    if x.ndim != 3:
        return x

    params = get_length_aware_regularization_params(sequence_length)
    gaussian_noise_std = params["gaussian_noise_std"]
    history_token_dropout_prob = params["history_token_dropout_prob"]
    last_window_replacement_prob = params["last_window_replacement_prob"]

    batch_size, seq_len, _ = x.shape
    if batch_size == 0:
        return x

    # Work on a cloned tensor so the DataLoader batch object remains side-effect safe.
    x_reg = x.clone()

    if gaussian_noise_std and gaussian_noise_std > 0:
        x_reg = x_reg + torch.randn_like(x_reg) * gaussian_noise_std

    if seq_len <= 1:
        return x_reg

    history = x_reg[:, :-1, :]

    if history_token_dropout_prob and history_token_dropout_prob > 0:
        hist_mask = (
            torch.rand(
                (batch_size, seq_len - 1, 1),
                device=x_reg.device,
                dtype=x_reg.dtype,
            ) < history_token_dropout_prob
        )
        x_reg[:, :-1, :] = torch.where(hist_mask, torch.zeros_like(history), history)

    if last_window_replacement_prob and last_window_replacement_prob > 0:
        sample_mask = (
            torch.rand(
                (batch_size, 1, 1),
                device=x_reg.device,
                dtype=x_reg.dtype,
            ) < last_window_replacement_prob
        )
        if TRAIN_LAST_WINDOW_REPLACEMENT == "history_mean":
            replacement = x_reg[:, :-1, :].mean(dim=1, keepdim=True)
        elif TRAIN_LAST_WINDOW_REPLACEMENT == "zero":
            replacement = torch.zeros_like(x_reg[:, -1:, :])
        else:
            raise ValueError(
                f"Unsupported TRAIN_LAST_WINDOW_REPLACEMENT={TRAIN_LAST_WINDOW_REPLACEMENT!r}. "
                "Use 'history_mean' or 'zero'."
            )
        x_reg[:, -1:, :] = torch.where(sample_mask, replacement, x_reg[:, -1:, :])

    return x_reg


def make_cross_entropy_loss(
    class_weights: Optional[torch.Tensor],
) -> nn.Module:
    """Create CrossEntropyLoss with optional label smoothing when supported."""
    try:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(LABEL_SMOOTHING))
    except TypeError:
        if LABEL_SMOOTHING and LABEL_SMOOTHING > 0:
            print(
                "[WARN] Current PyTorch does not support CrossEntropyLoss(label_smoothing=...). "
                "Falling back to standard CrossEntropyLoss.",
                file=sys.stderr,
            )
        return nn.CrossEntropyLoss(weight=class_weights)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_class_weights(y_train: np.ndarray, num_classes: int, device: torch.device) -> Optional[torch.Tensor]:
    if not USE_CLASS_WEIGHTS:
        return None
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Cannot use class weights when a class has zero training support: {counts}")
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[Any],
    sequence_length: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    correct = 0

    iterator = progress_bar(loader, total=len(loader), desc="train")
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        train_x = apply_missing_aware_training_augmentation(
            batch["x"],
            sequence_length=sequence_length,
        )
        train_x = apply_shortcut_resistant_training_regularization(
            train_x,
            sequence_length=sequence_length,
        )

        if device.type == "cuda" and scaler is not None:
            with autocast_context(device):
                logits = model(train_x)
                loss = criterion(logits, batch["labels"])
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(train_x)
            loss = criterion(logits, batch["labels"])
            loss.backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        preds = torch.argmax(logits.detach(), dim=1)
        correct += int((preds == batch["labels"]).sum().item())

    return {
        "loss": total_loss / total_count if total_count else 0.0,
        "accuracy": correct / total_count if total_count else 0.0,
    }


def apply_temporal_ablation_transform(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply deterministic test-time temporal ablations.

    The last window is the classification target state, so the ablations below
    keep the final position semantically comparable whenever possible.
    """
    mode = str(mode or "full").strip().lower()
    if mode in {"full", "none", "original"}:
        return x

    if x.ndim != 3:
        raise ValueError(f"Temporal ablation expects [B,L,D], got {tuple(x.shape)}")
    if x.shape[1] <= 1:
        return x

    if mode == "history_zero":
        z = x.clone()
        z[:, :-1, :] = 0.0
        return z

    if mode == "repeat_last":
        return x[:, -1:, :].expand_as(x).clone()

    if mode == "prefix_reverse":
        z = x.clone()
        z[:, :-1, :] = torch.flip(z[:, :-1, :], dims=[1])
        return z

    raise ValueError(
        f"Unsupported temporal ablation mode: {mode}. "
        f"Allowed: full, history_zero, repeat_last, prefix_reverse"
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    id_to_label: Dict[int, str],
    desc: str,
    input_transform: str = "full",
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
        x = apply_temporal_ablation_transform(batch["x"], input_transform)
        logits = model(x)
        loss = criterion(logits, batch["labels"])

        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(batch["labels"].detach().cpu().numpy())
        all_sequence_ids.append(batch["sequence_ids"].detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0).astype(np.int64)
    sequence_ids = np.concatenate(all_sequence_ids, axis=0).astype(np.int64)
    y_pred = logits_np.argmax(axis=1).astype(np.int64)
    probs = softmax_numpy(logits_np)

    summary, per_class_df, cm_df = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        id_to_label=id_to_label,
    )
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




def _float_lookup_by_length(table: Dict[int, float], sequence_length: int, default: float) -> float:
    value = table.get(int(sequence_length), default)
    return float(value)


def _int_lookup_by_length(table: Dict[int, int], sequence_length: int, default: int) -> int:
    value = table.get(int(sequence_length), default)
    return int(value)


def apply_robust_evidence_budget_transform(
    x: torch.Tensor,
    sequence_length: int,
    repeat_index: int,
    batch_index: int,
) -> torch.Tensor:
    """Legacy optional embedding stress transform, dormant in the clean run.

    Noise/dropout have no literal observable-field interpretation. The historical
    last_keep_prob variable controls a priority boost and does not exclude the
    endpoint when false; it must not be reported as the actual retention rate.
    This helper is separate from the authenticated counterfactual diagnostics.
    """
    if x.ndim != 3:
        return x

    batch_size, seq_len, embed_dim = x.shape
    if batch_size == 0 or seq_len <= 0:
        return x

    keep_windows = _int_lookup_by_length(
        ROBUST_KEEP_WINDOWS_BY_LENGTH,
        sequence_length,
        default=seq_len,
    )
    keep_windows = max(1, min(int(keep_windows), int(seq_len)))
    last_keep_prob = _float_lookup_by_length(
        ROBUST_LAST_WINDOW_KEEP_PROB_BY_LENGTH,
        sequence_length,
        default=1.0,
    )
    dim_dropout_prob = _float_lookup_by_length(
        ROBUST_DIM_DROPOUT_PROB_BY_LENGTH,
        sequence_length,
        default=0.0,
    )
    gaussian_noise_std = _float_lookup_by_length(
        ROBUST_GAUSSIAN_NOISE_STD_BY_LENGTH,
        sequence_length,
        default=0.0,
    )

    generator = torch.Generator(device=x.device)
    seed = (
        int(DEFAULT_RANDOM_SEED)
        + int(sequence_length) * 100_000
        + int(repeat_index) * 10_000
        + int(batch_index)
    )
    generator.manual_seed(seed)

    z = x.clone()

    if keep_windows < seq_len:
        scores = torch.rand((batch_size, seq_len), device=x.device, generator=generator)
        keep_last = torch.rand((batch_size,), device=x.device, generator=generator) < last_keep_prob
        scores[keep_last, -1] = scores[keep_last, -1] + 2.0

        topk = torch.topk(scores, k=keep_windows, dim=1).indices
        keep_mask = torch.zeros((batch_size, seq_len), device=x.device, dtype=torch.bool)
        keep_mask.scatter_(1, topk, True)
        keep_mask = keep_mask.unsqueeze(-1)
        z = torch.where(keep_mask, z, torch.zeros_like(z))

    if dim_dropout_prob > 0:
        dim_mask = (
            torch.rand((batch_size, 1, embed_dim), device=x.device, generator=generator)
            >= dim_dropout_prob
        )
        z = torch.where(dim_mask, z, torch.zeros_like(z))

    if gaussian_noise_std > 0:
        noise = torch.randn(z.shape, device=x.device, generator=generator, dtype=z.dtype)
        z = z + noise * gaussian_noise_std

    return z


@torch.no_grad()
def evaluate_model_robust_evidence_budget(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    id_to_label: Dict[int, str],
    sequence_length: int,
    desc: str,
) -> Dict[str, Any]:
    """Evaluate with MC-averaged robust evidence-budget perturbations."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_logits: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_sequence_ids: List[np.ndarray] = []

    repeats = max(1, int(ROBUST_MC_REPEATS))
    iterator = progress_bar(loader, total=len(loader), desc=desc)
    for batch_index, batch in enumerate(iterator):
        batch = move_batch_to_device(batch, device)
        logits_sum: Optional[torch.Tensor] = None
        for repeat_index in range(repeats):
            x_stress = apply_robust_evidence_budget_transform(
                batch["x"],
                sequence_length=sequence_length,
                repeat_index=repeat_index,
                batch_index=batch_index,
            )
            logits = model(x_stress)
            logits_sum = logits if logits_sum is None else logits_sum + logits

        assert logits_sum is not None
        logits_avg = logits_sum / float(repeats)
        loss = criterion(logits_avg, batch["labels"])

        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        all_logits.append(logits_avg.detach().cpu().numpy())
        all_y.append(batch["labels"].detach().cpu().numpy())
        all_sequence_ids.append(batch["sequence_ids"].detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0).astype(np.int64)
    sequence_ids = np.concatenate(all_sequence_ids, axis=0).astype(np.int64)
    y_pred = logits_np.argmax(axis=1).astype(np.int64)
    probs = softmax_numpy(logits_np)

    summary, per_class_df, cm_df = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        id_to_label=id_to_label,
    )
    summary["loss"] = total_loss / total_count if total_count else 0.0
    summary["robust_evidence_budget_eval"] = True
    summary["robust_mc_repeats"] = int(repeats)
    summary["robust_keep_windows"] = _int_lookup_by_length(
        ROBUST_KEEP_WINDOWS_BY_LENGTH,
        sequence_length,
        default=int(sequence_length),
    )
    summary["robust_last_window_keep_prob"] = _float_lookup_by_length(
        ROBUST_LAST_WINDOW_KEEP_PROB_BY_LENGTH,
        sequence_length,
        default=1.0,
    )
    summary["robust_dim_dropout_prob"] = _float_lookup_by_length(
        ROBUST_DIM_DROPOUT_PROB_BY_LENGTH,
        sequence_length,
        default=0.0,
    )
    summary["robust_gaussian_noise_std"] = _float_lookup_by_length(
        ROBUST_GAUSSIAN_NOISE_STD_BY_LENGTH,
        sequence_length,
        default=0.0,
    )

    return {
        "summary": summary,
        "per_class": per_class_df,
        "confusion_matrix": cm_df,
        "sequence_ids": sequence_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }


def metric_is_better(
    current: Dict[str, float],
    best: Optional[Dict[str, float]],
    metric_name: str = PRIMARY_SELECTION_METRIC,
) -> bool:
    if best is None:
        return True

    metric_key = normalize_selection_metric_name(metric_name)
    if metric_key not in current:
        raise KeyError(
            f"Selection metric '{metric_name}' was normalized to '{metric_key}', "
            f"but it is not present in current summary keys={sorted(current.keys())}."
        )
    if metric_key not in best:
        raise KeyError(
            f"Selection metric '{metric_name}' was normalized to '{metric_key}', "
            f"but it is not present in best summary keys={sorted(best.keys())}."
        )

    current_score = float(current.get(metric_key, 0.0))
    best_score = float(best.get(metric_key, 0.0))
    minimize = metric_key in {"loss", "val_loss", "test_loss"}

    if minimize:
        if current_score < best_score - 1e-12:
            return True
        if abs(current_score - best_score) <= 1e-12:
            return float(current.get("macro_f1", 0.0)) > float(best.get("macro_f1", 0.0))
        return False

    if current_score > best_score + 1e-12:
        return True
    if abs(current_score - best_score) <= 1e-12:
        return float(current.get("loss", math.inf)) < float(best.get("loss", math.inf))
    return False


# ============================================================
# 9. Output helpers
# ============================================================


def make_prediction_dataframe(
    eval_result: Dict[str, Any],
    id_to_label: Dict[int, str],
    metadata_df: Optional[pd.DataFrame],
    context: str = "prediction",
    observable_effects: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    sequence_ids = eval_result["sequence_ids"].astype(np.int64)
    y_true = eval_result["y_true"].astype(np.int64)
    y_pred = eval_result["y_pred"].astype(np.int64)
    probs = eval_result["probs"]

    df = pd.DataFrame({
        "sequence_id": sequence_ids,
        "true_label_id": y_true,
        "true_label": [id_to_label.get(int(x), str(int(x))) for x in y_true],
        "pred_label_id": y_pred,
        "pred_label": [id_to_label.get(int(x), str(int(x))) for x in y_pred],
        "correct": (y_true == y_pred).astype(np.int64),
        "confidence": probs.max(axis=1),
    })

    for class_id in range(probs.shape[1]):
        label = id_to_label.get(class_id, str(class_id))
        safe_label = label.lower().replace(" ", "_").replace("/", "_")
        df[f"prob_{safe_label}"] = probs[:, class_id]

    aligned_meta, align_info = align_metadata_to_sequence_ids(
        metadata_df=metadata_df,
        sequence_ids=sequence_ids,
        context=context,
    )
    df["metadata_alignment_method"] = align_info.get("alignment_method")

    if aligned_meta is not None:
        if len(aligned_meta) != len(df):
            raise ValueError(
                f"Prediction metadata length mismatch after alignment: "
                f"metadata={len(aligned_meta)}, predictions={len(df)}"
            )
        useful_cols = [
            col for col in [
                "episode_id",
                "episode_type",
                "scenario_id",
                "sequence_start_row",
                "sequence_end_row",
                "sequence_start_step",
                "sequence_end_step",
                "sequence_start_timestamp",
                "sequence_end_timestamp",
                "start_label",
                "label",
                "sequence_num_label_changes",
                "sequence_has_label_transition",
                "temporal_context_seconds",
            ] if col in aligned_meta.columns
        ]
        if useful_cols:
            df = pd.concat([df, aligned_meta[useful_cols].reset_index(drop=True)], axis=1)

    if aligned_meta is None or "source_split" not in aligned_meta:
        raise ValueError("Prediction provenance requires verified sequence metadata/source_split.")
    source_splits = aligned_meta["source_split"].unique()
    if len(source_splits) != 1:
        raise ValueError("Predictions mix source splits.")
    df["source_split"] = source_splits[0]
    df["common_population"] = verified_common_mask(aligned_meta, source_splits[0]).astype(np.int64)
    if observable_effects is not None:
        aligned_effects, _ = align_metadata_to_sequence_ids(observable_effects, sequence_ids, context + "_audit")
        try:
            pd.testing.assert_frame_equal(aligned_effects[list(aligned_meta.columns)], aligned_meta, check_exact=True)
        except (AssertionError, KeyError) as exc:
            raise ValueError("Prediction observable-effect diagnostics are not aligned with metadata.") from exc
        df = pd.concat([df, aligned_effects[OBSERVABLE_EFFECT_AUDIT_COLUMNS]], axis=1)
    for key, value in result_provenance().items():
        df[key] = value
    return df


def save_confusion_matrix_png(path: Path, cm_df: pd.DataFrame, title: str) -> None:
    if plt is None or not SAVE_TRAINING_CURVE_PNG:
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
    ax.set_xticklabels([c.replace("pred_", "") for c in cm_df.columns], rotation=45, ha="right")
    ax.set_yticklabels([i.replace("true_", "") for i in cm_df.index])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_training_curve_png(path: Path, history_df: pd.DataFrame, sequence_length: int) -> None:
    if plt is None or not SAVE_TRAINING_CURVE_PNG:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    if history_df.empty:
        return

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"TST-L{sequence_length} training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    acc_path = path.with_name(path.stem.replace("loss", "metrics") + path.suffix)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(history_df["epoch"], history_df["train_accuracy"], label="train_accuracy")
    ax.plot(history_df["epoch"], history_df["val_accuracy"], label="val_accuracy")
    ax.plot(history_df["epoch"], history_df["val_macro_f1"], label="val_macro_f1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"TST-L{sequence_length} validation metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(acc_path, dpi=200)
    plt.close(fig)


# ============================================================
# 10. Per-L training pipeline
# ============================================================


def train_and_evaluate_for_length(
    sequence_length: int,
    embedding_splits: Dict[str, Dict[str, np.ndarray]],
    extraction_report: Dict[str, Any],
    label_order: List[str],
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    expected_embedding_dim: int,
    device: torch.device,
) -> Dict[str, Any]:
    temporal_context_seconds = int(sequence_length * EXPECTED_OBSERVABLE_WINDOW_SECONDS)
    out_dir = length_output_dir(sequence_length)
    ckpt_dir = out_dir / "checkpoints"
    report_dir = out_dir / "reports"
    pred_dir = out_dir / "predictions"
    for path in [ckpt_dir, report_dir, pred_dir]:
        path.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 78)
    print(f"Train TST-L{sequence_length} | temporal context={temporal_context_seconds}s")
    print("-" * 78)

    num_classes = len(label_order)
    sequence_splits: Dict[str, Dict[str, np.ndarray]] = {}
    metadata_splits: Dict[str, Optional[pd.DataFrame]] = {}
    class_coverage_summaries: Dict[str, Dict[str, Any]] = {}
    for split_name in SPLITS:
        sequence_splits[split_name] = load_sequence_split(
            split_name=split_name,
            sequence_length=sequence_length,
            embedding_split=embedding_splits[split_name],
            expected_embedding_dim=expected_embedding_dim,
        )
        metadata_splits[split_name] = load_sequence_metadata_if_exists(split_name, sequence_length)
        class_coverage_summaries[split_name] = validate_class_coverage(
            split_name=split_name,
            sequence_length=sequence_length,
            y=sequence_splits[split_name]["y"],
            num_classes=num_classes,
            id_to_label=id_to_label,
        )
        print(
            f"  {split_name}: sequences={len(sequence_splits[split_name]['y']):,}, "
            f"shape={sequence_splits[split_name]['sequence_indices'].shape}, "
            f"missing_classes={class_coverage_summaries[split_name]['missing_labels']}"
        )

    strict_sequence_splits, strict_metadata_splits, strict_sequence_report = prepare_strict_sequence_views(
        sequence_splits=sequence_splits,
        metadata_splits=metadata_splits,
        sequence_length=sequence_length,
    )
    print("  strict sequence mode:")
    print(f"    enabled={strict_sequence_report.get('enabled')}, "
          f"train/val subset={strict_sequence_report.get('fit_subset')}, "
          f"test subset={strict_sequence_report.get('primary_test_subset')}")
    print(f"    selected_counts={strict_sequence_report.get('selected_counts')}")
    length_reg_params = get_length_aware_regularization_params(sequence_length)
    print("  train-time regularization for this L:")
    print(f"    gaussian_noise_std={length_reg_params['gaussian_noise_std']}")
    print(f"    history_token_dropout_prob={length_reg_params['history_token_dropout_prob']}")
    print(f"    last_window_replacement_prob={length_reg_params['last_window_replacement_prob']}")
    missing_aware_params = get_missing_aware_regularization_params(sequence_length)
    print("  missing-aware augmentation for this L:")
    print(f"    enabled={USE_MISSING_AWARE_TRAINING}")
    print(f"    gaussian_noise_std={missing_aware_params['gaussian_noise_std']}")
    print(f"    sequence_dim_dropout_prob={missing_aware_params['sequence_dim_dropout_prob']}")
    print(f"    window_dim_dropout_prob={missing_aware_params['window_dim_dropout_prob']}")
    print(f"    token_replacement_prob={missing_aware_params['token_replacement_prob']}")
    print(f"    last_token_replacement_prob={missing_aware_params['last_token_replacement_prob']}")
    print(f"    replacement={MISSING_AWARE_REPLACEMENT}")

    strict_class_coverage_summaries: Dict[str, Dict[str, Any]] = {}
    for split_name in SPLITS:
        strict_class_coverage_summaries[split_name] = validate_class_coverage(
            split_name=f"{split_name}_strict",
            sequence_length=sequence_length,
            y=strict_sequence_splits[split_name]["y"],
            num_classes=num_classes,
            id_to_label=id_to_label,
        )

    model_config = ModelConfig(
        input_dim=expected_embedding_dim,
        num_classes=num_classes,
        sequence_length=sequence_length,
        temporal_context_seconds=temporal_context_seconds,
    )
    train_config = TrainConfig(**{name: globals()["DEFAULT_RANDOM_SEED" if name == "random_seed" else name.upper()]
                                  for name in TrainConfig.__dataclass_fields__})

    train_loader = make_loader(
        embeddings=embedding_splits["train"]["embeddings"],
        sequence_data=strict_sequence_splits["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        device=device,
    )
    val_loader = make_loader(
        embeddings=embedding_splits["val"]["embeddings"],
        sequence_data=strict_sequence_splits["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        device=device,
    )
    test_loader = make_loader(
        embeddings=embedding_splits["test"]["embeddings"],
        sequence_data=strict_sequence_splits["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        device=device,
    )
    full_test_loader = make_loader(
        embeddings=embedding_splits["test"]["embeddings"],
        sequence_data=sequence_splits["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        device=device,
    )

    model = TimeSeriesTransformerClassifier(model_config).to(device)
    class_weights = compute_class_weights(strict_sequence_splits["train"]["y"], num_classes=num_classes, device=device)
    criterion = make_cross_entropy_loss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    scaler = make_grad_scaler(device)

    best_val_summary: Optional[Dict[str, float]] = None
    best_epoch = 0
    bad_epochs = 0
    history_rows: List[Dict[str, Any]] = []
    best_ckpt_path = ckpt_dir / f"best_tst_model_L{sequence_length}.pt"

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n[L={sequence_length}] Epoch {epoch}/{MAX_EPOCHS}")

        train_summary = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            sequence_length=sequence_length,
        )
        val_result = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            id_to_label=id_to_label,
            desc="val",
        )
        val_summary = val_result["summary"]
        scheduler.step(float(val_summary["macro_f1"]))

        lr = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = float(time.time() - epoch_start)
        row = {
            "epoch": int(epoch),
            "learning_rate": lr,
            "train_loss": float(train_summary["loss"]),
            "train_accuracy": float(train_summary["accuracy"]),
            "val_loss": float(val_summary["loss"]),
            "val_accuracy": float(val_summary["accuracy"]),
            "val_macro_f1": float(val_summary["macro_f1"]),
            "val_weighted_f1": float(val_summary["weighted_f1"]),
            "epoch_seconds": epoch_seconds,
        }
        history_rows.append(row)

        print(
            f"[L={sequence_length}] "
            f"train_loss={row['train_loss']:.4f}, train_acc={row['train_accuracy']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, val_acc={row['val_accuracy']:.4f}, "
            f"val_macro_f1={row['val_macro_f1']:.4f}, lr={lr:.2e}"
        )

        if metric_is_better(val_summary, best_val_summary, PRIMARY_SELECTION_METRIC):
            best_val_summary = {k: float(v) for k, v in val_summary.items() if isinstance(v, (int, float))}
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint_atomic(
                {
                    "provenance": result_provenance(),
                    "source_contract": SOURCE_CONTRACT,
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "train_config": asdict(train_config),
                    "label_order": label_order,
                    "label_to_id": label_to_id,
                    "id_to_label": {str(k): v for k, v in id_to_label.items()},
                    "best_epoch": best_epoch,
                    "best_val_summary": best_val_summary,
                    "source_embedding_dir": str(EMBEDDING_DIR),
                    "source_sequence_dir": str(SEQUENCE_DIR),
                },
                best_ckpt_path,
            )
            print(f"[L={sequence_length}] Best checkpoint saved: {best_ckpt_path.name}")
        else:
            bad_epochs += 1
            print(f"[L={sequence_length}] No improvement: {bad_epochs}/{PATIENCE}")
            if bad_epochs >= PATIENCE:
                print(f"[L={sequence_length}] Early stopping at epoch {epoch}")
                break

    elapsed_seconds = float(time.time() - start_time)

    history_df = pd.DataFrame(history_rows)
    history_path = report_dir / f"training_history_L{sequence_length}.csv"
    save_dataframe(history_path, history_df)
    save_training_curve_png(report_dir / f"training_loss_L{sequence_length}.png", history_df, sequence_length)

    if not best_ckpt_path.exists():
        raise RuntimeError(f"Best checkpoint was not saved for L={sequence_length}")

    checkpoint = load_checkpoint_safely(best_ckpt_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\n[L={sequence_length}] Evaluate best checkpoint on val/test")
    val_eval = evaluate_model(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        id_to_label=id_to_label,
        desc="val_best",
    )
    standard_test_eval = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        id_to_label=id_to_label,
        desc="test_standard_best",
    )
    full_test_eval = evaluate_model(
        model=model,
        loader=full_test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        id_to_label=id_to_label,
        desc="test_full_audit",
    )

    robust_test_eval: Optional[Dict[str, Any]] = None
    use_robust_primary = (
        bool(RUN_ROBUST_EVIDENCE_BUDGET_EVAL)
        and bool(USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS)
        and int(sequence_length) >= int(ROBUST_PRIMARY_MIN_SEQUENCE_LENGTH)
    )
    if RUN_ROBUST_EVIDENCE_BUDGET_EVAL:
        robust_test_eval = evaluate_model_robust_evidence_budget(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            id_to_label=id_to_label,
            sequence_length=sequence_length,
            desc="test_robust_evidence_budget",
        )

    test_eval = robust_test_eval if (use_robust_primary and robust_test_eval is not None) else standard_test_eval

    temporal_ablation_summaries = evaluate_temporal_ablations(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        id_to_label=id_to_label,
        sequence_length=sequence_length,
        split_name="test_primary",
        report_dir=report_dir,
        metadata_df=strict_metadata_splits["test"],
    )

    val_subset_summaries, val_subset_artifacts = build_subset_evaluation_report(
        eval_result=val_eval,
        metadata_df=strict_metadata_splits["val"],
        sequence_length=sequence_length,
        split_name="val",
        num_classes=num_classes,
        id_to_label=id_to_label,
        report_dir=report_dir,
    )
    test_subset_summaries, test_subset_artifacts = build_subset_evaluation_report(
        eval_result=test_eval,
        metadata_df=strict_metadata_splits["test"],
        sequence_length=sequence_length,
        split_name="test",
        num_classes=num_classes,
        id_to_label=id_to_label,
        report_dir=report_dir,
    )
    full_test_subset_summaries, full_test_subset_artifacts = build_subset_evaluation_report(
        eval_result=full_test_eval,
        metadata_df=metadata_splits["test"],
        sequence_length=sequence_length,
        split_name="test_full_audit",
        num_classes=num_classes,
        id_to_label=id_to_label,
        report_dir=report_dir,
    )

    # Save metrics and matrices.
    save_dataframe(report_dir / f"per_class_metrics_val_L{sequence_length}.csv", val_eval["per_class"])
    save_dataframe(report_dir / f"per_class_metrics_test_L{sequence_length}.csv", test_eval["per_class"])
    save_dataframe(report_dir / f"per_class_metrics_test_standard_L{sequence_length}.csv", standard_test_eval["per_class"])
    save_dataframe(report_dir / f"per_class_metrics_test_full_audit_L{sequence_length}.csv", full_test_eval["per_class"])
    if robust_test_eval is not None:
        save_dataframe(report_dir / f"per_class_metrics_test_robust_evidence_budget_L{sequence_length}.csv", robust_test_eval["per_class"])
    save_dataframe(report_dir / f"confusion_matrix_val_L{sequence_length}.csv", val_eval["confusion_matrix"].reset_index())
    save_dataframe(report_dir / f"confusion_matrix_test_L{sequence_length}.csv", test_eval["confusion_matrix"].reset_index())
    save_dataframe(report_dir / f"confusion_matrix_test_standard_L{sequence_length}.csv", standard_test_eval["confusion_matrix"].reset_index())
    save_dataframe(report_dir / f"confusion_matrix_test_full_audit_L{sequence_length}.csv", full_test_eval["confusion_matrix"].reset_index())
    if robust_test_eval is not None:
        save_dataframe(report_dir / f"confusion_matrix_test_robust_evidence_budget_L{sequence_length}.csv", robust_test_eval["confusion_matrix"].reset_index())
    save_confusion_matrix_png(
        report_dir / f"confusion_matrix_test_L{sequence_length}.png",
        test_eval["confusion_matrix"],
        title=f"TST-L{sequence_length} Test Confusion Matrix",
    )

    # Separate audit-only columns are attached only after model selection and
    # evaluation. Reuse existing predictions; no diagnostic forward passes.
    observable_effects = {}
    observable_effect_diagnostic_paths = {}
    for split, evaluated in (("val", val_eval), ("test", test_eval)):
        effects = load_sequence_observable_effects(split, sequence_length, strict_metadata_splits[split])
        observable_effects[split] = effects
        diagnostic = compute_observable_effect_diagnostics(evaluated, effects, sequence_length,
                                                           split, num_classes, id_to_label)
        diagnostic.update(provenance=result_provenance(), source_audit=SOURCE_CONTRACT["observability_audit"])
        path = report_dir / f"observable_effect_diagnostics_{split}_L{sequence_length}.json"
        save_json(path, diagnostic)
        observable_effect_diagnostic_paths[split] = str(path)

    if SAVE_VAL_PREDICTIONS:
        val_pred_df = make_prediction_dataframe(val_eval, id_to_label, strict_metadata_splits["val"], context=f"val_predictions_L{sequence_length}", observable_effects=observable_effects["val"])
        save_dataframe(pred_dir / f"val_predictions_L{sequence_length}.csv", val_pred_df)
    if SAVE_TEST_PREDICTIONS:
        test_pred_df = make_prediction_dataframe(test_eval, id_to_label, strict_metadata_splits["test"], context=f"test_predictions_L{sequence_length}", observable_effects=observable_effects["test"])
        save_dataframe(pred_dir / f"test_predictions_L{sequence_length}.csv", test_pred_df)

    evaluation_report = {
        "step": "11_train_tst_classifier",
        "provenance": result_provenance(),
        "common_endpoint_report": SOURCE_CONTRACT["common_endpoint_report"],
        "observable_effect_diagnostics": observable_effect_diagnostic_paths,
        "sequence_length": int(sequence_length),
        "temporal_context_seconds": int(temporal_context_seconds),
        "best_epoch": int(best_epoch),
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "training_regularization_rule": (
            "Optional embedding perturbations follow the recorded enable flags and are off by default. "
            "Label smoothing retains its original setting. Standard validation/test inputs are unperturbed; "
            "episode labels and available sequences are retained."
        ),
        "length_aware_regularization_enabled": bool(USE_LENGTH_AWARE_REGULARIZATION),
        "length_aware_regularization_params_for_this_length": length_reg_params,
        "length_aware_regularization_table": LENGTH_AWARE_REGULARIZATION,
        "missing_aware_training_enabled": bool(USE_MISSING_AWARE_TRAINING),
        "missing_aware_replacement": MISSING_AWARE_REPLACEMENT,
        "missing_aware_params_for_this_length": missing_aware_params,
        "missing_aware_regularization_table": MISSING_AWARE_REGULARIZATION,
        "missing_aware_training_rule": (
            "Optional training-only embedding-space augmentation, disabled for the clean main run. "
            "Embedding masking is not literal observable-field deletion and is not evidence of field-level robustness."
        ),
        "label_order": label_order,
        "label_to_id": label_to_id,
        "class_weights": class_weights.detach().cpu().numpy().tolist() if class_weights is not None else None,
        "class_coverage": class_coverage_summaries,
        "strict_class_coverage": strict_class_coverage_summaries,
        "strict_sequence_independence": strict_sequence_report,
        "primary_test_subset": PRIMARY_TEST_SUBSET,
        "fit_subset_for_train_and_validation": SEQUENCE_FIT_SUBSET,
        "input_rule": "TST input is embeddings[sequence_indices]. Metadata is not used as model input.",
        "label_rule": "The sequence label is the label_id of the last 1-second window in the sequence.",
        "boundary_rule": "All sequences are generated by Step 10 within episode_id and within the same split.",
        "evaluation_audit_rule": (
            "The primary test result is evaluated on the full test population. "
            "common_population metrics compare all L on the verified Step 10 endpoint intersection, "
            "while non_overlap metrics are reported "
            "as an additional dependence-reduction audit for overlapping sliding windows."
        ),
        "best_checkpoint_path": str(best_ckpt_path),
        "training_history_path": str(history_path),
        "val_summary": val_eval["summary"],
        "test_summary": test_eval["summary"],
        "test_metric_protocol": (
            "robust_evidence_budget" if use_robust_primary else "standard_unperturbed"
        ),
        "use_robust_evidence_budget_as_primary": bool(use_robust_primary),
        "test_standard_summary": standard_test_eval["summary"],
        "test_robust_evidence_budget_summary": (
            robust_test_eval["summary"] if robust_test_eval is not None else None
        ),
        "robust_evidence_budget_rule": (
            "Legacy embedding-space stress helper is disabled by default and the clean protocol "
            "rejects robust-as-primary. It is not a field-deletion experiment or the observable-effect "
            "diagnostic. Its last-window setting is a selection-priority probability, not calibrated "
            "actual retention; standard unperturbed test results remain the main results."
        ),
        "robust_evidence_budget_config": {
            "enabled": bool(RUN_ROBUST_EVIDENCE_BUDGET_EVAL),
            "use_as_primary_for_long_contexts": bool(USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS),
            "primary_min_sequence_length": int(ROBUST_PRIMARY_MIN_SEQUENCE_LENGTH),
            "mc_repeats": int(ROBUST_MC_REPEATS),
            "keep_windows_by_length": ROBUST_KEEP_WINDOWS_BY_LENGTH,
            "last_window_keep_prob_by_length": ROBUST_LAST_WINDOW_KEEP_PROB_BY_LENGTH,
            "dim_dropout_prob_by_length": ROBUST_DIM_DROPOUT_PROB_BY_LENGTH,
            "gaussian_noise_std_by_length": ROBUST_GAUSSIAN_NOISE_STD_BY_LENGTH,
        },
        "test_full_audit_summary": full_test_eval["summary"],
        "temporal_ablation_summaries": temporal_ablation_summaries,
        "val_subset_summaries": val_subset_summaries,
        "test_subset_summaries": test_subset_summaries,
        "test_full_audit_subset_summaries": full_test_subset_summaries,
        "artifacts": {
            "observable_effect_diagnostics": observable_effect_diagnostic_paths,
            "per_class_val_csv": str(report_dir / f"per_class_metrics_val_L{sequence_length}.csv"),
            "per_class_test_csv": str(report_dir / f"per_class_metrics_test_L{sequence_length}.csv"),
            "per_class_test_standard_csv": str(report_dir / f"per_class_metrics_test_standard_L{sequence_length}.csv"),
            "per_class_test_robust_evidence_budget_csv": (
                str(report_dir / f"per_class_metrics_test_robust_evidence_budget_L{sequence_length}.csv")
                if robust_test_eval is not None else None
            ),
            "confusion_matrix_val_csv": str(report_dir / f"confusion_matrix_val_L{sequence_length}.csv"),
            "confusion_matrix_test_csv": str(report_dir / f"confusion_matrix_test_L{sequence_length}.csv"),
            "confusion_matrix_test_standard_csv": str(report_dir / f"confusion_matrix_test_standard_L{sequence_length}.csv"),
            "confusion_matrix_test_robust_evidence_budget_csv": (
                str(report_dir / f"confusion_matrix_test_robust_evidence_budget_L{sequence_length}.csv")
                if robust_test_eval is not None else None
            ),
            "per_class_test_full_audit_csv": str(report_dir / f"per_class_metrics_test_full_audit_L{sequence_length}.csv"),
            "confusion_matrix_test_full_audit_csv": str(report_dir / f"confusion_matrix_test_full_audit_L{sequence_length}.csv"),
            "val_subset_artifacts": val_subset_artifacts,
            "test_subset_artifacts": test_subset_artifacts,
            "test_full_audit_subset_artifacts": full_test_subset_artifacts,
            "val_predictions_csv": str(pred_dir / f"val_predictions_L{sequence_length}.csv") if SAVE_VAL_PREDICTIONS else None,
            "test_predictions_csv": str(pred_dir / f"test_predictions_L{sequence_length}.csv") if SAVE_TEST_PREDICTIONS else None,
        },
    }
    report_path = report_dir / f"evaluation_metrics_L{sequence_length}.json"
    save_json(report_path, evaluation_report)

    print(
        f"[L={sequence_length}] Test ({'robust evidence-budget primary' if use_robust_primary else 'standard primary'}): "
        f"acc={test_eval['summary']['accuracy']:.4f}, "
        f"macro_f1={test_eval['summary']['macro_f1']:.4f}, "
        f"weighted_f1={test_eval['summary']['weighted_f1']:.4f}"
    )
    if robust_test_eval is not None:
        print(
            f"[L={sequence_length}] Robust evidence-budget audit: "
            f"acc={robust_test_eval['summary']['accuracy']:.4f}, "
            f"macro_f1={robust_test_eval['summary']['macro_f1']:.4f}; "
            f"standard_acc={standard_test_eval['summary']['accuracy']:.4f}, "
            f"standard_macro_f1={standard_test_eval['summary']['macro_f1']:.4f}"
        )

    return {
        **result_provenance(),
        "sequence_length": int(sequence_length),
        "temporal_context_seconds": int(temporal_context_seconds),
        "val_common_count": int(len(COMMON_ENDPOINTS["val"])),
        "test_common_count": int(len(COMMON_ENDPOINTS["test"])),
        "best_epoch": int(best_epoch),
        "elapsed_seconds": elapsed_seconds,
        "best_checkpoint_path": str(best_ckpt_path),
        "report_path": str(report_path),
        "val_accuracy": float(val_eval["summary"]["accuracy"]),
        "val_macro_f1": float(val_eval["summary"]["macro_f1"]),
        "val_weighted_f1": float(val_eval["summary"]["weighted_f1"]),
        "val_common_accuracy": get_subset_metric(val_subset_summaries, "common_population", "accuracy"),
        "val_common_macro_f1": get_subset_metric(val_subset_summaries, "common_population", "macro_f1"),
        "val_non_overlap_accuracy": get_subset_metric(val_subset_summaries, "non_overlap", "accuracy"),
        "val_non_overlap_macro_f1": get_subset_metric(val_subset_summaries, "non_overlap", "macro_f1"),
        "val_common_non_overlap_accuracy": get_subset_metric(val_subset_summaries, "common_non_overlap", "accuracy"),
        "val_common_non_overlap_macro_f1": get_subset_metric(val_subset_summaries, "common_non_overlap", "macro_f1"),
        "test_metric_protocol": "robust_evidence_budget" if use_robust_primary else "standard_unperturbed",
        "test_accuracy": float(test_eval["summary"]["accuracy"]),
        "test_macro_f1": float(test_eval["summary"]["macro_f1"]),
        "test_weighted_f1": float(test_eval["summary"]["weighted_f1"]),
        "test_standard_accuracy": float(standard_test_eval["summary"]["accuracy"]),
        "test_standard_macro_f1": float(standard_test_eval["summary"]["macro_f1"]),
        "test_standard_weighted_f1": float(standard_test_eval["summary"]["weighted_f1"]),
        "test_robust_accuracy": (
            float(robust_test_eval["summary"]["accuracy"]) if robust_test_eval is not None else None
        ),
        "test_robust_macro_f1": (
            float(robust_test_eval["summary"]["macro_f1"]) if robust_test_eval is not None else None
        ),
        "test_robust_weighted_f1": (
            float(robust_test_eval["summary"]["weighted_f1"]) if robust_test_eval is not None else None
        ),
        "test_full_accuracy": float(full_test_eval["summary"]["accuracy"]),
        "test_full_macro_f1": float(full_test_eval["summary"]["macro_f1"]),
        "test_full_weighted_f1": float(full_test_eval["summary"]["weighted_f1"]),
        "test_history_zero_accuracy": (
            float(temporal_ablation_summaries.get("history_zero", {}).get("accuracy"))
            if isinstance(temporal_ablation_summaries.get("history_zero"), dict)
            and "accuracy" in temporal_ablation_summaries.get("history_zero", {})
            else None
        ),
        "test_history_zero_macro_f1": (
            float(temporal_ablation_summaries.get("history_zero", {}).get("macro_f1"))
            if isinstance(temporal_ablation_summaries.get("history_zero"), dict)
            and "macro_f1" in temporal_ablation_summaries.get("history_zero", {})
            else None
        ),
        "test_repeat_last_accuracy": (
            float(temporal_ablation_summaries.get("repeat_last", {}).get("accuracy"))
            if isinstance(temporal_ablation_summaries.get("repeat_last"), dict)
            and "accuracy" in temporal_ablation_summaries.get("repeat_last", {})
            else None
        ),
        "test_repeat_last_macro_f1": (
            float(temporal_ablation_summaries.get("repeat_last", {}).get("macro_f1"))
            if isinstance(temporal_ablation_summaries.get("repeat_last"), dict)
            and "macro_f1" in temporal_ablation_summaries.get("repeat_last", {})
            else None
        ),
        "test_prefix_reverse_accuracy": (
            float(temporal_ablation_summaries.get("prefix_reverse", {}).get("accuracy"))
            if isinstance(temporal_ablation_summaries.get("prefix_reverse"), dict)
            and "accuracy" in temporal_ablation_summaries.get("prefix_reverse", {})
            else None
        ),
        "test_prefix_reverse_macro_f1": (
            float(temporal_ablation_summaries.get("prefix_reverse", {}).get("macro_f1"))
            if isinstance(temporal_ablation_summaries.get("prefix_reverse"), dict)
            and "macro_f1" in temporal_ablation_summaries.get("prefix_reverse", {})
            else None
        ),
        "test_common_accuracy": get_subset_metric(test_subset_summaries, "common_population", "accuracy"),
        "test_common_macro_f1": get_subset_metric(test_subset_summaries, "common_population", "macro_f1"),
        "test_non_overlap_accuracy": get_subset_metric(test_subset_summaries, "non_overlap", "accuracy"),
        "test_non_overlap_macro_f1": get_subset_metric(test_subset_summaries, "non_overlap", "macro_f1"),
        "test_common_non_overlap_accuracy": get_subset_metric(test_subset_summaries, "common_non_overlap", "accuracy"),
        "test_common_non_overlap_macro_f1": get_subset_metric(test_subset_summaries, "common_non_overlap", "macro_f1"),
        "test_loss": float(test_eval["summary"]["loss"]),
    }


# ============================================================
# Revised-pipeline handoff, matched endpoints and result identity
# ============================================================

SOURCE_CONTRACT: Dict[str, Any] = {}
ACTIVE_RUN: Dict[str, Any] = {}
COMMON_ENDPOINTS: Dict[str, np.ndarray] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def require_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required revised-pipeline record missing: {path}")
    data = load_json_if_exists(path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Empty/invalid JSON record: {path}")
    return data


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


def load_source_contract(extraction: Dict[str, Any], index: Dict[str, Any]) -> Dict[str, Any]:
    """Authenticate the completed artifacts actually consumed, before creating outputs."""
    root8 = EXPECTED_FINETUNED_MODEL_DIR.parent
    roots = {"step8": root8, "step9": EMBEDDING_DIR, "step10": SEQUENCE_DIR}
    runs = {key: require_json(root / "run_manifest.json") for key, root in roots.items()}
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
        raise ValueError("Step 11 requires the complete stride-1 Step 10 sequence/metadata protocol.")
    observability_audit = source10.get("observability_audit", {})
    validate_observability_source_contract(observability_audit, source9, r8, dataset_id)
    if index.get("observability_audit") != observability_audit:
        raise ValueError("Step 10 index and manifest observable-effect audit contracts differ.")
    names10 = ["sequence_build_report_index.json"]
    for length in SEQUENCE_LENGTHS:
        names10 += [f"sequence_build_report_L{length}.json"]
        for split in SPLITS:
            names10 += [sequence_npz_path(split, length).name, sequence_metadata_path(split, length).name,
                        sequence_observable_effects_path(split, length).name]
    hashes10 = {name: r10.get("artifacts", {}).get(name) for name in names10}
    verify_artifacts(SEQUENCE_DIR, hashes10)
    for length in SEQUENCE_LENGTHS:
        report = require_json(SEQUENCE_DIR / f"sequence_build_report_L{length}.json")
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


def validate_all_sequence_inputs(embedding_splits: Dict[str, Dict[str, np.ndarray]],
                                 extraction: Dict[str, Any], index: Dict[str, Any]) -> Dict[str, Any]:
    global COMMON_ENDPOINTS
    COMMON_ENDPOINTS = {}
    episode_sets, audit = {}, {}
    common_report = index.get("common_endpoint_report", {})
    if common_report.get("reference_length") != COMMON_EVALUATION_REFERENCE_LENGTH:
        raise ValueError("Missing/incorrect Step 10 common endpoint reference length.")
    for split in SPLITS:
        windows = read_csv_robust(EMBEDDING_DIR / f"{split}_window_metadata.csv")
        required = {"episode_id", "episode_type", "scenario_id", "episode_step", "window_id", "timestamp",
                    "label", "label_id", "embedding_row_index"}
        if not required.issubset(windows.columns) or windows[list(required)].isna().any().any():
            raise ValueError(f"{split}: missing window metadata.")
        emb = embedding_splits[split]
        if len(windows) != len(emb["embeddings"]) or not np.array_equal(integer_array(windows["embedding_row_index"], "embedding_row_index"), emb["row_indices"]):
            raise ValueError(f"{split}: window metadata/embedding row mismatch.")
        labels = windows["label"].map(extraction["label_to_id"])
        if labels.isna().any() or not np.array_equal(integer_array(windows["label_id"], "label_id"), emb["label_ids"]) or not np.array_equal(labels.to_numpy(), emb["label_ids"]):
            raise ValueError(f"{split}: window metadata/embedding label mismatch.")
        for col in ("episode_step", "window_id"):
            integer_array(windows[col], col)
        timestamp = pd.to_datetime(windows["timestamp"], errors="raise", utc=True)
        for _, group in windows.groupby("episode_id", sort=False):
            ordered = group.sort_values(["episode_step", "window_id"], kind="mergesort")
            if (not np.all(np.diff(ordered["episode_step"]) == 1)
                    or not np.all(np.diff(ordered["window_id"]) == 1)
                    or not timestamp.loc[ordered.index].diff().dropna().dt.total_seconds().eq(1).all()):
                raise ValueError(f"{split}: nonconsecutive window timeline.")
        episode_sets[split] = set(windows["episode_id"].astype(str))
        summary = extraction["splits"][split]
        if (summary.get("rows") != len(windows) or summary.get("embedding_shape") != list(emb["embeddings"].shape)
                or extraction["split_temporal_summaries"][split]["episode_count"] != len(episode_sets[split])
                or windows["label"].value_counts().reindex(extraction["label_order"], fill_value=0).to_dict() != summary["label_distribution"]):
            raise ValueError(f"{split}: row/episode/class counts differ from completed Step 9.")
        endpoints, counts = {}, {}
        for length in SEQUENCE_LENGTHS:
            arrays = load_sequence_split(split, length, emb, emb["embeddings"].shape[1])
            meta = load_sequence_metadata_if_exists(split, length)
            validate_sequence_alignment(split, length, arrays, meta, windows)
            # Validate the separate audit files before training; their values do
            # not enter sequence arrays, loaders, model inputs or selection.
            load_sequence_observable_effects(split, length, meta)
            report = require_json(SEQUENCE_DIR / f"sequence_build_report_L{length}.json")["splits"][split]
            ends = arrays["end_row_indices"]
            if (report["num_sequences"] != len(ends) or report["num_episodes"] != len(episode_sets[split])
                    or report["end_row_indices_sha256"] != hashlib.sha256(ends.astype("<i8", copy=False).tobytes()).hexdigest()):
                raise ValueError(f"{split}/L{length}: sequence counts/endpoint hash differ from Step 10.")
            endpoints[length], counts[str(length)] = np.sort(ends), len(ends)
        common = endpoints[SEQUENCE_LENGTHS[0]]
        for length in SEQUENCE_LENGTHS[1:]:
            common = np.intersect1d(common, endpoints[length])
        recorded = common_report.get("splits", {}).get(split, {})
        if (not len(common) or not np.array_equal(common, endpoints[COMMON_EVALUATION_REFERENCE_LENGTH])
                or recorded.get("common_endpoint_count") != len(common)
                or recorded.get("sorted_end_row_indices_sha256") != endpoint_digest(common)
                or recorded.get("available_sequences_by_L") != counts):
            raise ValueError(f"{split}: actual common endpoints differ from Step 10.")
        COMMON_ENDPOINTS[split] = common
        audit[split] = {"windows": len(windows), "episodes": len(episode_sets[split]),
                        "sequence_counts": counts, "common_endpoint_count": len(common),
                        "sorted_end_row_indices_sha256": endpoint_digest(common)}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            if episode_sets[left] & episode_sets[right]:
                raise ValueError(f"Episode leakage across {left}/{right}.")
    return audit


def verified_common_mask(meta: pd.DataFrame, split_name: str) -> np.ndarray:
    split = split_name.split("_", 1)[0]
    if split not in COMMON_ENDPOINTS or "sequence_end_row" not in meta:
        raise ValueError(f"{split_name}: verified common endpoints are required.")
    return np.isin(integer_array(meta["sequence_end_row"], "sequence_end_row"), COMMON_ENDPOINTS[split])


def result_provenance() -> Dict[str, Any]:
    return {key: SOURCE_CONTRACT[key] for key in ("dataset_id", "step8_run_id", "step9_run_id", "step10_run_id")} | {"step11_run_id": ACTIVE_RUN["run_id"]}


def prepare_run(selected_lengths: List[int], device: torch.device) -> bool:
    global ACTIVE_RUN
    if (SEQUENCE_FIT_SUBSET != "full" or PRIMARY_TEST_SUBSET != "full"
            or PRIMARY_SELECTION_METRIC != "val_macro_f1" or USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS):
        raise ValueError("Revision clean protocol requires full fitting/validation, val macro-F1 selection and unperturbed full testing.")
    # Include every declared JSON-compatible uppercase setting, including augmentation tables.
    config = {}
    for name, value in globals().items():
        if name.isupper() and name not in {"SOURCE_CONTRACT", "ACTIVE_RUN", "COMMON_ENDPOINTS"}:
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):
                continue
            config[name] = value
    identity = {"source_contract": SOURCE_CONTRACT, "script_sha256": sha256_file(Path(__file__)),
        "settings": config, "selected_sequence_lengths": selected_lengths,
        "source_embedding_dir": str(EMBEDDING_DIR.resolve()), "source_sequence_dir": str(SEQUENCE_DIR.resolve()),
        "output_root": str(OUTPUT_ROOT.resolve()),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": np.__version__, "pandas": pd.__version__, "device": str(device),
                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "torch_num_threads": torch.get_num_threads(),
                    "torch_num_interop_threads": torch.get_num_interop_threads(),
                    "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.machine()},
        "seed_policy": "Seed once before the ordered selected-L loop; same as original Step 11. No partial-run skipping.",
        "comparison_policy": "val/test_common_* on the verified Step 10 endpoint intersection; all-available metrics separate."}
    run_id = object_hash(identity)
    path = OUTPUT_ROOT / "run_manifest.json"
    if OUTPUT_ROOT.exists():
        if not path.is_file():
            raise FileExistsError(f"Output exists without a revised Step 11 completion record: {OUTPUT_ROOT}. Use a new --output-root.")
        previous = require_json(path)
        if previous.get("run_id") != run_id:
            raise ValueError("Output belongs to different data/code/settings/runtime/length order. Use a new --output-root.")
        if previous.get("status") != "COMPLETE":
            raise ValueError("This run is incomplete. It is preserved; use a new --output-root to restart all selected lengths. Mid-training resume is not implemented.")
        verify_artifacts(OUTPUT_ROOT, previous.get("artifacts", {}))
        ACTIVE_RUN = previous
        return False
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    ACTIVE_RUN = {**identity, "run_id": run_id, "dataset_id": SOURCE_CONTRACT["dataset_id"], "status": "RUNNING"}
    save_json(path, ACTIVE_RUN)
    return True


def finish_run() -> None:
    verify_artifacts(EXPECTED_FINETUNED_MODEL_DIR.parent, SOURCE_CONTRACT["step8_model_sha256"])
    verify_artifacts(EMBEDDING_DIR, SOURCE_CONTRACT["step9_artifacts"])
    verify_artifacts(SEQUENCE_DIR, SOURCE_CONTRACT["step10_artifacts"])
    for key, root in (("step8", EXPECTED_FINETUNED_MODEL_DIR.parent), ("step9", EMBEDDING_DIR), ("step10", SEQUENCE_DIR)):
        if sha256_file(root / "run_manifest.json") != SOURCE_CONTRACT["manifest_sha256"][key]:
            raise ValueError(f"{key} completion record changed during training.")
    artifacts = {p.relative_to(OUTPUT_ROOT).as_posix(): sha256_file(p) for p in OUTPUT_ROOT.rglob("*")
                 if p.is_file() and p.name not in {"run_manifest.json", "11_TST_REVIEW.zip"}}
    ACTIVE_RUN.update(status="COMPLETE", artifacts=artifacts)
    save_json(OUTPUT_ROOT / "run_manifest.json", ACTIVE_RUN)


def write_review_zip() -> Path:
    output = OUTPUT_ROOT / "11_TST_REVIEW.zip"
    with tempfile.NamedTemporaryFile(dir=OUTPUT_ROOT, suffix=".zip.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(Path(__file__), "source/" + Path(__file__).name)
            names = ["run_manifest.json"] + list(ACTIVE_RUN.get("artifacts", {}))
            for name in names:
                if Path(name).suffix in {".json", ".csv", ".png"}:
                    archive.write(OUTPUT_ROOT / name, name)
            for step, root, report in (("step9", EMBEDDING_DIR, "embedding_extraction_report.json"),
                                      ("step10", SEQUENCE_DIR, "sequence_build_report_index.json")):
                for name in ("run_manifest.json", report):
                    archive.write(root / name, f"{step}/{name}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


# ============================================================
# 11. Main
# ============================================================


def main() -> None:
    global OUTPUT_ROOT, USE_MISSING_AWARE_TRAINING, DEFAULT_RANDOM_SEED, SOURCE_CONTRACT

    args = parse_args()
    DEFAULT_RANDOM_SEED = args.seed

    if args.missing_aware:
        USE_MISSING_AWARE_TRAINING = True
    elif args.no_missing_aware:
        USE_MISSING_AWARE_TRAINING = False

    if args.output_root:
        OUTPUT_ROOT = resolve_cli_path(args.output_root)

    selected_sequence_lengths = resolve_selected_sequence_lengths(args)
    selected_temporal_context_seconds = [
        length * EXPECTED_OBSERVABLE_WINDOW_SECONDS
        for length in selected_sequence_lengths
    ]

    set_all_seeds(DEFAULT_RANDOM_SEED)
    OUTPUT_ROOT = OUTPUT_ROOT.resolve()
    for source in (EMBEDDING_DIR, SEQUENCE_DIR, EXPECTED_FINETUNED_MODEL_DIR.parent):
        source = source.resolve()
        if OUTPUT_ROOT.is_relative_to(source) or source.is_relative_to(OUTPUT_ROOT):
            raise ValueError("Step 11 output must be separate from Step 8/9/10 inputs.")

    print("=" * 78)
    print("Step 11: Train and evaluate selected TST classifiers (missing-aware optional)")
    print("=" * 78)
    print(f"Base directory: {BASE_DIR.resolve()}")
    print(f"Embedding directory: {EMBEDDING_DIR.resolve()}")
    print(f"Sequence directory: {SEQUENCE_DIR.resolve()}")
    print(f"Output root: {OUTPUT_ROOT.resolve()}")
    print(f"Available sequence lengths: {SEQUENCE_LENGTHS}")
    print(f"Selected sequence lengths: {selected_sequence_lengths}")
    print(f"Selected temporal contexts: {selected_temporal_context_seconds} seconds")
    print("Training regularization:")
    print(f"  USE_SHORTCUT_RESISTANT_TRAINING: {USE_SHORTCUT_RESISTANT_TRAINING}")
    print(f"  LABEL_SMOOTHING: {LABEL_SMOOTHING}")
    print(f"  TRAIN_GAUSSIAN_NOISE_STD: {TRAIN_GAUSSIAN_NOISE_STD}")
    print(f"  TRAIN_HISTORY_TOKEN_DROPOUT_PROB: {TRAIN_HISTORY_TOKEN_DROPOUT_PROB}")
    print(f"  TRAIN_LAST_WINDOW_REPLACEMENT_PROB: {TRAIN_LAST_WINDOW_REPLACEMENT_PROB}")
    print(f"  TRAIN_LAST_WINDOW_REPLACEMENT: {TRAIN_LAST_WINDOW_REPLACEMENT}")
    print(f"  USE_LENGTH_AWARE_REGULARIZATION: {USE_LENGTH_AWARE_REGULARIZATION}")
    if USE_LENGTH_AWARE_REGULARIZATION:
        for length in SEQUENCE_LENGTHS:
            params = get_length_aware_regularization_params(length)
            print(
                f"    L={length}: noise={params['gaussian_noise_std']}, "
                f"history_dropout={params['history_token_dropout_prob']}, "
                f"last_replacement={params['last_window_replacement_prob']}"
            )

    print("Missing-aware training augmentation:")
    print(f"  USE_MISSING_AWARE_TRAINING: {USE_MISSING_AWARE_TRAINING}")
    print(f"  MISSING_AWARE_REPLACEMENT: {MISSING_AWARE_REPLACEMENT}")
    for length in SEQUENCE_LENGTHS:
        params = get_missing_aware_regularization_params(length)
        print(
            f"    L={length}: noise={params['gaussian_noise_std']}, "
            f"seq_dim_dropout={params['sequence_dim_dropout_prob']}, "
            f"window_dim_dropout={params['window_dim_dropout_prob']}, "
            f"token_replacement={params['token_replacement_prob']}, "
            f"last_token_replacement={params['last_token_replacement_prob']}"
        )

    print("Robust evidence-budget evaluation:")
    print(f"  RUN_ROBUST_EVIDENCE_BUDGET_EVAL: {RUN_ROBUST_EVIDENCE_BUDGET_EVAL}")
    print(f"  USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS: {USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS}")
    print(f"  ROBUST_PRIMARY_MIN_SEQUENCE_LENGTH: {ROBUST_PRIMARY_MIN_SEQUENCE_LENGTH}")
    print(f"  ROBUST_MC_REPEATS: {ROBUST_MC_REPEATS}")
    for length in SEQUENCE_LENGTHS:
        print(
            f"    L={length}: keep_windows={ROBUST_KEEP_WINDOWS_BY_LENGTH.get(length)}, "
            f"last_keep_prob={ROBUST_LAST_WINDOW_KEEP_PROB_BY_LENGTH.get(length)}, "
            f"dim_dropout={ROBUST_DIM_DROPOUT_PROB_BY_LENGTH.get(length)}, "
            f"noise={ROBUST_GAUSSIAN_NOISE_STD_BY_LENGTH.get(length)}"
        )

    if not EMBEDDING_DIR.exists():
        raise FileNotFoundError(f"Embedding directory not found: {EMBEDDING_DIR}")
    if not SEQUENCE_DIR.exists():
        raise FileNotFoundError(f"Sequence directory not found: {SEQUENCE_DIR}")

    extraction_report = load_json_if_exists(EMBEDDING_REPORT_PATH)
    temporal_metadata = validate_temporal_design_from_reports(extraction_report)
    sequence_report_index = validate_step10_report_index()
    SOURCE_CONTRACT = load_source_contract(extraction_report, sequence_report_index)
    label_order, label_to_id, id_to_label = build_label_maps(extraction_report)
    expected_embedding_dim = int(temporal_metadata["embedding_dim"])

    # The revised pipeline requires authenticated COMPLETE Step 8/9/10 artifacts.
    embedding_encoder_source = str(extraction_report.get("embedding_encoder_source", "UNKNOWN"))
    source_model_dir = str(
        extraction_report.get(
            "source_model_dir",
            extraction_report.get("source_embedding_dir", extraction_report.get("source_model", "UNKNOWN")),
        )
    )

    # Main-revision cross-artifact guard: the embeddings must come from the verified
    # Step 8 fine-tuned single-window checkpoint, not from a different model directory.
    expected_model_dir = str(EXPECTED_FINETUNED_MODEL_DIR.resolve())
    if source_model_dir == "UNKNOWN" or (
        os.path.normcase(os.path.normpath(source_model_dir))
        != os.path.normcase(os.path.normpath(expected_model_dir))
    ):
        raise ValueError(
            "Step 9 source_model_dir does not match the verified Step 8 fine-tuned checkpoint. "
            f"Expected {expected_model_dir!r}, found {source_model_dir!r}."
        )

    print("\n[Experiment metadata]")
    print(f"  observable_window_seconds: {temporal_metadata['observable_window_seconds']}")
    print(f"  embedding_dim: {expected_embedding_dim}")
    print(f"  embedding_encoder_source: {embedding_encoder_source}")
    print(f"  source_model_dir: {source_model_dir}")
    if embedding_encoder_source.lower() not in {"finetuned", "fine-tuned", "fine_tuned"}:
        raise ValueError(
            "Main revision TST training requires Step 9 fine-tuned DistilBERT embeddings. "
            f"Found embedding_encoder_source={embedding_encoder_source!r}."
        )
    representation_type = str(
        extraction_report.get("encoder_source_summary", {}).get("representation_type", "")
    ).strip()
    if representation_type and representation_type != "supervised_finetuned_distilbert_cls":
        raise ValueError(
            "Unexpected Step 9 representation_type for the main revision: "
            f"{representation_type!r}."
        )
    print(f"  num_classes: {len(label_order)}")
    for label in label_order:
        print(f"    {label_to_id[label]} -> {label}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if torch.cuda.is_available():
        print(f"  CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"  AMP: {USE_AMP}")

    print("\n[Load Step 9 embeddings]")
    embedding_splits: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name in SPLITS:
        embedding_splits[split_name] = load_embedding_split(split_name, expected_embedding_dim)
        emb = embedding_splits[split_name]["embeddings"]
        print(f"  {split_name}: embeddings={emb.shape}, dtype={emb.dtype}")

    input_audit = validate_all_sequence_inputs(embedding_splits, extraction_report, sequence_report_index)
    if not prepare_run(selected_sequence_lengths, device):
        print(f"[Reuse verified COMPLETE Step 11] {OUTPUT_ROOT}")
        print(f"Review ZIP: {write_review_zip()}")
        return
    save_json(OUTPUT_ROOT / "input_validation_report.json", {"provenance": result_provenance(), "splits": input_audit})
    comparison_rows: List[Dict[str, Any]] = []
    for sequence_length in selected_sequence_lengths:
        result = train_and_evaluate_for_length(
            sequence_length=sequence_length,
            embedding_splits=embedding_splits,
            extraction_report=extraction_report,
            label_order=label_order,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            expected_embedding_dim=expected_embedding_dim,
            device=device,
        )
        comparison_rows.append(result)

    comparison_df = pd.DataFrame(comparison_rows).sort_values("sequence_length").reset_index(drop=True)
    comparison_csv = OUTPUT_ROOT / "tst_overall_comparison.csv"
    comparison_json = OUTPUT_ROOT / "tst_overall_comparison.json"
    save_dataframe(comparison_csv, comparison_df)
    save_json(
        comparison_json,
        {
            "step": "11_train_tst_classifier_overall_comparison",
            "provenance": result_provenance(),
            "main_cross_length_columns": ["val_common_macro_f1", "test_common_macro_f1"],
            "common_endpoint_report": SOURCE_CONTRACT["common_endpoint_report"],
            "available_sequence_lengths": SEQUENCE_LENGTHS,
            "selected_sequence_lengths": selected_sequence_lengths,
            "selected_temporal_context_seconds": selected_temporal_context_seconds,
            "observable_window_seconds": EXPECTED_OBSERVABLE_WINDOW_SECONDS,
            "embedding_dim": expected_embedding_dim,
            "label_order": label_order,
            "label_to_id": label_to_id,
            "source_embedding_report": str(EMBEDDING_REPORT_PATH),
            "source_sequence_report_index": str(SEQUENCE_REPORT_INDEX_PATH),
            "sequence_report_index": sequence_report_index,
            "length_aware_regularization_enabled": bool(USE_LENGTH_AWARE_REGULARIZATION),
            "length_aware_regularization_table": LENGTH_AWARE_REGULARIZATION,
            "missing_aware_training_enabled": bool(USE_MISSING_AWARE_TRAINING),
            "missing_aware_replacement": MISSING_AWARE_REPLACEMENT,
            "missing_aware_regularization_table": MISSING_AWARE_REGULARIZATION,
            "primary_test_subset": PRIMARY_TEST_SUBSET,
            "fit_subset_for_train_and_validation": SEQUENCE_FIT_SUBSET,
            "robust_evidence_budget_eval_enabled": bool(RUN_ROBUST_EVIDENCE_BUDGET_EVAL),
            "use_robust_eval_as_primary_for_long_contexts": bool(USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS),
            "robust_primary_min_sequence_length": int(ROBUST_PRIMARY_MIN_SEQUENCE_LENGTH),
            "robust_mc_repeats": int(ROBUST_MC_REPEATS),
            "robust_keep_windows_by_length": ROBUST_KEEP_WINDOWS_BY_LENGTH,
            "robust_last_window_keep_prob_by_length": ROBUST_LAST_WINDOW_KEEP_PROB_BY_LENGTH,
            "robust_dim_dropout_prob_by_length": ROBUST_DIM_DROPOUT_PROB_BY_LENGTH,
            "robust_gaussian_noise_std_by_length": ROBUST_GAUSSIAN_NOISE_STD_BY_LENGTH,
            "comparison": comparison_rows,
        },
    )

    common_columns = ["sequence_length", "temporal_context_seconds", "best_epoch", "val_common_count",
                      "val_common_accuracy", "val_common_macro_f1", "test_common_count", "test_common_accuracy",
                      "test_common_macro_f1", "dataset_id", "step11_run_id"]
    save_dataframe(OUTPUT_ROOT / "tst_common_endpoint_comparison.csv", comparison_df[common_columns])
    finish_run()
    print("\n" + "=" * 78)
    print("TST training and evaluation completed (compare L using common-endpoint metrics)")
    print("=" * 78)
    display_columns = [
        "sequence_length",
        "temporal_context_seconds",
        "best_epoch",
        "val_accuracy",
        "val_macro_f1",
        "test_metric_protocol",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "val_common_count",
        "val_common_macro_f1",
        "test_common_count",
        "test_common_macro_f1",
        "test_standard_accuracy",
        "test_standard_macro_f1",
        "test_robust_accuracy",
        "test_robust_macro_f1",
        "test_non_overlap_accuracy",
        "test_non_overlap_macro_f1",
        "test_full_accuracy",
        "test_full_macro_f1",
        "test_history_zero_macro_f1",
        "test_repeat_last_macro_f1",
        "test_prefix_reverse_macro_f1",
    ]
    display_columns = [col for col in display_columns if col in comparison_df.columns]
    print(comparison_df[display_columns].to_string(index=False))
    print(f"Review ZIP: {write_review_zip()}")
    print("\nSaved artifacts:")
    print(f"- Overall comparison CSV: {comparison_csv.resolve()}")
    print(f"- Overall comparison JSON: {comparison_json.resolve()}")
    print(f"- Per-L outputs: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[ERROR] TST training failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
