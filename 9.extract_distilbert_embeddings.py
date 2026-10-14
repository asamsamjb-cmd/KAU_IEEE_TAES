# 9. extract_distilbert_embeddings.py
# DistilBERT 1-second window-level embedding extractor for TST temporal learning.
#
# Pipeline position:
#   Step 8 trained a DistilBERT single-window 6-class baseline using
#   schema-preserving serialization of the 22 observable fields.
#   In the revised experiment, one row is one 1-second observable window,
#   and Step 10 will build TST contexts with L={1,2,3,5,10}, corresponding
#   to {1,2,3,5,10} seconds.
#
# This script performs:
#   1) Load train/val/test observable CSVs from Step 7.
#   2) Load the trained DistilBERT checkpoint from Step 8.
#   3) Reconstruct the exact observable-only serialization rule.
#   4) Extract one [CLS] embedding e_t per observation window.
#   5) Save split-specific embedding NPZ files and metadata CSV files.
#
# Critical rules:
#   - DistilBERT text input must be generated from OBSERVABLE_FIELDS only.
#   - Metadata is preserved only for TST sequence construction.
#   - Do not use scenario_id, episode_id, episode_step, window_id, timestamp,
#     label, split, raw fields, attack phase, severity, or other leakage fields
#     inside the serialized DistilBERT input text.
#   - Do not train or update DistilBERT in this script.
#   - Do not build TST sequences in this script; Step 10 handles sequence building.

from __future__ import annotations

import argparse
import hashlib
import platform
import tempfile
import zipfile
import transformers
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification, DistilBertModel, set_seed

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# ============================================================
# 0. Basic settings
# ============================================================

DEFAULT_RANDOM_SEED = 20260610

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "shared_nominal_v2"
SPLIT_DIR = DATASET_DIR / "splits"

DISTILBERT_BASELINE_DIR = BASE_DIR / "outputs" / "shared_nominal_v2" / "distilbert_window_baseline"
DISTILBERT_MODEL_DIR = DISTILBERT_BASELINE_DIR / "trained_distilbert_model"
DISTILBERT_LABEL_MAPPING_PATH = DISTILBERT_MODEL_DIR / "label_mapping.json"
STEP7_LABEL_MAPPING_PATH = SPLIT_DIR / "label_mapping.json"

OUTPUT_DIR = BASE_DIR / "outputs" / "shared_nominal_v2" / "tst_embeddings_finetuned_main"
REPORT_PATH = OUTPUT_DIR / "embedding_extraction_report.json"

TRAIN_INPUT_PATH = SPLIT_DIR / "train_observable.csv"
VAL_INPUT_PATH = SPLIT_DIR / "val_observable.csv"
TEST_INPUT_PATH = SPLIT_DIR / "test_observable.csv"

# Revised temporal design. These values must match Step 7 and Step 8 artifacts.
EXPECTED_OBSERVABLE_WINDOW_SECONDS = 1
TST_SEQUENCE_LENGTHS = [1, 2, 3, 5, 10]
TST_TEMPORAL_CONTEXT_SECONDS = [
    length * EXPECTED_OBSERVABLE_WINDOW_SECONDS for length in TST_SEQUENCE_LENGTHS
]
EXPECTED_ROWS_PER_CLASS = 50000
EXPECTED_TOTAL_ROWS = EXPECTED_ROWS_PER_CLASS * 6

# Extraction defaults. Increase BATCH_SIZE if GPU memory is sufficient.
BATCH_SIZE = 64
NUM_WORKERS = 0  # Keep 0 for Windows compatibility.
SAVE_COMPRESSED_NPZ = True
EMBEDDING_DTYPE = "float32"  # Recommended for stable downstream TST training.
USE_FP16_INFERENCE = False  # main revision: deterministic/stable FP32 embedding extraction by default

# The default is the Step 8 validation-selected, supervised encoder.
# Optional base mode is a separately identified pretrained-encoder comparison.
# Neither mode changes weights during extraction; neither is chosen by test scores.
DEFAULT_EMBEDDING_ENCODER_SOURCE = "finetuned"
EMBEDDING_ENCODER_SOURCE = DEFAULT_EMBEDDING_ENCODER_SOURCE
BASE_ENCODER_NAME = "distilbert-base-uncased"
SERIALIZER_VERSION = "observable22_fixed_order_is_v1"
SAVE_SERIALIZATION_AUDIT_SAMPLES = True
ACTIVE_RUN: Dict[str, Any] | None = None
SOURCE_CONTRACT: Dict[str, Any] | None = None

# Exact shared schema; model artifacts are checked against it, never trusted to redefine it.
DEFAULT_METADATA_FIELDS: List[str] = [
    "scenario_id",
    "episode_id",
    "episode_type",
    "episode_step",
    "window_id",
    "timestamp",
    "label",
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

DEFAULT_LABEL_ORDER: List[str] = [
    "NORMAL",
    "RF_LINK_JAMMING",
    "GNSS_SPOOFING",
    "UNAUTHORIZED_TELECOMMAND_INJECTION",
    "TELECOMMAND_REPLAY",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE",
]


# ============================================================
# 1. Reproducibility and IO utilities
# ============================================================

def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}\n"
            f"Run Step 7 split generation first, and confirm dataset/splits exists."
        )
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}\n"
            f"Run Step 8 DistilBERT baseline first and confirm trained_distilbert_model/label_mapping.json exists."
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


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


def progress_bar(iterable: Iterable[Any], total: int | None, desc: str) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DistilBERT [CLS] embeddings for TST temporal learning. "
            "The default encoder source is the Step 8 fine-tuned DistilBERT checkpoint for the main revision experiment."
        )
    )
    parser.add_argument(
        "--encoder-source",
        choices=["finetuned", "base"],
        default=DEFAULT_EMBEDDING_ENCODER_SOURCE,
        help=(
            "DistilBERT encoder source for embedding extraction. Use 'finetuned' for the main revision experiment. Use 'base' for a separate pretrained-encoder comparison."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional embedding output directory. Default is outputs/shared_nominal_v2/tst_embeddings_finetuned_main, "
            "to be consumed by the revised Step 10."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Embedding extraction batch size.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable CUDA FP16 autocast during embedding extraction.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--step8-dir", type=Path, default=DISTILBERT_BASELINE_DIR)
    return parser.parse_args()


def apply_runtime_args(args: argparse.Namespace) -> None:
    global EMBEDDING_ENCODER_SOURCE, OUTPUT_DIR, REPORT_PATH, BATCH_SIZE, USE_FP16_INFERENCE
    global DATASET_DIR, SPLIT_DIR, DISTILBERT_BASELINE_DIR, DISTILBERT_MODEL_DIR
    global DISTILBERT_LABEL_MAPPING_PATH, STEP7_LABEL_MAPPING_PATH
    EMBEDDING_ENCODER_SOURCE = str(args.encoder_source)
    def resolve(value):
        # Explicit relative CLI paths follow the working directory, as in Steps 7/8.
        return Path(value).resolve()
    DATASET_DIR = resolve(args.dataset_dir)
    SPLIT_DIR = DATASET_DIR / "splits"
    DISTILBERT_BASELINE_DIR = resolve(args.step8_dir)
    DISTILBERT_MODEL_DIR = DISTILBERT_BASELINE_DIR / "trained_distilbert_model"
    DISTILBERT_LABEL_MAPPING_PATH = DISTILBERT_MODEL_DIR / "label_mapping.json"
    STEP7_LABEL_MAPPING_PATH = SPLIT_DIR / "label_mapping.json"
    for split in ("train", "val", "test"):
        globals()[f"{split.upper()}_INPUT_PATH"] = SPLIT_DIR / f"{split}_observable.csv"
    default_name = "tst_embeddings_finetuned_main" if EMBEDDING_ENCODER_SOURCE == "finetuned" else "tst_embeddings_base_ablation"
    OUTPUT_DIR = resolve(args.output_dir) if args.output_dir else BASE_DIR / "outputs" / "shared_nominal_v2" / default_name
    OUTPUT_DIR = OUTPUT_DIR.resolve()
    for source in (DATASET_DIR, DISTILBERT_BASELINE_DIR):
        if OUTPUT_DIR == source or OUTPUT_DIR.is_relative_to(source) or source.is_relative_to(OUTPUT_DIR):
            raise ValueError("Embedding output must be separate from the Step 7/8 source directories.")
    REPORT_PATH = OUTPUT_DIR / "embedding_extraction_report.json"
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    BATCH_SIZE = int(args.batch_size)
    if args.no_fp16: USE_FP16_INFERENCE = False


def has_tokenizer_files(model_dir: Path) -> bool:
    return any(
        (model_dir / name).exists()
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
        ]
    )


def has_model_weight_files(model_dir: Path) -> bool:
    return any(
        (model_dir / name).exists()
        for name in [
            "model.safetensors",
            "pytorch_model.bin",
        ]
    )


def validate_finetuned_checkpoint_artifacts(model_dir: Path) -> Dict[str, Any]:
    missing: List[str] = []
    if not model_dir.exists():
        missing.append("trained model directory")
    if not (model_dir / "config.json").exists():
        missing.append("config.json")
    if not has_model_weight_files(model_dir):
        missing.append("model weights: model.safetensors or pytorch_model.bin")
    if not (model_dir / "label_mapping.json").exists():
        missing.append("label_mapping.json")

    if not (model_dir / "tokenizer_config.json").is_file() or not any(
            (model_dir / name).is_file() for name in ("tokenizer.json", "vocab.txt")):
        missing.append("saved tokenizer vocabulary/configuration (no fallback)")
    if missing:
        raise FileNotFoundError(
            "Step 8 fine-tuned DistilBERT checkpoint is incomplete. Missing: "
            + ", ".join(missing)
            + f"\nChecked directory: {model_dir}\n"
            + "Run Step 8 DistilBERT window baseline first, then rerun this Step 9 script."
        )

    return {
        "model_dir_exists": True,
        "config_json_exists": True,
        "model_weight_files_found": [
            name for name in ["model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack"]
            if (model_dir / name).exists()
        ],
        "tokenizer_files_found": [
            name for name in ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]
            if (model_dir / name).exists()
        ],
    }


# ============================================================
# 2. Configuration loader from Step 8 artifacts
# ============================================================

def validate_label_mapping(label_order: List[str], label_to_id: Dict[str, int]) -> None:
    expected = dict(zip(DEFAULT_LABEL_ORDER, range(len(DEFAULT_LABEL_ORDER))))
    if label_order != DEFAULT_LABEL_ORDER or label_to_id != expected:
        raise ValueError("The canonical six-class order and IDs 0..5 are required.")
    if any(type(value) is not int for value in label_to_id.values()):
        raise ValueError("Label IDs must be integers.")


def normalize_int_list(values: Any, name: str) -> List[int]:
    if values is None:
        raise ValueError(f"{name} is missing from label_mapping.json.")
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list, got {type(values).__name__}.")
    try:
        return [int(v) for v in values]
    except Exception as exc:
        raise ValueError(f"{name} must contain integer-like values: {values}") from exc


def load_extraction_config(model_dir: Path) -> Dict[str, Any]:
    """Load Step 8 model metadata and enforce the revised 1-second TST design."""
    data = load_json(model_dir / "label_mapping.json")

    required = ("label_order", "label_to_id", "metadata_fields", "observable_fields", "max_length",
                "model_name", "dataset_id", "serializer_version", "encoder_source")
    if any(key not in data for key in required):
        raise ValueError("Revised Step 8 label mapping with dataset/serializer identity is required.")
    label_order, label_to_id = data["label_order"], data["label_to_id"]
    validate_label_mapping(label_order, label_to_id)
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    metadata_fields, observable_fields = data["metadata_fields"], data["observable_fields"]
    if metadata_fields != DEFAULT_METADATA_FIELDS or observable_fields != DEFAULT_OBSERVABLE_FIELDS:
        raise ValueError("Step 8 schema differs from the exact 7 metadata + 22 observable fields.")
    if data["serializer_version"] != SERIALIZER_VERSION or data["encoder_source"] != "finetuned":
        raise ValueError("Step 8 serializer/encoder identity differs from this pipeline.")
    max_length = data["max_length"]
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("Step 8 max_length must be a positive integer.")
    model_name = str(data["model_name"])

    # These keys are written by the revised Step 8 script. Requiring them prevents
    # accidental use of an old 10-second DistilBERT checkpoint.
    if "observable_window_seconds" not in data:
        raise ValueError(
            "Step 8 label_mapping.json does not contain observable_window_seconds. "
            "This usually means an old 10-second DistilBERT checkpoint is being used. "
            "Run the revised Step 8 1-second DistilBERT baseline first."
        )

    observable_window_seconds = int(data["observable_window_seconds"])
    if observable_window_seconds != EXPECTED_OBSERVABLE_WINDOW_SECONDS:
        raise ValueError(
            f"Observable window mismatch in Step 8 model: expected "
            f"{EXPECTED_OBSERVABLE_WINDOW_SECONDS}s, got {observable_window_seconds}s."
        )

    tst_sequence_lengths = normalize_int_list(data.get("tst_sequence_lengths"), "tst_sequence_lengths")
    if tst_sequence_lengths != TST_SEQUENCE_LENGTHS:
        raise ValueError(
            f"TST sequence length mismatch in Step 8 model: expected "
            f"{TST_SEQUENCE_LENGTHS}, got {tst_sequence_lengths}."
        )

    tst_temporal_context_seconds = normalize_int_list(
        data.get("tst_temporal_context_seconds"),
        "tst_temporal_context_seconds",
    )
    if tst_temporal_context_seconds != TST_TEMPORAL_CONTEXT_SECONDS:
        raise ValueError(
            f"TST temporal-context mismatch in Step 8 model: expected "
            f"{TST_TEMPORAL_CONTEXT_SECONDS}, got {tst_temporal_context_seconds}."
        )

    return {
        "dataset_id": data["dataset_id"],
        "serializer_version": data["serializer_version"],
        "label_order": label_order,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "metadata_fields": metadata_fields,
        "observable_fields": observable_fields,
        "max_length": max_length,
        "model_name": model_name,
        "observable_window_seconds": observable_window_seconds,
        "tst_sequence_lengths": tst_sequence_lengths,
        "tst_temporal_context_seconds": tst_temporal_context_seconds,
        "distilbert_input_rule": data.get(
            "distilbert_input_rule",
            "Serialize OBSERVABLE_FIELDS only. Metadata/raw/split/label columns are never serialized.",
        ),
        "distilbert_baseline_context": data.get(
            "distilbert_baseline_context",
            "single 1-second observable window; no temporal sequence is used in Step 8.",
        ),
    }


# ============================================================
# 3. Dataset validation
# ============================================================

def validate_observable_dataframe(
    df: pd.DataFrame,
    split_name: str,
    metadata_fields: List[str],
    observable_fields: List[str],
    label_order: List[str],
) -> pd.DataFrame:
    if metadata_fields != DEFAULT_METADATA_FIELDS or observable_fields != DEFAULT_OBSERVABLE_FIELDS:
        raise ValueError("Unexpected observable/metadata schema.")
    canonical_columns = metadata_fields + observable_fields
    if df.empty or df.columns.duplicated().any():
        raise ValueError(f"[{split_name}] Empty data or duplicate columns.")

    missing = [col for col in canonical_columns if col not in df.columns]
    if missing:
        raise ValueError(f"[{split_name}] Missing required columns: {missing}")

    extra = [col for col in df.columns if col not in canonical_columns]
    if extra:
        raise ValueError(
            f"[{split_name}] Unexpected extra columns found: {extra}\n"
            f"Expected exactly metadata_fields + observable_fields from Step 7."
        )

    forbidden_raw_cols = [col for col in df.columns if col.startswith("raw_")]
    if forbidden_raw_cols:
        raise ValueError(f"[{split_name}] Raw-audit columns found: {forbidden_raw_cols}")

    null_counts = df[canonical_columns].isnull().sum()
    if int(null_counts.sum()) > 0:
        raise ValueError(f"[{split_name}] Null values found: {null_counts[null_counts > 0].to_dict()}")

    unknown_labels = sorted(set(df["label"].astype(str).unique()) - set(label_order))
    if unknown_labels:
        raise ValueError(f"[{split_name}] Unknown labels found: {unknown_labels}")

    out = df[canonical_columns].copy().reset_index(drop=True)
    out["label"] = out["label"].astype(str)
    out["episode_id"] = out["episode_id"].astype(str)
    out["episode_type"] = out["episode_type"].astype(str)
    out["scenario_id"] = out["scenario_id"].astype(str)
    out["timestamp"] = out["timestamp"].astype(str)
    for field in ("episode_step", "window_id"):
        values = pd.to_numeric(out[field], errors="raise")
        if (not np.isfinite(values).all() or (values < 0).any()
                or (values >= 2**63).any() or (values % 1 != 0).any()):
            raise ValueError(f"[{split_name}] {field} must be a nonnegative int64 index.")
        out[field] = values.astype(np.int64)
    if any(out[field].astype(str).str.strip().eq("").any() for field in canonical_columns):
        raise ValueError(f"[{split_name}] Blank metadata/observable value.")
    if out.duplicated(["episode_id", "episode_step"]).any():
        raise ValueError(f"[{split_name}] Duplicate episode/step key.")
    if (out.groupby("episode_id")[["label", "scenario_id", "episode_type"]].nunique() > 1).any().any():
        raise ValueError(f"[{split_name}] Mixed identity/label within an episode.")

    return out


def validate_no_episode_overlap(split_dfs: Dict[str, pd.DataFrame]) -> None:
    split_eps = {
        split: set(df["episode_id"].astype(str).unique())
        for split, df in split_dfs.items()
    }

    names = list(split_eps.keys())
    overlaps: Dict[str, set[str]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{a}_{b}"
            overlaps[key] = split_eps[a] & split_eps[b]

    bad = {name: eps for name, eps in overlaps.items() if eps}
    if bad:
        example = {name: sorted(list(eps))[:5] for name, eps in bad.items()}
        raise ValueError(f"Episode leakage detected across splits: {example}")


def validate_temporal_consistency(
    df: pd.DataFrame,
    split_name: str,
    expected_window_seconds: int,
) -> Dict[str, Any]:
    """Validate that each episode is a 1-second ordered observable sequence."""
    required = ["episode_id", "episode_step", "window_id", "timestamp"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"[{split_name}] Missing temporal metadata columns: {missing}")

    parsed_timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if parsed_timestamps.isnull().any():
        bad_count = int(parsed_timestamps.isnull().sum())
        raise ValueError(f"[{split_name}] Found {bad_count} unparsable timestamp values.")

    work = df[["episode_id", "episode_step", "window_id"]].copy()
    work["timestamp_dt"] = parsed_timestamps

    invalid_timestamp_episodes: List[str] = []
    invalid_step_episodes: List[str] = []
    invalid_window_episodes: List[str] = []
    duplicate_step_episodes: List[str] = []
    episode_lengths: List[int] = []

    for episode_id, group in work.groupby("episode_id", sort=False):
        ordered = group.sort_values(["episode_step", "window_id"]).reset_index(drop=True)
        episode_lengths.append(int(len(ordered)))

        step_diff = ordered["episode_step"].diff().dropna().to_numpy(dtype=np.int64)
        if len(step_diff) and not np.all(step_diff == 1):
            invalid_step_episodes.append(str(episode_id))

        if ordered["episode_step"].duplicated().any():
            duplicate_step_episodes.append(str(episode_id))

        window_diff = ordered["window_id"].diff().dropna().to_numpy(dtype=np.int64)
        if len(window_diff) and not np.all(window_diff == 1):
            invalid_window_episodes.append(str(episode_id))

        ts_diff = ordered["timestamp_dt"].diff().dropna().dt.total_seconds().to_numpy(dtype=float)
        if len(ts_diff) and not np.all(ts_diff == float(expected_window_seconds)):
            invalid_timestamp_episodes.append(str(episode_id))

    if duplicate_step_episodes:
        raise ValueError(f"[{split_name}] Duplicate episode_step values. Examples={duplicate_step_episodes[:10]}")
    if invalid_step_episodes:
        raise ValueError(f"[{split_name}] Non-consecutive episode_step values. Examples={invalid_step_episodes[:10]}")
    if invalid_window_episodes:
        raise ValueError(f"[{split_name}] Non-consecutive window_id values within episodes. Examples={invalid_window_episodes[:10]}")
    if invalid_timestamp_episodes:
        raise ValueError(
            f"[{split_name}] Timestamp spacing is not {expected_window_seconds}s within episodes. "
            f"Examples={invalid_timestamp_episodes[:10]}"
        )

    lengths = np.asarray(episode_lengths, dtype=np.int64)
    return {
        "split": split_name,
        "episode_count": int(len(lengths)),
        "min_episode_length_rows": int(lengths.min()) if len(lengths) else 0,
        "mean_episode_length_rows": float(lengths.mean()) if len(lengths) else 0.0,
        "max_episode_length_rows": int(lengths.max()) if len(lengths) else 0,
        "mean_episode_duration_seconds": float(lengths.mean() * expected_window_seconds) if len(lengths) else 0.0,
        "timestamp_spacing_seconds": int(expected_window_seconds),
    }


def validate_total_label_distribution(split_dfs: Dict[str, pd.DataFrame], label_order: List[str]) -> Dict[str, Any]:
    combined = pd.concat(split_dfs.values(), axis=0, ignore_index=True)
    total_rows = int(len(combined))
    if total_rows != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Total row count mismatch: expected {EXPECTED_TOTAL_ROWS}, got {total_rows}.")

    label_counts = combined["label"].value_counts().reindex(label_order).fillna(0).astype(int)
    bad_counts = {
        str(label): int(count)
        for label, count in label_counts.items()
        if int(count) != EXPECTED_ROWS_PER_CLASS
    }
    if bad_counts:
        raise ValueError(
            f"Class row count mismatch. Expected {EXPECTED_ROWS_PER_CLASS} per class, got {bad_counts}."
        )

    return {
        "total_rows": total_rows,
        "expected_total_rows": EXPECTED_TOTAL_ROWS,
        "expected_rows_per_class": EXPECTED_ROWS_PER_CLASS,
        "label_distribution": {str(k): int(v) for k, v in label_counts.to_dict().items()},
    }


def validate_step7_temporal_metadata() -> Dict[str, Any]:
    """Cross-check Step 7 split metadata against the revised Step 9 assumptions."""
    if not STEP7_LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Step 7 label mapping not found: {STEP7_LABEL_MAPPING_PATH}. Run Step 7 first."
        )
    data = load_json(STEP7_LABEL_MAPPING_PATH)

    if "observable_window_seconds" not in data:
        raise ValueError(
            "Step 7 label_mapping.json does not include observable_window_seconds. "
            "Run the revised 1-second Step 7 builder."
        )

    step7_window = int(data["observable_window_seconds"])
    if step7_window != EXPECTED_OBSERVABLE_WINDOW_SECONDS:
        raise ValueError(
            f"Step 7 observable_window_seconds mismatch: expected "
            f"{EXPECTED_OBSERVABLE_WINDOW_SECONDS}, got {step7_window}."
        )

    step7_lengths = normalize_int_list(data.get("tst_sequence_lengths"), "Step 7 tst_sequence_lengths")
    if step7_lengths != TST_SEQUENCE_LENGTHS:
        raise ValueError(
            f"Step 7 TST sequence lengths mismatch: expected {TST_SEQUENCE_LENGTHS}, got {step7_lengths}."
        )

    step7_context = normalize_int_list(
        data.get("tst_temporal_context_seconds"),
        "Step 7 tst_temporal_context_seconds",
    )
    if step7_context != TST_TEMPORAL_CONTEXT_SECONDS:
        raise ValueError(
            f"Step 7 TST temporal contexts mismatch: expected "
            f"{TST_TEMPORAL_CONTEXT_SECONDS}, got {step7_context}."
        )

    return {
        "path": str(STEP7_LABEL_MAPPING_PATH),
        "observable_window_seconds": step7_window,
        "tst_sequence_lengths": step7_lengths,
        "tst_temporal_context_seconds": step7_context,
    }


# ============================================================
# 4. Schema-preserving serializer
# ============================================================

def normalize_value_for_text(value: Any) -> str:
    """Return a stable string representation of one observable value."""
    if pd.isna(value):
        raise ValueError("Observable value is NaN; serialization is not allowed.")
    text = str(value).strip()
    if not text:
        raise ValueError("Observable value is empty; serialization is not allowed.")
    return text


def serialize_observable_row(row: pd.Series, observable_fields: List[str]) -> str:
    """
    Serialize one observable record into a fixed-order sentence-like string.

    Critical constraints:
      - Use observable_fields only.
      - Preserve field names and canonical values.
      - Use fixed field order for reproducibility.
      - Do not include label or metadata.
    """
    parts = []
    for field in observable_fields:
        value = normalize_value_for_text(row[field])
        parts.append(f"{field} is {value}")
    return "; ".join(parts) + "."


def validate_serialized_text(
    text: str,
    metadata_fields: List[str],
    observable_fields: List[str],
) -> None:
    if observable_fields != DEFAULT_OBSERVABLE_FIELDS or metadata_fields != DEFAULT_METADATA_FIELDS:
        raise ValueError("Unexpected serialization schema.")
    clauses = text[:-1].split("; ") if text.endswith(".") else []
    if len(clauses) != len(observable_fields) or any(
            not clause.startswith(field + " is ") or not clause[len(field) + 4:].strip()
            for clause, field in zip(clauses, observable_fields)):
        raise ValueError("Serialized text must contain exactly the 22 fields in canonical order.")
    lowered = text.lower()

    forbidden_patterns = []
    for col in metadata_fields:
        col_lower = col.lower()
        forbidden_patterns.append(f"{col_lower} is ")
        forbidden_patterns.append(f"{col_lower} =")

    forbidden_patterns += [
        "raw_",
        "label_id is ",
        "label_id =",
        "split is ",
        "split =",
        "attack_phase is ",
        "attack_phase =",
        "severity is ",
        "severity =",
        "case_type is ",
        "case_type =",
        "training_role is ",
        "training_role =",
    ]

    found = [pattern for pattern in forbidden_patterns if pattern in lowered]
    if found:
        raise ValueError(f"Serialized text contains forbidden leakage pattern(s): {found}")

    missing_fields = [field for field in observable_fields if f"{field} is " not in text]
    if missing_fields:
        raise ValueError(f"Serialized text missing observable fields: {missing_fields}")

    duplicated_fields = [field for field in observable_fields if text.count(f"{field} is ") != 1]
    if duplicated_fields:
        counts = {field: text.count(f"{field} is ") for field in duplicated_fields}
        raise ValueError(f"Serialized text has non-unique observable fields: {counts}")


def add_label_id_and_text(
    df: pd.DataFrame,
    split_name: str,
    label_to_id: Dict[str, int],
    metadata_fields: List[str],
    observable_fields: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy().reset_index(drop=True)
    out["label_id"] = out["label"].map(label_to_id)

    if out["label_id"].isnull().any():
        bad_labels = sorted(out.loc[out["label_id"].isnull(), "label"].unique().tolist())
        raise ValueError(f"[{split_name}] Failed to map label_id for labels: {bad_labels}")
    out["label_id"] = out["label_id"].astype(np.int64)

    out["text"] = out.apply(lambda row: serialize_observable_row(row, observable_fields), axis=1)

    for text in out["text"].unique():
        validate_serialized_text(text, metadata_fields, observable_fields)

    metadata_df = out[metadata_fields].copy()
    metadata_df["label_id"] = out["label_id"].astype(np.int64)
    metadata_df["embedding_row_index"] = np.arange(len(out), dtype=np.int64)

    return out, metadata_df


# ============================================================
# 5. PyTorch dataset and collator
# ============================================================

class TextWindowDataset(Dataset):
    def __init__(self, texts: List[str], label_ids: np.ndarray):
        if len(texts) != len(label_ids):
            raise ValueError(f"texts and label_ids length mismatch: {len(texts)} vs {len(label_ids)}")
        self.texts = texts
        self.label_ids = label_ids.astype(np.int64)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "text": self.texts[idx],
            "label_id": int(self.label_ids[idx]),
            "row_index": int(idx),
        }


def make_collate_fn(tokenizer: Any, max_length: int) -> Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]:
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        label_ids = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
        row_indices = torch.tensor([item["row_index"] for item in batch], dtype=torch.long)

        encoded = tokenizer(
            texts,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )

        lengths = encoded["attention_mask"].sum(dim=1)
        if (lengths > max_length).any():
            raise ValueError(f"Input exceeds Step 8 max_length={max_length}; truncation is not allowed.")
        if tokenizer.unk_token_id is not None and (encoded["input_ids"] == tokenizer.unk_token_id).any():
            raise ValueError("Tokenizer [UNK] encountered; check vocabulary/serialization identity.")
        if tokenizer.cls_token_id is None or not (encoded["input_ids"][:, 0] == tokenizer.cls_token_id).all():
            raise ValueError("First token is not CLS; its hidden state cannot represent this window.")
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label_ids": label_ids,
            "row_indices": row_indices,
        }

    return collate_fn


# ============================================================
# 6. Model loading and embedding extraction
# ============================================================

def load_model_and_tokenizer(model_dir: Path, device: torch.device,
                             config: Dict[str, Any]) -> Tuple[Any, Any, int, Dict[str, Any]]:
    """Load authenticated weights; extract only the last-layer CLS state, without fitting."""
    if SOURCE_CONTRACT is None:
        raise ValueError("Step 7/8 source contract must be validated before loading a model.")
    validate_finetuned_checkpoint_artifacts(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    if tokenizer_fingerprint(tokenizer) != SOURCE_CONTRACT["tokenizer_sha256"]:
        raise ValueError("Saved tokenizer differs from the tokenizer used by Step 8.")
    run8 = SOURCE_CONTRACT["step8_run"]
    if EMBEDDING_ENCODER_SOURCE == "finetuned":
        model_source = str(model_dir)
        model, loading = DistilBertForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, output_loading_info=True)
        if (model.config.id2label != config["id_to_label"]
                or model.config.label2id != config["label_to_id"]
                or getattr(model.config, "step7_dataset_id", None) != config["dataset_id"]
                or getattr(model.config, "observable_serializer_version", None) != SERIALIZER_VERSION):
            raise ValueError("Loaded classifier schema/dataset identity differs from Step 8 metadata.")
        allowed_unexpected = ()
    elif EMBEDDING_ENCODER_SOURCE == "base":
        model_source = run8["protocol"]["model_name"]
        local_hashes = run8.get("local_initializer_sha256")
        revision = run8.get("initializer_commit")
        if local_hashes is not None:
            actual_files = {path.relative_to(Path(model_source)).as_posix()
                            for path in Path(model_source).rglob("*") if path.is_file()}
            if actual_files != set(local_hashes):
                raise ValueError("Local initializer file inventory differs from Step 8.")
            verify_artifacts(Path(model_source), local_hashes)
        else:
            if Path(model_source).is_dir():
                raise ValueError("Recorded Hub initializer is shadowed by a local directory.")
            if not revision:
                raise ValueError("Base comparison requires the exact initializer revision recorded by Step 8.")
        model, loading = DistilBertModel.from_pretrained(
            model_source, revision=revision, output_loading_info=True)
        if local_hashes is None and getattr(model.config, "_commit_hash", None) != revision:
            raise ValueError("Loaded base encoder revision differs from Step 8's recorded initializer.")
        # A base bundle may also contain an unused MLM head or classification head.
        allowed_unexpected = ("vocab_transform.", "vocab_layer_norm.", "vocab_projector.",
                              "pre_classifier.", "classifier.")
    else:
        raise ValueError("Unknown encoder source.")
    unexpected = [key for key in loading.get("unexpected_keys", [])
                  if not key.startswith(allowed_unexpected)]
    if unexpected or any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint did not load exactly: {loading}")
    if config["max_length"] > model.config.max_position_embeddings:
        raise ValueError("Step 8 max_length exceeds encoder position capacity.")
    model.to(device)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    embedding_dim = int(model.config.dim)
    summary = {"embedding_encoder_source": EMBEDDING_ENCODER_SOURCE,
               "model_source": model_source, "tokenizer_source": str(model_dir),
               "tokenizer_sha256": SOURCE_CONTRACT["tokenizer_sha256"],
               "initializer_commit": run8.get("initializer_commit"),
               "representation_type": ("supervised_finetuned_distilbert_cls" if EMBEDDING_ENCODER_SOURCE == "finetuned"
                                       else "pretrained_distilbert_cls_ablation"),
               "embedding_dim": embedding_dim, "weight_updates": False,
               "pooling": "last_hidden_state[:, 0, :]", "normalization": "none"}
    return tokenizer, model, embedding_dim, summary

def infer_cls_embeddings(
    model: DistilBertForSequenceClassification,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    use_fp16: bool,
) -> torch.Tensor:
    """Return DistilBERT [CLS] embeddings with shape [batch_size, hidden_dim]."""
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)

    encoder = model.distilbert if hasattr(model, "distilbert") else model
    with torch.inference_mode():
        if device.type == "cuda" and use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                encoder_outputs = encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        cls_embeddings = encoder_outputs.last_hidden_state[:, 0, :]

    return cls_embeddings


def save_embedding_npz(path: Path, embeddings: np.ndarray, label_ids: np.ndarray,
                       row_indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    arrays = {"embeddings": embeddings, "label_ids": label_ids, "row_indices": row_indices}
    try:
        with temporary.open("wb") as stream:
            writer = np.savez_compressed if SAVE_COMPRESSED_NPZ else np.savez
            writer(stream, **arrays)
        with np.load(temporary, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays): raise ValueError("Incomplete embedding NPZ keys.")
            for name, expected in arrays.items():
                actual = saved[name]
                if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                    raise ValueError(f"Saved NPZ differs from extracted array: {name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_extraction_outputs(
    split_name: str,
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    row_indices: np.ndarray,
    metadata_df: pd.DataFrame,
    expected_embedding_dim: int,
) -> Dict[str, Any]:
    if (embeddings.dtype != np.dtype(EMBEDDING_DTYPE) or label_ids.dtype != np.int64
            or row_indices.dtype != np.int64 or label_ids.ndim != 1 or row_indices.ndim != 1):
        raise ValueError(f"[{split_name}] Unexpected embedding/index dtype or rank.")
    if embeddings.ndim != 2:
        raise ValueError(f"[{split_name}] embeddings must be 2-D, got shape {embeddings.shape}")

    if embeddings.shape[1] != expected_embedding_dim:
        raise ValueError(
            f"[{split_name}] embedding dim mismatch: expected {expected_embedding_dim}, got {embeddings.shape[1]}"
        )
    if "embedding_row_index" not in metadata_df.columns:
        raise ValueError(f"[{split_name}] metadata_df is missing embedding_row_index.")

    metadata_row_indices = metadata_df["embedding_row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(row_indices.astype(np.int64), metadata_row_indices):
        bad_positions = np.where(row_indices.astype(np.int64) != metadata_row_indices)[0][:10].tolist()
        raise ValueError(f"[{split_name}] metadata embedding_row_index alignment failed at positions: {bad_positions}")

    n_rows = embeddings.shape[0]
    if len(label_ids) != n_rows:
        raise ValueError(f"[{split_name}] label_ids length mismatch: {len(label_ids)} vs {n_rows}")
    if len(row_indices) != n_rows:
        raise ValueError(f"[{split_name}] row_indices length mismatch: {len(row_indices)} vs {n_rows}")
    if len(metadata_df) != n_rows:
        raise ValueError(f"[{split_name}] metadata rows mismatch: {len(metadata_df)} vs {n_rows}")

    expected_row_indices = np.arange(n_rows, dtype=np.int64)
    if not np.array_equal(row_indices.astype(np.int64), expected_row_indices):
        bad_positions = np.where(row_indices.astype(np.int64) != expected_row_indices)[0][:10].tolist()
        raise ValueError(f"[{split_name}] row index alignment failed at positions: {bad_positions}")

    metadata_label_ids = metadata_df["label_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(label_ids.astype(np.int64), metadata_label_ids):
        bad_positions = np.where(label_ids.astype(np.int64) != metadata_label_ids)[0][:10].tolist()
        raise ValueError(f"[{split_name}] label alignment failed at positions: {bad_positions}")

    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())
    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"[{split_name}] invalid embeddings: nan_count={nan_count}, inf_count={inf_count}")

    return {
        "alignment_check_passed": True,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }




def extract_embeddings_for_split(
    split_name: str,
    df: pd.DataFrame,
    tokenizer: Any,
    model: DistilBertForSequenceClassification,
    device: torch.device,
    config: Dict[str, Any],
    embedding_dim: int,
) -> Dict[str, Any]:
    label_to_id: Dict[str, int] = config["label_to_id"]
    label_order: List[str] = config["label_order"]
    metadata_fields: List[str] = config["metadata_fields"]
    observable_fields: List[str] = config["observable_fields"]
    max_length: int = int(config["max_length"])

    print(f"\n[{split_name}] Build serialized observable-only inputs")
    serialized_df, metadata_df = add_label_id_and_text(
        df=df,
        split_name=split_name,
        label_to_id=label_to_id,
        metadata_fields=metadata_fields,
        observable_fields=observable_fields,
    )

    serialization_match = verify_step8_serialization(split_name, serialized_df)

    if SAVE_SERIALIZATION_AUDIT_SAMPLES:
        sample_cols = metadata_fields + ["label_id", "text"]
        sample_n = min(10, len(serialized_df))
        sample_path = OUTPUT_DIR / f"{split_name}_serialization_samples.csv"
        save_verified_csv(sample_path, serialized_df[sample_cols].head(sample_n))

    texts = serialized_df["text"].tolist()
    label_ids_source = serialized_df["label_id"].to_numpy(dtype=np.int64)

    dataset = TextWindowDataset(texts=texts, label_ids=label_ids_source)
    collate_fn = make_collate_fn(tokenizer=tokenizer, max_length=max_length)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # Allocate once; avoid holding both a batch list and its concatenated full matrix.
    embeddings = np.empty((len(df), embedding_dim), dtype=EMBEDDING_DTYPE)
    label_ids = np.empty(len(df), dtype=np.int64)
    row_indices = np.empty(len(df), dtype=np.int64)
    offset = 0
    max_tokens = 0

    print(f"[{split_name}] Extract CLS embeddings")
    model.eval()
    iterator = progress_bar(loader, total=len(loader), desc=f"Extract {split_name}")

    for batch in iterator:
        cls_embeddings = infer_cls_embeddings(
            model=model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            device=device,
            use_fp16=USE_FP16_INFERENCE,
        )

        cls_np = cls_embeddings.detach().cpu().numpy()
        if EMBEDDING_DTYPE == "float16":
            cls_np = cls_np.astype(np.float16)
        elif EMBEDDING_DTYPE == "float32":
            cls_np = cls_np.astype(np.float32)
        else:
            raise ValueError(f"Unsupported EMBEDDING_DTYPE: {EMBEDDING_DTYPE}")

        count = len(cls_np)
        batch_indices = batch["row_indices"].cpu().numpy().astype(np.int64)
        if cls_np.shape != (count, embedding_dim) or not np.isfinite(cls_np).all():
            raise ValueError(f"[{split_name}] Invalid encoder output at row {offset}.")
        if not np.array_equal(batch_indices, np.arange(offset, offset + count)):
            raise ValueError(f"[{split_name}] Batch order changed at row {offset}.")
        embeddings[offset:offset + count] = cls_np
        label_ids[offset:offset + count] = batch["label_ids"].cpu().numpy()
        row_indices[offset:offset + count] = batch_indices
        offset += count
        max_tokens = max(max_tokens, int(batch["attention_mask"].sum(dim=1).max()))
    if offset != len(df): raise ValueError(f"[{split_name}] Incomplete extraction.")

    validation_summary = validate_extraction_outputs(
        split_name=split_name,
        embeddings=embeddings,
        label_ids=label_ids,
        row_indices=row_indices,
        metadata_df=metadata_df,
        expected_embedding_dim=embedding_dim,
    )

    npz_path = OUTPUT_DIR / f"{split_name}_window_embeddings.npz"
    metadata_path = OUTPUT_DIR / f"{split_name}_window_metadata.csv"

    save_embedding_npz(
        path=npz_path,
        embeddings=embeddings,
        label_ids=label_ids,
        row_indices=row_indices,
    )
    save_verified_csv(metadata_path, metadata_df)

    label_distribution = (
        metadata_df["label"]
        .value_counts()
        .reindex(label_order)
        .fillna(0)
        .astype(int)
        .to_dict()
    )

    summary: Dict[str, Any] = {
        "split": split_name,
        "rows": int(len(metadata_df)),
        "serialization_matches_step8": serialization_match,
        "max_tokens": max_tokens,
        "truncated_rows": 0,
        "tokenizer_unk_rows": 0,
        "embedding_shape": [int(x) for x in embeddings.shape],
        "embedding_dtype": str(embeddings.dtype),
        "embedding_npz_path": str(npz_path),
        "metadata_csv_path": str(metadata_path),
        "label_distribution": {str(k): int(v) for k, v in label_distribution.items()},
        "batch_size": BATCH_SIZE,
        "max_length": max_length,
        "save_compressed_npz": SAVE_COMPRESSED_NPZ,
        **validation_summary,
    }

    print(
        f"[{split_name}] Saved embeddings: {npz_path.name} | "
        f"shape={embeddings.shape} | metadata={metadata_path.name}"
    )

    return summary


# ============================================================
# 7. Main
# ============================================================

def load_source_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    """Authenticate exactly the Step 7 inputs and completed Step 8 artifacts consumed here."""
    build_path = DATASET_DIR / "build_manifest.json"
    build = load_json(build_path)
    if build.get("status") != "complete": raise ValueError("Step 7 build is not complete.")
    if build.get("dataset_id") != object_hash({"policy": build["policy"], "sources": build["sources"]}):
        raise ValueError("Step 7 dataset identity does not match its sources/policy.")
    if build["policy"].get("nominal_substrate") != "shared_nominal_v2":
        raise ValueError("Step 7 must use the shared_nominal_v2 nominal substrate; rebuild the dataset with the revised makers.")
    if (build["policy"].get("columns") != DEFAULT_METADATA_FIELDS + DEFAULT_OBSERVABLE_FIELDS
            or build["policy"].get("label_order") != DEFAULT_LABEL_ORDER
            or build["policy"].get("window_seconds") != EXPECTED_OBSERVABLE_WINDOW_SECONDS):
        raise ValueError("Step 7 schema/labels/window duration do not match Step 9.")
    required_inputs = [f"splits/{split}_observable.csv" for split in ("train", "val", "test")]
    required_inputs += ["splits/label_mapping.json", "splits/episode_split_manifest.csv"]
    input_hashes = {name: build.get("artifacts", {}).get(name) for name in required_inputs}
    verify_artifacts(DATASET_DIR, input_hashes)
    run_path = DISTILBERT_BASELINE_DIR / "run_manifest.json"
    run8 = load_json(run_path)
    if run8.get("status") != "COMPLETE":
        raise ValueError("Step 8 is not COMPLETE. Wait for Step 8 training/evaluation to finish, then run Step 9.")
    if (run8.get("dataset_id") != build["dataset_id"] or config["dataset_id"] != build["dataset_id"]
            or run8.get("input_sha256") != input_hashes
            or run8.get("step7_build_manifest_sha256") != sha256_file(build_path)):
        raise ValueError("Step 7 inputs and the completed Step 8 model belong to different data builds.")
    if run8["protocol"].get("sample_limits") != [None, None, None]:
        raise ValueError("A subsampled Step 8 pilot is not a main-experiment encoder.")
    required8 = {name: digest for name, digest in run8.get("artifacts", {}).items()
                 if name.startswith("trained_distilbert_model/")}
    present = {p.relative_to(DISTILBERT_BASELINE_DIR).as_posix()
               for p in DISTILBERT_MODEL_DIR.rglob("*") if p.is_file()}
    if present != set(required8):
        raise ValueError("Model directory contains missing/unrecorded files relative to completed Step 8.")
    for name in ("config.json", "label_mapping.json", "training_provenance.json", "tokenizer_config.json"):
        if f"trained_distilbert_model/{name}" not in required8:
            raise ValueError(f"Step 8 did not authenticate required model artifact: {name}")
    validate_finetuned_checkpoint_artifacts(DISTILBERT_MODEL_DIR)
    for split in ("train", "val", "test"):
        name = f"serialized/{split}_serialized.csv"
        digest = run8.get("artifacts", {}).get(name)
        if not digest: raise ValueError("Step 8 must save all serialized inputs for an exact handoff.")
        required8[name] = digest
    verify_artifacts(DISTILBERT_BASELINE_DIR, required8)
    provenance = load_json(DISTILBERT_MODEL_DIR / "training_provenance.json")
    for name, expected in (("dataset_id", build["dataset_id"]), ("run_id", run8["run_id"]),
                           ("input_sha256", input_hashes), ("serializer_version", SERIALIZER_VERSION),
                           ("max_length", config["max_length"]), ("encoder_source", "finetuned"),
                           ("tokenizer_sha256", run8.get("tokenizer_sha256")),
                           ("step7_build_manifest_sha256", sha256_file(build_path))):
        if provenance.get(name) != expected: raise ValueError(f"Step 8 provenance mismatch: {name}")
    if (run8["protocol"].get("serializer_version") != SERIALIZER_VERSION
            or run8["protocol"].get("max_length") != config["max_length"]
            or not run8.get("tokenizer_sha256")):
        raise ValueError("Step 8 tokenizer/serializer/length protocol differs.")
    model_hashes = {name[len("trained_distilbert_model/"):]: digest for name,digest in required8.items()
                    if name.startswith("trained_distilbert_model/") and not name.endswith("/training_provenance.json")}
    if provenance.get("model_artifacts") != model_hashes:
        raise ValueError("Step 8 model hashes differ between completion and model provenance.")
    return {"dataset_id": build["dataset_id"], "input_sha256": input_hashes,
            "source_dataset_dir": str(DATASET_DIR.resolve()),
            "step7_build_manifest_sha256": sha256_file(build_path),
            "step8_run_id": run8["run_id"], "step8_manifest_sha256": sha256_file(run_path),
            "step8_artifacts": required8, "tokenizer_sha256": run8["tokenizer_sha256"],
            "step7_build": build, "step8_run": run8}


def validate_frames_against_build(frames: Dict[str, pd.DataFrame]) -> None:
    records = []
    for split, frame in frames.items():
        expected = SOURCE_CONTRACT["step7_build"]["counts"][split]
        if len(frame) != expected["rows"] or frame.episode_id.nunique() != expected["episodes"]:
            raise ValueError(f"{split} counts differ from the authenticated Step 7 build.")
        part = frame.groupby("episode_id").agg(label=("label", "first"), row_count=("label", "size"))
        part["split"] = split
        records.append(part.reset_index())
    actual = pd.concat(records, ignore_index=True).sort_values("episode_id").reset_index(drop=True)
    manifest = read_csv_robust(SPLIT_DIR / "episode_split_manifest.csv")
    known_ids = set(actual.episode_id.astype(str))
    manifest["episode_id"] = manifest.apply(
        lambda row: str(row.original_episode_id) if str(row.original_episode_id) in known_ids
        else str(row.global_episode_id), axis=1)
    expected = manifest[["episode_id", "label", "row_count", "split"]].sort_values("episode_id").reset_index(drop=True)
    if not actual.equals(expected): raise ValueError("Episode identity/split/count differs from Step 7.")


def verify_step8_serialization(split: str, serialized: pd.DataFrame) -> bool:
    path = DISTILBERT_BASELINE_DIR / "serialized" / f"{split}_serialized.csv"
    columns = DEFAULT_METADATA_FIELDS + ["label_id", "text"]
    offset = 0
    for chunk in pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False, chunksize=10000):
        expected = serialized.iloc[offset:offset + len(chunk)][columns].astype(str).reset_index(drop=True)
        if not chunk.reset_index(drop=True).equals(expected):
            raise ValueError(f"{split}: Step 9 text/metadata differs from Step 8 at row {offset}.")
        offset += len(chunk)
    if offset != len(serialized): raise ValueError(f"{split}: Step 8 serialization row count differs.")
    return True


def prepare_extraction_run(encoder_summary: Dict[str, Any]) -> bool:
    global ACTIVE_RUN
    # Step 10 authenticates analysis-only sidecars using this exact Step 7 build.
    contract = {key: value for key,value in SOURCE_CONTRACT.items() if key != "step8_run"}
    identity = {"source_contract": contract, "script_sha256": sha256_file(Path(__file__)),
                "encoder_source": EMBEDDING_ENCODER_SOURCE, "encoder": encoder_summary,
                "protocol": {"seed": DEFAULT_RANDOM_SEED, "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS,
                             "dtype": EMBEDDING_DTYPE, "fp16": USE_FP16_INFERENCE,
                             "compressed_npz": SAVE_COMPRESSED_NPZ, "serializer_version": SERIALIZER_VERSION},
                "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                            "transformers": transformers.__version__, "numpy": np.__version__,
                            "pandas": pd.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}}
    run_id = object_hash(identity)
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    if OUTPUT_DIR.exists() and not manifest_path.exists():
        raise FileExistsError(f"Output already exists without a Step 9 manifest: {OUTPUT_DIR}. Use a new --output-dir.")
    previous = load_json(manifest_path) if manifest_path.exists() else {}
    if previous and previous.get("run_id") != run_id:
        raise ValueError("Output belongs to different data/model/code/settings/runtime. Use a new --output-dir.")
    completed = previous.get("completed_splits", {})
    for split, record in completed.items():
        if split not in ("train", "val", "test"): raise ValueError("Unknown completed split.")
        verify_artifacts(OUTPUT_DIR, record.get("artifacts", {}))
    if previous.get("status") == "COMPLETE":
        if set(completed) != {"train", "val", "test"}: raise ValueError("Incomplete split completion record.")
        verify_artifacts(OUTPUT_DIR, previous.get("artifacts", {}))
        ACTIVE_RUN = previous
        return False
    ACTIVE_RUN = {**identity, "dataset_id": SOURCE_CONTRACT["dataset_id"],
                  "step8_run_id": SOURCE_CONTRACT["step8_run_id"], "run_id": run_id,
                  "status": "RUNNING", "completed_splits": completed}
    ensure_directories()
    save_json(manifest_path, ACTIVE_RUN)
    return True


def record_completed_split(split: str, summary: Dict[str, Any]) -> None:
    names = [f"{split}_window_embeddings.npz", f"{split}_window_metadata.csv"]
    if SAVE_SERIALIZATION_AUDIT_SAMPLES: names.append(f"{split}_serialization_samples.csv")
    hashes = {name: sha256_file(OUTPUT_DIR / name) for name in names}
    ACTIVE_RUN["completed_splits"][split] = {"summary": summary, "artifacts": hashes}
    save_json(OUTPUT_DIR / "run_manifest.json", ACTIVE_RUN)


def finish_extraction_run() -> None:
    # Detect input/model changes during a long extraction before marking outputs complete.
    verify_artifacts(DATASET_DIR, SOURCE_CONTRACT["input_sha256"])
    verify_artifacts(DISTILBERT_BASELINE_DIR, SOURCE_CONTRACT["step8_artifacts"])
    if (sha256_file(DATASET_DIR / "build_manifest.json") != SOURCE_CONTRACT["step7_build_manifest_sha256"]
            or sha256_file(DISTILBERT_BASELINE_DIR / "run_manifest.json") != SOURCE_CONTRACT["step8_manifest_sha256"]):
        raise ValueError("Source completion manifests changed during embedding extraction.")
    if EMBEDDING_ENCODER_SOURCE == "base":
        run8 = SOURCE_CONTRACT["step8_run"]
        if run8.get("local_initializer_sha256") is not None:
            verify_artifacts(Path(run8["protocol"]["model_name"]), run8["local_initializer_sha256"])
    hashes = {REPORT_PATH.name: sha256_file(REPORT_PATH)}
    if set(ACTIVE_RUN["completed_splits"]) != {"train", "val", "test"}:
        raise ValueError("Cannot complete Step 9 without all three split artifacts.")
    for record in ACTIVE_RUN["completed_splits"].values(): hashes.update(record["artifacts"])
    verify_artifacts(OUTPUT_DIR, hashes)
    ACTIVE_RUN.update(status="COMPLETE", artifacts=hashes)
    save_json(OUTPUT_DIR / "run_manifest.json", ACTIVE_RUN)


def write_review_zip() -> Path:
    output = OUTPUT_DIR / "09_EMBEDDINGS_REVIEW.zip"
    with tempfile.NamedTemporaryFile(dir=OUTPUT_DIR, suffix=".zip.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(Path(__file__), "source/9.extract_distilbert_embeddings.py")
            paths = [OUTPUT_DIR / "run_manifest.json", REPORT_PATH]
            paths += list(OUTPUT_DIR.glob("*_window_metadata.csv")) + list(OUTPUT_DIR.glob("*_serialization_samples.csv"))
            for path in paths:
                if path.is_file(): archive.write(path, path.relative_to(OUTPUT_DIR).as_posix())
            for name in ("config.json", "label_mapping.json", "training_provenance.json"):
                path = DISTILBERT_MODEL_DIR / name
                if path.is_file(): archive.write(path, "step8_model/" + name)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    global SOURCE_CONTRACT
    args = parse_args()
    apply_runtime_args(args)

    set_all_seeds(DEFAULT_RANDOM_SEED)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run Step 9 as a single Python process.")
    # Check completion before reading model files, allocating arrays, or creating outputs.
    run8 = load_json(DISTILBERT_BASELINE_DIR / "run_manifest.json")
    if run8.get("status") != "COMPLETE":
        raise ValueError("Step 8 is not COMPLETE. Finish Step 8 first, then run Step 9.")

    print("=" * 78)
    print("Step 9: Extract fine-tuned DistilBERT CLS embeddings for 1-second TST")
    print("=" * 78)
    print(f"Base directory: {BASE_DIR.resolve()}")
    print(f"Split directory: {SPLIT_DIR.resolve()}")
    print(f"DistilBERT model directory: {DISTILBERT_MODEL_DIR.resolve()}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    step7_temporal_metadata = validate_step7_temporal_metadata()
    config = load_extraction_config(DISTILBERT_MODEL_DIR)
    SOURCE_CONTRACT = load_source_contract(config)

    print("\n[Temporal design check]")
    print(f"  observable window: {EXPECTED_OBSERVABLE_WINDOW_SECONDS} second")
    print(f"  TST sequence lengths: {TST_SEQUENCE_LENGTHS}")
    print(f"  TST temporal contexts: {TST_TEMPORAL_CONTEXT_SECONDS} seconds")

    print("\n[Loaded extraction config]")
    print(f"  model_name from Step 8: {config['model_name']}")
    print(f"  max_length: {config['max_length']}")
    print(f"  metadata fields: {len(config['metadata_fields'])}")
    print(f"  observable fields: {len(config['observable_fields'])}")
    print(f"  observable window seconds: {config['observable_window_seconds']}")
    print(f"  TST sequence lengths: {config['tst_sequence_lengths']}")
    print(f"  TST temporal contexts: {config['tst_temporal_context_seconds']}")
    print("  label mapping:")
    for label in config["label_order"]:
        print(f"    {config['label_to_id'][label]} -> {label}")

    split_paths = {
        "train": TRAIN_INPUT_PATH,
        "val": VAL_INPUT_PATH,
        "test": TEST_INPUT_PATH,
    }

    print("\n[Load and validate observable split CSVs]")
    split_dfs: Dict[str, pd.DataFrame] = {}
    temporal_summaries: Dict[str, Any] = {}
    for split_name, path in split_paths.items():
        df_raw = read_csv_robust(path)
        df_validated = validate_observable_dataframe(
            df=df_raw,
            split_name=split_name,
            metadata_fields=config["metadata_fields"],
            observable_fields=config["observable_fields"],
            label_order=config["label_order"],
        )
        temporal_summaries[split_name] = validate_temporal_consistency(
            df_validated,
            split_name=split_name,
            expected_window_seconds=EXPECTED_OBSERVABLE_WINDOW_SECONDS,
        )
        split_dfs[split_name] = df_validated
        print(
            f"  {split_name}: {len(df_validated):,} rows | "
            f"episodes={temporal_summaries[split_name]['episode_count']:,} | "
            f"mean_episode_duration={temporal_summaries[split_name]['mean_episode_duration_seconds']:.2f}s"
        )

    validate_frames_against_build(split_dfs)
    validate_no_episode_overlap(split_dfs)
    print("  Episode overlap check: passed")
    label_distribution_summary = validate_total_label_distribution(split_dfs, config["label_order"])
    print("  Total/class row-count check: passed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if torch.cuda.is_available():
        print(f"  CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"  FP16 inference: {USE_FP16_INFERENCE}")

    print(f"\n[Embedding encoder source] {EMBEDDING_ENCODER_SOURCE}")
    tokenizer, model, embedding_dim, encoder_source_summary = load_model_and_tokenizer(
        DISTILBERT_MODEL_DIR,
        device=device,
        config=config,
    )
    print(f"[Model loaded] embedding_dim={embedding_dim}")
    print(f"  model_source: {encoder_source_summary.get('model_source')}")
    print(f"  tokenizer_source: {encoder_source_summary.get('tokenizer_source')}")
    print(f"  representation_type: {encoder_source_summary.get('representation_type')}")

    if not prepare_extraction_run(encoder_source_summary):
        print(f"[Reuse complete Step 9] {OUTPUT_DIR}")
        print(f"Review ZIP: {write_review_zip()}")
        return
    split_summaries: Dict[str, Any] = {}
    for split_name in ["train", "val", "test"]:
        completed = ACTIVE_RUN["completed_splits"].get(split_name)
        if completed:
            split_summaries[split_name] = completed["summary"]
            print(f"[Reuse verified split] {split_name}")
            continue
        split_summaries[split_name] = extract_embeddings_for_split(
            split_name=split_name,
            df=split_dfs[split_name],
            tokenizer=tokenizer,
            model=model,
            device=device,
            config=config,
            embedding_dim=embedding_dim,
        )
        record_completed_split(split_name, split_summaries[split_name])

    report = {
        "step": "9_extract_distilbert_embeddings",
        "status": "COMPLETE",
        "dataset_id": SOURCE_CONTRACT["dataset_id"],
        "step8_run_id": SOURCE_CONTRACT["step8_run_id"],
        "run_id": ACTIVE_RUN["run_id"],
        "input_sha256": SOURCE_CONTRACT["input_sha256"],
        "step8_model_sha256": {name: value for name,value in SOURCE_CONTRACT["step8_artifacts"].items()
                                if name.startswith("trained_distilbert_model/")},
        "serializer_version": SERIALIZER_VERSION,
        "label_semantics": "supplied scenario/episode class; not instantaneous attack presence",
        "input_order": "unchanged Step 7 CSV row order; embedding_row_index is split-local",
        "extraction_rule": "eval + inference mode, no fitting, no normalization, all rows retained",
        "downstream_rule": "Use the same encoder and endpoint set for TST, LSTM, and pooling comparisons.",
        "source_model_dir": str(DISTILBERT_MODEL_DIR),
        "embedding_encoder_source": EMBEDDING_ENCODER_SOURCE,
        "encoder_source_summary": encoder_source_summary,
        "base_encoder_name": SOURCE_CONTRACT["step8_run"]["protocol"]["model_name"],
        "source_label_mapping": str(DISTILBERT_LABEL_MAPPING_PATH),
        "output_dir": str(OUTPUT_DIR),
        "random_seed": DEFAULT_RANDOM_SEED,
        "observable_window_seconds": EXPECTED_OBSERVABLE_WINDOW_SECONDS,
        "tst_sequence_lengths": TST_SEQUENCE_LENGTHS,
        "tst_temporal_context_seconds": TST_TEMPORAL_CONTEXT_SECONDS,
        "step7_temporal_metadata": step7_temporal_metadata,
        "split_temporal_summaries": temporal_summaries,
        "label_distribution_summary": label_distribution_summary,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "use_fp16_inference": bool(USE_FP16_INFERENCE and device.type == "cuda"),
        "embedding_dim": int(embedding_dim),
        "embedding_dtype": EMBEDDING_DTYPE,
        "save_compressed_npz": SAVE_COMPRESSED_NPZ,
        "max_length": int(config["max_length"]),
        "metadata_fields": config["metadata_fields"],
        "observable_fields": config["observable_fields"],
        "label_order": config["label_order"],
        "label_to_id": config["label_to_id"],
        "input_rule": (
            "Serialize observable fields only. Metadata is preserved only in metadata CSV "
            "for episode-level TST sequence construction. Each metadata row corresponds "
            "to one 1-second observable window."
        ),
        "splits": split_summaries,
    }
    save_json(REPORT_PATH, report)
    finish_extraction_run()

    print("\n" + "=" * 78)
    print("DistilBERT embedding extraction completed")
    print("=" * 78)
    for split_name, summary in split_summaries.items():
        print(
            f"{split_name}: rows={summary['rows']:,}, "
            f"shape={summary['embedding_shape']}, "
            f"alignment={summary['alignment_check_passed']}"
        )
    print("\nSaved artifacts:")
    print(f"- Embeddings and metadata: {OUTPUT_DIR.resolve()}")
    print(f"- Extraction report: {REPORT_PATH.resolve()}")
    print("\nNext step:")
    print("Next: run the revised Step 10 to build episode-bounded contexts from these embeddings.")
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
        print("\n[ERROR] DistilBERT embedding extraction failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
