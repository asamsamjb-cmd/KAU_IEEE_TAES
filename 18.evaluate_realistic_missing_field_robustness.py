#!/usr/bin/env python3
"""Step 18: evaluate structured missing-field stress with the fixed shared-nominal models.

Save as 18.evaluate_realistic_missing_field_robustness.py beside the existing scripts.
Requires numpy, pandas, torch and transformers from training.
Default seed 20260610 is fixed in advance; L3 and L10 use the SAME L10 test endpoints.
No retraining, checkpoint selection on test data, base-encoder fallback or automatic
source regeneration. The COMPLETE shared-nominal missing-field dataset is required.
Generate it with the existing generate_realistic_missing_field_tests script.

The fine-tuned Step 8 classifier and its own raw CLS encoder serve both baselines.
A clean-control gate first reproduces Step 9 embeddings, Step 8 predictions and
Step 11 TST predictions. Failure stops evaluation before missing-condition inference.
Outputs are per-run, preserving previous results and all upstream inputs.

'REALISTIC' in legacy CSV names is a compatibility label only: these are synthetic
structured masking conditions, not field-calibrated outages. One mask realization
and one prespecified head seed do not establish multiseed/operational robustness.
Primary comparison: 22-field representation + temporal aggregation at common endpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
SEQUENCE_LENGTHS = [1, 2, 3, 5, 10]
SPLITS = ['train', 'val', 'test']
D_MODEL, NUM_HEADS, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD = 256, 8, 2, 512
DROPOUT, POOLING = 0.10, 'last'
IDENTITY_KEYS = ('dataset_id', 'step8_run_id', 'step9_run_id', 'step10_run_id')
LABELS = ['NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING', 'UNAUTHORIZED_TELECOMMAND_INJECTION',
          'TELECOMMAND_REPLAY', 'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE']
META = ['scenario_id', 'episode_id', 'episode_type', 'episode_step', 'window_id', 'timestamp', 'label']
FIELDS = ['observation_stage', 'path_type', 'direction', 'source_role', 'destination_role',
          'protocol_context', 'operational_phase', 'message_type', 'service_context', 'command_intent',
          'payload_or_parameter_context', 'sequence_freshness_status', 'control_security_state',
          'signal_quality_state', 'tracking_stability', 'navigation_consistency', 'timing_consistency',
          'decoder_status', 'command_execution_result', 'retry_timeout_state', 'session_state',
          'state_transition_consistency']
KEYS = ['episode_id', 'episode_step', 'window_id']
PROTOCOL_VERSION = 'shared_nominal_structured_missing_evaluation_v3'
EXPECTED_FINETUNED_MODEL_DIR = BASE_DIR / 'outputs/shared_nominal_v2/distilbert_window_baseline/trained_distilbert_model'
EMBEDDING_DIR = BASE_DIR / 'outputs/shared_nominal_v2/tst_embeddings_finetuned_main'
SEQUENCE_DIR = BASE_DIR / 'outputs/shared_nominal_v2/tst_sequences_finetuned_main'
PINNED_INPUTS = {}
# Fixed numerical tolerance handles floating-point kernels; class decisions must match exactly.
CLEAN_ATOL, CLEAN_RTOL = 1e-4, 1e-4

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

def sequence_npz_path(split_name: str, sequence_length: int) -> Path:
    return SEQUENCE_DIR / f"{split_name}_tst_sequences_L{sequence_length}.npz"

def sequence_metadata_path(split_name: str, sequence_length: int) -> Path:
    return SEQUENCE_DIR / f"{split_name}_tst_sequence_metadata_L{sequence_length}.csv"

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

def integer_array(value: Any, context: str, ndim: int = 1) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != ndim or arr.dtype.kind not in "iu" or not arr.size:
        raise ValueError(f"{context}: expected a nonempty {ndim}-D integer array; got {arr.shape}/{arr.dtype}")
    if arr.dtype.kind == "u" and arr.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{context}: integer overflow")
    return arr.astype(np.int64, copy=False)

def endpoint_digest(endpoints: np.ndarray) -> str:
    return hashlib.sha256(np.sort(endpoints).astype("<i8", copy=False).tobytes()).hexdigest()


OBSERVABLE_EFFECT_AUDIT_COLUMNS = [
    "audit_current_observable_effect", "audit_history_effect_window_count",
    "audit_context_effect_window_count", "audit_effect_stratum",
]
MASK_COLUMNS = ['mask_current_missing_field_count', 'mask_history_affected_window_count',
                'mask_context_affected_window_count', 'mask_context_missing_field_count', 'mask_exposure_stratum']
MASK_STRATA = ['current_masked', 'history_only_masked', 'no_context_masked']


def sequence_observable_effects_path(split_name: str, sequence_length: int) -> Path:
    return SEQUENCE_DIR / f"{split_name}_tst_sequence_observable_effects_L{sequence_length}.csv"


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

def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

def safe_label_name(label: str) -> str:
    return (
        str(label)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("&", "and")
        .replace("-", "_")
    )

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
            "class_id": class_id,
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
    }
    per_class_df = pd.DataFrame(rows)
    labels = [id_to_label.get(i, str(i)) for i in range(num_classes)]
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    return summary, per_class_df, cm_df

def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path, expected=None):
    path = Path(path).resolve()
    digest = sha256_file(path)
    require(expected is None or digest == expected, f'Changed source artifact: {path}')
    require(str(path) not in PINNED_INPUTS or PINNED_INPUTS[str(path)] == digest, f'Source changed while in use: {path}')
    PINNED_INPUTS[str(path)] = digest
    return digest


def require_json(path):
    pin(path)
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    pin(path)
    require(isinstance(value, dict) and value, f'Invalid JSON record: {path}')
    return value


def read_csv(path):
    pin(path)
    frame = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False,
                        dtype={key: str for key in META if key not in ('episode_step', 'window_id')})
    pin(path)
    return frame


def read_npz(path):
    pin(path)
    with np.load(path, allow_pickle=False) as archive:
        result = {k: archive[k] for k in archive.files}
    pin(path)
    return result


def verify_artifacts(root, hashes):
    require(bool(hashes), f'Empty source artifact inventory: {root}')
    root = Path(root).resolve()
    for name, digest in hashes.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and isinstance(digest, str) and len(digest) == 64,
                f'Invalid artifact identity: {name}')
        pin(path, digest)


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def resolve_path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def require_input_file(path, purpose, action):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{purpose} not found: {path}\n{action}')
    return path


def align_windows(frame, reference):
    """Identity join, never assume CSV row order equals embedding row order."""
    require(not frame.duplicated(KEYS).any() and not reference.duplicated(KEYS).any(), 'Duplicate window identity')
    left, right = pd.MultiIndex.from_frame(frame[KEYS]), pd.MultiIndex.from_frame(reference[KEYS])
    require(len(left) == len(right), 'Window population size mismatch')
    positions = left.get_indexer(right)
    require((positions >= 0).all(), 'A Step 9 window is absent from the condition')
    aligned = frame.iloc[positions].reset_index(drop=True)
    for col in META:
        require(aligned[col].astype(str).equals(reference[col].astype(str).reset_index(drop=True)),
                f'Changed/misaligned window metadata: {col}')
    return aligned


def validate_condition(base, frame, k, placeholder='UNK'):
    require(list(frame.columns) == META + FIELDS and len(frame) == len(base), 'Changed condition schema/population')
    require(not frame.isna().any().any(), 'Null condition values')
    for col in META + FIELDS[:13]:
        require(frame[col].astype(str).equals(base[col].astype(str)), f'Changed protected/metadata field: {col}')
    before, after = base[FIELDS].astype(str).to_numpy(), frame[FIELDS].astype(str).to_numpy()
    changed = before != after
    require((before != placeholder).all(), 'Complete data already has missing placeholders')
    require(np.isin(changed.sum(1), [0, k]).all() and (after[changed] == placeholder).all(),
            'A condition changed values other than the declared k-field missing mask')
    return changed


def choose_tst_root(root, explicit, seed, source_contract=None):
    """Choose by recorded seed/source identity, never by observed performance."""
    if explicit is not None:
        return resolve_path(root, explicit)
    base = root / 'outputs/shared_nominal_v2'
    main = base / 'tst_training_finetuned_main'
    if seed == 20260610 and (main / 'run_manifest.json').is_file():
        run = require_json(main / 'run_manifest.json')
        if (run.get('status') == 'COMPLETE' and run.get('settings', {}).get('DEFAULT_RANDOM_SEED') == seed
                and (source_contract is None or run.get('source_contract') == source_contract)):
            return main
    multi = base / 'tst_training_finetuned_multiseed'
    require((multi/'multiseed_run_manifest.json').is_file(),
            'No matching complete Step 11 main run. Finish it, or supply --tst-root for the prescribed completed seed.')
    record = require_json(multi / 'multiseed_run_manifest.json')
    require(record.get('status') == 'COMPLETE', 'Step 12 must be COMPLETE, or specify --tst-root for a complete Step 11 run')
    identity = record.get('identity', {})
    require(record.get('job_id') == hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                                             allow_nan=False).encode()).hexdigest(),
            'Step 12 identity differs from its completed record')
    runs = [r for r in record.get('runs', []) if r.get('seed') == seed]
    require(len(runs) == 1, f'Step 12 does not identify one run for seed {seed}')
    folder = resolve_path(root, runs[0]['output_root'])
    pin(folder / 'run_manifest.json', runs[0]['manifest_sha256'])
    run = require_json(folder/'run_manifest.json')
    require(run.get('status') == 'COMPLETE' and run.get('run_id') == runs[0].get('step11_run_id')
            and run.get('settings', {}).get('DEFAULT_RANDOM_SEED') == seed
            and (source_contract is None or run.get('source_contract') == source_contract),
            'Recorded Step 12 seed uses incomplete or different source artifacts')
    return folder


def validate_shared_build(root, contract, r8, r9):
    dataset = root/'dataset/shared_nominal_v2'
    path = dataset/'build_manifest.json'
    build = require_json(path)
    policy, source9 = build.get('policy', {}), r9.get('source_contract', {})
    require(build == source9.get('step7_build') and build.get('status') == 'complete'
            and build.get('dataset_id') == contract['dataset_id']
            and build.get('model_input_fields') == FIELDS
            and policy.get('nominal_substrate') == 'shared_nominal_v2'
            and policy.get('columns') == META+FIELDS and policy.get('label_order') == LABELS
            and policy.get('window_seconds') == 1, 'Use the current shared_nominal_v2 Step 7 build.')
    digest = hashlib.sha256(json.dumps({'policy': policy, 'sources': build.get('sources')},
                                     sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    require(build['dataset_id'] == digest and pin(path) == source9.get('step7_build_manifest_sha256')
            == r8.get('step7_build_manifest_sha256') == contract['observability_audit']['step7_build_manifest_sha256']
            and Path(source9.get('source_dataset_dir', '')).resolve() == dataset.resolve(),
            'Step 7/8/9 shared nominal build identity mismatch')
    return build


def required_missing_artifacts(counts):
    names = {'test_complete_observable.csv', 'realistic_missing_field_manifest_step18.json',
             'realistic_missing_field_summary_step18.csv'}
    for k in counts:
        names.update({f'test_realistic_missing{k}_observable.csv',
            f'audit/test_realistic_missing{k}_mask_audit.csv', f'audit/test_realistic_missing{k}_burst_audit.csv'})
    return names


def preflight(root, missing_dir, tst_root, seed, lengths):
    global EXPECTED_FINETUNED_MODEL_DIR, EMBEDDING_DIR, SEQUENCE_DIR
    root = Path(root).resolve()
    base = root / 'outputs/shared_nominal_v2'
    EXPECTED_FINETUNED_MODEL_DIR = base / 'distilbert_window_baseline/trained_distilbert_model'
    EMBEDDING_DIR = base / 'tst_embeddings_finetuned_main'
    SEQUENCE_DIR = base / 'tst_sequences_finetuned_main'
    require_input_file(EMBEDDING_DIR / 'embedding_extraction_report.json', 'Current Step 9 extraction report',
        'Keep this script in the project folder containing outputs/shared_nominal_v2, '
        'or specify --project-root for that folder. Reuse the completed Step 9 output.')
    require_input_file(SEQUENCE_DIR / 'sequence_build_report_index.json', 'Current Step 10 sequence report',
        'Use the same project root and completed Step 10 output as the current Step 11/12 run.')
    extraction = require_json(EMBEDDING_DIR / 'embedding_extraction_report.json')
    index = require_json(SEQUENCE_DIR / 'sequence_build_report_index.json')
    contract = load_source_contract(extraction, index)
    r8 = require_json(EXPECTED_FINETUNED_MODEL_DIR.parent / 'run_manifest.json')
    r9 = require_json(EMBEDDING_DIR / 'run_manifest.json')
    build = validate_shared_build(root, contract, r8, r9)
    model_map = require_json(EXPECTED_FINETUNED_MODEL_DIR / 'label_mapping.json')
    require(model_map.get('dataset_id') == contract['dataset_id']
            and model_map.get('encoder_source') == 'finetuned'
            and model_map.get('serializer_version') == 'observable22_fixed_order_is_v1', 'Step 8 model identity mismatch')
    require(model_map.get('label_order') == LABELS and model_map.get('label_to_id') == dict(zip(LABELS, range(6)))
            and model_map.get('observable_fields') == FIELDS and model_map.get('metadata_fields') == META,
            'Step 8 exact schema/label order mismatch')
    require(extraction.get('label_order') == LABELS and extraction.get('label_to_id') == model_map['label_to_id'],
            'Step 9 label order mismatch')
    max_length = model_map.get('max_length')
    require(type(max_length) is int and max_length > 0 and extraction.get('max_length') == max_length
            and r8.get('protocol', {}).get('max_length') == max_length, 'Tokenizer length contract mismatch')
    require(extraction.get('embedding_dtype') == 'float32', 'Expected raw float32 Step 9 embeddings')
    require(extraction.get('use_fp16_inference') is False and not r9.get('protocol', {}).get('fp16'),
            'This main-revision evaluation requires the FP32 Step 9 extraction protocol')
    folder = choose_tst_root(root, tst_root, seed, contract)
    r11 = require_json(folder / 'run_manifest.json')
    require(r11.get('status') == 'COMPLETE' and r11.get('source_contract') == contract, 'Step 11 is incomplete or uses different source artifacts')
    require(r11.get('dataset_id') == contract['dataset_id'] and r11.get('run_id'), 'Missing Step 11 identity')
    settings = r11.get('settings', {})
    require(settings.get('DEFAULT_RANDOM_SEED') == seed, 'TST head seed differs from prespecified seed')
    require(settings.get('USE_MISSING_AWARE_TRAINING') is False, 'Expected the main clean-trained TST (no missing-aware training)')
    require(settings.get('SEQUENCE_FIT_SUBSET') == 'full' and settings.get('PRIMARY_TEST_SUBSET') == 'full'
            and settings.get('PRIMARY_SELECTION_METRIC') == 'val_macro_f1'
            and settings.get('USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS') is False,
            'Use the main clean-training/full-population Step 11 protocol selected on validation.')
    require(set(lengths) <= set(r11.get('selected_sequence_lengths', [])), 'Requested lengths were not trained in this run')
    for length in lengths:
        names = [f'L{length}/checkpoints/best_tst_model_L{length}.pt',
                 f'L{length}/predictions/test_predictions_L{length}.csv',
                 f'L{length}/reports/observable_effect_diagnostics_test_L{length}.json']
        verify_artifacts(folder, {name: r11.get('artifacts', {}).get(name) for name in names})
        diagnostic = require_json(folder/names[-1])
        require(diagnostic.get('provenance') == {**{k:contract[k] for k in IDENTITY_KEYS}, 'step11_run_id':r11['run_id']}
                and diagnostic.get('source_audit') == contract['observability_audit']
                and diagnostic.get('split') == 'test' and diagnostic.get('sequence_length') == length,
                f'L{length}: Step 11 observable diagnostic context/source mismatch')
    missing = resolve_path(root, missing_dir) if missing_dir else root / 'dataset/shared_nominal_v2/realistic_missing_field_tests'
    require_input_file(missing / 'run_manifest.json', 'Shared-nominal missing-field dataset',
        'Run the shared_nominal_v2 generate_realistic_missing_field_tests.py script first, '
        'or use --missing-dir for its existing completed output. This generates masking conditions; '
        'it does not retrain Steps 8-16. The evaluator does not generate or replace input data.')
    # Existing artifact keys/files identify the generator as step18. Retain that
    # schema even though this evaluation file is numbered 18 in the user's project.
    r18 = require_json(missing / 'run_manifest.json')
    require(r18.get('status') == 'COMPLETE' and r18.get('dataset_id') == contract['dataset_id'], 'Step 18 is incomplete or uses another dataset')
    identity18 = r18.get('identity', {})
    require(r18.get('run_id') == hashlib.sha256(json.dumps(identity18, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            'Invalid Step 18 identity')
    require(identity18.get('protocol_version') == 'label_independent_structured_missing_v2',
            'Regenerate with revised Step 18: legacy label-conditioned masks are incompatible')
    require(identity18.get('input_contract_version') == 'shared_nominal_v2',
            'Use Step 18 generated from the current shared nominal pipeline.')
    counts = identity18.get('missing_counts')
    require(isinstance(counts, list) and counts and all(type(k) is int and 1 <= k <= 9 for k in counts)
            and counts == sorted(set(counts)), 'Invalid missing-count list')
    require(set(r18.get('artifacts', {})) == required_missing_artifacts(counts),
            'Step 18 generated artifact inventory is incomplete or unexpected')
    verify_artifacts(missing, r18.get('artifacts', {}))
    protocol18 = require_json(missing / 'realistic_missing_field_manifest_step18.json')
    require(protocol18.get('run_id') == r18['run_id'] and protocol18.get('dataset_id') == contract['dataset_id']
            and protocol18.get('protocol_version') == identity18['protocol_version'], 'Step 18 report identity mismatch')
    expected18 = {'dataset_id': contract['dataset_id'], 'nominal_substrate': 'shared_nominal_v2',
                  'step7_build_manifest_sha256': contract['observability_audit']['step7_build_manifest_sha256'],
                  'observability_audit': contract['observability_audit']}
    require(identity18.get('source_contract') == r18.get('source_contract') == protocol18.get('source_contract') == expected18
            and protocol18.get('input_contract_version') == 'shared_nominal_v2',
            'Step 18 is not bound to the same shared nominal build/observable-effect lineage')
    require(protocol18.get('seed') == identity18.get('seed')
            and protocol18.get('placeholder') == identity18.get('placeholder') == 'UNK'
            and protocol18.get('schema') == {'metadata_fields': META, 'observable_fields': FIELDS,
                'canonical_columns': META+FIELDS, 'label_order': LABELS, 'observable_window_seconds':1},
            'Step 18 report schema/seed/placeholder differs from its generation record')
    dataset = root / 'dataset/shared_nominal_v2'
    raw_path = dataset / 'splits/test_observable.csv'
    raw_digest = pin(raw_path, contract['input_sha256'].get('splits/test_observable.csv'))
    build_path, map_path = dataset / 'build_manifest.json', dataset / 'splits/label_mapping.json'
    build = require_json(build_path)
    require(build.get('status') == 'complete' and build.get('dataset_id') == contract['dataset_id']
            and build.get('model_input_fields') == FIELDS, 'Current Step 7 build mismatch')
    for path in (raw_path, map_path, build_path):
        require(identity18.get('input_sha256', {}).get(str(path)) == pin(path), f'Step 18 uses different input: {path}')
    require(build.get('artifacts', {}).get('splits/test_observable.csv') == raw_digest
            and protocol18.get('source_test_sha256') == raw_digest, 'Step 7/18 test hash mismatch')
    base_frame = read_csv(raw_path)
    require(list(base_frame.columns) == META + FIELDS and not base_frame.isna().any().any(), 'Invalid complete schema')
    require(len(base_frame) == build['counts']['test']['rows'] and base_frame.episode_id.nunique() == build['counts']['test']['episodes'],
            'Step 7 raw test population mismatch')
    counts = identity18.get('missing_counts')
    require(isinstance(counts, list) and counts and all(type(k) is int and 1 <= k <= 9 for k in counts), 'Invalid missing-count list')
    conditions = ['complete'] + [f'realistic_missing{k}' for k in counts]
    names = {'complete': 'test_complete_observable.csv', **{f'realistic_missing{k}': f'test_realistic_missing{k}_observable.csv' for k in counts}}
    require(protocol18.get('condition_paths') == names and protocol18.get('placeholder') == 'UNK', 'Unexpected condition mapping')
    require(pin(missing / names['complete']) == raw_digest, 'The complete control must be byte-identical to Step 7')
    frames = {}
    for condition in conditions:
        frame = read_csv(missing / names[condition])
        k = 0 if condition == 'complete' else int(condition.removeprefix('realistic_missing'))
        validate_condition(base_frame, frame, k)
        frames[condition] = frame
    windows = read_csv(EMBEDDING_DIR / 'test_window_metadata.csv')
    require(np.array_equal(integer_array(windows.embedding_row_index, 'embedding_row_index'), np.arange(len(windows))),
            'Step 9 embedding row index mismatch')
    aligned = {name: align_windows(frame, windows) for name, frame in frames.items()}
    expected_embeddings = read_npz(EMBEDDING_DIR / 'test_window_embeddings.npz')
    require(expected_embeddings['embeddings'].dtype == np.float32 and expected_embeddings['embeddings'].shape == (len(windows), 768)
            and np.isfinite(expected_embeddings['embeddings']).all(), 'Invalid Step 9 embeddings')
    require(np.array_equal(expected_embeddings['row_indices'], np.arange(len(windows)))
            and np.array_equal(expected_embeddings['label_ids'], windows.label.map(model_map['label_to_id']))
            and np.array_equal(expected_embeddings['label_ids'], windows.label_id), 'Step 9 embedding labels/order mismatch')
    # Mask exposure is calculated from the actual condition values only. Original
    # counterfactual effect flags cannot establish what evidence survives masking.
    mask_counts = {name: np.count_nonzero(frame[FIELDS].to_numpy(dtype=str) != aligned['complete'][FIELDS].to_numpy(dtype=str), axis=1)
                   for name, frame in aligned.items()}
    effects = {}
    sequences, common = {}, contract['common_endpoint_report']
    require(common.get('reference_length') == 10, 'Expected common L10 population')
    reference = read_npz(sequence_npz_path('test', 10))['end_row_indices']
    require(len(reference) == len(np.unique(reference)) and len(reference) == common['splits']['test']['common_endpoint_count']
            and endpoint_digest(reference) == common['splits']['test']['sorted_end_row_indices_sha256'], 'Common endpoint hash/count mismatch')
    for length in lengths:
        arrays = read_npz(sequence_npz_path('test', length))
        meta = read_csv(sequence_metadata_path('test', length))
        seq = integer_array(arrays['sequence_indices'], 'sequence_indices', 2)
        require(seq.shape == (len(meta), length) and (seq >= 0).all() and (seq < len(windows)).all(), 'Invalid sequence index shape/range')
        require(np.array_equal(seq[:, 0], arrays['start_row_indices']) and np.array_equal(seq[:, -1], arrays['end_row_indices'])
                and np.array_equal(arrays['y'], expected_embeddings['label_ids'][seq[:, -1]]), 'Sequence index/label mismatch')
        validate_sequence_alignment('test', length, arrays, meta, windows)
        effects[length] = read_csv(sequence_observable_effects_path('test', length))
        validate_sequence_observable_effects(effects[length], meta, length, 'test')
        positions = pd.Index(arrays['end_row_indices']).get_indexer(reference)
        require((positions >= 0).all(), f'L{length}: missing common endpoints')
        sequences[length] = (arrays, meta, positions)
    clean8_name = 'predictions/test_predictions.csv'
    verify_artifacts(EXPECTED_FINETUNED_MODEL_DIR.parent, {clean8_name: r8.get('artifacts', {}).get(clean8_name)})
    clean8_original = read_csv(EXPECTED_FINETUNED_MODEL_DIR.parent / clean8_name)
    clean8 = align_windows(clean8_original, windows)
    classifier_order = pd.MultiIndex.from_frame(windows[KEYS]).get_indexer(pd.MultiIndex.from_frame(clean8_original[KEYS]))
    classifier_batch = r8.get('protocol', {}).get('eval_batch')
    require(type(classifier_batch) is int and classifier_batch > 0, 'Step 8 evaluation batch size missing')
    require(isinstance(r8.get('runtime', {}).get('gpu'), str), 'Step 8 runtime/precision record missing')
    classifier_fp16 = r8['runtime']['gpu'] != 'CPU'  # Revised Step 8 fp16 = torch.cuda.is_available().
    tokenizer_hash = r8.get('tokenizer_sha256')
    require(isinstance(tokenizer_hash, str) and len(tokenizer_hash) == 64
            and r9.get('source_contract', {}).get('tokenizer_sha256') == tokenizer_hash
            and extraction.get('encoder_source_summary', {}).get('tokenizer_sha256') == tokenizer_hash,
            'Step 8/9 tokenizer fingerprint contract mismatch')
    extraction_batch = r9.get('protocol', {}).get('batch_size')
    require(type(extraction_batch) is int and extraction_batch > 0, 'Step 9 batch size missing')
    require(type(settings.get('BATCH_SIZE')) is int and settings['BATCH_SIZE'] > 0, 'Step 11 batch size missing')
    return {'root': root, 'source_contract': contract, 'r11': r11, 'tst_root': folder,
            'step18_run_id': r18['run_id'], 'missing_protocol': protocol18, 'conditions': conditions,
            'frames': aligned, 'windows': windows, 'embeddings9': expected_embeddings['embeddings'],
            'sequences': sequences, 'common_endpoints': reference, 'mapping': model_map,
            'clean_observable_effects': effects, 'mask_counts': mask_counts,
            'encoder_batch_size': extraction_batch, 'clean8': clean8,
            'classifier_batch_size': classifier_batch, 'classifier_fp16': classifier_fp16,
            'classifier_order': classifier_order, 'tokenizer_sha256': tokenizer_hash}
def serialize_observable_row(row):
    values = [str(row[field]).strip() for field in FIELDS]
    require(all(values) and not row[FIELDS].isna().any(), 'Empty/null observable cannot be serialized')
    return '; '.join(f'{field} is {value}' for field, value in zip(FIELDS, values)) + '.'


def tokenize_batch(tokenizer, texts, max_length):
    encoded = tokenizer(texts, truncation=False, padding=True, return_tensors='pt')
    require(not (encoded['attention_mask'].sum(1) > max_length).any(),
            f'Input exceeds the trained max_length={max_length}; silent truncation is forbidden')
    require(tokenizer.cls_token_id is not None and (encoded['input_ids'][:, 0] == tokenizer.cls_token_id).all(),
            'First token must be CLS')
    require(tokenizer.unk_token_id is None or not (encoded['input_ids'] == tokenizer.unk_token_id).any(),
            'Tokenizer [UNK] encountered. The literal field value UNK must use the saved tokenizer vocabulary')
    return {key: encoded[key] for key in ('input_ids', 'attention_mask')}



def infer_windows(frame, tokenizer, classifier, context, device):
    texts = frame.apply(serialize_observable_row, axis=1).tolist()
    embeddings = []
    batch_size = context['encoder_batch_size']
    classifier.eval()
    for lo in range(0, len(frame), batch_size):
        encoded = tokenize_batch(tokenizer, texts[lo:lo + batch_size], context['mapping']['max_length'])
        with torch.inference_mode():
            output = classifier.distilbert(**{key: value.to(device) for key, value in encoded.items()})
        embeddings.append(output.last_hidden_state[:, 0, :].float().cpu().numpy())
        if lo == 0 or (lo // batch_size) % 100 == 0:
            print(f'  Encoder {min(lo + batch_size, len(frame)):,}/{len(frame):,}', flush=True)
    # Step 8 Trainer prediction uses its evaluation batches and CUDA FP16 policy;
    # Step 9 CLS and Step 11 evaluation use FP32. Reuse weights, not precision modes.
    probabilities = np.empty((len(frame), 6), dtype=np.float32)
    order = context['classifier_order']
    for lo in range(0, len(frame), context['classifier_batch_size']):
        ids = order[lo:lo + context['classifier_batch_size']]
        encoded = tokenize_batch(tokenizer, [texts[i] for i in ids], context['mapping']['max_length'])
        amp = torch.autocast(device_type='cuda', dtype=torch.float16) if context['classifier_fp16'] else nullcontext()
        with torch.inference_mode(), amp:
            scores = classifier(**{key: value.to(device) for key, value in encoded.items()}).logits
        probabilities[ids] = torch.softmax(scores.float(), dim=1).cpu().numpy()
    emb = np.concatenate(embeddings)
    require(emb.shape == (len(frame), 768) and np.isfinite(emb).all() and np.isfinite(probabilities).all(),
            'Invalid encoder/classifier output')
    return emb, probabilities


def load_tst_checkpoint(length, context, device):
    path = context['tst_root'] / f'L{length}/checkpoints/best_tst_model_L{length}.pt'
    expected = context['r11']['artifacts'][path.relative_to(context['tst_root']).as_posix()]
    pin(path, expected)
    # Current Step 11 saves tensors and plain JSON-compatible metadata.
    # Never fall back to unrestricted pickle deserialization.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    pin(path, expected)
    provenance = {k: context['source_contract'][k] for k in IDENTITY_KEYS}
    provenance['step11_run_id'] = context['r11']['run_id']
    require(checkpoint.get('provenance') == provenance and checkpoint.get('source_contract') == context['source_contract'],
            f'L{length}: checkpoint provenance mismatch')
    require(checkpoint.get('label_order') == LABELS and checkpoint.get('label_to_id') == dict(zip(LABELS, range(6))),
            f'L{length}: checkpoint class order mismatch')
    train = checkpoint.get('train_config', {})
    settings = context['r11']['settings']
    require(train.get('random_seed') == settings['DEFAULT_RANDOM_SEED'] and train.get('use_missing_aware_training') is False
            and train.get('use_amp') == settings['USE_AMP'], f'L{length}: checkpoint training settings mismatch')
    config = ModelConfig(**checkpoint['model_config'])
    require(config.input_dim == 768 and config.num_classes == 6 and config.sequence_length == length
            and config.temporal_context_seconds == length, f'L{length}: checkpoint model dimensions mismatch')
    model = TimeSeriesTransformerClassifier(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model


@torch.inference_mode()
def infer_tst(model, embeddings, arrays, batch_size, device):
    # Evaluate all available sequences in Step 11 order/batches, then select common
    # endpoints for comparisons. This also makes the clean reproduction gate precise.
    scores = []
    indices = arrays['sequence_indices']
    for lo in range(0, len(indices), batch_size):
        x = torch.from_numpy(embeddings[indices[lo:lo + batch_size]].astype(np.float32, copy=False)).to(device)
        # Step 11 evaluate_model does not use autocast, even when training uses AMP.
        logits = model(x)
        scores.append(logits.float().cpu().numpy())
    values = np.concatenate(scores)
    require(values.shape == (len(indices), 6) and np.isfinite(values).all(), 'Invalid TST logits')
    return softmax_numpy(values)


def check_clean_probabilities(probs, saved, probability_columns, truth, context_name):
    validate_probabilities(probs, len(truth))
    require(len(probs) == len(saved) and set(probability_columns) <= set(saved), f'{context_name}: incomplete clean predictions')
    require(np.array_equal(integer_array(saved.true_label_id, 'true_label_id'), truth), f'{context_name}: clean truth mismatch')
    original = saved[probability_columns].to_numpy(dtype=np.float64)
    require(np.isfinite(original).all() and ((original >= 0) & (original <= 1)).all()
            and np.allclose(original.sum(1), 1, atol=2e-6, rtol=0), f'{context_name}: invalid saved probabilities')
    require(np.array_equal(original.argmax(1), integer_array(saved.pred_label_id, 'pred_label_id')),
            f'{context_name}: saved decision/probability mismatch')
    require(np.array_equal(probs.argmax(1), saved.pred_label_id)
            and np.allclose(probs, original, atol=CLEAN_ATOL, rtol=CLEAN_RTOL),
            f'{context_name}: clean predictions do not reproduce. Check the original model/runtime/precision; '
            'do not proceed with missing-condition results or select another seed based on the test score')
    return {'decision_mismatches': 0, 'max_abs_probability_difference': float(np.max(np.abs(probs - original)))}


def check_clean_control(context, embeddings, distilbert_probs, tst_probs):
    require(set(tst_probs) == set(context['sequences']), 'Clean control must reproduce every selected TST length')
    expected = context['embeddings9']
    require(embeddings.shape == expected.shape and np.allclose(embeddings, expected, atol=CLEAN_ATOL, rtol=CLEAN_RTOL),
            'Clean CLS embeddings do not reproduce Step 9. Check saved fine-tuned encoder, tokenizer, '
            'serialization, precision and runtime before evaluating missing conditions')
    truth = context['windows'].label_id.to_numpy(dtype=np.int64)
    report = {'status': 'PASS', 'embedding_max_abs_difference': float(np.max(np.abs(embeddings - expected))),
              'atol': CLEAN_ATOL, 'rtol': CLEAN_RTOL,
              'distilbert': check_clean_probabilities(distilbert_probs, context['clean8'],
                        ['prob_' + label for label in LABELS], truth, 'Step 8 DistilBERT'), 'tst': {}}
    for length, probs in tst_probs.items():
        arrays, meta, common_positions = context['sequences'][length]
        saved = read_csv(context['tst_root'] / f'L{length}/predictions/test_predictions_L{length}.csv')
        require(saved.sequence_id.is_unique and len(saved) == len(meta), f'L{length}: incomplete saved predictions')
        saved = saved.sort_values('sequence_id').reset_index(drop=True)
        for col in ('sequence_id', 'episode_id', 'sequence_start_row', 'sequence_end_row', 'sequence_start_step',
                    'sequence_end_step', 'sequence_start_timestamp', 'sequence_end_timestamp'):
            require(saved[col].astype(str).equals(meta[col].astype(str)), f'L{length}: saved prediction alignment mismatch: {col}')
        for key in IDENTITY_KEYS:
            require(saved[key].eq(context['source_contract'][key]).all(), f'L{length}: saved prediction identity mismatch')
        require(saved.step11_run_id.eq(context['r11']['run_id']).all(), f'L{length}: saved TST run mismatch')
        require(saved.get('source_split', pd.Series(dtype=str)).eq('test').all() and 'source_split' in saved,
                f'L{length}: saved prediction split mismatch')
        effects = context['clean_observable_effects'][length]
        for column in OBSERVABLE_EFFECT_AUDIT_COLUMNS:
            require(column in saved and np.array_equal(saved[column].astype(str), effects[column].astype(str)),
                    f'L{length}: saved observable diagnostic mismatch: {column}')
        expected_common = np.zeros(len(saved), dtype=int); expected_common[common_positions] = 1
        require(np.array_equal(integer_array(saved.common_population, 'common_population'), expected_common),
                f'L{length}: saved common evaluation flags mismatch')
        report['tst'][str(length)] = check_clean_probabilities(probs, saved,
                    ['prob_' + safe_label_name(label) for label in LABELS], arrays['y'], f'Step 11 L{length}')
    return report



def validate_probabilities(probs, n_rows):
    probs = np.asarray(probs)
    require(probs.shape == (n_rows, 6) and n_rows > 0 and np.isfinite(probs).all()
            and ((probs >= 0) & (probs <= 1)).all() and np.allclose(probs.sum(1), 1, atol=2e-6, rtol=0),
            'Prediction probabilities must be finite N x 6 distributions for the complete evaluation population')


def mask_exposure(window_missing_counts, indices):
    """Post-hoc bookkeeping of masked inputs, not attack observability or evidence."""
    counts = integer_array(window_missing_counts, 'window missing-field counts')
    indices = integer_array(indices, 'model context indices', 2)
    require((counts >= 0).all() and (counts <= 9).all() and (indices >= 0).all()
            and (indices < len(counts)).all(), 'Invalid masked-input context')
    values = counts[indices]
    current, history = values[:, -1], np.count_nonzero(values[:, :-1], axis=1)
    return pd.DataFrame({MASK_COLUMNS[0]:current, MASK_COLUMNS[1]:history,
        MASK_COLUMNS[2]:history+(current > 0).astype(int), MASK_COLUMNS[3]:values.sum(1),
        MASK_COLUMNS[4]:np.where(current > 0, 'current_masked', np.where(history > 0, 'history_only_masked', 'no_context_masked'))})


def evaluation_mask_exposure(context, condition, meta, population, length):
    if population == 'all_test_windows':
        require(length == 1 and len(meta) == len(context['windows']), 'Invalid single-window population')
        indices = np.arange(len(meta), dtype=np.int64)[:, None]
    else:
        require(population == 'common_L10_endpoints' and np.array_equal(meta.sequence_end_row, context['common_endpoints']),
                'Mask diagnostics must use exactly the primary common endpoints')
        if length == 1:
            indices = integer_array(meta.sequence_end_row, 'classifier endpoints')[:, None]
        else:
            arrays = context['sequences'][length][0]
            ids = integer_array(meta.sequence_id, 'TST sequence IDs')
            require((ids >= 0).all() and (ids < len(arrays['sequence_indices'])).all(), 'Invalid TST diagnostic IDs')
            indices = arrays['sequence_indices'][ids]
            require(np.array_equal(indices[:, -1], meta.sequence_end_row), 'Mask diagnostic endpoint mismatch')
    return mask_exposure(context['mask_counts'][condition], indices)


def mask_diagnostics(truth, pred, exposure, population, length):
    strata = {}
    for name in MASK_STRATA:
        selected = exposure.mask_exposure_stratum.eq(name).to_numpy()
        n = int(selected.sum())
        summary, per_class, cm = compute_metrics(truth[selected], pred[selected], 6, dict(enumerate(LABELS)))
        present = per_class.support.gt(0)
        strata[name] = {'num_sequences': n, 'fraction_of_population': n/len(truth),
            'support_by_label': dict(zip(LABELS, per_class.support.astype(int).tolist())),
            'accuracy': summary['accuracy'] if n else None,
            'normal_prediction_fraction': float((pred[selected] == 0).mean()) if n else None,
            'macro_f1_present_true_classes': float(per_class.loc[present, 'f1'].mean()) if n else None,
            'present_true_labels': per_class.loc[present, 'label'].tolist(),
            'absent_true_labels': per_class.loc[~present, 'label'].tolist(), 'confusion_matrix':cm.to_numpy().tolist()}
    return {'evaluation_population':population, 'model_context_length':length,
        'total_sequences':len(truth), 'strata':strata,
        'interpretation':'Descriptive conditioning on actual input masking after fixed-model prediction. '
            'Mask exposure is not remaining attack evidence. Step 7/10 complete-data counterfactual effect '
            'flags are not recomputed or relabeled as post-mask observability. Strata may have different '
            'class/context composition; present-class F1 has a changing class basis. Primary metrics retain every endpoint.'}


def save_evaluation(out, condition, model, prefix, meta, truth, probs, population, context, length):
    truth = integer_array(truth, 'evaluation truth')
    validate_probabilities(probs, len(truth))
    require(len(meta) == len(truth) and np.isin(truth, range(6)).all()
            and np.array_equal(meta.label.to_numpy(), np.array(LABELS)[truth]), 'Evaluation metadata/label mismatch')
    pred = probs.argmax(1).astype(np.int64)
    summary, per_class, cm = compute_metrics(truth, pred, 6, dict(enumerate(LABELS)))
    identity = {k: context['source_contract'][k] for k in IDENTITY_KEYS}
    identity.update(step11_run_id=context['r11']['run_id'], step18_run_id=context['step18_run_id'],
                    head_seed=context['r11']['settings']['DEFAULT_RANDOM_SEED'])
    row = {**summary, **identity, 'condition': condition, 'model': model, 'sequence_length': length,
           'temporal_context_seconds': length, 'evaluation_population': population}
    exposure = evaluation_mask_exposure(context, condition, meta, population, length)
    diagnostic_path = out/'reports'/f'{prefix}_mask_exposure_diagnostics.json'
    diagnostic = mask_diagnostics(truth, pred, exposure, population, length)
    diagnostic.update(provenance=identity, condition=condition, model=model)
    save_json(diagnostic_path, diagnostic)
    row['mask_exposure_diagnostics'] = str(diagnostic_path.resolve())
    save_json(out / 'reports' / f'{prefix}_metrics.json', row)
    per_class.to_csv(out / 'reports' / f'{prefix}_per_class.csv', index=False, encoding='utf-8-sig')
    cm.to_csv(out / 'reports' / f'{prefix}_confusion_matrix.csv', encoding='utf-8-sig')
    frame = meta.copy().reset_index(drop=True)
    for key, value in identity.items():
        frame[key] = value
    frame['condition'], frame['model'], frame['evaluation_population'] = condition, model, population
    frame['common_population'] = 1 if population == 'common_L10_endpoints' else np.isin(
        context['windows'].embedding_row_index, context['common_endpoints']).astype(int)
    frame['model_context_length'] = length
    for column in MASK_COLUMNS: frame[column] = exposure[column].to_numpy()
    frame['true_label_id'], frame['pred_label_id'] = truth, pred
    frame['true_label'], frame['pred_label'] = np.array(LABELS)[truth], np.array(LABELS)[pred]
    frame['correct'], frame['confidence'] = (truth == pred).astype(int), probs.max(1)
    for i, label in enumerate(LABELS):
        frame['prob_' + safe_label_name(label)] = probs[:, i]
    frame.to_csv(out / 'predictions' / f'{prefix}_predictions.csv', index=False, encoding='utf-8-sig')
    return row


def run(root, missing_dir=None, tst_root=None, seed=20260610, lengths=(3, 10), output_dir=None, audit_only=False):
    global PINNED_INPUTS
    PINNED_INPUTS = {}
    root = Path(root).expanduser().resolve()
    lengths = sorted(set(lengths))
    require(lengths and set(lengths) <= {1, 2, 3, 5, 10}, 'Lengths must be a nonempty subset of 1,2,3,5,10')
    require(seed >= 0, 'Seed must be nonnegative')
    print(f'Step 18: verifying shared-nominal inputs, fixed checkpoints and common endpoints...\n'
          f'Project root: {root}', flush=True)
    context = preflight(root, missing_dir, tst_root, seed, lengths)
    print(f"Source audit PASS: {len(context['windows']):,} windows, "
          f"{len(context['common_endpoints']):,} common L10 endpoints, head seed {seed}", flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = resolve_path(root, output_dir) if output_dir else root / 'outputs/shared_nominal_v2/realistic_missing_field_robustness' / ('run_' + stamp)
    # Never delete prior results, incomplete runs, generated conditions or upstream inputs.
    out.mkdir(parents=True, exist_ok=False)
    for name in ('reports', 'predictions'):
        (out / name).mkdir()
    record = {'status': 'RUNNING', 'protocol_version': PROTOCOL_VERSION, 'source_contract': context['source_contract'],
              'step11_run_id': context['r11']['run_id'], 'step18_run_id': context['step18_run_id'],
              'head_seed': seed, 'tst_root': str(context['tst_root']), 'lengths': lengths,
              'script_sha256': sha256_file(Path(__file__)), 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'condition_order': context['conditions'], 'evaluation_population': 'common_L10_endpoints',
              'distilbert_all_windows': 'separate descriptive population; not the primary TST comparison',
              'scope': context['missing_protocol']['scope'], 'encoder_source': 'verified fine-tuned Step 8 raw CLS, FP32, no normalization',
              'inference_precision': {'distilbert_classifier': 'CUDA FP16' if context['classifier_fp16'] else 'FP32',
                                      'cls_embeddings': 'FP32', 'tst_evaluation': 'FP32'},
              'observable_effect_audit_use': 'Verify complete-data source/prediction lineage only; not post-mask evidence availability.',
              'mask_diagnostics': 'Actual current/history masking in each model input; post-hoc only, no endpoint filtering.',
              'input_sha256': dict(PINNED_INPUTS), 'audit_only': bool(audit_only)}
    save_json(out / 'run_manifest.json', record)
    try:
        if audit_only:
            record['status'] = 'AUDIT_ONLY_COMPLETE'
        else:
            from transformers import AutoTokenizer, DistilBertForSequenceClassification
            import transformers
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            settings = context['r11']['settings']
            record['runtime'] = {'device': str(device), 'python': platform.python_version(),
                                 'torch': str(torch.__version__), 'transformers': transformers.__version__,
                                 'numpy': np.__version__, 'cuda': torch.version.cuda}
            require(not context['classifier_fp16'] or device.type == 'cuda',
                    'Step 8 used CUDA FP16 prediction. Run this evaluation on CUDA to retain that precision protocol')
            tokenizer = AutoTokenizer.from_pretrained(str(EXPECTED_FINETUNED_MODEL_DIR), use_fast=True, local_files_only=True)
            backend = json.loads(tokenizer.backend_tokenizer.to_str())
            backend.pop('padding', None); backend.pop('truncation', None)
            tokenizer_identity = {'backend': backend, 'special_tokens_map': tokenizer.special_tokens_map,
                                  'model_input_names': tokenizer.model_input_names}
            require(hashlib.sha256(json.dumps(tokenizer_identity, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                    == context['tokenizer_sha256'], 'Loaded tokenizer differs from Step 8/9')
            classifier, loading = DistilBertForSequenceClassification.from_pretrained(
                str(EXPECTED_FINETUNED_MODEL_DIR), local_files_only=True, output_loading_info=True)
            require(not any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')),
                    f'DistilBERT checkpoint did not load exactly: {loading}')
            classifier = classifier.to(device).eval()
            require(classifier.config.dim == 768 and classifier.config.num_labels == 6, 'Unexpected DistilBERT dimensions')
            require(classifier.config.label2id == dict(zip(LABELS, range(6)))
                    and classifier.config.id2label == dict(enumerate(LABELS))
                    and getattr(classifier.config, 'step7_dataset_id', None) == context['source_contract']['dataset_id']
                    and getattr(classifier.config, 'observable_serializer_version', None) == 'observable22_fixed_order_is_v1',
                    'Loaded model label/dataset/serializer identity mismatch')
            require(context['mapping']['max_length'] <= classifier.config.max_position_embeddings,
                    'Trained tokenizer length exceeds encoder capacity')
            models = {length: load_tst_checkpoint(length, context, device) for length in lengths}
            rows = []
            for condition in context['conditions']:
                print(f'[{condition}] fine-tuned encoder -> fixed TST heads', flush=True)
                embeddings, probabilities = infer_windows(context['frames'][condition], tokenizer, classifier, context, device)
                tst = {length: infer_tst(model, embeddings, context['sequences'][length][0], settings['BATCH_SIZE'],
                                         device) for length, model in models.items()}
                if condition == 'complete':
                    gate = check_clean_control(context, embeddings, probabilities, tst)
                    save_json(out / 'clean_control_check.json', gate)
                    record['clean_control_check'] = gate
                    print('Clean control: Step 8 predictions + Step 9 CLS + Step 11 predictions PASS', flush=True)
                require(record.get('clean_control_check', {}).get('status') == 'PASS', 'Clean control gate must pass first')
                truth = context['windows'].label_id.to_numpy(dtype=np.int64)
                rows.append(save_evaluation(out, condition, 'DistilBERT-only', f'distilbert_only_{condition}',
                     context['windows'], truth, probabilities, 'all_test_windows', context, 1))
                for length in lengths:
                    arrays, meta, positions = context['sequences'][length]
                    common_meta = meta.iloc[positions].reset_index(drop=True)
                    ends, y = arrays['end_row_indices'][positions], arrays['y'][positions]
                    require(np.array_equal(ends, context['common_endpoints']), 'Model/condition changed common endpoints')
                    rows.append(save_evaluation(out, condition, f'DistilBERT-only-L{length}-aligned',
                         f'distilbert_only_L{length}_endpoint_{condition}', common_meta, y, probabilities[ends],
                         'common_L10_endpoints', context, 1))
                    rows.append(save_evaluation(out, condition, f'DistilBERT--TST-L{length}',
                         f'tst_L{length}_{condition}', common_meta, y, tst[length][positions],
                         'common_L10_endpoints', context, length))
                del embeddings, probabilities, tst
            summary = pd.DataFrame(rows)
            summary.to_csv(out / 'realistic_missing_field_robustness_summary.csv', index=False, encoding='utf-8-sig')
            for metric in ('macro_f1', 'accuracy', 'weighted_f1'):
                pivot = summary.pivot(index='model', columns='condition', values=metric).reindex(columns=context['conditions'])
                pivot.to_csv(out / f'realistic_missing_field_{metric}_pivot.csv', encoding='utf-8-sig')
            # Both alignment labels for DistilBERT have identical common-endpoint metrics.
            # Keep legacy labels for graph/table compatibility, never treat them as separate classifiers.
            paper = summary[summary.model.isin(['DistilBERT-only-L10-aligned', 'DistilBERT--TST-L10'])]
            paper.to_csv(out / 'realistic_missing_field_l10_aligned_paper_summary.csv', index=False, encoding='utf-8-sig')
            primary = summary[(summary.evaluation_population == 'common_L10_endpoints') &
                              (~summary.model.str.startswith('DistilBERT-only-L') | summary.model.eq(f'DistilBERT-only-L{lengths[-1]}-aligned'))]
            primary.to_csv(out / 'common_endpoint_robustness_summary.csv', index=False, encoding='utf-8-sig')
            record.update(status='COMPLETE', summary_rows=rows)
        for path, digest in PINNED_INPUTS.items():
            require(sha256_file(path) == digest, f'Input changed during evaluation: {path}')
        require(sha256_file(Path(__file__)) == record['script_sha256'], 'Evaluation code changed during the run')
        if not audit_only:
            save_json(out / 'realistic_missing_field_robustness_summary.json', record)
        record.update(input_sha256=dict(PINNED_INPUTS), finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      artifacts={p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob('*'))
                                 if p.is_file() and p.name != 'run_manifest.json'})
        save_json(out / 'run_manifest.json', record)
        print(f'{record["status"]}: {out}', flush=True)
        return out
    except Exception as exc:
        record.update(status='FAILED', error_type=type(exc).__name__, error=str(exc), input_sha256=dict(PINNED_INPUTS))
        save_json(out / 'run_manifest.json', record)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=BASE_DIR)
    parser.add_argument('--missing-dir', type=Path)
    parser.add_argument('--tst-root', type=Path)
    parser.add_argument('--seed', type=int, default=20260610)
    parser.add_argument('--lengths', type=int, nargs='+', default=[3, 10])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--audit-only', action='store_true', help='Check files/provenance/alignment without model inference')
    args = parser.parse_args(argv)
    run(args.project_root, args.missing_dir, args.tst_root, args.seed, args.lengths, args.output_dir, args.audit_only)


if __name__ == '__main__':
    main()
