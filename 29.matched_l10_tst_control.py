# -*- coding: utf-8 -*-
"""Step 29: matched-training L10 TST control for the completed Step 27 L20 pilot.

Place beside 10.build_tst_sequences.py, 11.train_tst_classifier.py,
12.multiseed_tst_runner.py, and 27.l20_tst_pilot.py in the ORIGINAL project.
No original source, embeddings, checkpoint, or result is modified.

Commands (same project venv):
    python "29.matched_l10_tst_control.py" --preflight
    python "29.matched_l10_tst_control.py"
    python "29.matched_l10_tst_control.py" --retry-failed  # previous run stopped
    python "29.matched_l10_tst_control.py" --final-test

New training: three L10 TST heads, one per fixed Step27 seed. For each split,
use EXACTLY Step27's L20-eligible endpoints, but only the most recent ten
of each twenty-window input. Assert exact equality to original Step10's L10
sequence at that endpoint. Step27 L20 heads are never retrained.

Validation selects each L10 checkpoint using original Step11 macro-F1 rule;
only the explicitly requested --final-test evaluates held-out test endpoints.
NOTE: The L20 test was already examined before Step29 was designed, so these
comparisons are EXPLORATORY, not prospectively confirmed on a fresh holdout.

References:
    Step11 supplies the TST class, DataLoader, train/eval, loss, scheduler
    settings, checkpoint I/O and seed setup. Step27 supplies authenticated
    Step8/9/10 provenance, completed L20 sequence cache, and prediction schema.

Important interpretation:
    Same train/val/test endpoints and labels remove the training-cohort
    difference. Context length (and hence the length-specific positional
    parameters) differs by design. Numeric seed identity is NOT evidence
    that L10 and L20 weights start from identical initializations.
    Evidence-location groups use audit-only paired-nominal changes; they are
    not model inputs, definitive attack indicators, or Bayes-error bounds.

Each finished validation run writes 29_L10_MATCHED_VALIDATION_REVIEW.zip.
An explicit final test writes 29_L10_MATCHED_FINAL_TEST_REVIEW.zip.
Checkpoints are saved locally but omitted from review ZIP to keep it small.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
from types import SimpleNamespace
import shutil
import socket
import sys
import tempfile
import time
from typing import Any
import zipfile

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DEFAULT_27_ROOT = BASE / "outputs/shared_nominal_v2/27_L20_TST_VALIDATION_PILOT"
DEFAULT_29_ROOT = BASE / "outputs/shared_nominal_v2/29_L10_MATCHED_TST_CONTROL"
DEFAULT_MAIN = BASE / "outputs/shared_nominal_v2/tst_training_finetuned_main"
DEFAULT_MULTI = BASE / "outputs/shared_nominal_v2/tst_training_finetuned_multiseed"
SEEDS = (20260610, 20260611, 20260612)
SPLITS = ("train", "val", "test")
L10, L20 = 10, 20
SCHEMA = "taes_step29_matched_l10_tst_control_v1"
EXPECTED_DATASET_ID = "478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b"
N_CLASSES = 6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def index_sha(index: np.ndarray) -> str:
    return hashlib.sha256(np.sort(np.asarray(index, dtype="<i8")).tobytes()).hexdigest()


def matrix_sha(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(matrix, dtype="<i8").tobytes()).hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file(), f"Required JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(data, dict), f"Expected JSON object: {path}")
    return data


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                     dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                     suffix=".tmp", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        frame.to_csv(stream, index=False)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def import_path(module_name: str, filename: Path):
    require(filename.is_file(), f"Required Python source not found: {filename}")
    spec = importlib.util.spec_from_file_location(module_name, filename)
    require(spec is not None and spec.loader is not None, f"Cannot import {filename}")
    obj = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = obj
    spec.loader.exec_module(obj)
    return obj


def choose_step27(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        require(path.is_file(), f"Step27 source missing: {path}")
        return path
    candidates = [BASE / x for x in ("27.l20_tst_pilot.py", "27. l20_tst_pilot.py") if (BASE / x).is_file()]
    require(candidates, "Place 27.l20_tst_pilot.py beside Step29 or supply --step27.")
    require(len({sha256(path) for path in candidates}) == 1,
            "Different Step27 sources found; supply --step27 explicitly.")
    return candidates[0].resolve()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="Input and model-forward validation, no files written.")
    mode.add_argument("--final-test", action="store_true", help="Only evaluate existing validation-selected L10 checkpoints.")
    parser.add_argument("--retry-failed", action="store_true", help="Archive and restart incomplete local seeds only after old process stopped.")
    parser.add_argument("--step27-root", type=Path, default=DEFAULT_27_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_29_ROOT)
    parser.add_argument("--step27", type=Path, default=None)
    parser.add_argument("--step10", type=str, default=None)
    parser.add_argument("--step11", type=str, default=None)
    parser.add_argument("--step12", type=str, default=None)
    parser.add_argument("--baseline-main-root", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--baseline-multiseed-root", type=Path, default=DEFAULT_MULTI)
    args = parser.parse_args()
    require(not (args.retry_failed and (args.preflight or args.final_test)),
            "--retry-failed is only for validation training.")
    for attr in ("step27_root", "output_root", "baseline_main_root", "baseline_multiseed_root"):
        setattr(args, attr, getattr(args, attr).expanduser().resolve())
    return args


class ExclusiveLock:
    def __init__(self, root: Path):
        self.root = root
        self.file = root / ".29_active.lock"

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with self.file.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "hostname": socket.gethostname(), "started": utc_now()}, stream)
        except FileExistsError as exc:
            raise RuntimeError(f"Step29 lock already exists: {self.file}. Confirm previous process stopped first.") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        self.file.unlink(missing_ok=True)


def protect_inputs(args, s11) -> None:
    output = args.output_root
    require(output != BASE and not BASE.is_relative_to(output), "Output root cannot contain the project code directory.")
    protected = (args.step27_root, args.baseline_main_root, args.baseline_multiseed_root,
                 s11.EMBEDDING_DIR, s11.SEQUENCE_DIR, s11.EXPECTED_FINETUNED_MODEL_DIR.parent)
    for path in protected:
        root = Path(path).resolve()
        require(not (output == root or output.is_relative_to(root) or root.is_relative_to(output)),
                f"Step29 output overlaps original input or prior result: {root}")


def validate_step27(args, s27, paths, mod, source) -> tuple[dict, dict, dict]:
    root = args.step27_root
    p = root / "27_run_manifest.json"
    pilot = read_json(p)
    require(pilot.get("status") == "COMPLETE", "Step27 validation pilot must be COMPLETE.")
    identity = pilot.get("identity", {})
    require(pilot.get("identity_hash") == s27.digest_json(identity), "Step27 identity hash invalid.")
    require(identity.get("schema") == s27.SCHEMA, "Wrong Step27 schema.")
    require(identity.get("selected_seeds") == list(SEEDS), "Step27 seeds are not the locked three seeds.")
    require(identity.get("reference_length") == L10 and identity.get("new_length") == L20,
            "Wrong reference/extension length in Step27.")
    require(identity.get("source_contract") == source and source.get("dataset_id") == EXPECTED_DATASET_ID,
            "Step27 uses different dataset, encoder or Step10 source contract.")
    for name, source_path in (("pilot_source_sha256", Path(s27.__file__)),
                              ("step10_source_sha256", paths["step10"]),
                              ("step11_source_sha256", paths["step11"]),
                              ("step12_source_sha256", paths["step12"])):
        require(identity.get(name) == sha256(source_path),
                f"Original Step27/{name} source changed. Restore the source used in Step27.")
    require(pilot.get("val_summary_sha256") == sha256(root / "27_val_summary.json"),
            "Step27 validation summary hash changed.")
    require(pilot.get("val_comparison_sha256") == sha256(root / "27_val_seed_comparison.csv"),
            "Step27 validation comparison hash changed.")
    cache = root / "sequence_cache"
    cache_manifest = cache / "27_L20_sequence_manifest.json"
    require(pilot.get("L20_sequences_sha256") == sha256(cache_manifest),
            "Step27 L20 sequence cache manifest hash changed.")
    stored = read_json(cache_manifest)
    require(stored.get("status") == "COMPLETE" and
            stored.get("identity") == s27.sequence_cache_identity(source, paths),
            "Step27 sequence cache identity mismatch.")
    required_cache = {"27_L20_sequence_report.json"}
    for split in SPLITS:
        required_cache |= {f"{split}_tst_sequences_L20.npz",
                           f"{split}_tst_sequence_metadata_L20.csv",
                           f"{split}_tst_sequence_observable_effects_L20.csv"}
    require(required_cache.issubset(stored.get("artifacts", {})), "Missing L20 cache artifact inventory.")
    for relative, expected in stored["artifacts"].items():
        item = (cache / relative).resolve()
        require(item.is_relative_to(cache.resolve()) and item.is_file() and sha256(item) == expected,
                f"Step27 L20 cache artifact missing/changed: {item}")
    seed_records = {}
    for seed in SEEDS:
        record = s27.verify_seed_complete(root / f"seed_{seed}", pilot["identity_hash"], seed)
        metrics = read_json(root / f"seed_{seed}" / "val_metrics.json")
        require(metrics.get("seed") == seed and metrics.get("n_train_sequences") == 144321
                and metrics.get("n_val_sequences") == 29841,
                f"Step27 seed {seed} training/validation population changed.")
        require(record.get("best_epoch") == metrics.get("best_epoch"), "Step27 best epoch mismatch.")
        seed_records[str(seed)] = {
            "manifest_sha256": sha256(root / f"seed_{seed}" / "27_seed_manifest.json"),
            "checkpoint_sha256": record["artifacts"]["best_L20.pt"],
            "val_predictions_sha256": record["artifacts"]["val_predictions.csv"],
            "best_epoch": record["best_epoch"],
        }
    with (root / "27_val_seed_comparison.csv").open("r", encoding="utf-8-sig") as stream:
        comparison = pd.read_csv(stream)
    require(len(comparison) == 3 and sorted(comparison.seed.astype(int).tolist()) == list(SEEDS),
            "Step27 validation comparison lacks the complete 3-seed grid.")
    summary = read_json(root / "27_val_summary.json")
    require(summary.get("num_head_seeds") == 3 and summary.get("dataset_id") == source["dataset_id"]
            and summary.get("identity_hash") == pilot["identity_hash"],
            "Step27 validation summary identity incorrect.")
    return pilot, stored, seed_records


def build_matched_l10(mod, raw, seq20: dict, seqmeta: dict) -> tuple[dict, dict]:
    """Build no new NPZ: same L20 endpoint rows, restricted to last ten columns.

    Assert every matched sequence is the original Step10 L10 sequence at the
    corresponding endpoint, for all train/val/test splits.
    """
    s11 = mod["step11"]
    matched, report = {}, {}
    for split in SPLITS:
        full = seq20[split]
        x20 = full["sequence_indices"]
        ends20 = full["end_row_indices"]
        y20 = full["y"]
        require(x20.ndim == 2 and x20.shape[1] == 20 and len(y20) == len(x20) and
                np.array_equal(x20[:, -1], ends20), f"{split}: invalid L20 shape/endpoints.")
        x10 = np.ascontiguousarray(x20[:, -L10:].astype(np.int64, copy=False))
        original = s11.load_sequence_split(split, L10, raw[split], raw[split]["embeddings"].shape[1])
        lookup = pd.Index(original["end_row_indices"])
        require(lookup.is_unique, f"{split}: duplicated original L10 endpoints.")
        positions = lookup.get_indexer(ends20)
        require((positions >= 0).all(), f"{split}: missing L10 endpoints required for L20 matching.")
        require(np.array_equal(original["sequence_indices"][positions], x10),
                f"{split}: L20[-10:] differs from authenticated original L10 history.")
        require(np.array_equal(original["y"][positions], y20) and
                np.array_equal(x10[:, -1], ends20), f"{split}: matched labels or endpoints differ.")
        m = seqmeta[split]
        require(len(m) == len(y20) and
                np.array_equal(m["label_id"].to_numpy(dtype=np.int64), y20) and
                np.array_equal(m["sequence_end_row"].to_numpy(dtype=np.int64), ends20),
                f"{split}: metadata mismatch against Step27 L20 sequences.")
        require(not m["episode_id"].astype(str).duplicated().all(), f"{split}: implausible episode IDs.")
        require(set(y20.tolist()) == set(range(N_CLASSES)), f"{split}: missing one of six episode-scenario classes.")
        matched[split] = {"sequence_indices": x10, "y": y20.astype(np.int64, copy=False)}
        report[split] = {
            "n_endpoints": int(len(y20)), "n_episodes": int(m.episode_id.nunique()),
            "n_trainable_windows_per_sequence": L10, "source_L20_length": L20,
            "endpoint_sha256": index_sha(ends20), "matched_index_sha256": matrix_sha(x10),
            "label_sha256": matrix_sha(y20),
            "original_step10_L10_history_exactly_matches": True,
        }
        print(f"[Matched {split}] {len(y20):,} endpoints, {m.episode_id.nunique():,} episodes, L20[-10:] == original L10", flush=True)
    # Lock benchmark endpoint population, but derive its identity rather than assuming row order.
    expected = {"train": 144321, "val": 29841, "test": 30561}
    for split in SPLITS:
        require(report[split]["n_endpoints"] == expected[split],
                f"{split}: Step27 experimental population unexpectedly changed.")
    return matched, report


def read_existing_l20_prediction(args, pilot: dict, seq: dict, meta: pd.DataFrame,
                                 seed: int, split: str, s11) -> pd.DataFrame:
    require(split in ("val", "test"), "Only val/test L20 predictions are scored.")
    path = args.step27_root / f"seed_{seed}" / f"{split}_predictions.csv"
    require(path.is_file(), f"Step27 prediction missing: {path}")
    if split == "val":
        record = read_json(args.step27_root / f"seed_{seed}" / "27_seed_manifest.json")
        require(sha256(path) == record["artifacts"].get("val_predictions.csv"),
                "Step27 validation prediction SHA256 mismatch.")
    else:
        # Original Step27 did not add post-test predictions to 27_seed_manifest.
        # Bind each actual test CSV to the immutable completed Step27 review ZIP.
        final_zip = args.step27_root / "27_L20_FINAL_TEST_REVIEW.zip"
        require(final_zip.is_file(), f"Step27 final test archive missing: {final_zip}")
        name = f"seed_{seed}/test_predictions.csv"
        with zipfile.ZipFile(final_zip) as archive:
            require(name in archive.namelist() and archive.testzip() is None,
                    "Step27 final-test ZIP is missing/corrupt prediction records.")
            archived_hash = hashlib.sha256(archive.read(name)).hexdigest()
        require(sha256(path) == archived_hash,
                "Step27 Test prediction differs from its completed final-test review ZIP.")
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    require(not frame.columns.duplicated().any() and len(frame) == len(seq["y"]),
            f"Step27 L20/{split}/{seed}: duplicate columns or incomplete population.")
    for name, target in (("sequence_end_row", seq["end_row_indices"]),
                         ("true_label_id", seq["y"]),
                         ("seed", np.full(len(frame), seed)),
                         ("sequence_length", np.full(len(frame), L20))):
        require(name in frame and pd.api.types.is_integer_dtype(frame[name].dtype) and
                np.array_equal(frame[name].to_numpy(dtype=np.int64), target),
                f"Step27 {split}/{seed}: {name} is wrong/out of order.")
    require(np.array_equal(frame["episode_id"].astype(str).to_numpy(), meta.episode_id.astype(str).to_numpy()),
            f"Step27 {split}/{seed}: episode alignment changed.")
    for name, value in (("source_split", split), ("dataset_id", pilot["identity"]["source_contract"]["dataset_id"]),
                        ("step8_run_id", pilot["identity"]["source_contract"]["step8_run_id"]),
                        ("step9_run_id", pilot["identity"]["source_contract"]["step9_run_id"]),
                        ("original_step10_run_id", pilot["identity"]["source_contract"]["step10_run_id"])):
        require(name in frame and frame[name].eq(value).all(),
                f"Step27 {split}/{seed}: prediction provenance mismatch: {name}")
    pred = frame["pred_label_id"]
    require(pd.api.types.is_integer_dtype(pred.dtype) and pred.between(0, N_CLASSES - 1).all(),
            "Invalid L20 prediction label ID.")
    score = s11.compute_metrics(seq["y"], pred.to_numpy(dtype=np.int64), N_CLASSES,
                                {i: str(i) for i in range(N_CLASSES)})[0]
    if split == "val":
        report = read_json(args.step27_root / f"seed_{seed}" / "val_metrics.json")
        require(math.isclose(score["macro_f1"], report["val"]["macro_f1"], abs_tol=1e-10),
                "Step27 L20 validation predictions disagree with its selected checkpoint report.")
    return frame


def predict_frame(evaluation: dict, meta: pd.DataFrame, seed: int, split: str,
                  source: dict, identity_hash: str) -> pd.DataFrame:
    n = len(meta)
    require(n == len(evaluation["y_true"]) and
            np.array_equal(evaluation["sequence_ids"], np.arange(n)),
            f"Step29 {split}/{seed}: predictions not aligned with matched L20 endpoints.")
    y = evaluation["y_true"].astype(np.int64)
    require(np.array_equal(y, meta.label_id.to_numpy(dtype=np.int64)), "Step29 labels do not match metadata.")
    return pd.DataFrame({
        "sequence_end_row": meta.sequence_end_row.to_numpy(dtype=np.int64),
        "episode_id": meta.episode_id.astype(str).to_numpy(),
        "true_label_id": y,
        "pred_label_id": evaluation["y_pred"].astype(np.int64),
        "source_split": split, "sequence_length": L10, "seed": seed,
        "dataset_id": source["dataset_id"], "step8_run_id": source["step8_run_id"],
        "step9_run_id": source["step9_run_id"], "original_step10_run_id": source["step10_run_id"],
        "step29_identity_hash": identity_hash,
    })


def verified_new_prediction(path: Path, meta: pd.DataFrame, seed: int,
                            split: str, source: dict, identity_hash: str) -> pd.DataFrame:
    require(path.is_file(), f"Missing Step29 {split} prediction: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    require(not frame.columns.duplicated().any() and len(frame) == len(meta),
            f"Step29 {split}/{seed}: wrong prediction table shape.")
    checks = {"sequence_end_row": meta.sequence_end_row.to_numpy(dtype=np.int64),
              "true_label_id": meta.label_id.to_numpy(dtype=np.int64),
              "seed": np.full(len(meta), seed), "sequence_length": np.full(len(meta), L10)}
    for name, expect in checks.items():
        require(name in frame and pd.api.types.is_integer_dtype(frame[name].dtype) and
                np.array_equal(frame[name].to_numpy(dtype=np.int64), expect),
                f"Step29 {split}/{seed} wrong {name}.")
    require(np.array_equal(frame.episode_id.astype(str).to_numpy(), meta.episode_id.astype(str).to_numpy()),
            "Step29 episode alignment mismatch.")
    for name, value in (("source_split", split), ("dataset_id", source["dataset_id"]),
                        ("step8_run_id", source["step8_run_id"]),
                        ("step9_run_id", source["step9_run_id"]),
                        ("original_step10_run_id", source["step10_run_id"]),
                        ("step29_identity_hash", identity_hash)):
        require(name in frame and frame[name].eq(value).all(), f"Step29 prediction provenance: {name}")
    require(pd.api.types.is_integer_dtype(frame.pred_label_id.dtype) and
            frame.pred_label_id.between(0, N_CLASSES - 1).all(), "Step29 prediction class outside 0..5.")
    return frame


def compare_model_predictions(old_l20: pd.DataFrame, new_l10: pd.DataFrame,
                              meta: pd.DataFrame, seed: int, split: str, s11) -> dict:
    require(len(old_l20) == len(new_l10) == len(meta), "Comparison rows differ.")
    for col in ("sequence_end_row", "true_label_id", "episode_id"):
        require(np.array_equal(old_l20[col].astype(str).to_numpy(), new_l10[col].astype(str).to_numpy()),
                f"L10/L20 comparison is not paired on {col}.")
    y = new_l10.true_label_id.to_numpy(dtype=np.int64)
    p10 = new_l10.pred_label_id.to_numpy(dtype=np.int64)
    p20 = old_l20.pred_label_id.to_numpy(dtype=np.int64)
    m10 = s11.compute_metrics(y, p10, N_CLASSES, {i: str(i) for i in range(N_CLASSES)})[0]
    m20 = s11.compute_metrics(y, p20, N_CLASSES, {i: str(i) for i in range(N_CLASSES)})[0]
    result = {"seed": seed, "split": split, "endpoints": int(len(y)),
              "episodes": int(meta.episode_id.nunique()),
              "endpoint_sha256": index_sha(meta.sequence_end_row.to_numpy(dtype=np.int64))}
    for name, metric in (("matched_L10", m10), ("original_L20", m20)):
        for key in ("macro_f1", "accuracy", "weighted_f1"):
            result[f"{name}_{key}"] = float(metric[key])
    result["delta_L20_minus_matched_L10_macro_f1"] = float(m20["macro_f1"] - m10["macro_f1"])
    normal = y == 0
    require(normal.any(), "Matched population has no NORMAL class for false-alarm calculation.")
    result["matched_L10_normal_false_alarm_rate"] = float(np.mean(p10[normal] != 0))
    result["original_L20_normal_false_alarm_rate"] = float(np.mean(p20[normal] != 0))
    result["correct_L10_wrong_L20"] = int(np.count_nonzero((p10 == y) & (p20 != y)))
    result["wrong_L10_correct_L20"] = int(np.count_nonzero((p10 != y) & (p20 == y)))
    return result


def summarize(rows: list[dict], cohort: dict, stage: str, source: dict, identity_hash: str) -> tuple[pd.DataFrame, dict]:
    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    require(len(frame) == len(SEEDS) and frame.seed.astype(int).tolist() == list(SEEDS),
            "Missing/duplicated head seed in Step29 comparison.")
    require(frame.endpoints.nunique() == 1 and frame.episodes.nunique() == 1 and
            frame.endpoint_sha256.nunique() == 1 and
            frame.endpoints.iloc[0] == cohort[stage]["n_endpoints"] and
            frame.endpoint_sha256.iloc[0] == cohort[stage]["endpoint_sha256"],
            "Step29 seeds have mismatched endpoint population.")
    summary = {"schema": SCHEMA, "stage": stage, "exploratory_post_test_design": True,
               "dataset_id": source["dataset_id"], "identity_hash": identity_hash,
               "head_seeds": list(SEEDS), "num_seeds": len(SEEDS),
               "endpoints": int(frame.endpoints.iloc[0]), "episodes": int(frame.episodes.iloc[0]),
               "endpoint_sha256": frame.endpoint_sha256.iloc[0],
               "train_population": cohort["train"], "val_population": cohort["val"],
               "interpretation": "L10 and L20 heads have identical eligible train/val/test endpoints and labels; "
                                 "input history and positional length differ. L20 weights are reused from Step27.",
               "seed_scope": "Seed labels correspond by number; weights are NOT paired initializations. "
                             "Sample SD across three fixed runs, not a bootstrap CI.",
               "test_scope": "This follow-up was designed after earlier L20 test inspection; exploratory, not fresh confirmatory holdout.",
               "recorded_at": utc_now()}
    for col in ("matched_L10_macro_f1", "original_L20_macro_f1",
                "delta_L20_minus_matched_L10_macro_f1", "matched_L10_accuracy", "original_L20_accuracy",
                "matched_L10_weighted_f1", "original_L20_weighted_f1",
                "matched_L10_normal_false_alarm_rate", "original_L20_normal_false_alarm_rate"):
        summary[f"{col}_mean"] = float(frame[col].mean())
        summary[f"{col}_sd_ddof1"] = float(frame[col].std(ddof=1))
    return frame, summary


def training_identity(args, paths: dict, source: dict, pilot: dict,
                      step27_seed_records: dict, cohort: dict, s11) -> dict:
    import torch
    return {
        "schema": SCHEMA,
        "step29_source_sha256": sha256(Path(__file__)),
        "source_contract": source,
        "step27_root": str(args.step27_root),
        "step27_run_manifest_sha256": sha256(args.step27_root / "27_run_manifest.json"),
        "step27_identity_hash": pilot["identity_hash"],
        "step27_seed_records": step27_seed_records,
        "original_scripts_sha256": {key: sha256(value) for key, value in paths.items()},
        "step27_source_sha256": sha256(Path(args.step27).resolve() if args.step27 else choose_step27(None)),
        "locked_seeds": list(SEEDS), "train_and_validation_population": cohort,
        "new_length": L10, "reference_length": L20,
        "training_protocol": {
            "model": "Original Step11 TimeSeriesTransformerClassifier, no architecture modifications",
            "selection": "val_macro_f1 (unperturbed, full matched L20-eligible validation population)",
            "fit_subset": "all L20-eligible train endpoints; embeddings from Step9",
            "max_epochs": s11.MAX_EPOCHS, "patience": s11.PATIENCE,
            "batch_size": s11.BATCH_SIZE, "learning_rate": s11.LEARNING_RATE,
            "weight_decay": s11.WEIGHT_DECAY, "label_smoothing": s11.LABEL_SMOOTHING,
            "d_model": s11.D_MODEL, "num_heads": s11.NUM_HEADS,
            "num_encoder_layers": s11.NUM_ENCODER_LAYERS,
            "dim_feedforward": s11.DIM_FEEDFORWARD, "dropout": s11.DROPOUT,
            "pooling": s11.POOLING, "amp": bool(s11.USE_AMP),
            "regularizers": {k: getattr(s11, k) for k in ("USE_MISSING_AWARE_TRAINING",
                            "USE_SHORTCUT_RESISTANT_TRAINING", "USE_LENGTH_AWARE_REGULARIZATION")},
            "lr_scheduler": "ReduceLROnPlateau(max, factor=0.5, patience=2, min_lr=1e-6)",
        },
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }


def verified_complete_seed(seed_dir: Path, seed: int, identity_hash: str) -> dict:
    record = read_json(seed_dir / "29_seed_manifest.json")
    require(record.get("status") == "COMPLETE" and record.get("seed") == seed and
            record.get("identity_hash") == identity_hash, f"Invalid Step29 completed seed: {seed_dir}")
    must = {"best_L10.pt", "history.csv", "val_predictions.csv", "val_metrics.json"}
    require(must.issubset(record.get("artifacts", {})), f"Incomplete Step29 seed artifacts: {seed_dir}")
    for relative, expected in record["artifacts"].items():
        path = (seed_dir / relative).resolve()
        require(path.is_relative_to(seed_dir.resolve()) and path.is_file() and sha256(path) == expected,
                f"Step29 seed artifact changed: {path}")
    return record


def train_seed(seed: int, args, identity_hash: str, mod, source: dict,
               raw: dict, matched: dict, seqmeta: dict, s27) -> dict:
    import torch
    s11 = mod["step11"]
    seed_dir = args.output_root / f"seed_{seed}"
    manifest_path = seed_dir / "29_seed_manifest.json"
    if seed_dir.exists():
        if manifest_path.is_file():
            prior = read_json(manifest_path)
            if prior.get("status") == "COMPLETE":
                return verified_complete_seed(seed_dir, seed, identity_hash)
            require(prior.get("identity_hash") == identity_hash and prior.get("seed") == seed,
                    f"Existing incomplete seed belongs to different protocol: {seed_dir}")
            require(args.retry_failed, f"Incomplete Step29 seed {seed}. After stopping prior process, pass --retry-failed.")
            archived = args.output_root / "_incomplete" / (f"seed_{seed}_" +
                         datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
            archived.parent.mkdir(parents=True, exist_ok=True)
            seed_dir.rename(archived)
            print(f"[Archived failed seed {seed}] {archived}", flush=True)
        else:
            raise ValueError(f"Seed directory exists without manifest: {seed_dir}")
    seed_dir.mkdir(parents=True, exist_ok=False)
    write_json(manifest_path, {"status": "RUNNING", "seed": seed,
                               "identity_hash": identity_hash, "started": utc_now()})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        s11.set_all_seeds(seed)
        config = s11.ModelConfig(input_dim=raw["train"]["embeddings"].shape[1],
                                 num_classes=N_CLASSES, sequence_length=L10,
                                 temporal_context_seconds=L10)
        model = s11.TimeSeriesTransformerClassifier(config).to(device)
        criterion = s11.make_cross_entropy_loss(None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=s11.LEARNING_RATE, weight_decay=s11.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
        scaler = s11.make_grad_scaler(device)
        train_loader = s11.make_loader(raw["train"]["embeddings"], matched["train"], s11.BATCH_SIZE, True, device)
        val_loader = s11.make_loader(raw["val"]["embeddings"], matched["val"], s11.BATCH_SIZE, False, device)
        best, best_epoch, bad_epochs, history = None, 0, 0, []
        best_path = seed_dir / "best_L10.pt"
        started = time.monotonic()
        print(f"\n[STEP29 TRAIN seed={seed}] L10 matched cohort train={len(matched['train']['y']):,}, "
              f"val={len(matched['val']['y']):,} device={device}", flush=True)
        for epoch in range(1, s11.MAX_EPOCHS + 1):
            t0 = time.monotonic()
            tr = s11.train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, L10)
            val = s11.evaluate_model(model, val_loader, criterion, device, N_CLASSES,
                                     {i: str(i) for i in range(N_CLASSES)}, desc="matched_L10_val")
            sc = val["summary"]
            scheduler.step(float(sc["macro_f1"]))
            history.append({"epoch": epoch, "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "train_loss": float(tr["loss"]), "train_accuracy": float(tr["accuracy"]),
                            "val_loss": float(sc["loss"]), "val_accuracy": float(sc["accuracy"]),
                            "val_macro_f1": float(sc["macro_f1"]), "epoch_seconds": time.monotonic() - t0})
            print(f"[L10 matched seed={seed}] epoch={epoch}/{s11.MAX_EPOCHS}, "
                  f"val_macro_f1={sc['macro_f1']:.6f}, best_epoch={best_epoch}", flush=True)
            if s11.metric_is_better(sc, best, s11.PRIMARY_SELECTION_METRIC):
                best = {k: float(v) for k, v in sc.items() if isinstance(v, (int, float))}
                best_epoch, bad_epochs = epoch, 0
                s11.save_checkpoint_atomic({
                    "model_state_dict": model.state_dict(), "model_config": asdict(config),
                    "train_config": asdict(s11.TrainConfig(random_seed=seed)),
                    "best_epoch": best_epoch, "best_val_summary": best,
                    "identity_hash": identity_hash, "seed": seed, "schema": SCHEMA,
                    "dataset_id": source["dataset_id"], "source_contract": source,
                    "training_population": "L20-eligible train/val endpoints; last ten embeddings only",
                }, best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= s11.PATIENCE:
                    print(f"[L10 matched seed={seed}] Early stopping epoch {epoch}", flush=True)
                    break
        elapsed = time.monotonic() - started
        require(best_path.is_file(), "Best L10 checkpoint was never created.")
        checkpoint = s11.load_checkpoint_safely(best_path, device)
        require(checkpoint.get("identity_hash") == identity_hash and checkpoint.get("seed") == seed and
                checkpoint.get("model_config") == asdict(config) and checkpoint.get("best_epoch") == best_epoch,
                "Selected L10 checkpoint provenance differs.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        final = s11.evaluate_model(model, val_loader, criterion, device, N_CLASSES,
                                   {i: str(i) for i in range(N_CLASSES)}, desc="matched_L10_val_selected")
        require(math.isclose(final["summary"]["macro_f1"], best["macro_f1"], abs_tol=1e-8),
                "Reloaded matched L10 checkpoint fails best Validation F1 reproduction.")
        write_csv(seed_dir / "history.csv", pd.DataFrame(history))
        write_csv(seed_dir / "val_predictions.csv",
                  predict_frame(final, seqmeta["val"], seed, "val", source, identity_hash))
        write_json(seed_dir / "val_metrics.json", {
            "seed": seed, "best_epoch": best_epoch, "trained_epochs": len(history),
            "training_elapsed_seconds": elapsed, "val": final["summary"],
            "n_train_sequences": len(matched["train"]["y"]),
            "n_val_sequences": len(matched["val"]["y"]),
            "length": L10, "source_L20_cohort": True,
            "selection": "val_macro_f1, Step11 original training protocol",
        })
        contents = ["best_L10.pt", "history.csv", "val_predictions.csv", "val_metrics.json"]
        manifest = {"status": "COMPLETE", "identity_hash": identity_hash, "seed": seed,
                    "finished": utc_now(), "best_epoch": best_epoch, "training_elapsed_seconds": elapsed,
                    "device": str(device), "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    "amp": bool(s11.USE_AMP and device.type == "cuda"),
                    "artifacts": {name: sha256(seed_dir / name) for name in contents}}
        write_json(manifest_path, manifest)
        return verified_complete_seed(seed_dir, seed, identity_hash)
    except BaseException as exc:
        write_json(manifest_path, {"status": "FAILED", "identity_hash": identity_hash,
                                   "seed": seed, "time": utc_now(),
                                   "error": type(exc).__name__ + ": " + str(exc)})
        raise
    finally:
        if "model" in locals():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def evidence_groups(window_audit: pd.DataFrame, seq20: dict, matched10: dict,
                    meta: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    """Audit-only: A recent changed, B old-only changed, C no 20-window change."""
    flags = window_audit.audit_observable_effect_present.to_numpy(dtype=np.int64)
    require(np.isin(flags, (0, 1)).all(), "Nonbinary paired-nominal audit change flags.")
    hit20 = flags[seq20["sequence_indices"]].any(axis=1)
    hit10 = flags[matched10["sequence_indices"]].any(axis=1)
    require(not np.any(hit10 & ~hit20), "Last ten changes not included in twenty-window history.")
    attack = y != 0
    groups = {"A_recent10_change": attack & hit10,
              "B_older11to20_only_change": attack & ~hit10 & hit20,
              "C_no_change_in20": attack & ~hit20,
              "NORMAL": ~attack}
    require(np.all(np.sum(np.stack([x.astype(np.int8) for x in groups.values()]), axis=0) == 1),
            "Audit-only groups do not partition the matched evaluation population.")
    return groups


def summarize_evidence_groups(args, mod, pilot, source, seq20, matched, metadata,
                              raw_audit, identity_hash: str, split: str) -> pd.DataFrame:
    s11 = mod["step11"]
    y = matched[split]["y"]
    meta = metadata[split]
    groups = evidence_groups(raw_audit[split], seq20[split], matched[split], meta, y)
    rows = []
    old_reference = None
    for seed in SEEDS:
        p20 = read_existing_l20_prediction(args, pilot, seq20[split], meta, seed, split, s11)
        p10 = verified_new_prediction(args.output_root / f"seed_{seed}" / f"{split}_predictions.csv",
                                      meta, seed, split, source, identity_hash)
        y = p10.true_label_id.to_numpy(dtype=np.int64)
        hit10 = p10.pred_label_id.to_numpy(dtype=np.int64) == y
        hit20 = p20.pred_label_id.to_numpy(dtype=np.int64) == y
        for name, mask in groups.items():
            eps = int(meta.loc[mask, "episode_id"].nunique())
            n = int(mask.sum())
            require(n > 0, f"Empty Step29 evidence stratum: {name}")
            l10_correct = int(np.count_nonzero(hit10 & mask))
            l20_correct = int(np.count_nonzero(hit20 & mask))
            rows.append({"split": split, "seed": seed, "group": name,
                         "endpoints": n, "episodes": eps,
                         "matched_L10_correct": l10_correct,
                         "original_L20_correct": l20_correct,
                         "matched_L10_correct_rate": float(l10_correct / n),
                         "original_L20_correct_rate": float(l20_correct / n),
                         "L10_wrong_L20_correct": int(np.count_nonzero(~hit10 & hit20 & mask)),
                         "L10_correct_L20_wrong": int(np.count_nonzero(hit10 & ~hit20 & mask)),
                         "delta_correct_rate": float((l20_correct - l10_correct) / n),
                         "definition": "posthoc paired-nominal 22-field projection change; not an oracle"})
        if old_reference is None:
            old_reference = {name: int(mask.sum()) for name, mask in groups.items()}
    if split == "test":
        report = read_json(args.step27_root / "27_evidence_availability_test.json")
        require(old_reference["A_recent10_change"] == report["L10_in_context_change_attack"] and
                old_reference["B_older11to20_only_change"] == report["L20_additional_evidence_attack"] and
                old_reference["A_recent10_change"] + old_reference["B_older11to20_only_change"] ==
                    report["L20_in_context_change_attack"],
                "Step29 grouped audit counts disagree with Step27 Test evidence report.")
    return pd.DataFrame(rows)


def make_zip(root: Path, final: bool) -> Path:
    name = "29_L10_MATCHED_FINAL_TEST_REVIEW.zip" if final else "29_L10_MATCHED_VALIDATION_REVIEW.zip"
    target = root / name
    with tempfile.NamedTemporaryFile(dir=root, suffix=".zip.tmp", delete=False) as stream:
        temp = Path(stream.name)
    try:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(Path(__file__), "source/" + Path(__file__).name)
            common = ["29_run_manifest.json", "29_protocol.json", "29_input_audit.json",
                      "29_val_seed_comparison.csv", "29_val_summary.json"]
            if final:
                common.extend(["29_final_test.json", "29_test_seed_comparison.csv",
                               "29_test_summary.json", "29_test_evidence_groups.csv"])
            for item in common:
                file = root / item
                require(file.is_file(), f"Review artifact missing: {file}")
                archive.write(file, item)
            for seed in SEEDS:
                seed_dir = root / f"seed_{seed}"
                for item in ("29_seed_manifest.json", "history.csv", "val_metrics.json", "val_predictions.csv"):
                    archive.write(seed_dir / item, f"seed_{seed}/{item}")
                if final:
                    archive.write(seed_dir / "test_predictions.csv", f"seed_{seed}/test_predictions.csv")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def verify_cached_step27_population(cache_manifest: dict, cache_report: dict,
                                    windows: dict, seq20: dict, seqmeta: dict,
                                    matched_cohort: dict) -> None:
    """Compare Step27's *all-episode* count with eligible L20 episode counts.

    Step10.process_split_for_length reports ``num_episodes`` for all source
    windows, including episodes shorter than L20. They produce no sequences.
    Step29's matched cohort intentionally counts only eligible episodes.
    Validating those unlike definitions for direct equality is incorrect.
    """
    require(cache_report.get("status") == "COMPLETE" and
            cache_report.get("identity") == cache_manifest.get("identity"),
            "Step27 L20 sequence report/manifest identity mismatch.")
    for split in SPLITS:
        recorded = cache_manifest.get("splits", {}).get(split, {})
        described = cache_report.get("splits", {}).get(split, {})
        cohort = matched_cohort[split]
        window_meta = windows[split]
        sequence_meta = seqmeta[split]
        arrays = seq20[split]

        # Independent reconstruction from Step9's authenticated window rows.
        # Episode metadata are only used for this audit, never model features.
        sizes = window_meta.groupby("episode_id", sort=False).size()
        short = sizes[sizes < L20]
        eligible = sizes[sizes >= L20]
        total_episodes = int(len(sizes))
        skipped_episodes = int(len(short))
        skipped_windows = int(short.sum())
        eligible_episodes = int(len(eligible))
        expected_sequences = int((eligible - L20 + 1).sum())

        actual_episode_ids = set(sequence_meta["episode_id"].astype(str).unique())
        expected_episode_ids = set(map(str, eligible.index.tolist()))
        require(actual_episode_ids == expected_episode_ids,
                f"{split}: L20 sequence episodes disagree with eligible Step9 episodes.")

        require(recorded.get("num_episodes") == described.get("num_episodes") == total_episodes,
                f"{split}: Step27 all-episode count differs from Step9 windows "
                f"(cached={recorded.get('num_episodes')}, actual={total_episodes}).")
        require(described.get("skipped_short_episodes") == skipped_episodes and
                described.get("skipped_short_windows") == skipped_windows,
                f"{split}: Step27 short-episode exclusion count differs from Step9 windows.")
        require(cohort["n_episodes"] == eligible_episodes == total_episodes - skipped_episodes,
                f"{split}: incorrect matched L20-eligible episode count "
                f"(eligible={eligible_episodes}, matched={cohort['n_episodes']}).")
        require(recorded.get("num_sequences") == described.get("num_sequences") ==
                cohort["n_endpoints"] == expected_sequences == len(arrays["y"]) == len(sequence_meta),
                f"{split}: Step27/Step29 L20 sequence population mismatch "
                f"(cache={recorded.get('num_sequences')}, matched={cohort['n_endpoints']}, "
                f"reconstructed={expected_sequences}).")
        require(described.get("input_windows") == len(window_meta),
                f"{split}: Step27 input-window count differs from Step9 metadata.")
        require(cohort["endpoint_sha256"] == index_sha(arrays["end_row_indices"]),
                f"{split}: Step29 L20 endpoint SHA-256 differs from Step27 cached arrays.")
        print(f"[Verified Step27 {split}] {total_episodes:,} total episodes "
              f"- {skipped_episodes:,} short = {eligible_episodes:,} L20-eligible; "
              f"{expected_sequences:,} matched endpoints", flush=True)


def prepare_context(args):
    step27_path = choose_step27(args.step27)
    require(step27_path.parent == BASE, "Step27 must be alongside Step29 to avoid mixed project sources.")
    s27 = import_path("taes29_step27", step27_path)
    proxy = SimpleNamespace(step10=args.step10, step11=args.step11, step12=args.step12,
                            output_root=args.step27_root,
                            baseline_multiseed_root=args.baseline_multiseed_root,
                            baseline_main_root=args.baseline_main_root,
                            seeds=list(SEEDS))
    paths, mod = s27.load_source_modules(proxy)
    protect_inputs(args, mod["step11"])
    extraction, report_index, source = s27.authenticate_inputs(proxy, mod)
    require(source["dataset_id"] == EXPECTED_DATASET_ID, "Unexpected benchmark dataset ID.")
    pilot, cache_manifest, seed_records = validate_step27(args, s27, paths, mod, source)
    raw, windows, audit, window_checks, input_checks = s27.get_raw_splits(mod, extraction)
    # Step27 history/sidecar storage is immutable; load_cache also validates the
    # causal episode and sequence metadata against original Step9 rows.
    sequences, seq_metadata, seq_effects = s27.load_cache(args.step27_root / "sequence_cache", mod, windows, raw)
    matched, report = build_matched_l10(mod, raw, sequences, seq_metadata)
    # Confirm exact same Step27 eligible populations and end index digests.
    val_summary = read_json(args.step27_root / "27_val_summary.json")
    require(report["val"]["endpoint_sha256"] == val_summary["endpoint_sha256"],
            "Step27 validation endpoint hash mismatch.")
    cache_report = read_json(args.step27_root / "sequence_cache" / "27_L20_sequence_report.json")
    # The Step27 cache includes ALL episodes (including short ones), whereas
    # matched L20 sequences include only episodes with >=20 windows.
    # Audit both populations without conflating their definitions.
    verify_cached_step27_population(cache_manifest, cache_report, windows,
                                    sequences, seq_metadata, report)
    return dict(s27=s27, paths=paths, mod=mod, source=source, pilot=pilot,
                step27_seed_records=seed_records, raw=raw, windows=windows, audit=audit,
                l20=sequences, meta=seq_metadata, matched=matched, cohort=report)


def preflight(ctx):
    import torch
    s11 = ctx["mod"]["step11"]
    seq = ctx["matched"]["val"]
    s11.set_all_seeds(SEEDS[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = s11.ModelConfig(input_dim=ctx["raw"]["val"]["embeddings"].shape[1],
                                num_classes=N_CLASSES, sequence_length=L10, temporal_context_seconds=L10)
    model = s11.TimeSeriesTransformerClassifier(model_cfg).to(device)
    batch = torch.from_numpy(ctx["raw"]["val"]["embeddings"][seq["sequence_indices"][:2]]).to(device)
    with torch.no_grad():
        output = model(batch)
    require(tuple(output.shape) == (2, N_CLASSES) and torch.isfinite(output).all().item(),
            "Step29 L10 TST forward test failed.")
    for seed in SEEDS:
        read_existing_l20_prediction(ctx["args"], ctx["pilot"], ctx["l20"]["val"],
                                     ctx["meta"]["val"], seed, "val", s11)
    print("\n" + "=" * 72)
    print("STEP 29 PREFLIGHT PASS: NO MODEL TRAINED, NO OUTPUT WRITTEN")
    print(f"Dataset: {ctx['source']['dataset_id']}")
    for split, stats in ctx["cohort"].items():
        print(f"{split}: {stats['n_endpoints']:,} endpoints, {stats['n_episodes']:,} episodes")
    print(f"TST L10 forward: {tuple(output.shape)} on {device}; 3 saved L20 validation seeds verified.")
    print("Train/val/test indices are 100% matched to Step27 L20 endpoints.")
    print("=" * 72)


def validation(args, ctx, identity: dict, identity_hash: str):
    s11, s27 = ctx["mod"]["step11"], ctx["s27"]
    root = args.output_root
    with ExclusiveLock(root):
        run_path = root / "29_run_manifest.json"
        old = read_json(run_path) if run_path.is_file() else None
        if old is None:
            require(not any(path.name != ".29_active.lock" for path in root.iterdir()),
                    "Step29 output folder contains foreign data with no manifest; use new --output-root.")
        else:
            require(old.get("identity_hash") == identity_hash and old.get("identity") == identity,
                    "Existing Step29 output has different code, source data, runtime or protocol.")
            require(not (root / "29_final_test.json").exists(),
                    "Final Test has been run/attempted; validation cannot be retrained afterward.")
            if old.get("status") == "COMPLETE":
                for key, name in (("val_summary_sha256", "29_val_summary.json"),
                                  ("val_comparison_sha256", "29_val_seed_comparison.csv"),
                                  ("protocol_sha256", "29_protocol.json"),
                                  ("input_audit_sha256", "29_input_audit.json")):
                    require(old.get(key) == sha256(root / name), f"Verified COMPLETE Step29 {name} changed.")
                for seed in SEEDS:
                    verified_complete_seed(root / f"seed_{seed}", seed, identity_hash)
                print(f"[Step29 completed validation reused] {root}")
                print(f"Review ZIP: {make_zip(root, final=False)}")
                return
        manifest = old or {"schema": SCHEMA, "identity": identity, "identity_hash": identity_hash,
                           "seeds": list(SEEDS), "started_at": utc_now()}
        manifest.update(status="RUNNING", last_error=None)
        write_json(run_path, manifest)
        try:
            if not (root / "29_protocol.json").exists():
                write_json(root / "29_protocol.json", {
                    "identity_hash": identity_hash, "experiment": SCHEMA,
                    "stage": "validation", "exploratory": True,
                    "selection": "val_macro_f1 of matched L10 checkpoint only; no test used to select epoch",
                    "training_population": "Exactly Step27 L20-eligible training endpoints, with last-ten embedding inputs",
                    "validation_population": "Exactly Step27 L20-eligible validation endpoints",
                    "comparison": "Stored Step27 L20 validation-selected head versus new matched L10 head",
                    "test_note": "L20 Test already inspected before this follow-up; not new independent confirmation",
                    "fixed_seeds": list(SEEDS), "different_model_lengths_not_paired_initialization": True,
                    "bootstrap": "No inferential bootstrap in Step29; Step28's original L10 bootstrap does not apply to matched L10",
                })
            if not (root / "29_input_audit.json").exists():
                write_json(root / "29_input_audit.json", {
                    "data": ctx["source"], "cohort": ctx["cohort"],
                    "step27_identity_hash": ctx["pilot"]["identity_hash"],
                    "step27_manifest_sha256": sha256(args.step27_root / "27_run_manifest.json"),
                    "step27_seed_records": ctx["step27_seed_records"],
                    "source_files_sha256": {k: sha256(v) for k, v in ctx["paths"].items()},
                    "step29_source_sha256": sha256(Path(__file__)),
                    "test_prediction_access": "No Step27 test prediction was scored during validation",
                })
            manifest.update(protocol_sha256=sha256(root / "29_protocol.json"),
                            input_audit_sha256=sha256(root / "29_input_audit.json"))
            write_json(run_path, manifest)
            for seed in SEEDS:
                read_existing_l20_prediction(args, ctx["pilot"], ctx["l20"]["val"],
                                             ctx["meta"]["val"], seed, "val", s11)
                train_seed(seed, args, identity_hash, ctx["mod"], ctx["source"],
                           ctx["raw"], ctx["matched"], ctx["meta"], s27)
            rows = []
            for seed in SEEDS:
                verified_complete_seed(root / f"seed_{seed}", seed, identity_hash)
                p10 = verified_new_prediction(root / f"seed_{seed}/val_predictions.csv",
                                              ctx["meta"]["val"], seed, "val", ctx["source"], identity_hash)
                p20 = read_existing_l20_prediction(args, ctx["pilot"], ctx["l20"]["val"],
                                                  ctx["meta"]["val"], seed, "val", s11)
                row = compare_model_predictions(p20, p10, ctx["meta"]["val"], seed, "val", s11)
                m = read_json(root / f"seed_{seed}/val_metrics.json")
                require(math.isclose(row["matched_L10_macro_f1"], m["val"]["macro_f1"], abs_tol=1e-10),
                        "Step29 predictions do not reproduce selected checkpoint macro-F1.")
                row.update({key: m[key] for key in ("best_epoch", "trained_epochs", "training_elapsed_seconds")})
                rows.append(row)
            frame, summary = summarize(rows, ctx["cohort"], "val", ctx["source"], identity_hash)
            saved_l20 = read_json(args.step27_root / "27_val_summary.json")
            require(math.isclose(summary["original_L20_macro_f1_mean"],
                                 saved_l20["L20_macro_f1_mean"], abs_tol=1e-10),
                    "Selected original L20 Validation mean differs from Step27 reported mean.")
            write_csv(root / "29_val_seed_comparison.csv", frame)
            write_json(root / "29_val_summary.json", summary)
            manifest.update(status="COMPLETE", completed_at=utc_now(),
                            val_summary_sha256=sha256(root / "29_val_summary.json"),
                            val_comparison_sha256=sha256(root / "29_val_seed_comparison.csv"))
            write_json(run_path, manifest)
            review = make_zip(root, final=False)
        except BaseException as exc:
            manifest.update(status="FAILED", last_error=type(exc).__name__ + ": " + str(exc),
                            updated_at=utc_now())
            write_json(run_path, manifest)
            raise
    print("\nSTEP 29 VALIDATION COMPLETE -- TEST NOT SCORED")
    print(frame[["seed", "endpoints", "matched_L10_macro_f1", "original_L20_macro_f1",
                 "delta_L20_minus_matched_L10_macro_f1", "best_epoch", "training_elapsed_seconds"]].to_string(index=False))
    print(f"Mean matched L10 F1={summary['matched_L10_macro_f1_mean']:.6f}; "
          f"stored L20 F1={summary['original_L20_macro_f1_mean']:.6f}; "
          f"delta={summary['delta_L20_minus_matched_L10_macro_f1_mean']:+.6f}")
    print(f"Review ZIP: {review}")
    print("Only after reviewing validation, run --final-test; do not retrain after test inspection.")


def validate_27_final_archive(args):
    final = read_json(args.step27_root / "27_final_test.json")
    require(final.get("status") == "COMPLETE", "Step27 final Test was not COMPLETE.")
    pilot = read_json(args.step27_root / "27_run_manifest.json")
    require(final.get("identity_hash") == pilot.get("identity_hash") and
            final.get("val_summary_sha256") == pilot.get("val_summary_sha256"),
            "Step27 Test does not belong to the same validation-selected heads.")
    require(final.get("test_summary_sha256") == sha256(args.step27_root / "27_test_summary.json") and
            final.get("test_comparison_sha256") == sha256(args.step27_root / "27_test_seed_comparison.csv"),
            "Step27 original final Test reports were modified.")
    return final


def final_test(args, ctx, identity_hash: str):
    import torch
    root = args.output_root
    run = read_json(root / "29_run_manifest.json")
    require(run.get("status") == "COMPLETE" and run.get("identity_hash") == identity_hash,
            "Final Test requires completed and identical Step29 Validation stage.")
    for key, path in (("val_summary_sha256", "29_val_summary.json"),
                      ("val_comparison_sha256", "29_val_seed_comparison.csv"),
                      ("protocol_sha256", "29_protocol.json"),
                      ("input_audit_sha256", "29_input_audit.json")):
        require(run.get(key) == sha256(root / path), f"Frozen Step29 validation artifact modified: {path}")
    for seed in SEEDS:
        verified_complete_seed(root / f"seed_{seed}", seed, identity_hash)
    validate_27_final_archive(args)  # test data only after explicit --final-test
    s11 = ctx["mod"]["step11"]
    with ExclusiveLock(root):
        path = root / "29_final_test.json"
        if path.exists():
            prior = read_json(path)
            require(prior.get("identity_hash") == identity_hash,
                    "Another Step29 final test has used this output folder.")
            if prior.get("status") == "COMPLETE":
                require(prior.get("test_summary_sha256") == sha256(root / "29_test_summary.json") and
                        prior.get("test_comparison_sha256") == sha256(root / "29_test_seed_comparison.csv") and
                        prior.get("test_group_sha256") == sha256(root / "29_test_evidence_groups.csv"),
                        "Completed Step29 Test output was altered.")
                print(f"[Step29 final Test already COMPLETE] {root}")
                print(f"Review ZIP: {make_zip(root, final=True)}")
                return
            raise ValueError("An incomplete Test stage exists. Preserve it; do not rerun Test on this output root.")
        write_json(path, {"status": "RUNNING", "identity_hash": identity_hash,
                          "unlocked_at": utc_now(),
                          "selection_frozen_before_test": True,
                          "test_already_inspected_for_step27": True})
        try:
            rows = []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for seed in SEEDS:
                seed_dir = root / f"seed_{seed}"
                model_record = verified_complete_seed(seed_dir, seed, identity_hash)
                checkpoint = s11.load_checkpoint_safely(seed_dir / "best_L10.pt", device)
                require(checkpoint.get("identity_hash") == identity_hash and
                        checkpoint.get("seed") == seed and checkpoint.get("schema") == SCHEMA and
                        checkpoint.get("best_epoch") == model_record["best_epoch"],
                        "Validation-selected L10 checkpoint identity mismatch.")
                config = s11.ModelConfig(**checkpoint["model_config"])
                require(config.sequence_length == L10 and config.num_classes == N_CLASSES and
                        config.input_dim == ctx["raw"]["test"]["embeddings"].shape[1],
                        "Saved L10 model config incompatible with test inputs.")
                network = s11.TimeSeriesTransformerClassifier(config).to(device)
                network.load_state_dict(checkpoint["model_state_dict"], strict=True)
                loader = s11.make_loader(ctx["raw"]["test"]["embeddings"], ctx["matched"]["test"],
                                         s11.BATCH_SIZE, False, device)
                evaluation = s11.evaluate_model(network, loader, s11.make_cross_entropy_loss(None),
                                                device, N_CLASSES, {i: str(i) for i in range(N_CLASSES)},
                                                desc=f"Step29_matched_L10_test_seed{seed}")
                preds = predict_frame(evaluation, ctx["meta"]["test"], seed, "test", ctx["source"], identity_hash)
                write_csv(seed_dir / "test_predictions.csv", preds)
                p10 = verified_new_prediction(seed_dir / "test_predictions.csv", ctx["meta"]["test"],
                                              seed, "test", ctx["source"], identity_hash)
                p20 = read_existing_l20_prediction(args, ctx["pilot"], ctx["l20"]["test"],
                                                  ctx["meta"]["test"], seed, "test", s11)
                rows.append(compare_model_predictions(p20, p10, ctx["meta"]["test"], seed, "test", s11))
                del network
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            frame, summary = summarize(rows, ctx["cohort"], "test", ctx["source"], identity_hash)
            saved_l20 = read_json(args.step27_root / "27_test_summary.json")
            require(math.isclose(summary["original_L20_macro_f1_mean"],
                                 saved_l20["L20_macro_f1_mean"], abs_tol=1e-10),
                    "Reused original L20 Test mean differs from Step27 reported mean.")
            group = summarize_evidence_groups(args, ctx["mod"], ctx["pilot"], ctx["source"],
                                             ctx["l20"], ctx["matched"], ctx["meta"], ctx["audit"],
                                             identity_hash, "test")
            write_csv(root / "29_test_seed_comparison.csv", frame)
            write_csv(root / "29_test_evidence_groups.csv", group)
            summary["evidence_group_file"] = "29_test_evidence_groups.csv"
            summary["evidence_group_notes"] = (
                "Attack-only groups A/B/C are disjoint; NORMAL is separate. "
                "Correct-scenario rate is NOT within-stratum six-class macro-F1. "
                "Audit uses paired-nominal changes and the grouping is posthoc; exploratory only.")
            write_json(root / "29_test_summary.json", summary)
            write_json(path, {"status": "COMPLETE", "identity_hash": identity_hash,
                              "completed_at": utc_now(), "selected_on_val_only": True,
                              "test_summary_sha256": sha256(root / "29_test_summary.json"),
                              "test_comparison_sha256": sha256(root / "29_test_seed_comparison.csv"),
                              "test_group_sha256": sha256(root / "29_test_evidence_groups.csv"),
                              "test_prediction_sha256": {str(seed): sha256(root / f"seed_{seed}/test_predictions.csv")
                                                         for seed in SEEDS}})
            review = make_zip(root, final=True)
        except BaseException as exc:
            # Retain incomplete results without permitting test-driven reruns.
            write_json(path, {"status": "FAILED", "identity_hash": identity_hash,
                              "time": utc_now(), "error": type(exc).__name__ + ": " + str(exc)})
            raise
    print("\nSTEP29 EXPLICIT FINAL TEST COMPLETE")
    print(frame[["seed", "endpoints", "matched_L10_macro_f1", "original_L20_macro_f1",
                 "delta_L20_minus_matched_L10_macro_f1"]].to_string(index=False))
    print(f"Matched L20 minus matched L10 (mean of 3 heads): "
          f"{summary['delta_L20_minus_matched_L10_macro_f1_mean']:+.6f}")
    print("Evidence groups saved for A (recent change), B (old-only change), C (no 20-window change).")
    print(f"Review ZIP: {review}")
    print("Exploratory follow-up designed after previous L20 Test inspection; no new independent holdout.")


def main() -> None:
    args = arguments()
    context = prepare_context(args)
    context["args"] = args
    identity = training_identity(args, context["paths"], context["source"], context["pilot"],
                                 context["step27_seed_records"], context["cohort"], context["mod"]["step11"])
    identity_hash = object_sha(identity)
    print("=" * 72)
    print("STEP29 matched L10 TST control versus previously trained Step27 L20")
    print("MODE:", "PREFLIGHT" if args.preflight else "FINAL TEST" if args.final_test else "VALIDATION TRAIN")
    print("Fixed seeds:", list(SEEDS), "| dataset ID:", context["source"]["dataset_id"])
    print("Step27 root:", args.step27_root, "| new Step29 root:", args.output_root)
    if args.preflight:
        preflight(context)
    elif args.final_test:
        final_test(args, context, identity_hash)
    else:
        validation(args, context, identity, identity_hash)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[STEP29 STOPPED] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
