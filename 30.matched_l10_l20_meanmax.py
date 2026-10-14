#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 30: matched-cohort L10/L20 Mean+Max pooling + MLP control.

Run next to the ORIGINAL Steps 10, 11, 12, 13 and 27 in the TAES project.
Original datasets, embeddings, checkpoints, Step 27/29 results and Python
sources are READ-ONLY; all new artifacts go to a separate Step 30 folder.

    python "30.matched_l10_l20_meanmax.py" --version
    python "30.matched_l10_l20_meanmax.py" --preflight
    python "30.matched_l10_l20_meanmax.py"
    python "30.matched_l10_l20_meanmax.py" --retry-failed  # only after old process stops
    python "30.matched_l10_l20_meanmax.py" --final-test
    python "30.matched_l10_l20_meanmax.py" --final-test --retry-final-test  # if prior test stopped

Training: exactly 2 lengths x 3 fixed head seeds (20260610--20260612).
Both lengths use Step27's L20-eligible train and validation endpoints.
L10 is L20_sequence_indices[:, -10:] and MUST match the historical Step10
L10 sequence at every selected endpoint. No new encodings/sequences are trained.

Reuse, without modifying, Step13's:
  - float32 mean+max feature extraction (concatenated 768+768 = 1536),
  - two-layer LayerNorm/Linear/GELU/Dropout/Linear MLP,
  - per-seed initialization, original AdamW, loss, scheduler, optimizer,
  - 30 epochs, patience 7, batch 256, lr 7e-5, weight decay 5e-4,
    label smoothing 0.03, maximum validation macro-F1 checkpoint choice.

Validation training NEVER scores or predicts TEST; --final-test is an explicit
separate action which only reloads frozen validation-selected checkpoints.
Note that L20's Test performance was already inspected before this experiment
was designed. This is an exploratory matched-population extension, not a new
independent confirmation. Identically numbered seeds are NOT paired model
initializations; paired statistical inference must resample EPISODES.

Protocol decisions:
  - No hyperparameter or model selection on Test.
  - No audit/label/episode metadata reaches the classifier features.
  - Endpoints and labels are identical for both lengths; MLP architecture is
    identical at both lengths, unlike length-dependent TST positional vectors.
  - Keep all train/val endpoints eligible at L20, with stride 1; no augmentation.
  - Per-seed completion is hash-authenticated; failed seeds may be archived and
    restarted from epoch one; finished seeds are reused only after verification.
  - No large pooled-feature arrays are saved to disk; one length at a time.
  - Read historical TST L10/L20 summaries only for four-row context comparison,
    never to select a pooling checkpoint or alter its protocol.

Output:
  outputs/shared_nominal_v2/30_MATCHED_MEANMAX_L10_L20/
      30_run_manifest.json, 30_protocol.json, 30_input_audit.json,
      L10/seed_*/best_meanmax.pt, history.csv, val_predictions.csv,...
      L20/seed_*/best_meanmax.pt, history.csv, val_predictions.csv,...
      30_val_seed_results.csv, 30_val_2x2.csv, 30_val_summary.json,
      30_MATCHED_MEANMAX_VALIDATION_REVIEW.zip
  --final-test adds test predictions, 30_test_2x2.csv and
      30_MATCHED_MEANMAX_FINAL_TEST_REVIEW.zip.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any
import zipfile

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DEFAULT_27 = BASE / "outputs/shared_nominal_v2/27_L20_TST_VALIDATION_PILOT"
DEFAULT_29 = BASE / "outputs/shared_nominal_v2/29_L10_MATCHED_TST_CONTROL"
DEFAULT_30 = BASE / "outputs/shared_nominal_v2/30_MATCHED_MEANMAX_L10_L20"
DEFAULT_MAIN = BASE / "outputs/shared_nominal_v2/tst_training_finetuned_main"
DEFAULT_MULTI = BASE / "outputs/shared_nominal_v2/tst_training_finetuned_multiseed"
DEFAULT_13_SUMMARY = BASE / "outputs/shared_nominal_v2/39_pooling_mlp_finetuned_main/39_pooling_mlp_summary.json"
SEEDS = (20260610, 20260611, 20260612)
LENGTHS = (10, 20)
SPLITS = ("train", "val", "test")
SCHEMA = "taes_step30_matched_l10_l20_meanmax_v1"
VERSION = "STEP30_MATCHED_MEANMAX_20261013_V1"
EXPECTED_DATASET_ID = "478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b"
EXPECTED_LABELS = (
    "NORMAL", "RF_LINK_JAMMING", "GNSS_SPOOFING",
    "UNAUTHORIZED_TELECOMMAND_INJECTION", "TELECOMMAND_REPLAY",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE",
)


def require(ok: Any, message: str) -> None:
    if not bool(ok):
        raise ValueError(message)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def objsha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def arrsha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<i8").tobytes()).hexdigest()


def endsha(a: np.ndarray) -> str:
    return arrsha(np.sort(np.asarray(a, dtype=np.int64)))


def read_json(path: Path) -> dict:
    require(path.is_file(), f"Required JSON file not found: {path}")
    item = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(item, dict), f"Expected a JSON object: {path}")
    return item


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as f:
        temp = Path(f.name)
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent,
                                     suffix=".tmp", delete=False) as f:
        temp = Path(f.name)
        frame.to_csv(f, index=False)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def source(name: str, candidates: tuple[str, ...], explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        require(p.is_file(), f"{name} source file missing: {p}")
        return p
    found = [(BASE / n).resolve() for n in candidates if (BASE / n).is_file()]
    require(found, f"{name} source missing beside Step30: {candidates}")
    require(len({sha(p) for p in found}) == 1,
            f"Different versions of {name} found: {found}; select via explicit --{name}.")
    return found[0]


def import_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=VERSION)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--preflight", action="store_true", help="Verify sources/data and one MeanMax forward pass; no outputs.")
    g.add_argument("--final-test", action="store_true", help="Read frozen 6 checkpoints and evaluate held-out test only.")
    p.add_argument("--retry-failed", action="store_true", help="Only after process stopped: archive incomplete training seed and restart.")
    p.add_argument("--retry-final-test", action="store_true", help="Only after stopped FAILED test: archive partial Test files, rerun inference (no training).")
    p.add_argument("--output-root", type=Path, default=DEFAULT_30)
    p.add_argument("--step27-root", type=Path, default=DEFAULT_27)
    p.add_argument("--step29-root", type=Path, default=DEFAULT_29)
    p.add_argument("--step13-summary", type=Path, default=DEFAULT_13_SUMMARY)
    p.add_argument("--step13", type=Path, default=None)
    p.add_argument("--step27", type=Path, default=None)
    p.add_argument("--step10", type=str, default=None)
    p.add_argument("--step11", type=str, default=None)
    p.add_argument("--step12", type=str, default=None)
    p.add_argument("--baseline-main-root", type=Path, default=DEFAULT_MAIN)
    p.add_argument("--baseline-multiseed-root", type=Path, default=DEFAULT_MULTI)
    a = p.parse_args()
    require(not (a.retry_failed and (a.preflight or a.final_test)),
            "--retry-failed applies only to validation training, not preflight/final-test.")
    require(not a.retry_final_test or a.final_test,
            "--retry-final-test must be used together with --final-test.")
    for key in ("output_root", "step27_root", "step29_root", "step13_summary",
                "baseline_main_root", "baseline_multiseed_root"):
        setattr(a, key, getattr(a, key).expanduser().resolve())
    return a


class RunLock:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / ".30_active.lock"

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "hostname": socket.gethostname(), "at": utc()}, f)
        except FileExistsError as e:
            raise RuntimeError(f"Step30 lock exists: {self.path}. Verify the previous process stopped.") from e
        return self

    def __exit__(self, exc_type, exc, tb):
        self.path.unlink(missing_ok=True)


def check_protected(a, s13, s27_mod) -> None:
    output = a.output_root
    require(output != BASE and not BASE.is_relative_to(output), "Output cannot contain project source directory.")
    protect = (a.step27_root, a.step29_root, a.step13_summary.parent, a.baseline_main_root,
               a.baseline_multiseed_root, s13.EMBEDDING_DIR, s13.SEQUENCE_DIR,
               s13.EXPECTED_FINETUNED_MODEL_DIR.parent, s27_mod["step11"].EMBEDDING_DIR)
    for p in protect:
        p = p.resolve()
        require(not (output == p or output.is_relative_to(p) or p.is_relative_to(output)),
                f"Step30 output overlaps original input or another experiment: {p}")


def check_step13_config(s13, report: dict) -> None:
    require(s13.POOLING_MODES == ("mean", "max", "meanmax") and s13.SEQUENCE_LENGTH == 10,
            "Original Step13 pooling protocol was changed.")
    require(s13.SEEDS == tuple(range(20260610, 20260620)), "Original Step13 seed protocol changed.")
    expected = {
        "MAX_EPOCHS": 30, "PATIENCE": 7, "BATCH_SIZE": 256,
        "LEARNING_RATE": 7e-5, "WEIGHT_DECAY": 5e-4, "GRAD_CLIP_NORM": 1.0,
        "HIDDEN_DIM": 256, "MLP_LAYERS": 2, "DROPOUT": 0.25,
        "USE_INPUT_LAYERNORM": True, "USE_STANDARDIZATION": False,
        "LABEL_SMOOTHING": .03, "POOLING_CHUNK_SIZE": 4096,
    }
    for key, value in expected.items():
        require(getattr(s13, key) == value, f"Original Step13 {key} differs from locked value {value}.")
    manifest = report.get("manifest", {})
    require(manifest.get("pooling_modes") == list(s13.POOLING_MODES) and
            manifest.get("seeds") == list(s13.SEEDS) and manifest.get("sequence_length") == 10,
            "Historical Step13 summary has a different pooling/seed/length protocol.")
    saved = manifest.get("train_config", {})
    fields = {"MAX_EPOCHS": "max_epochs", "PATIENCE": "patience", "BATCH_SIZE": "batch_size",
              "LEARNING_RATE": "learning_rate", "WEIGHT_DECAY": "weight_decay",
              "GRAD_CLIP_NORM": "grad_clip_norm", "HIDDEN_DIM": "hidden_dim",
              "MLP_LAYERS": "mlp_layers", "DROPOUT": "dropout",
              "USE_INPUT_LAYERNORM": "use_input_layernorm", "USE_STANDARDIZATION": "use_standardization"}
    for key, field in fields.items():
        require(saved.get(field) == expected[key], f"Step13 summary training config mismatch: {field}")
    require(manifest.get("label_smoothing") == .03 and
            tuple(manifest.get("revised_pipeline_provenance", {}).get("step10_sequence_lengths", [])) == (1, 2, 3, 5, 10),
            "Historical Step13 summary uses a different training/sequence policy.")
    require(s13.MLPClassifier.__module__ == s13.__name__ and s13.SAVE_MODEL_CHECKPOINTS and s13.SAVE_PREDICTIONS,
            "Step13 model source or saved-artifact protocol changed.")


def verify_prior_experiments(a, s27, paths: dict, source_contract: dict) -> dict:
    """Read-only Step27/29 identity/hash checks, with NO predictions/TEST scoring."""
    r27 = a.step27_root
    pilot = read_json(r27 / "27_run_manifest.json")
    require(pilot.get("status") == "COMPLETE" and pilot.get("schema") == s27.SCHEMA,
            "Step27 validation pilot is not a completed expected run.")
    identity = pilot.get("identity", {})
    require(pilot.get("identity_hash") == s27.digest_json(identity) and
            identity.get("source_contract") == source_contract and
            identity.get("selected_seeds") == list(SEEDS) and
            identity.get("reference_length") == 10 and identity.get("new_length") == 20,
            "Step27 run identity/dataset/seeds/lengths differ from Step30 protocol.")
    for key, p in (("pilot_source_sha256", Path(s27.__file__)),
                   ("step10_source_sha256", paths["step10"]),
                   ("step11_source_sha256", paths["step11"]),
                   ("step12_source_sha256", paths["step12"])):
        require(identity.get(key) == sha(p), f"Step27 source provenance mismatch: {key}")
    for field, name in (("val_summary_sha256", "27_val_summary.json"),
                        ("val_comparison_sha256", "27_val_seed_comparison.csv"),
                        ("L20_sequences_sha256", "sequence_cache/27_L20_sequence_manifest.json")):
        require(pilot.get(field) == sha(r27 / name), f"Step27 output hash changed: {name}")
    cache = r27 / "sequence_cache"
    manifest = read_json(cache / "27_L20_sequence_manifest.json")
    require(manifest.get("status") == "COMPLETE" and
            manifest.get("identity") == s27.sequence_cache_identity(source_contract, paths),
            "Step27 sequence cache provenance differs from Step30 inputs.")
    required = {"27_L20_sequence_report.json"}
    for split in SPLITS:
        required.update((f"{split}_tst_sequences_L20.npz",
                         f"{split}_tst_sequence_metadata_L20.csv",
                         f"{split}_tst_sequence_observable_effects_L20.csv"))
    require(required.issubset(manifest.get("artifacts", {})), "Step27 L20 sequence cache is incomplete.")
    for name, digest in manifest["artifacts"].items():
        p = (cache / name).resolve()
        require(p.is_relative_to(cache.resolve()) and p.is_file() and sha(p) == digest,
                f"Step27 cache artifact changed: {p}")
    seeds = {}
    for seed in SEEDS:
        verified = s27.verify_seed_complete(r27 / f"seed_{seed}", pilot["identity_hash"], seed)
        seeds[str(seed)] = {"seed_manifest_sha256": sha(r27 / f"seed_{seed}/27_seed_manifest.json"),
                            "best_epoch": verified["best_epoch"]}
    v27 = read_json(r27 / "27_val_summary.json")
    require(v27.get("identity_hash") == pilot["identity_hash"] and
            v27.get("dataset_id") == source_contract["dataset_id"] and
            v27.get("num_head_seeds") == 3,
            "Step27 validation summary does not match its completed run.")

    r29 = a.step29_root
    prior29 = read_json(r29 / "29_run_manifest.json")
    require(prior29.get("status") == "COMPLETE" and
            prior29.get("identity_hash") == objsha(prior29.get("identity", {})) and
            prior29["identity"].get("source_contract") == source_contract and
            prior29["identity"].get("step27_identity_hash") == pilot["identity_hash"] and
            prior29["identity"].get("locked_seeds") == list(SEEDS),
            "Step29 matched L10 TST is not a comparable completed validation run.")
    require(prior29["identity"].get("step27_run_manifest_sha256") == sha(r27 / "27_run_manifest.json"),
            "Step29 referenced a different Step27 manifest.")
    for key, name in (("val_summary_sha256", "29_val_summary.json"),
                      ("val_comparison_sha256", "29_val_seed_comparison.csv"),
                      ("protocol_sha256", "29_protocol.json"),
                      ("input_audit_sha256", "29_input_audit.json")):
        require(prior29.get(key) == sha(r29 / name), f"Step29 validation artifact changed: {name}")
    v29 = read_json(r29 / "29_val_summary.json")
    require(v29.get("identity_hash") == prior29["identity_hash"] and
            v29.get("dataset_id") == source_contract["dataset_id"] and
            v29.get("head_seeds") == list(SEEDS) and v29.get("num_seeds") == 3,
            "Step29 saved Validation results have incorrect provenance.")
    require(v29.get("endpoint_sha256") == v27.get("endpoint_sha256") and
            v29.get("endpoints") == v27.get("common_endpoints"),
            "Step27 and Step29 validation cohorts do not match.")
    return {"step27": pilot, "step27_cache": manifest, "step27_val": v27,
            "step29": prior29, "step29_val": v29, "step27_seeds": seeds}


def assemble_sequences(raw: dict, meta: dict, windows: dict, seq20: dict, s11) -> tuple[dict, dict]:
    sequences, audit = {}, {}
    ep_sets = {}
    for split in SPLITS:
        s20 = seq20[split]
        indices20 = s20["sequence_indices"]
        ends = s20["end_row_indices"]
        labels = s20["y"]
        require(indices20.ndim == 2 and indices20.shape[1] == 20 and
                np.array_equal(indices20[:, -1], ends) and
                len(np.unique(ends)) == len(ends), f"Invalid L20 sequence coverage: {split}")
        indices10 = np.ascontiguousarray(indices20[:, -10:].astype(np.int64, copy=False))
        historical = s11.load_sequence_split(split, 10, raw[split], raw[split]["embeddings"].shape[1])
        lookup = pd.Index(historical["end_row_indices"])
        require(lookup.is_unique, f"Historical L10 endpoint duplicated in {split}")
        pos = lookup.get_indexer(ends)
        require((pos >= 0).all() and
                np.array_equal(historical["sequence_indices"][pos], indices10) and
                np.array_equal(historical["y"][pos], labels),
                f"L20[-10:] != original verified L10 at the same {split} endpoints.")
        m = meta[split]
        require(len(m) == len(labels) and
                np.array_equal(m["sequence_end_row"].to_numpy(np.int64), ends) and
                np.array_equal(m["label_id"].to_numpy(np.int64), labels) and
                (set(labels.tolist()) == set(range(6))), f"Invalid L20/Step9 metadata: {split}")
        ep_sets[split] = set(m["episode_id"].astype(str))
        sequences[split] = {10: indices10, 20: indices20}
        audit[split] = {"n_endpoints": int(len(ends)),
                        "eligible_episodes": int(m.episode_id.nunique()),
                        "total_source_episodes": int(windows[split].episode_id.nunique()),
                        "short_episodes_excluded": int(windows[split].episode_id.nunique() - m.episode_id.nunique()),
                        "endpoint_sha256": endsha(ends),
                        "labels_sha256": arrsha(labels),
                        "L10_indices_sha256": arrsha(indices10),
                        "L20_indices_sha256": arrsha(indices20),
                        "original_step10_L10_exact_match": True,
                        "class_counts": {EXPECTED_LABELS[i]: int(np.sum(labels == i)) for i in range(6)}}
        print(f"[Matched {split}] {len(ends):,} endpoints, {m.episode_id.nunique():,} eligible episodes; L10 == L20[-10:]", flush=True)
    for i, split in enumerate(SPLITS):
        for other in SPLITS[i + 1:]:
            require(not (ep_sets[split] & ep_sets[other]), f"Episode overlap {split}/{other}")
    return sequences, audit


def create_context(a) -> dict:
    s27_path = source("step27", ("27.l20_tst_pilot.py", "27. l20_tst_pilot.py"), a.step27)
    s13_path = source("step13", ("13.pooling_mlp_temporal_aggregation_baseline.py",
                                     "13. pooling_mlp_temporal_aggregation_baseline.py"), a.step13)
    require(s27_path.parent == BASE and s13_path.parent == BASE,
            "Step27 and Step13 sources must be beside Step30 (avoid ambiguous project roots).")
    s27 = import_source("taes30_step27", s27_path)
    s13 = import_source("taes30_step13", s13_path)
    args27 = SimpleNamespace(step10=a.step10, step11=a.step11, step12=a.step12,
                             output_root=a.output_root, baseline_main_root=a.baseline_main_root,
                             baseline_multiseed_root=a.baseline_multiseed_root, seeds=list(SEEDS))
    paths, mod = s27.load_source_modules(args27)
    check_protected(a, s13, mod)
    report13 = read_json(a.step13_summary)
    check_step13_config(s13, report13)
    extraction, _, contract = s27.authenticate_inputs(args27, mod)
    require(contract.get("dataset_id") == EXPECTED_DATASET_ID,
            "Dataset identity differs from the frozen shared_nominal_v2 benchmark.")
    p13 = report13.get("manifest", {}).get("revised_pipeline_provenance", {})
    for key in ("dataset_id", "step8_run_id", "step9_run_id", "step10_run_id"):
        require(p13.get(key) == contract[key], f"Step13 historical summary differs at {key}.")
    prior = verify_prior_experiments(a, s27, paths, contract)
    raw, windows, audit, _, _ = s27.get_raw_splits(mod, extraction)
    seq20, seq_meta, _ = s27.load_cache(a.step27_root / "sequence_cache", mod, windows, raw)
    sequences, cohort = assemble_sequences(raw, seq_meta, windows, seq20, mod["step11"])
    cache_report = read_json(a.step27_root / "sequence_cache/27_L20_sequence_report.json")
    require(cache_report.get("status") == "COMPLETE", "Step27 cached population report incomplete.")
    for split in SPLITS:
        saved = cache_report.get("splits", {}).get(split, {})
        report = cohort[split]
        require(saved.get("num_sequences") == report["n_endpoints"] and
                saved.get("num_episodes") == report["total_source_episodes"] and
                saved.get("skipped_short_episodes") == report["short_episodes_excluded"],
                f"Step27 cache population differs from Step30 independently reconstructed {split} cohort.")
    for split in ("val", "test"):
        summary = prior["step27_val"] if split == "val" else None
        if summary:
            require(summary["common_endpoints"] == cohort[split]["n_endpoints"] and
                    summary["endpoint_sha256"] == cohort[split]["endpoint_sha256"],
                    f"Step27 completed validation summary differs for {split}")
    return {"args": a, "s27": s27, "s13": s13, "paths": {**paths, "step13": s13_path,
            "step27": s27_path}, "contract": contract, "prior": prior,
            "raw": raw, "windows": windows, "window_audit": audit,
            "seq20": seq20, "meta": seq_meta, "sequences": sequences, "cohort": cohort}


def identity_for(ctx: dict) -> dict:
    import torch
    a = ctx["args"]
    s13 = ctx["s13"]
    return {
        "schema": SCHEMA,
        "source_sha256": sha(Path(__file__)),
        "original_source_sha256": {key: sha(path) for key, path in ctx["paths"].items()},
        "source_contract": ctx["contract"],
        "step27_identity_hash": ctx["prior"]["step27"]["identity_hash"],
        "step27_manifest_sha256": sha(a.step27_root / "27_run_manifest.json"),
        "step27_sequence_manifest_sha256": sha(a.step27_root / "sequence_cache/27_L20_sequence_manifest.json"),
        "step29_manifest_sha256": sha(a.step29_root / "29_run_manifest.json"),
        "step13_historical_summary_sha256": sha(a.step13_summary),
        "seeds": list(SEEDS), "lengths": list(LENGTHS), "pooling": "meanmax",
        "train_populations": {split: ctx["cohort"][split] for split in ("train", "val", "test")},
        "train_config": {
            "epochs": s13.MAX_EPOCHS, "patience": s13.PATIENCE, "batch_size": s13.BATCH_SIZE,
            "lr": s13.LEARNING_RATE, "weight_decay": s13.WEIGHT_DECAY,
            "grad_clip": s13.GRAD_CLIP_NORM, "hidden_dim": s13.HIDDEN_DIM,
            "mlp_layers": s13.MLP_LAYERS, "dropout": s13.DROPOUT,
            "input_norm": s13.USE_INPUT_LAYERNORM,
            "standardization": s13.USE_STANDARDIZATION,
            "label_smoothing": s13.LABEL_SMOOTHING,
            "amp": bool(s13.USE_AMP),
            "checkpoint_selection": "highest unperturbed validation six-class macro-F1",
            "train_and_val_fit": "all L20 eligible endpoints",
            "test_input_access": "test source files authenticated for cohort integrity; no new Test inference/scoring until explicit --final-test",
        },
        "runtime": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "pandas": pd.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.machine(),
        },
        "interpretation": "exploratory post-L20-test; seeds are fixed heads, not paired initializations",
    }


def new_checkpoint(model, optimizer, s13, seed: int, length: int, epoch: int,
                   best: dict, identity_hash: str, path: Path) -> None:
    import torch
    config = s13.ModelConfig(input_dim=1536, hidden_dim=s13.HIDDEN_DIM,
                             num_classes=6, mlp_layers=s13.MLP_LAYERS,
                             dropout=s13.DROPOUT, use_input_layernorm=s13.USE_INPUT_LAYERNORM)
    payload = {"model_state_dict": model.state_dict(),
               "model_config": asdict(config),
               "pooling_mode": "meanmax", "sequence_length": length, "seed": seed,
               "best_epoch": epoch, "best_val_summary": best,
               "source_identity_hash": identity_hash,
               "validation_selected": True,
               "input_rule": "concatenate mean and max of frozen original [CLS] window embeddings"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt.tmp", delete=False) as f:
        temp = Path(f.name)
    try:
        torch.save(payload, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def safe_load_checkpoint(path: Path, device):
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # older PyTorch
        return torch.load(path, map_location=device)
    # Do not fall back to unrestricted unpickling for corrupted/untrusted files.


def check_checkpoint(path: Path, ctx: dict, identity_hash: str,
                     length: int, seed: int, device):
    s13 = ctx["s13"]
    ckpt = safe_load_checkpoint(path, device)
    require(ckpt.get("source_identity_hash") == identity_hash and
            ckpt.get("seed") == seed and ckpt.get("sequence_length") == length and
            ckpt.get("pooling_mode") == "meanmax" and
            ckpt.get("validation_selected") is True,
            f"Checkpoint source/length/seed mismatch: {path}")
    cfg = s13.ModelConfig(input_dim=1536, hidden_dim=s13.HIDDEN_DIM,
                           num_classes=6, mlp_layers=2, dropout=.25,
                           use_input_layernorm=True)
    require(ckpt.get("model_config") == asdict(cfg), "Saved MLP architecture differs from original Step13.")
    model = s13.MLPClassifier(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt


def features(ctx: dict, split: str, length: int) -> tuple[np.ndarray, np.ndarray]:
    s13 = ctx["s13"]
    source_embeddings = ctx["raw"][split]["embeddings"]
    idx = ctx["sequences"][split][length]
    result = s13.build_pooled_features(source_embeddings, idx, "meanmax", f"{split}_L{length}")
    require(result.dtype == np.float32 and result.shape == (len(idx), 1536) and
            np.isfinite(result).all(), f"Invalid meanmax feature matrix for {split}/L{length}")
    # Independent sample: mean+max must be the exact Step13 computation.
    n = min(len(idx), 9)
    x = source_embeddings[idx[:n]]
    expected = np.concatenate([x.mean(axis=1, dtype=np.float32), x.max(axis=1)], axis=1)
    require(np.array_equal(result[:n], expected),
            f"MeanMax result differs from direct NumPy mean+max: {split}/L{length}")
    y = np.asarray(ctx["seq20"][split]["y"], dtype=np.int64)
    require(np.isin(y, np.arange(6)).all(), f"Invalid labels for {split}")
    return result, y


def build_prediction(ctx: dict, inference: dict, split: str, length: int, seed: int,
                     identity_hash: str) -> pd.DataFrame:
    s13 = ctx["s13"]
    seq = ctx["sequences"][split][length]
    meta = ctx["meta"][split]
    y = np.asarray(ctx["seq20"][split]["y"], dtype=np.int64)
    true = np.asarray(inference["y_true"], dtype=np.int64)
    pred = np.asarray(inference["y_pred"], dtype=np.int64)
    probs = np.asarray(inference["probs"])
    require(np.array_equal(y, true) and len(meta) == len(y) and probs.shape == (len(y), 6) and
            np.array_equal(probs.argmax(1), pred) and np.isfinite(probs).all() and
            np.allclose(probs.sum(1), 1, atol=2e-6, rtol=0),
            f"Prediction/label/softmax mismatch {split}/L{length}/seed{seed}")
    start = seq[:, 0].astype(np.int64)
    end = seq[:, -1].astype(np.int64)
    windows = ctx["windows"][split]
    data = {"sequence_id": np.arange(len(y), dtype=np.int64),
            "seed": np.full(len(y), seed, dtype=np.int64),
            "pooling_mode": ["meanmax"] * len(y),
            "sequence_length": np.full(len(y), length, dtype=np.int64),
            "source_split": [split] * len(y),
            "episode_id": meta["episode_id"].astype(str).to_numpy(),
            "sequence_start_row": start,
            "sequence_end_row": end,
            "sequence_start_step": windows["episode_step"].to_numpy(dtype=np.int64)[start],
            "sequence_end_step": windows["episode_step"].to_numpy(dtype=np.int64)[end],
            "true_label_id": true, "pred_label_id": pred,
            "true_label": [EXPECTED_LABELS[i] for i in true],
            "pred_label": [EXPECTED_LABELS[i] for i in pred],
            "correct": (true == pred).astype(np.int64),
            "confidence": probs.max(1)}
    for i, label in enumerate(EXPECTED_LABELS):
        data["prob_" + s13.safe_name(label)] = probs[:, i]
    data["dataset_id"] = [ctx["contract"]["dataset_id"]] * len(y)
    for key in ("step8_run_id", "step9_run_id", "step10_run_id"):
        data[key] = [ctx["contract"][key]] * len(y)
    data["step30_identity_hash"] = [identity_hash] * len(y)
    result = pd.DataFrame(data)
    require(not result.duplicated("sequence_end_row").any() and
            np.array_equal(result["sequence_end_row"].to_numpy(),
                           ctx["seq20"][split]["end_row_indices"]),
            "Prediction endpoints have duplicates or changed order.")
    return result


def metric_of(y: np.ndarray, pred: np.ndarray, s13) -> dict:
    return s13.compute_metrics(np.asarray(y, np.int64), np.asarray(pred, np.int64),
                               6, dict(enumerate(EXPECTED_LABELS)))[0]


def seed_directory(root: Path, length: int, seed: int) -> Path:
    return root / f"L{length}" / f"seed_{seed}"


def validate_saved_predictions(path: Path, ctx: dict, length: int,
                               seed: int, split: str, identity_hash: str) -> dict:
    f = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    expected_cols = ("sequence_id", "seed", "sequence_length", "source_split", "episode_id",
                     "sequence_start_row", "sequence_end_row", "true_label_id", "pred_label_id",
                     "correct", "step30_identity_hash", "dataset_id")
    require(not f.columns.duplicated().any() and all(col in f for col in expected_cols),
            f"Prediction schema incomplete: {path}")
    seq = ctx["sequences"][split][length]
    meta = ctx["meta"][split]
    y = ctx["seq20"][split]["y"]
    require(len(f) == len(seq) and np.array_equal(f["sequence_id"].to_numpy(np.int64), np.arange(len(seq))) and
            f["seed"].eq(seed).all() and f["sequence_length"].eq(length).all() and
            f["source_split"].eq(split).all() and
            f["step30_identity_hash"].eq(identity_hash).all() and
            f["dataset_id"].eq(ctx["contract"]["dataset_id"]).all() and
            np.array_equal(f["episode_id"].astype(str).to_numpy(), meta["episode_id"].astype(str).to_numpy()) and
            np.array_equal(f["sequence_start_row"].to_numpy(np.int64), seq[:, 0]) and
            np.array_equal(f["sequence_end_row"].to_numpy(np.int64), seq[:, -1]) and
            np.array_equal(f["true_label_id"].to_numpy(np.int64), y),
            f"Prediction cohort, seed, provenance or endpoint mismatch: {path}")
    pred = f["pred_label_id"].to_numpy(dtype=np.int64)
    require(np.isin(pred, range(6)).all() and
            np.array_equal(f["correct"].to_numpy(dtype=np.int64), (pred == y).astype(np.int64)) and
            np.array_equal(f["pred_label"].to_numpy(), np.array(EXPECTED_LABELS)[pred]),
            f"Invalid predicted class/correctness: {path}")
    prob = f[["prob_" + ctx["s13"].safe_name(x) for x in EXPECTED_LABELS]].to_numpy(dtype=float)
    require(np.isfinite(prob).all() and np.all((prob >= 0) & (prob <= 1)) and
            np.allclose(prob.sum(1), 1, atol=2e-6, rtol=0) and np.array_equal(prob.argmax(1), pred),
            f"Saved probability simplex/prediction mismatch: {path}")
    return metric_of(y, pred, ctx["s13"])


def verify_complete_seed(ctx: dict, root: Path, length: int, seed: int,
                         identity_hash: str) -> dict:
    folder = seed_directory(root, length, seed)
    saved = read_json(folder / "30_seed_manifest.json")
    require(saved.get("status") == "COMPLETE" and
            saved.get("identity_hash") == identity_hash and
            saved.get("seed") == seed and saved.get("length") == length,
            f"Incomplete or foreign L{length}/seed{seed}.")
    needed = {"best_meanmax.pt", "history.csv", "val_predictions.csv", "val_metrics.json"}
    require(needed.issubset(saved.get("artifacts", {})), "Completed seed manifest lacks required files.")
    for name, digest in saved["artifacts"].items():
        p = folder / name
        require(p.is_file() and sha(p) == digest, f"Completed seed file changed: {p}")
    report = read_json(folder / "val_metrics.json")
    pred = validate_saved_predictions(folder / "val_predictions.csv", ctx, length, seed, "val", identity_hash)
    require(report.get("best_epoch") == saved.get("best_epoch") and
            math.isclose(report["validation"]["macro_f1"], pred["macro_f1"], abs_tol=1e-10),
            "Saved validation scores differ from audited predictions.")
    return saved


def make_model(ctx: dict, device):
    s13 = ctx["s13"]
    cfg = s13.ModelConfig(input_dim=1536, hidden_dim=s13.HIDDEN_DIM,
                           num_classes=6, mlp_layers=s13.MLP_LAYERS,
                           dropout=s13.DROPOUT, use_input_layernorm=s13.USE_INPUT_LAYERNORM)
    return s13.MLPClassifier(cfg).to(device)


def train_seed(ctx: dict, root: Path, length: int, seed: int, identity_hash: str,
               Xtrain: np.ndarray, ytrain: np.ndarray, Xval: np.ndarray, yval: np.ndarray, device):
    import torch
    s13 = ctx["s13"]
    folder = seed_directory(root, length, seed)
    if folder.exists():
        return verify_complete_seed(ctx, root, length, seed, identity_hash)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "30_seed_manifest.json", {"status": "RUNNING", "identity_hash": identity_hash,
                                                    "length": length, "seed": seed, "started": utc()})
    started = time.perf_counter()
    s13.set_all_seeds(seed)
    model = make_model(ctx, device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=s13.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=s13.LEARNING_RATE, weight_decay=s13.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=.5, patience=2, min_lr=1e-6)
    scaler = s13.make_grad_scaler(device)
    train_loader = s13.make_loader(Xtrain, ytrain, s13.BATCH_SIZE, shuffle=True, device=device)
    val_loader = s13.make_loader(Xval, yval, s13.BATCH_SIZE * 2, shuffle=False, device=device)
    best_score, best_epoch, bad = -1.0, 0, 0
    history = []
    ckpt_path = folder / "best_meanmax.pt"
    for epoch in range(1, s13.MAX_EPOCHS + 1):
        summary = s13.train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val = s13.predict_model(model, val_loader, device)
        vm = metric_of(val["y_true"], val["y_pred"], s13)
        scheduler.step(float(vm["macro_f1"]))
        history.append({"seed": seed, "length": length, "epoch": epoch,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "train_loss": float(summary["loss"]),
                        "train_accuracy": float(summary["accuracy"]),
                        "val_accuracy": float(vm["accuracy"]),
                        "val_macro_f1": float(vm["macro_f1"]),
                        "val_weighted_f1": float(vm["weighted_f1"])})
        print(f"[L{length} seed={seed}] epoch {epoch}/{s13.MAX_EPOCHS} "
              f"train_loss={summary['loss']:.5f} val_macro_f1={vm['macro_f1']:.5f}", flush=True)
        if vm["macro_f1"] > best_score + 1e-12:
            best_score = float(vm["macro_f1"])
            best_epoch = epoch
            bad = 0
            new_checkpoint(model, optimizer, s13, seed, length, epoch, vm, identity_hash, ckpt_path)
        else:
            bad += 1
            if bad >= s13.PATIENCE:
                print(f"[L{length} seed={seed}] early stop at epoch {epoch}", flush=True)
                break
    model, ckpt = check_checkpoint(ckpt_path, ctx, identity_hash, length, seed, device)
    val = s13.predict_model(model, val_loader, device)
    valmetrics = metric_of(val["y_true"], val["y_pred"], s13)
    require(math.isclose(best_score, valmetrics["macro_f1"], abs_tol=1e-10),
            f"Reloaded selected model yields different Validation F1: L{length}, seed {seed}.")
    pred = build_prediction(ctx, val, "val", length, seed, identity_hash)
    write_csv(folder / "history.csv", pd.DataFrame(history))
    write_csv(folder / "val_predictions.csv", pred)
    duration = time.perf_counter() - started
    write_json(folder / "val_metrics.json", {"length": length, "seed": seed,
        "validation": valmetrics, "best_epoch": best_epoch, "trained_epochs": len(history),
        "training_elapsed_seconds": duration,
        "n_train": int(len(ytrain)), "n_val": int(len(yval)),
        "train_episode_count": ctx["cohort"]["train"]["eligible_episodes"],
        "val_episode_count": ctx["cohort"]["val"]["eligible_episodes"],
        "endpoint_sha256": ctx["cohort"]["val"]["endpoint_sha256"],
        "selected_by": "maximum validation macro-F1; no test evaluation"})
    artifacts = {name: sha(folder / name) for name in
                 ("best_meanmax.pt", "history.csv", "val_predictions.csv", "val_metrics.json")}
    write_json(folder / "30_seed_manifest.json", {
        "status": "COMPLETE", "identity_hash": identity_hash,
        "length": length, "seed": seed, "best_epoch": best_epoch,
        "trained_epochs": len(history), "training_elapsed_seconds": duration,
        "artifacts": artifacts, "completed": utc()})
    print(f"[COMPLETE] L{length} seed={seed}: val F1={valmetrics['macro_f1']:.6f}, "
          f"best epoch={best_epoch}, train={duration:.1f}s", flush=True)
    return verify_complete_seed(ctx, root, length, seed, identity_hash)


def archive_failed_seed(root: Path, length: int, seed: int) -> Path:
    original = seed_directory(root, length, seed)
    archive = root / "_incomplete" / f"L{length}_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    original.rename(archive)
    return archive


def metric_table(ctx: dict, root: Path, stage: str, identity_hash: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    require(stage in ("val", "test"), "Bad evaluation stage.")
    rows = []
    for length in LENGTHS:
        for seed in SEEDS:
            folder = seed_directory(root, length, seed)
            if stage == "val":
                verified = verify_complete_seed(ctx, root, length, seed, identity_hash)
                metric = read_json(folder / "val_metrics.json")["validation"]
                best_epoch = verified["best_epoch"]
            else:
                metric = validate_saved_predictions(folder / "test_predictions.csv", ctx,
                                                    length, seed, stage, identity_hash)
                best_epoch = verify_complete_seed(ctx, root, length, seed, identity_hash)["best_epoch"]
            rows.append({"model": "MeanMax_MLP", "length": length, "seed": seed,
                         "n_endpoints": ctx["cohort"][stage]["n_endpoints"],
                         "best_epoch": best_epoch,
                         "accuracy": float(metric["accuracy"]),
                         "macro_f1": float(metric["macro_f1"]),
                         "weighted_f1": float(metric["weighted_f1"])})
    per_seed = pd.DataFrame(rows).sort_values(["length", "seed"]).reset_index(drop=True)
    grouped = per_seed.groupby("length", sort=True)
    summary = grouped[["accuracy", "macro_f1", "weighted_f1"]].agg(["mean", "std"])
    summary.columns = ["_".join(x) for x in summary.columns]
    summary = summary.reset_index()
    summary.insert(1, "num_head_seeds", 3)
    summary.insert(2, "n_endpoints", ctx["cohort"][stage]["n_endpoints"])
    line = {"stage": stage, "dataset_id": ctx["contract"]["dataset_id"],
            "identity_hash": identity_hash, "cohort": ctx["cohort"][stage],
            "model": "MeanMax_MLP", "num_seeds": 3,
            "values": {str(int(row["length"])): {k: float(row[k]) for k in
              ("accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std",
               "weighted_f1_mean", "weighted_f1_std")} for _, row in summary.iterrows()},
            "interpretation": "exploratory, matched train/validation/test endpoint cohort",
            "scope": "Head-seed SD conditional on fixed encoder/dataset, not paired episode CI."}
    return per_seed, summary, line


def tst_rows(ctx: dict, stage: str) -> list[dict]:
    require(stage in ("val", "test"), "Invalid TST stage.")
    a = ctx["args"]
    p = ctx["prior"]
    if stage == "val":
        t27, t29 = p["step27_val"], p["step29_val"]
    else:
        f27 = read_json(a.step27_root / "27_final_test.json")
        f29 = read_json(a.step29_root / "29_final_test.json")
        require(f27.get("status") == "COMPLETE" and f29.get("status") == "COMPLETE" and
                f27.get("identity_hash") == p["step27"]["identity_hash"] and
                f29.get("identity_hash") == p["step29"]["identity_hash"],
                "Step27/29 held-out Test results not COMPLETE/matched.")
        require(f27.get("test_summary_sha256") == sha(a.step27_root / "27_test_summary.json") and
                f29.get("test_summary_sha256") == sha(a.step29_root / "29_test_summary.json"),
                "Original TST final Test summaries were modified.")
        t27 = read_json(a.step27_root / "27_test_summary.json")
        t29 = read_json(a.step29_root / "29_test_summary.json")
    require(t27["endpoint_sha256"] == t29["endpoint_sha256"] == ctx["cohort"][stage]["endpoint_sha256"] and
            t27["common_endpoints"] == t29["endpoints"] == ctx["cohort"][stage]["n_endpoints"],
            f"TST/{stage} endpoint cohort differs from new MeanMax model.")
    return [
        {"model": "TST", "length": 10, "n_seeds": 3,
         "macro_f1_mean": t29["matched_L10_macro_f1_mean"],
         "macro_f1_sd": t29["matched_L10_macro_f1_sd_ddof1"],
         "accuracy_mean": t29["matched_L10_accuracy_mean"], "source": "Step29 matched L10"},
        {"model": "TST", "length": 20, "n_seeds": 3,
         "macro_f1_mean": t27["L20_macro_f1_mean"],
         "macro_f1_sd": t27["L20_macro_f1_sd"],
         "accuracy_mean": t27["L20_accuracy_mean"], "source": "Step27 L20"},
    ]


def cross_table(ctx: dict, summary: pd.DataFrame, stage: str) -> pd.DataFrame:
    rows = tst_rows(ctx, stage)
    for _, record in summary.iterrows():
        rows.append({"model": "MeanMax_MLP", "length": int(record["length"]),
                     "n_seeds": 3, "macro_f1_mean": float(record["macro_f1_mean"]),
                     "macro_f1_sd": float(record["macro_f1_std"]),
                     "accuracy_mean": float(record["accuracy_mean"]),
                     "source": "Step30 new matched Mean+Max"})
    out = pd.DataFrame(rows).sort_values(["model", "length"], kind="stable").reset_index(drop=True)
    require(len(out) == 4 and not out.duplicated(["model", "length"]).any() and
            np.isfinite(out[["macro_f1_mean", "macro_f1_sd", "accuracy_mean"]].to_numpy(float)).all(),
            "Four-row 2x2 model/length table is incomplete.")
    out["stage"] = stage
    out["n_endpoints"] = ctx["cohort"][stage]["n_endpoints"]
    out["n_episodes"] = ctx["cohort"][stage]["eligible_episodes"]
    return out


def save_review(root: Path, stage: str) -> Path:
    filename = "30_MATCHED_MEANMAX_VALIDATION_REVIEW.zip" if stage == "val" else "30_MATCHED_MEANMAX_FINAL_TEST_REVIEW.zip"
    archive_path = root / filename
    with tempfile.NamedTemporaryFile(dir=root, suffix=".zip.tmp", delete=False) as f:
        temp = Path(f.name)
    try:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(Path(__file__), "source/" + Path(__file__).name)
            filenames = ["30_run_manifest.json", "30_protocol.json", "30_input_audit.json",
                         "30_val_seed_results.csv", "30_val_model_summary.csv",
                         "30_val_2x2.csv", "30_val_summary.json"]
            if stage == "test":
                filenames += ["30_final_test.json", "30_test_seed_results.csv",
                              "30_test_model_summary.csv", "30_test_2x2.csv", "30_test_summary.json"]
            for name in filenames:
                p = root / name
                require(p.is_file(), f"Cannot create review archive; missing: {p}")
                archive.write(p, name)
            for length in LENGTHS:
                for seed in SEEDS:
                    f = seed_directory(root, length, seed)
                    filenames = ["30_seed_manifest.json", "history.csv", "val_metrics.json", "val_predictions.csv"]
                    if stage == "test":
                        filenames += ["test_metrics.json", "test_predictions.csv"]
                    for name in filenames:
                        p = f / name
                        require(p.is_file(), f"Missing required review source: {p}")
                        archive.write(p, p.relative_to(root).as_posix())
        temp.replace(archive_path)
    finally:
        temp.unlink(missing_ok=True)
    return archive_path


def preflight(ctx: dict) -> None:
    import torch
    s13 = ctx["s13"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for length in LENGTHS:
        indices = ctx["sequences"]["val"][length][:4]
        raw = ctx["raw"]["val"]["embeddings"]
        pooled = np.concatenate([raw[indices].mean(axis=1, dtype=np.float32),
                                 raw[indices].max(axis=1)], axis=1).astype(np.float32)
        s13.set_all_seeds(SEEDS[0])
        model = make_model(ctx, device)
        with torch.no_grad():
            result = model(torch.from_numpy(pooled).to(device))
        require(result.shape == (4, 6) and torch.isfinite(result).all().item(),
                f"MeanMax MLP forward preflight failed at L{length}.")
    print("\nSTEP 30 PREFLIGHT PASS -- NO TRAINING/OUTPUT FILES WRITTEN", flush=True)
    print("Historical Step13 MeanMax settings/provenance verified.")
    print("Historical Step27/29 model/dataset/cohort provenance verified.")
    for split, item in ctx["cohort"].items():
        print(f"{split}: {item['n_endpoints']:,} endpoints / "
              f"{item['eligible_episodes']:,} eligible episodes, both L10/L20")
    print(f"MLP: LayerNorm(1536)->Linear(1536,256)->GELU->Dropout(.25)->Linear(256,6) on {device}")
    print("No new encoder training. No Test score computed.")


def run_validation(ctx: dict, identity: dict, identity_hash: str) -> None:
    import torch
    a = ctx["args"]
    root = a.output_root
    s13 = ctx["s13"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with RunLock(root):
        manifest_path = root / "30_run_manifest.json"
        existing = read_json(manifest_path) if manifest_path.is_file() else None
        if existing is None:
            require(not any(p.name != ".30_active.lock" for p in root.iterdir()),
                    "Unrecognized Step30 output folder. Choose a new --output-root.")
        else:
            require(existing.get("identity") == identity and existing.get("identity_hash") == identity_hash,
                    "Existing Step30 output uses different source/data/settings/runtime. Use another --output-root.")
            require(not (root / "30_final_test.json").exists(),
                    "Final Test has been attempted. Do not retrain after Test; use preserved results.")
            if existing.get("status") == "COMPLETE":
                for length in LENGTHS:
                    for seed in SEEDS:
                        verify_complete_seed(ctx, root, length, seed, identity_hash)
                for key, name in (("val_summary_sha256", "30_val_summary.json"),
                                  ("val_table_sha256", "30_val_2x2.csv")):
                    require(existing.get(key) == sha(root / name), "Completed Step30 Validation file changed: " + name)
                review = save_review(root, "val")
                print("Step30 Validation already COMPLETE and verified. No training.\nReview ZIP: " + str(review))
                return
        manifest = existing or {"schema": SCHEMA, "identity": identity,
                                "identity_hash": identity_hash, "seeds": list(SEEDS), "lengths": list(LENGTHS),
                                "started": utc(), "archived_incomplete": []}
        manifest.update(status="RUNNING", last_error=None)
        write_json(manifest_path, manifest)
        write_json(root / "30_protocol.json", {
            "experiment": SCHEMA, "identity_hash": identity_hash,
            "prior_context": "Step27 L20 Test was inspected before Step30 design; exploratory post-test extension.",
            "hypothesis": "Additional context improves episode-scenario classification with an order-insensitive aggregator too.",
            "input": "fine-tuned DistilBERT frozen CLS embeddings; mean+max concatenation",
            "training_policy": "L10 and L20 use exactly the same L20-eligible train/validation endpoints",
            "selected_models": "Both prespecified lengths and all three seeds; no best-length/seed cherry-picking",
            "training_settings": identity["train_config"],
            "test_policy": "No Test inference during training/validation; Test evaluated only by --final-test.",
            "uncertainty": "Seed sample SD, not CI. No paired-bootstrap computed in Step30.",
            "seed_note": "Head-seed numbers match as labels, not equal initializations or paired training.",
            "scope": "Controlled generator-specific episode-scenario classification; not live attack onset detection."})
        write_json(root / "30_input_audit.json", {"identity_hash": identity_hash,
            "source_contract": ctx["contract"], "cohort": ctx["cohort"],
            "original_source_sha256": identity["original_source_sha256"],
            "step27_run_manifest_sha256": identity["step27_manifest_sha256"],
            "step29_run_manifest_sha256": identity["step29_manifest_sha256"],
            "historical_step13_summary_sha256": identity["step13_historical_summary_sha256"],
            "test_prediction_access": "Existing test files authenticated for alignment only; no new Step30 Test inference/scoring or checkpoint selection."})
        try:
            for length in LENGTHS:
                pending = []
                for seed in SEEDS:
                    seedroot = seed_directory(root, length, seed)
                    if not seedroot.exists():
                        pending.append(seed)
                        continue
                    seedinfo = read_json(seedroot / "30_seed_manifest.json") if (seedroot / "30_seed_manifest.json").exists() else {}
                    if seedinfo.get("status") == "COMPLETE":
                        verify_complete_seed(ctx, root, length, seed, identity_hash)
                        print(f"[Verified complete reused] L{length}/seed{seed}", flush=True)
                    else:
                        require(a.retry_failed,
                                f"Incomplete L{length}/seed{seed} preserved; rerun --retry-failed after old process stopped.")
                        old = archive_failed_seed(root, length, seed)
                        manifest["archived_incomplete"].append(str(old))
                        write_json(manifest_path, manifest)
                        pending.append(seed)
                if not pending:
                    continue
                # Build ONLY one length's train/val pooled arrays and discard before next length.
                Xtrain, ytrain = features(ctx, "train", length)
                Xval, yval = features(ctx, "val", length)
                require(len(ytrain) == ctx["cohort"]["train"]["n_endpoints"] and
                        len(yval) == ctx["cohort"]["val"]["n_endpoints"], "Wrong fit population.")
                for seed in pending:
                    if sha(Path(__file__)) != identity["source_sha256"]:
                        raise ValueError("Step30 source changed during training; preserving partial run.")
                    manifest["active_trial"] = {"length": length, "seed": seed}
                    write_json(manifest_path, manifest)
                    train_seed(ctx, root, length, seed, identity_hash, Xtrain, ytrain, Xval, yval, device)
                    manifest["active_trial"] = None
                    write_json(manifest_path, manifest)
                del Xtrain, ytrain, Xval, yval
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            seeds, summary, report = metric_table(ctx, root, "val", identity_hash)
            two = cross_table(ctx, summary, "val")
            write_csv(root / "30_val_seed_results.csv", seeds)
            write_csv(root / "30_val_model_summary.csv", summary)
            write_csv(root / "30_val_2x2.csv", two)
            report["2x2"] = two.to_dict(orient="records")
            write_json(root / "30_val_summary.json", report)
            manifest.update(status="COMPLETE", completed=utc(), active_trial=None,
                            val_summary_sha256=sha(root / "30_val_summary.json"),
                            val_table_sha256=sha(root / "30_val_2x2.csv"))
            write_json(manifest_path, manifest)
            archive = save_review(root, "val")
        except BaseException as exc:
            manifest.update(status="FAILED", last_error=f"{type(exc).__name__}: {exc}", updated=utc())
            write_json(manifest_path, manifest)
            raise
    print("\nSTEP30 VALIDATION COMPLETE (no Test predictions generated)", flush=True)
    print(two[["model", "length", "macro_f1_mean", "macro_f1_sd"]].to_string(index=False))
    print("Review ZIP: " + str(archive), flush=True)


def run_final_test(ctx: dict, identity: dict, identity_hash: str) -> None:
    import torch
    a, root, s13 = ctx["args"], ctx["args"].output_root, ctx["s13"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with RunLock(root):
        manifest = read_json(root / "30_run_manifest.json")
        require(manifest.get("status") == "COMPLETE" and
                manifest.get("identity") == identity and manifest.get("identity_hash") == identity_hash and
                manifest.get("val_summary_sha256") == sha(root / "30_val_summary.json") and
                manifest.get("val_table_sha256") == sha(root / "30_val_2x2.csv"),
                "Step30 Validation must be COMPLETE and unchanged before final Test.")
        for length in LENGTHS:
            for seed in SEEDS:
                verify_complete_seed(ctx, root, length, seed, identity_hash)
        prior_test = root / "30_final_test.json"
        if prior_test.exists():
            done = read_json(prior_test)
            require(done.get("identity_hash") == identity_hash,
                    "A previous Test attempt belongs to different code/data/protocol. Preserve it.")
            if done.get("status") == "COMPLETE":
                require(done.get("test_summary_sha256") == sha(root / "30_test_summary.json") and
                        done.get("test_2x2_sha256") == sha(root / "30_test_2x2.csv"),
                        "Completed Step30 Test summary/2x2 report changed.")
                for length in LENGTHS:
                    for seed in SEEDS:
                        folder = seed_directory(root, length, seed)
                        saved = read_json(folder / "test_metrics.json")
                        actual = sha(folder / "test_predictions.csv")
                        require(saved.get("prediction_sha256") == actual and
                                done.get("test_prediction_sha256", {}).get(f"L{length}/seed_{seed}") == actual,
                                "Completed Test prediction file changed from its recorded hashes.")
                        pred = validate_saved_predictions(folder / "test_predictions.csv", ctx,
                                                          length, seed, "test", identity_hash)
                        require(math.isclose(saved["test"]["macro_f1"], pred["macro_f1"], abs_tol=1e-10),
                                "Completed Test metrics differ from actual saved predictions.")
                archive = save_review(root, "test")
                print("Existing Step30 Final Test verified; no inference rerun.\nReview ZIP: " + str(archive))
                return
            # A failed Test is not retrained; preserve every partial report and
            # restart inference of the same six immutable best checkpoints.
            require(a.retry_final_test and done.get("status") in ("RUNNING", "FAILED"),
                    "Earlier Step30 final Test is incomplete. After confirming the previous "
                    "process stopped, use --final-test --retry-final-test.")
            rescue = root / "_incomplete_final" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
            files = [prior_test]
            files += [root / name for name in (
                "30_test_seed_results.csv", "30_test_model_summary.csv", "30_test_2x2.csv", "30_test_summary.json")]
            for length in LENGTHS:
                for seed in SEEDS:
                    folder = seed_directory(root, length, seed)
                    files += [folder / "test_metrics.json", folder / "test_predictions.csv"]
            for path in files:
                if path.is_file():
                    relative = path.relative_to(root)
                    saved_to = rescue / relative
                    saved_to.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(saved_to)
            print(f"[Archived incomplete final Test; checkpoints untouched] {rescue}", flush=True)
        # Freeze a test invocation record BEFORE loading Test pooled features.
        write_json(prior_test, {"status": "RUNNING", "identity_hash": identity_hash,
                               "started_at": utc(), "selection_fixed_before_test": True,
                               "training": False})
        try:
            for length in LENGTHS:
                Xt, yt = features(ctx, "test", length)
                loader = s13.make_loader(Xt, yt, s13.BATCH_SIZE * 2, shuffle=False, device=device)
                for seed in SEEDS:
                    folder = seed_directory(root, length, seed)
                    model, selected = check_checkpoint(folder / "best_meanmax.pt", ctx, identity_hash,
                                                       length, seed, device)
                    inference = s13.predict_model(model, loader, device)
                    frame = build_prediction(ctx, inference, "test", length, seed, identity_hash)
                    write_csv(folder / "test_predictions.csv", frame)
                    measured = validate_saved_predictions(folder / "test_predictions.csv", ctx,
                                                          length, seed, "test", identity_hash)
                    write_json(folder / "test_metrics.json", {
                        "identity_hash": identity_hash, "length": length, "seed": seed,
                        "test": measured, "test_endpoints": int(len(frame)),
                        "best_epoch_selected_on_validation": selected["best_epoch"],
                        "validation_checkpoint_sha256": sha(folder / "best_meanmax.pt"),
                        "prediction_sha256": sha(folder / "test_predictions.csv"),
                    })
                    print(f"[Final Test] MeanMax L{length} seed{seed} "
                          f"macro-F1={measured['macro_f1']:.6f}", flush=True)
                del Xt, yt, loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            seeds, summary, report = metric_table(ctx, root, "test", identity_hash)
            table = cross_table(ctx, summary, "test")
            write_csv(root / "30_test_seed_results.csv", seeds)
            write_csv(root / "30_test_model_summary.csv", summary)
            write_csv(root / "30_test_2x2.csv", table)
            report["2x2"] = table.to_dict(orient="records")
            report["post_test_design"] = True
            write_json(root / "30_test_summary.json", report)
            write_json(prior_test, {"status": "COMPLETE", "identity_hash": identity_hash,
                "completed_at": utc(), "selection_fixed_before_test": True,
                "test_summary_sha256": sha(root / "30_test_summary.json"),
                "test_2x2_sha256": sha(root / "30_test_2x2.csv"),
                "test_prediction_sha256": {f"L{length}/seed_{seed}": sha(seed_directory(root, length, seed) / "test_predictions.csv")
                                           for length in LENGTHS for seed in SEEDS}})
            archive = save_review(root, "test")
        except BaseException as exc:
            previous = read_json(prior_test)
            previous.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", at=utc())
            write_json(prior_test, previous)
            raise
    print("\nSTEP30 FINAL TEST COMPLETE", flush=True)
    print(table[["model", "length", "macro_f1_mean", "macro_f1_sd"]].to_string(index=False))
    print("Review ZIP: " + str(archive), flush=True)


def main() -> None:
    a = arguments()
    context = create_context(a)
    identity = identity_for(context)
    run_id = objsha(identity)
    print(f"Step30 source SHA256: {sha(Path(__file__))}")
    print(f"Dataset: {context['contract']['dataset_id']}")
    if a.preflight:
        preflight(context)
    elif a.final_test:
        run_final_test(context, identity, run_id)
    else:
        run_validation(context, identity, run_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[STEP30 STOPPED] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
