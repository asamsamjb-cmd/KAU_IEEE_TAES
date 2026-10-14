# -*- coding: utf-8 -*-
"""Step 27 -- isolated L20 TST validation pilot for the current shared_nominal_v2 benchmark.

Place this file BESIDE the original Step 10, 11 and 12 scripts; DO NOT edit them.
The completed Step 8/9/10 and historical L10 runs are always read-only.

Recommended commands (from the project's code directory, with its .venv active):
    python "27.l20_tst_pilot.py" --preflight
    python "27.l20_tst_pilot.py"
    python "27.l20_tst_pilot.py" --retry-failed   # only after stopped incomplete run

Only after deciding, using the VALIDATION results, to unlock the held-out test:
    python "27.l20_tst_pilot.py" --final-test

What this experiment DOES:
  - Authenticate existing fine-tuned DistilBERT embeddings and Step 8/9/10 lineage.
  - Construct L20 episode-bounded index-only sequences; do not copy embedding cubes.
  - Reuse training class, optimizer, augmentation policy and selection rule from Step 11.
  - Train exactly one L20 TST head for each of three preselected seeds.
  - Recompute pre-existing L10 validation predictions on L20-eligible endpoints.
  - Record macro-F1, accuracy, head-seed SD, elapsed time and best epoch.
  - Compute descriptive paired-nominal evidence availability L10 vs L20.
  - Keep all test scores inaccessible in the default validation pilot.

Limits:
  - Historical L10 and new L20 heads have different TRAIN eligibility populations.
  - Matching the numeric seed DOES NOT guarantee identical random initializations:
    the original L10 head followed other length trainings in the same RNG stream.
  - The evidence-availability reference uses ground truth; it is nondeployable and
    is NOT a theoretical upper bound or a suitable selection target.
  - A three-head-seed pilot is exploratory, not a formal statistical comparison.
  - No data or code in Steps 8--12 are changed by this script.
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
import shutil
import socket
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DEFAULT_ROOT = BASE / "outputs" / "shared_nominal_v2" / "27_L20_TST_VALIDATION_PILOT"
DEFAULT_MULTI = BASE / "outputs" / "shared_nominal_v2" / "tst_training_finetuned_multiseed"
DEFAULT_MAIN = BASE / "outputs" / "shared_nominal_v2" / "tst_training_finetuned_main"
ORIGINAL_LENGTHS = [1, 2, 3, 5, 10]
L10, L20 = 10, 20
DEFAULT_SEEDS = [20260610, 20260611, 20260612]
SCHEMA = "taes_step27_l20_validation_pilot_v1"
SPLITS = ("train", "val", "test")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for part in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def digest_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False, ensure_ascii=False).encode()).hexdigest()


def digest_ends(a: np.ndarray) -> str:
    return hashlib.sha256(np.sort(np.asarray(a, dtype="<i8")).tobytes()).hexdigest()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", suffix=".tmp", delete=False) as fp:
        tmp = Path(fp.name)
        json.dump(obj, fp, indent=2, ensure_ascii=False, allow_nan=False)
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8-sig", suffix=".tmp", delete=False, newline="") as fp:
        tmp = Path(fp.name)
        df.to_csv(fp, index=False)
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def resolve_source(option: str | None, patterns: tuple[str, ...]) -> Path:
    if option:
        p = Path(option).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    candidates = [BASE / s for s in patterns if (BASE / s).is_file()]
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(f"Missing dependency beside Step 27: {patterns}. Supply --step10/--step11/--step12.")
    digests = {digest_file(p) for p in candidates}
    if len(digests) > 1:
        raise ValueError(f"Different copies of original source detected: {candidates}. Specify the exact path.")
    return candidates[0].resolve()


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj  # required for Step 11 @dataclass
    spec.loader.exec_module(obj)
    return obj


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone L20 validation pilot with historical L10 matched-endpoint controls.")
    parser.add_argument("--preflight", action="store_true", help="Audit inputs and one L20 training batch; create NO output.")
    parser.add_argument("--final-test", action="store_true", help="Explicitly unlock held-out test after a finished validation pilot.")
    parser.add_argument("--retry-failed", action="store_true", help="Archive and restart an incomplete seed (no epoch resume).")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline-multiseed-root", type=Path, default=DEFAULT_MULTI)
    parser.add_argument("--baseline-main-root", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--step10", type=str, default=None)
    parser.add_argument("--step11", type=str, default=None)
    parser.add_argument("--step12", type=str, default=None)
    a = parser.parse_args()
    a.seeds = list(a.seeds)
    if len(a.seeds) != 3 or len(set(a.seeds)) != 3 or any(not 0 <= x < 2**32 for x in a.seeds):
        raise ValueError("Pilot protocol requires exactly 3 different, preselected seeds in [0,2**32).")
    if a.preflight and a.final_test:
        raise ValueError("Choose --preflight OR --final-test.")
    if a.final_test and a.retry_failed:
        raise ValueError("--retry-failed only applies to pilot training.")
    for name in ("output_root", "baseline_multiseed_root", "baseline_main_root"):
        setattr(a, name, getattr(a, name).expanduser().resolve())
    return a


def load_source_modules(args):
    paths = {
        "step10": resolve_source(args.step10, ("10.build_tst_sequences.py", "10. build_tst_sequences.py", "10.build_tst_sequences(1).py", "10.build_tst_sequences(3).py", "10. build_tst_sequences(3).py")),
        "step11": resolve_source(args.step11, ("11.train_tst_classifier.py", "11. train_tst_classifier.py", "11_train_tst_classifier.py")),
        "step12": resolve_source(args.step12, ("12.multiseed_tst_runner.py", "12. multiseed_tst_runner.py")),
    }
    if any(p.parent != BASE for p in paths.values()):
        raise ValueError("For safety, Steps 10/11/12 must be located beside Step 27 in the project directory.")
    if args.output_root == BASE or BASE.is_relative_to(args.output_root):
        raise ValueError("Output must not contain the source-code project.")
    modules = {name: import_file("taes27_" + name, path) for name, path in paths.items()}
    s10, s11, s12 = modules["step10"], modules["step11"], modules["step12"]
    if s10.SEQUENCE_LENGTHS != ORIGINAL_LENGTHS or s11.SEQUENCE_LENGTHS != ORIGINAL_LENGTHS:
        raise ValueError("Original sequence-length contract changed. Do not patch Steps 10/11 for this pilot.")
    if s11.SEQUENCE_FIT_SUBSET != "full" or s11.PRIMARY_TEST_SUBSET != "full" or s11.PRIMARY_SELECTION_METRIC != "val_macro_f1":
        raise ValueError("Step 11 no longer follows full-fit / validation-macro-F1 selection protocol.")
    for flag in ("USE_MISSING_AWARE_TRAINING", "USE_SHORTCUT_RESISTANT_TRAINING", "USE_LENGTH_AWARE_REGULARIZATION",
                 "RUN_ROBUST_EVIDENCE_BUDGET_EVAL", "USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS"):
        if getattr(s11, flag):
            raise ValueError(f"Step 27 clean pilot requires {flag}=False in Step 11 source.")
    if s11.SAVE_VAL_PREDICTIONS is not True:
        raise ValueError("Original Step 11 must support validation predictions.")
    expected = {"MAX_EPOCHS": 30, "PATIENCE": 7, "BATCH_SIZE": 256, "LEARNING_RATE": 7e-5,
                "WEIGHT_DECAY": 5e-4, "D_MODEL": 128, "NUM_HEADS": 4, "NUM_ENCODER_LAYERS": 1,
                "DIM_FEEDFORWARD": 256, "DROPOUT": 0.25, "POOLING": "last", "LABEL_SMOOTHING": 0.03}
    for name, wanted in expected.items():
        if getattr(s11, name) != wanted:
            raise ValueError(f"Original Step 11 setting {name}={getattr(s11, name)!r} differs from locked protocol {wanted!r}.")
    # The baseline run may have been generated by another revision of Step 11.
    # We authenticate its actual stored protocol rather than silently assuming a source match.
    return paths, modules


def authenticate_inputs(args, mod):
    s10, s11 = mod["step10"], mod["step11"]
    if s10.EMBEDDING_DIR.resolve() != s11.EMBEDDING_DIR.resolve():
        raise ValueError("Step 10 and 11 embedding directories disagree.")
    for path in (s11.EMBEDDING_DIR, s11.SEQUENCE_DIR, s11.EXPECTED_FINETUNED_MODEL_DIR.parent,
                 args.baseline_multiseed_root, args.baseline_main_root):
        if args.output_root == path or args.output_root.is_relative_to(path) or path.is_relative_to(args.output_root):
            raise ValueError(f"Output would overlap original inputs/results: {path}")
    extraction = s11.require_json(s11.EMBEDDING_REPORT_PATH)
    index = s11.validate_step10_report_index()
    s11.validate_temporal_design_from_reports(extraction)
    source = s11.load_source_contract(extraction, index)
    if source["common_endpoint_report"].get("reference_length") != L10:
        raise ValueError("Historical benchmark's original common endpoint must be L10.")
    # Step 10 source is also authenticated against the same Step 9 and audit artifacts.
    rep10, src10 = s10.load_source_contract()
    s10.SOURCE_CONTRACT = src10
    if (src10["dataset_id"] != source["dataset_id"]
            or src10["step8_run_id"] != source["step8_run_id"]
            or src10["step9_run_id"] != source["step9_run_id"]
            or src10["observability_audit"] != source["observability_audit"]):
        raise ValueError("Step 10/11 provenance mismatch for Step 9 or counterfactual audit.")
    if not rep10 or rep10.get("run_id") != source["step9_run_id"]:
        raise ValueError("Step 10 selected a different Step 9 embedding report.")
    return extraction, index, source


def get_raw_splits(mod, extraction):
    s10, s11 = mod["step10"], mod["step11"]
    config = s10.validate_step9_report(extraction)
    splits = {}
    metas = {}
    audit = {}
    checks = {}
    for split in SPLITS:
        emb, labels, indices, meta, validation = s10.load_step9_split(split, config["embedding_dim"], config["metadata_fields"])
        s10.verify_split_counts(split, meta, validation, extraction)
        splits[split] = {"embeddings": emb, "label_ids": labels, "row_indices": indices}
        metas[split] = meta
        audit[split] = s10.load_window_observable_effects(split, meta)
        checks[split] = validation
    s10.validate_no_episode_overlap(metas)
    # Cross-check that all old L1/L2/L3/L5/L10 artifacts remain in their verified states.
    seq_validation = s11.validate_all_sequence_inputs(splits, extraction, s11.validate_step10_report_index())
    return splits, metas, audit, checks, seq_validation


def baseline_roots(args, mod):
    """Discover reference seeds by manifests, NEVER by their validation/test score."""
    candidates = {}
    manifest_path = args.baseline_multiseed_root / "multiseed_run_manifest.json"
    if manifest_path.is_file():
        multi = read_json(manifest_path)
        if multi.get("status") != "COMPLETE":
            raise ValueError(f"Historical multiseed runner is not COMPLETE: {manifest_path}")
        for info in multi.get("runs", []):
            if int(info["seed"]) in args.seeds:
                candidates[int(info["seed"])] = Path(info["output_root"]).resolve()
    for seed in args.seeds:
        candidates.setdefault(seed, (args.baseline_multiseed_root / f"seed_{seed}").resolve())
        if not (candidates[seed] / "run_manifest.json").is_file():
            run_path = args.baseline_main_root / "run_manifest.json"
            if run_path.is_file():
                record = read_json(run_path)
                if record.get("settings", {}).get("DEFAULT_RANDOM_SEED") == seed:
                    candidates[seed] = args.baseline_main_root.resolve()
    return candidates


def as_int_array(df: pd.DataFrame, key: str) -> np.ndarray:
    if key not in df:
        raise ValueError(f"Prediction missing column: {key}")
    a = df[key].to_numpy()
    if not pd.api.types.is_integer_dtype(a.dtype):
        raise ValueError(f"Prediction column must be integer: {key} ({a.dtype})")
    return a.astype(np.int64)


def authenticate_baselines(args, mod, source, split_names=("val",)):
    s11, s12 = mod["step11"], mod["step12"]
    roots = baseline_roots(args, mod)
    info = {}
    first_signature = None
    # Load one verified L10 sequence/metadata file for label and endpoint identity.
    old_targets = {}
    for split in split_names:
        with np.load(s11.sequence_npz_path(split, L10), allow_pickle=False) as z:
            old_targets[split] = {"end": z["end_row_indices"].astype(np.int64), "y": z["y"].astype(np.int64)}
        meta = s11.read_csv_robust(s11.sequence_metadata_path(split, L10))
        old_targets[split]["episode"] = meta["episode_id"].astype(str).to_numpy()
        if len(meta) != len(old_targets[split]["end"]):
            raise ValueError(f"Historical L10/{split} metadata length mismatch.")
    for seed in args.seeds:
        root = roots[seed]
        # validates status, seed, run identity, settings and upstream audit contract;
        # no requirement that the current Step 11 source hash equal its archival hash.
        run = s12.validate_seed_manifest(root, seed, ORIGINAL_LENGTHS, original_script=None, missing_aware=False)
        if run["source_contract"] != source:
            raise ValueError(f"Historical L10 seed {seed} uses different data/encoder/Step 10 provenance.")
        if Path(run["source_embedding_dir"]).resolve() != s11.EMBEDDING_DIR.resolve() or Path(run["source_sequence_dir"]).resolve() != s11.SEQUENCE_DIR.resolve():
            raise ValueError(f"Historical L10 seed {seed} points at different input directories.")
        locked = ("MAX_EPOCHS", "PATIENCE", "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY", "GRAD_CLIP_NORM",
                  "USE_CLASS_WEIGHTS", "LABEL_SMOOTHING", "D_MODEL", "NUM_HEADS", "NUM_ENCODER_LAYERS",
                  "DIM_FEEDFORWARD", "DROPOUT", "POOLING", "USE_MISSING_AWARE_TRAINING",
                  "USE_SHORTCUT_RESISTANT_TRAINING", "USE_LENGTH_AWARE_REGULARIZATION")
        for name in locked:
            if run["settings"].get(name) != getattr(s11, name):
                raise ValueError(f"Historical L10 seed {seed}: training setting {name} differs from current protocol.")
        signature = s12.comparable_protocol(run)
        if first_signature is not None and signature != first_signature:
            raise ValueError("Historical L10 seeds have different settings, source or runtime; cannot compare.")
        first_signature = signature
        pred_by_split = {}
        for split in split_names:
            rel = f"L10/predictions/{split}_predictions_L10.csv"
            path = root / rel
            expected_hash = run.get("artifacts", {}).get(rel)
            if not expected_hash or not path.is_file() or digest_file(path) != expected_hash:
                raise ValueError(f"Unverified original L10 predictions for seed {seed}: {path}")
            pred = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
            ends = as_int_array(pred, "sequence_end_row")
            y, yp = as_int_array(pred, "true_label_id"), as_int_array(pred, "pred_label_id")
            if (pred.columns.duplicated().any() or len(ends) != len(old_targets[split]["end"])
                    or len(np.unique(ends)) != len(ends)
                    or digest_ends(ends) != digest_ends(old_targets[split]["end"])):
                raise ValueError(f"L10/{split} endpoint IDs incomplete or duplicated for seed {seed}.")
            ordering = pd.DataFrame({"end": old_targets[split]["end"], "y": old_targets[split]["y"],
                                     "ep": old_targets[split]["episode"]})
            lookup = ordering.set_index("end")
            if (not np.array_equal(y, lookup.loc[ends, "y"].to_numpy(dtype=np.int64))
                    or not np.array_equal(pred["episode_id"].astype(str).to_numpy(), lookup.loc[ends, "ep"].to_numpy())):
                raise ValueError(f"L10/{split} label or episode identity mismatch for seed {seed}.")
            for field, expected in (("source_split", split), ("dataset_id", source["dataset_id"]),
                                    ("step8_run_id", source["step8_run_id"]),
                                    ("step9_run_id", source["step9_run_id"]),
                                    ("step10_run_id", source["step10_run_id"]),
                                    ("step11_run_id", run["run_id"])):
                if field not in pred or not pred[field].eq(expected).all():
                    raise ValueError(f"Historical L10/{split} prediction provenance mismatch for seed {seed}: {field}")
            if not np.isin(y, np.arange(6)).all() or not np.isin(yp, np.arange(6)).all():
                raise ValueError("L10 prediction labels must be canonical six-class indices.")
            pred_by_split[split] = pred
        info[seed] = {"root": root, "manifest_sha256": digest_file(root / "run_manifest.json"),
                      "run_id": run["run_id"], "trained_script_sha256": run["script_sha256"],
                      "matches_current_step11_source": run["script_sha256"] == digest_file(Path(mod["step11"].__file__)),
                      "predictions": pred_by_split}
        print(f"[Verified historical L10] seed={seed} run={run['run_id'][:12]} root={root}", flush=True)
    return info


def sequence_cache_identity(source: dict, paths: dict) -> dict:
    return {
        "schema": SCHEMA, "length": L20, "stride": 1, "materialize_X": False,
        "dataset_id": source["dataset_id"],
        "step8_run_id": source["step8_run_id"], "step9_run_id": source["step9_run_id"],
        "original_step10_run_id": source["step10_run_id"],
        "step9_artifacts": source["step9_artifacts"],
        "observability_audit": source["observability_audit"],
        "step10_source_sha256": digest_file(paths["step10"]),
        "step27_source_sha256": digest_file(Path(__file__)),
    }


def checked_cache(args, mod, source, paths, raw, metadata, audit):
    """Create and authenticate ONLY L20 indices/metadata/audit in a separate dir."""
    s10, s11 = mod["step10"], mod["step11"]
    cache = args.output_root / "sequence_cache"
    record_path = cache / "27_L20_sequence_manifest.json"
    identity = sequence_cache_identity(source, paths)
    if record_path.exists():
        prior = read_json(record_path)
        if prior.get("identity") != identity:
            raise ValueError("L20 sequence cache belongs to a different code/data identity. Use a different --output-root.")
        if prior.get("status") == "COMPLETE":
            required = {"27_L20_sequence_report.json"}
            for split in SPLITS:
                required.update({f"{split}_tst_sequences_L20.npz",
                                 f"{split}_tst_sequence_metadata_L20.csv",
                                 f"{split}_tst_sequence_observable_effects_L20.csv"})
            if not required.issubset(prior.get("artifacts", {})):
                raise ValueError("L20 cache manifest misses one or more required split artifacts.")
            for rel, expected in prior.get("artifacts", {}).items():
                p = cache / rel
                if not p.is_file() or digest_file(p) != expected:
                    raise ValueError(f"Corrupt L20 cached artifact, never silently reused: {p}")
            print("[Verified L20 sequence cache]", cache, flush=True)
            return load_cache(cache, mod, metadata, raw)
    elif cache.exists() and any(cache.iterdir()):
        raise ValueError(f"Foreign L20 sequence cache without manifest: {cache}")
    cache.mkdir(parents=True, exist_ok=True)
    save_json(record_path, {"identity": identity, "status": "RUNNING", "started": now_utc()})
    reports = {}
    s10.OUTPUT_DIR = cache
    if s10.MATERIALIZE_X or s10.STRIDE != 1 or not s10.SAVE_SEQUENCE_METADATA:
        raise ValueError("Step 10 source changed the required index-only/stride-one/metadata protocol.")
    id_to_label = {i: v for i, v in enumerate(s10.DEFAULT_LABEL_ORDER)}
    try:
        for split in SPLITS:
            report = s10.process_split_for_length(
                split_name=split, embeddings=raw[split]["embeddings"],
                label_ids=raw[split]["label_ids"], metadata_df=metadata[split],
                sequence_length=L20, id_to_label=id_to_label, observable_effects=audit[split])
            reports[split] = report
        save_json(cache / "27_L20_sequence_report.json", {"status": "COMPLETE", "identity": identity, "splits": reports})
        files = [p for p in cache.rglob("*") if p.is_file() and p != record_path]
        hashes = {p.relative_to(cache).as_posix(): digest_file(p) for p in files}
        save_json(record_path, {"identity": identity, "status": "COMPLETE", "completed": now_utc(),
                                "artifacts": hashes, "splits": {split: {"num_sequences": int(reports[split]["num_sequences"]),
                                                                        "num_episodes": int(reports[split]["num_episodes"])} for split in SPLITS}})
    except BaseException as exc:
        save_json(record_path, {"identity": identity, "status": "FAILED", "error": repr(exc), "updated": now_utc()})
        raise
    return load_cache(cache, mod, metadata, raw)


def load_cache(cache: Path, mod, metadata, raw):
    s10, s11 = mod["step10"], mod["step11"]
    sequences, seq_metas, effects = {}, {}, {}
    for split in SPLITS:
        name = cache / f"{split}_tst_sequences_L20.npz"
        with np.load(name, allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        if "X" in data:
            raise ValueError("L20 cache materialized full embedding cubes; expected indices only.")
        seq_metas[split] = s11.read_csv_robust(cache / f"{split}_tst_sequence_metadata_L20.csv")
        effects[split] = s11.read_csv_robust(cache / f"{split}_tst_sequence_observable_effects_L20.csv")
        s11.validate_sequence_alignment(split, L20, data, seq_metas[split], metadata[split])
        s11.validate_sequence_observable_effects(effects[split], seq_metas[split], L20, split)
        if not np.array_equal(data["y"], raw[split]["label_ids"][data["end_row_indices"]]):
            raise ValueError(f"L20 {split} labels do not equal Step 9 episode-scenario endpoint labels.")
        if not {"y", "label_ids", "sequence_indices", "start_row_indices", "end_row_indices"}.issubset(data):
            raise ValueError("L20 sequence data missing required arrays.")
        sequences[split] = data
    all_sets = {split: set(seq_metas[split]["episode_id"].astype(str).tolist()) for split in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            if all_sets[a] & all_sets[b]:
                raise ValueError(f"L20 episode split leakage: {a}/{b}")
    return sequences, seq_metas, effects


def paired_reference(mod, raw, window_audit, seq_data, split: str) -> dict:
    """Descriptive references and coverage; never model inputs or model-selection scores."""
    s11 = mod["step11"]
    newest = seq_data[split]
    ends20 = newest["end_row_indices"]
    y20 = newest["y"].astype(np.int64)
    l10 = s11.load_sequence_split(split, L10, raw[split], raw[split]["embeddings"].shape[1])
    old_index = pd.Series(np.arange(len(l10["end_row_indices"])), index=l10["end_row_indices"])
    indexes10 = old_index.reindex(ends20)
    if indexes10.isna().any():
        raise ValueError(f"{split}: L20 endpoints are not a strict subset of L10 endpoints.")
    positions10 = indexes10.to_numpy(dtype=np.int64)
    indices10 = l10["sequence_indices"][positions10]
    if not np.array_equal(l10["y"][positions10], y20) or not np.array_equal(indices10[:, -1], ends20):
        raise ValueError("Matched L10/L20 endpoints differ in labels or final observation.")
    flags = window_audit[split]["audit_observable_effect_present"].to_numpy(dtype=np.int64)
    hit10 = (flags[indices10].sum(axis=1) > 0)
    hit20 = (flags[newest["sequence_indices"]].sum(axis=1) > 0)
    if np.any(hit10 & ~hit20):
        raise ValueError("Adding earlier context cannot remove an in-context paired-nominal change.")
    atk = y20 != 0
    results = {"split": split, "common_endpoints": int(len(ends20)),
               "note": "Nondeployable descriptive paired-nominal change; not a bound, model input or selection target.",
               "attack_labeled_endpoints": int(atk.sum()),
               "L10_in_context_change_attack": int(np.count_nonzero(hit10 & atk)),
               "L20_in_context_change_attack": int(np.count_nonzero(hit20 & atk)),
               "L20_additional_evidence_attack": int(np.count_nonzero((~hit10) & hit20 & atk)),
               "L10_coverage_attack": float(hit10[atk].mean()) if atk.any() else None,
               "L20_coverage_attack": float(hit20[atk].mean()) if atk.any() else None,
               "L20_additional_coverage_attack": float(((~hit10) & hit20)[atk].mean()) if atk.any() else None,
               "endpoint_sha256": digest_ends(ends20)}
    for length, hit in ((L10, hit10), (L20, hit20)):
        prediction = np.where(hit & atk, y20, 0).astype(np.int64)
        summary = s11.compute_metrics(y20, prediction, 6, {i: str(i) for i in range(6)})[0]
        results[f"L{length}_nondeployable_reference_macro_f1"] = float(summary["macro_f1"])
    return results


def check_baseline_on_new_endpoints(baseline: pd.DataFrame, seq: dict, seqmeta: pd.DataFrame, mod):
    s11 = mod["step11"]
    ends = seq["end_row_indices"].astype(np.int64)
    compare = baseline.set_index("sequence_end_row", verify_integrity=True)
    if not pd.Index(ends).isin(compare.index).all():
        raise ValueError("Historical L10 predictions do not cover all L20 eligible endpoints.")
    matched = compare.loc[ends].reset_index()
    if (not np.array_equal(as_int_array(matched, "sequence_end_row"), ends)
            or not np.array_equal(as_int_array(matched, "true_label_id"), seq["y"])
            or not np.array_equal(matched["episode_id"].astype(str).to_numpy(), seqmeta["episode_id"].astype(str).to_numpy())):
        raise ValueError("Historical L10 predictions disagree with new L20 endpoints/episode labels.")
    return matched


def training_identity(args, paths, source, baselines):
    s11_hash = digest_file(paths["step11"])
    return {
        "schema": SCHEMA, "pilot_source_sha256": digest_file(Path(__file__)),
        "step10_source_sha256": digest_file(paths["step10"]),
        "step11_source_sha256": s11_hash, "step12_source_sha256": digest_file(paths["step12"]),
        "source_contract": source, "selected_seeds": args.seeds,
        "reference_length": L10, "new_length": L20,
        "training_population": "all L20-eligible training endpoints (different from historical L10 training endpoints)",
        "selection": "L20 validation macro-F1 only; not evidence availability or held-out test",
        "historical_L10_seed_roots": {str(seed): str(baselines[seed]["root"]) for seed in args.seeds},
        "historical_L10_seed_manifests": {str(seed): baselines[seed]["manifest_sha256"] for seed in args.seeds},
        "historical_L10_runs": {str(seed): baselines[seed]["run_id"] for seed in args.seeds},
        "head_config": {"d_model": 128, "heads": 4, "encoder_layers": 1, "feedforward": 256,
                        "dropout": 0.25, "readout": "last"},
        "training_config": {"epochs": 30, "patience": 7, "batch": 256, "lr": 7e-5,
                            "weight_decay": 5e-4, "label_smoothing": 0.03,
                            "lr_scheduler": "ReduceLROnPlateau(max, factor=.5, patience=2, min_lr=1e-6)",
                            "selection": "val_macro_f1", "missing_aware": False},
    }


def verify_seed_complete(seedroot: Path, identity_hash: str, seed: int) -> dict:
    manifest = read_json(seedroot / "27_seed_manifest.json")
    if manifest.get("identity_hash") != identity_hash or manifest.get("seed") != seed or manifest.get("status") != "COMPLETE":
        raise ValueError(f"Existing seed is not a verified COMPLETE run: {seedroot}")
    for rel, expected in manifest.get("artifacts", {}).items():
        path = seedroot / rel
        if not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"Corrupt prior seed artifact: {path}")
    if not manifest.get("artifacts") or not {"best_L20.pt", "history.csv", "val_predictions.csv", "val_metrics.json"}.issubset(manifest["artifacts"]):
        raise ValueError("Incomplete seed manifest inventory.")
    return manifest


def predicted_dataframe(evaluated: dict, seqmeta: pd.DataFrame, seed: int, split: str, source: dict) -> pd.DataFrame:
    sequence_ids = evaluated["sequence_ids"].astype(np.int64)
    if len(sequence_ids) != len(seqmeta) or not np.array_equal(sequence_ids, np.arange(len(seqmeta))):
        raise ValueError("Model evaluation order does not match L20 sequence metadata.")
    truth = evaluated["y_true"].astype(np.int64)
    pred = evaluated["y_pred"].astype(np.int64)
    if not np.array_equal(truth, pd.to_numeric(seqmeta["label_id"], errors="raise").to_numpy(dtype=np.int64)):
        raise ValueError("Evaluation ground truth differs from L20 sequence metadata.")
    return pd.DataFrame({
        "sequence_end_row": pd.to_numeric(seqmeta["sequence_end_row"], errors="raise").to_numpy(dtype=np.int64),
        "episode_id": seqmeta["episode_id"].astype(str).to_numpy(),
        "true_label_id": truth, "pred_label_id": pred,
        "confidence": evaluated["probs"].max(axis=1),
        "source_split": split, "sequence_length": L20, "seed": seed,
        "dataset_id": source["dataset_id"], "step8_run_id": source["step8_run_id"],
        "step9_run_id": source["step9_run_id"], "original_step10_run_id": source["step10_run_id"],
    })


def train_seed(seed: int, root: Path, identity_hash: str, mod, source, raw, seq, seqmeta, args):
    import torch
    s11 = mod["step11"]
    seedroot = root / f"seed_{seed}"
    seedmanifest = seedroot / "27_seed_manifest.json"
    if seedmanifest.exists():
        previous = read_json(seedmanifest)
        if previous.get("status") == "COMPLETE":
            return verify_seed_complete(seedroot, identity_hash, seed)
        if not args.retry_failed:
            raise ValueError(f"Incomplete local seed {seed}; stop old job and rerun --retry-failed to archive/restart.")
        archived = root / "_incomplete" / (f"seed_{seed}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        archived.parent.mkdir(parents=True, exist_ok=True)
        seedroot.rename(archived)
        print(f"[Archived incomplete seed] {archived}", flush=True)
    elif seedroot.exists():
        raise ValueError(f"Seed directory exists without own manifest: {seedroot}")
    seedroot.mkdir(parents=True, exist_ok=False)
    save_json(seedmanifest, {"status": "RUNNING", "identity_hash": identity_hash, "seed": seed, "started": now_utc()})
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s11.set_all_seeds(seed)
        num_classes = 6
        model_config = s11.ModelConfig(
            input_dim=raw["train"]["embeddings"].shape[1], num_classes=num_classes,
            sequence_length=L20, temporal_context_seconds=L20)
        model = s11.TimeSeriesTransformerClassifier(model_config).to(device)
        criterion = s11.make_cross_entropy_loss(None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=s11.LEARNING_RATE, weight_decay=s11.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
        scaler = s11.make_grad_scaler(device)
        tr = s11.make_loader(raw["train"]["embeddings"], seq["train"], s11.BATCH_SIZE, True, device)
        va = s11.make_loader(raw["val"]["embeddings"], seq["val"], s11.BATCH_SIZE, False, device)
        selected, best_epoch, bad, epochs = None, 0, 0, []
        ckpt = seedroot / "best_L20.pt"
        started = time.monotonic()
        print(f"\n[Step 27 L20 TRAIN] seed={seed}  train={len(seq['train']['y']):,}  val={len(seq['val']['y']):,}  device={device}", flush=True)
        for epoch in range(1, s11.MAX_EPOCHS + 1):
            t0 = time.monotonic()
            train = s11.train_one_epoch(model, tr, optimizer, criterion, device, scaler, L20)
            validation = s11.evaluate_model(model, va, criterion, device, num_classes,
                                             {i: str(i) for i in range(num_classes)}, desc="val")
            summary = validation["summary"]
            scheduler.step(float(summary["macro_f1"]))
            epochs.append({"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]),
                           "train_loss": float(train["loss"]), "train_accuracy": float(train["accuracy"]),
                           "val_loss": float(summary["loss"]), "val_accuracy": float(summary["accuracy"]),
                           "val_macro_f1": float(summary["macro_f1"]), "epoch_seconds": time.monotonic() - t0})
            print(f"[L20 seed {seed}] epoch {epoch}/{s11.MAX_EPOCHS}, val F1={summary['macro_f1']:.6f}, "
                  f"best={best_epoch}, time={epochs[-1]['epoch_seconds']:.1f}s", flush=True)
            if s11.metric_is_better(summary, selected, s11.PRIMARY_SELECTION_METRIC):
                selected = {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}
                best_epoch, bad = epoch, 0
                s11.save_checkpoint_atomic({
                    "model_state_dict": model.state_dict(), "model_config": asdict(model_config),
                    "train_config": asdict(s11.TrainConfig(random_seed=seed)),
                    "best_epoch": best_epoch, "best_val_summary": selected,
                    "seed": seed, "schema": SCHEMA, "identity_hash": identity_hash,
                    "dataset_id": source["dataset_id"], "source_contract": source,
                }, ckpt)
            else:
                bad += 1
                if bad >= s11.PATIENCE:
                    print(f"[L20 seed {seed}] Early stop at {epoch}", flush=True)
                    break
        elapsed = time.monotonic() - started
        saved = s11.load_checkpoint_safely(ckpt, device)
        if saved.get("seed") != seed or saved.get("identity_hash") != identity_hash or saved.get("model_config") != asdict(model_config):
            raise ValueError("Saved L20 checkpoint identity/config mismatch.")
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        final_val = s11.evaluate_model(model, va, criterion, device, num_classes,
                                       {i: str(i) for i in range(num_classes)}, desc="val_selected")
        if not math.isclose(final_val["summary"]["macro_f1"], selected["macro_f1"], abs_tol=1e-8):
            raise ValueError("Reloaded selected L20 checkpoint does not reproduce validation macro-F1.")
        save_csv(seedroot / "history.csv", pd.DataFrame(epochs))
        save_csv(seedroot / "val_predictions.csv", predicted_dataframe(final_val, seqmeta["val"], seed, "val", source))
        save_json(seedroot / "val_metrics.json", {"seed": seed, "best_epoch": best_epoch, "trained_epochs": len(epochs),
                   "training_elapsed_seconds": elapsed, "val": final_val["summary"],
                   "n_train_sequences": len(seq["train"]["y"]), "n_val_sequences": len(seq["val"]["y"]),
                   "head_training_protocol": "Step 11 exact optimizer/epoch/early-stopping/augmentation flags; L20 only"})
        hashes = {name: digest_file(seedroot / name) for name in
                  ("best_L20.pt", "history.csv", "val_predictions.csv", "val_metrics.json")}
        manifest = {"status": "COMPLETE", "identity_hash": identity_hash, "seed": seed, "completed": now_utc(),
                    "artifacts": hashes, "best_epoch": best_epoch, "training_elapsed_seconds": elapsed,
                    "device": str(device), "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    "amp": bool(s11.USE_AMP and device.type == "cuda")}
        save_json(seedmanifest, manifest)
        return verify_seed_complete(seedroot, identity_hash, seed)
    except BaseException as exc:
        save_json(seedmanifest, {"status": "FAILED", "identity_hash": identity_hash, "seed": seed,
                                 "failure_time": now_utc(), "error": type(exc).__name__ + ": " + str(exc)})
        raise


def compare_one(mod, old: pd.DataFrame, new: pd.DataFrame, seq: dict, meta: pd.DataFrame, seed: int, split: str):
    s11 = mod["step11"]
    ref = check_baseline_on_new_endpoints(old, seq, meta, mod)
    if not np.array_equal(as_int_array(new, "sequence_end_row"), seq["end_row_indices"]):
        raise ValueError(f"L20/{split} predictions out of sequence-end order.")
    if (not np.array_equal(as_int_array(new, "true_label_id"), as_int_array(ref, "true_label_id"))
            or not np.array_equal(new["episode_id"].astype(str), ref["episode_id"].astype(str))):
        raise ValueError("L10/L20 truths or episode identifiers differ on matched endpoints.")
    truth = as_int_array(ref, "true_label_id")
    l10_pred, l20_pred = as_int_array(ref, "pred_label_id"), as_int_array(new, "pred_label_id")
    labels = {i: str(i) for i in range(6)}
    m10 = s11.compute_metrics(truth, l10_pred, 6, labels)[0]
    m20 = s11.compute_metrics(truth, l20_pred, 6, labels)[0]
    if not np.isin(truth, np.arange(6)).all() or set(truth.tolist()) != set(range(6)):
        raise ValueError("Not all six classes are represented in matched evaluation population.")
    return {"seed": seed, "split": split, "common_endpoints": int(len(truth)),
            "common_episodes": int(len(np.unique(meta["episode_id"].astype(str)))),
            "endpoint_sha256": digest_ends(seq["end_row_indices"]),
            "L10_macro_f1": float(m10["macro_f1"]), "L20_macro_f1": float(m20["macro_f1"]),
            "delta_L20_minus_L10": float(m20["macro_f1"] - m10["macro_f1"]),
            "L10_accuracy": float(m10["accuracy"]), "L20_accuracy": float(m20["accuracy"]),
            "L10_weighted_f1": float(m10["weighted_f1"]), "L20_weighted_f1": float(m20["weighted_f1"])}


def aggregate_rows(rows: list[dict], evidence: dict, split: str):
    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    if (len(frame) != 3 or frame["common_endpoints"].nunique() != 1
            or frame["common_episodes"].nunique() != 1 or frame["endpoint_sha256"].nunique() != 1):
        raise ValueError("Three seeds must use exactly the same evaluation population.")
    summary = {"split": split, "num_head_seeds": 3,
               "common_endpoints": int(frame.common_endpoints.iloc[0]),
               "common_episodes": int(frame.common_episodes.iloc[0]),
               "endpoint_sha256": frame.endpoint_sha256.iloc[0],
               "evidence_availability": evidence,
               "statistical_scope": "Exploratory 3-seed mean/sample SD conditional on fixed dataset/encoder; NOT bootstrap CI.",
               "limitations": ["L10 and L20 training cohorts differ in early episode endpoints.",
                               "Identical numeric seed IDs do not guarantee paired parameter initializations.",
                               "No test-set-based selection is permitted by the pilot stage."]}
    for metric in ("macro_f1", "accuracy", "weighted_f1"):
        for length in (L10, L20):
            name = f"L{length}_{metric}"
            summary[f"{name}_mean"] = float(frame[name].mean())
            summary[f"{name}_sd"] = float(frame[name].std(ddof=1))
    summary["delta_L20_minus_L10_mean"] = float(frame.delta_L20_minus_L10.mean())
    summary["delta_L20_minus_L10_sd"] = float(frame.delta_L20_minus_L10.std(ddof=1))
    return frame, summary


def write_review_zip(root: Path, name: str, seeds: list[int], include_test: bool) -> Path:
    import zipfile
    output = root / name
    with tempfile.NamedTemporaryFile(dir=root, suffix=".zip.tmp", delete=False) as fp:
        tmp = Path(fp.name)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.write(Path(__file__), f"source/{Path(__file__).name}")
            top_files = ["27_run_manifest.json", "27_val_seed_comparison.csv", "27_val_summary.json",
                         "27_evidence_availability_val.json", "sequence_cache/27_L20_sequence_manifest.json",
                         "sequence_cache/27_L20_sequence_report.json"]
            if include_test:
                top_files += ["27_final_test.json", "27_test_seed_comparison.csv", "27_test_summary.json",
                              "27_evidence_availability_test.json"]
            for rel in top_files:
                p = root / rel
                if p.is_file():
                    z.write(p, rel)
            for seed in seeds:
                for rel in ("27_seed_manifest.json", "val_metrics.json", "history.csv", "val_predictions.csv"):
                    p = root / f"seed_{seed}" / rel
                    if p.is_file():
                        z.write(p, f"seed_{seed}/{rel}")
                if include_test:
                    p = root / f"seed_{seed}" / "test_predictions.csv"
                    if p.is_file():
                        z.write(p, f"seed_{seed}/test_predictions.csv")
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return output


class ExclusiveJobLock:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / ".27_running.lock"

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x", encoding="utf-8") as fp:
                json.dump({"pid": os.getpid(), "hostname": socket.gethostname(), "time": now_utc()}, fp)
        except FileExistsError as exc:
            raise RuntimeError(f"Step 27 appears active: {self.path}. Check process has stopped before removing stale lock.") from exc
        return self

    def __exit__(self, typ, value, tb):
        self.path.unlink(missing_ok=True)


def preflight(mod, extraction, source, raw, metas, window_audit, baselines):
    import torch
    s10, s11 = mod["step10"], mod["step11"]
    id_to_label = {i: str(s) for i, s in enumerate(extraction["label_order"])}
    val_arrays, val_meta, summary = s10.build_sequences_for_split(
        "val", raw["val"]["embeddings"], raw["val"]["label_ids"], metas["val"],
        L20, 1, 1, id_to_label)
    evidence = paired_reference(mod, raw, window_audit, {"val": val_arrays}, "val")
    for seed, base in baselines.items():
        check_baseline_on_new_endpoints(base["predictions"]["val"], val_arrays, val_meta, mod)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s11.set_all_seeds(20260610)
    configuration = s11.ModelConfig(input_dim=int(extraction["embedding_dim"]), num_classes=6,
                                    sequence_length=L20, temporal_context_seconds=20)
    network = s11.TimeSeriesTransformerClassifier(configuration).to(device)
    model_indices = val_arrays["sequence_indices"][:2]
    minibatch = torch.from_numpy(raw["val"]["embeddings"][model_indices]).to(device)
    with torch.no_grad():
        result = network(minibatch)
    if result.shape != (len(model_indices), 6) or not torch.isfinite(result).all():
        raise ValueError("L20 GPU/CPU forward smoke test returned invalid logits.")
    print("\n" + "=" * 68)
    print("STEP 27 PREFLIGHT PASSED -- NO OUTPUT FILES CREATED")
    print(f"dataset_id: {source['dataset_id']}")
    print(f"embedding_dim: {extraction['embedding_dim']}; L20 val endpoints: {len(val_arrays['y']):,}")
    print(f"val episodes: {val_meta['episode_id'].nunique():,}; original L10 seeds: {list(baselines)}")
    print(f"L10 -> L20 availability: {evidence['L10_coverage_attack']:.4f} -> {evidence['L20_coverage_attack']:.4f}")
    print(f"TST forward batch: {tuple(result.shape)}, device={device}")
    print("The historical L10 runs and all Step 8/9/10 input checks passed.")
    print("No L20 head has been trained; no test score has been read.")
    print("=" * 68)


def pilot(args, mod, paths, source, raw, metas, audit, baselines):
    root = args.output_root
    identity = training_identity(args, paths, source, baselines)
    identity_hash = digest_json(identity)
    job_manifest = root / "27_run_manifest.json"
    with ExclusiveJobLock(root):
        sealed_test = root / "27_final_test.json"
        if sealed_test.is_file() and read_json(sealed_test).get("status") == "COMPLETE":
            raise ValueError("This pilot has already unlocked its held-out test. Do not rerun validation after test inspection.")
        existing = read_json(job_manifest) if job_manifest.is_file() else None
        if existing is None and any(x.name != ".27_running.lock" for x in root.iterdir()):
            raise ValueError(f"Output root already contains foreign files without Step 27 manifest: {root}")
        if existing is not None and existing.get("identity_hash") != identity_hash:
            raise ValueError("Output root belongs to different Step27 code, seed list, provenance or protocol.")
        manifest = existing or {"schema": SCHEMA, "identity": identity, "identity_hash": identity_hash,
                                "started": now_utc(), "seeds": args.seeds}
        manifest["status"] = "RUNNING"
        save_json(job_manifest, manifest)
        try:
            seq, seqmeta, cache_audit = checked_cache(args, mod, source, paths, raw, metas, audit)
            evidence = paired_reference(mod, raw, audit, seq, "val")
            evidence["episodes"] = int(seqmeta["val"]["episode_id"].nunique())
            save_json(root / "27_evidence_availability_val.json", evidence)
            # Baseline coverage rechecked before any new GPU training.
            for seed in args.seeds:
                if not baselines[seed]["matches_current_step11_source"]:
                    print(f"[SOURCE CODE WARNING] Historical L10 seed {seed} was trained by another Step 11 source hash. "
                          "Hyperparameters and original data lineage were checked, but exact code equivalence "
                          "is NOT established. Interpret L10/L20 as exploratory, not a controlled source-identical comparison.", flush=True)
            for seed in args.seeds:
                check_baseline_on_new_endpoints(baselines[seed]["predictions"]["val"],
                                                seq["val"], seqmeta["val"], mod)
            for seed in args.seeds:
                train_seed(seed, root, identity_hash, mod, source, raw, seq, seqmeta, args)
            rows = []
            for seed in args.seeds:
                verify_seed_complete(root / f"seed_{seed}", identity_hash, seed)
                new = pd.read_csv(root / f"seed_{seed}" / "val_predictions.csv", encoding="utf-8-sig", keep_default_na=False)
                row = compare_one(mod, baselines[seed]["predictions"]["val"], new,
                                  seq["val"], seqmeta["val"], seed, "val")
                row.update({k: read_json(root / f"seed_{seed}" / "val_metrics.json")[k]
                            for k in ("best_epoch", "trained_epochs", "training_elapsed_seconds")})
                rows.append(row)
            frame, summary = aggregate_rows(rows, evidence, "val")
            summary.update({"selected_head_seeds": args.seeds, "dataset_id": source["dataset_id"],
                            "identity_hash": identity_hash, "created": now_utc(),
                            "decision_rule": "Inspect validation only; separately decide whether to run --final-test."})
            save_csv(root / "27_val_seed_comparison.csv", frame)
            save_json(root / "27_val_summary.json", summary)
            manifest.update({"status": "COMPLETE", "completed": now_utc(),
                            "val_summary_sha256": digest_file(root / "27_val_summary.json"),
                            "val_comparison_sha256": digest_file(root / "27_val_seed_comparison.csv"),
                            "L20_sequences_sha256": digest_file(root / "sequence_cache" / "27_L20_sequence_manifest.json"),
                            "baseline_source_match": {str(seed): baselines[seed]["matches_current_step11_source"] for seed in args.seeds}})
            save_json(job_manifest, manifest)
            review = write_review_zip(root, "27_L20_VALIDATION_REVIEW.zip", args.seeds, include_test=False)
        except BaseException as exc:
            manifest.update(status="FAILED", last_error=f"{type(exc).__name__}: {exc}", updated=now_utc())
            save_json(job_manifest, manifest)
            raise
    print("\n" + "=" * 75)
    print("STEP 27 VALIDATION PILOT COMPLETED -- TEST NOT EVALUATED")
    print(frame[["seed", "common_endpoints", "L10_macro_f1", "L20_macro_f1", "delta_L20_minus_L10", "best_epoch", "training_elapsed_seconds"]].to_string(index=False))
    print(f"L10 mean val macro-F1: {summary['L10_macro_f1_mean']:.6f} +/- {summary['L10_macro_f1_sd']:.6f}")
    print(f"L20 mean val macro-F1: {summary['L20_macro_f1_mean']:.6f} +/- {summary['L20_macro_f1_sd']:.6f}")
    print(f"Mean delta (L20-L10): {summary['delta_L20_minus_L10_mean']:+.6f}")
    print(f"Matched attack evidence coverage L10/L20: {evidence['L10_coverage_attack']:.4f} / {evidence['L20_coverage_attack']:.4f}")
    print(f"Results: {root}\nReview ZIP: {review}")
    print("Do NOT use test to decide whether L20 merits inclusion; decide from validation first.")
    print("=" * 75)


def final_test(args, mod, paths, source, raw, metas, audit, baselines_val):
    """Locked post-pilot inspection: this option must be explicitly requested."""
    import torch
    root = args.output_root
    identity = training_identity(args, paths, source, baselines_val)
    identity_hash = digest_json(identity)
    pilot_record = read_json(root / "27_run_manifest.json")
    if pilot_record.get("status") != "COMPLETE" or pilot_record.get("identity_hash") != identity_hash:
        raise ValueError("Held-out test requires a COMPLETE matching validation pilot and identical seeds/code/data.")
    if (digest_file(root / "27_val_summary.json") != pilot_record.get("val_summary_sha256")
            or digest_file(root / "27_val_seed_comparison.csv") != pilot_record.get("val_comparison_sha256")):
        raise ValueError("Validation summary changed after selection. Test remains locked.")
    # Test baseline files are not read until the explicit --final-test user action.
    baselines_test = authenticate_baselines(args, mod, source, split_names=("test",))
    with ExclusiveJobLock(root):
        final_path = root / "27_final_test.json"
        if final_path.exists():
            prior = read_json(final_path)
            if prior.get("status") == "COMPLETE":
                print(f"[Final test previously completed] {root / '27_test_summary.json'}")
                return
            if prior.get("identity_hash") != identity_hash:
                raise ValueError("Existing final test record belongs to different pilot identity.")
        save_json(final_path, {"status": "RUNNING", "identity_hash": identity_hash,
                               "unlocked_explicitly": now_utc(),
                               "note": "Validation-selected exploratory pilot; test was not used for L20 selection."})
        try:
            seq, seqmeta, _ = load_cache(root / "sequence_cache", mod, metas, raw)
            evidence = paired_reference(mod, raw, audit, seq, "test")
            evidence["episodes"] = int(seqmeta["test"]["episode_id"].nunique())
            save_json(root / "27_evidence_availability_test.json", evidence)
            rows = []
            for seed in args.seeds:
                seedroot = root / f"seed_{seed}"
                verify_seed_complete(seedroot, identity_hash, seed)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ckpt = mod["step11"].load_checkpoint_safely(seedroot / "best_L20.pt", device)
                if ckpt.get("identity_hash") != identity_hash or ckpt.get("seed") != seed:
                    raise ValueError("Selected L20 checkpoint differs from completed validation pilot.")
                model_config = mod["step11"].ModelConfig(**ckpt["model_config"])
                if model_config.sequence_length != L20 or model_config.num_classes != 6:
                    raise ValueError("Invalid selected L20 model configuration.")
                model = mod["step11"].TimeSeriesTransformerClassifier(model_config).to(device)
                model.load_state_dict(ckpt["model_state_dict"])
                loader = mod["step11"].make_loader(raw["test"]["embeddings"], seq["test"],
                                                    mod["step11"].BATCH_SIZE, False, device)
                evaluation = mod["step11"].evaluate_model(
                    model, loader, mod["step11"].make_cross_entropy_loss(None), device, 6,
                    {i: str(i) for i in range(6)}, desc="test_explicit_final")
                new = predicted_dataframe(evaluation, seqmeta["test"], seed, "test", source)
                save_csv(seedroot / "test_predictions.csv", new)
                rows.append(compare_one(mod, baselines_test[seed]["predictions"]["test"],
                                        new, seq["test"], seqmeta["test"], seed, "test"))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            frame, summary = aggregate_rows(rows, evidence, "test")
            summary.update({"identity_hash": identity_hash, "validation_selected_before_test": True,
                            "test_unlocked": now_utc(),
                            "statistical_scope": "Exploratory held-out check conditional on post-hoc decision to extend L; three head seeds only."})
            save_csv(root / "27_test_seed_comparison.csv", frame)
            save_json(root / "27_test_summary.json", summary)
            save_json(final_path, {"status": "COMPLETE", "identity_hash": identity_hash,
                                   "completed": now_utc(), "val_summary_sha256": pilot_record["val_summary_sha256"],
                                   "test_summary_sha256": digest_file(root / "27_test_summary.json"),
                                   "test_comparison_sha256": digest_file(root / "27_test_seed_comparison.csv")})
            review = write_review_zip(root, "27_L20_FINAL_TEST_REVIEW.zip", args.seeds, include_test=True)
        except BaseException as exc:
            save_json(final_path, {"status": "FAILED", "identity_hash": identity_hash,
                                   "updated": now_utc(), "error": f"{type(exc).__name__}: {exc}"})
            raise
    print("\nSTEP 27 EXPLICIT TEST EVALUATION COMPLETE")
    print(frame[["seed", "common_endpoints", "L10_macro_f1", "L20_macro_f1", "delta_L20_minus_L10"]].to_string(index=False))
    print(f"Test delta L20-L10 (mean over 3 head seeds): {summary['delta_L20_minus_L10_mean']:+.6f}")
    print(f"Review ZIP: {review}")
    print("Exploratory three-seed results are NOT a formal paired-bootstrap confidence interval.")


def main():
    args = parse_args()
    paths, mod = load_source_modules(args)
    print("=" * 75)
    print("STEP 27 | L20 TST extension (fixed fine-tuned embeddings; old runs read-only)")
    print(f"mode={'PRE-FLIGHT' if args.preflight else 'FINAL-TEST' if args.final_test else 'VALIDATION PILOT'}")
    print(f"seeds={args.seeds}, output={args.output_root}")
    print("Original sources:", {key: str(p) for key, p in paths.items()})
    extraction, old_index, source = authenticate_inputs(args, mod)
    raw, metas, window_audit, window_checks, sequence_checks = get_raw_splits(mod, extraction)
    baselines = authenticate_baselines(args, mod, source, split_names=("val",))
    if args.preflight:
        preflight(mod, extraction, source, raw, metas, window_audit, baselines)
    elif args.final_test:
        final_test(args, mod, paths, source, raw, metas, window_audit, baselines)
    else:
        pilot(args, mod, paths, source, raw, metas, window_audit, baselines)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[STEP 27 STOPPED] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
