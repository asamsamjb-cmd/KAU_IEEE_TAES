# -*- coding: utf-8 -*-
"""Step 12: repeated TST-head seeds on one verified Step 8/9/10 representation.

Run AFTER the current Step 11 finishes, from the project directory:
    python "12.multiseed_tst_runner.py"

Keep the delivered 11.train_tst_classifier.py beside this file in the project.
The delivered name and legacy spaced/underscore names are checked;
only a revised, seed-capable source can be selected. An explicit --train_script
is never silently replaced. Source compatibility and actual --help are checked
BEFORE creating the output directory or launching training.

The revised Step 11 is called unchanged with --seed; no patched copies are made.
The completed main Step 11 run is reused read-only for its matching seed by default.
Other seeds run sequentially under outputs/shared_nominal_v2/tst_training_finetuned_multiseed.
All requested seed x L combinations must be COMPLETE and comparable before aggregation.
Paper tables use the SAME Step 10 common endpoints. SD is sample SD across head seeds,
conditional on the fixed encoder/data; it is NOT a bootstrap confidence interval.

Useful options (legacy underscore spellings remain supported):
    --train_script "11.train_tst_classifier.py"
    --seeds 20260610 20260611 --lengths 1 2 3 5 10
    --aggregate_only                  # validate and aggregate; never train
    --skip_existing                   # compatibility flag; reuse is always verified
    --main_run PATH                   # completed first/main run to include read-only
    --no_reuse_main                   # train all requested seeds in the new root
    --retry_failed                    # only after the previous child has stopped

Completed seeds survive interruption. --retry_failed archives an incomplete local
seed directory and restarts that entire seed (all selected L); no epoch-level resume.
Do not run two copies against the same output root. Existing foreign/corrupt results
are rejected, not silently overwritten. Change --output_root for a different protocol.
A reused main seed may live outside seed_<seed>/; authoritative paths are recorded in
multiseed_run_manifest.json and multiseed_raw_results.csv (seed_output_root).
Counterfactual observable-change diagnostics remain per-seed, post-evaluation
outputs. Their annotations must agree across seeds for each L. They are never
used as features, seed-selection criteria, or claims of an oracle error bound.
Aggregate-only verifies saved artifacts without requiring a model backend.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
import zipfile

import numpy as np
import pandas as pd

DEFAULT_TRAIN_SCRIPT = "11.train_tst_classifier.py"
LEGACY_TRAIN_SCRIPT = "11. train_tst_classifier.py"
OLDER_TRAIN_SCRIPT = "11_train_tst_classifier.py"
REQUIRED_TRAIN_FLAGS = {"--seed", "--lengths", "--output-root", "--missing-aware", "--no-missing-aware"}
DEFAULT_SEEDS = list(range(20260610, 20260620))
DEFAULT_LENGTHS = [1, 2, 3, 5, 10]
DEFAULT_OUTPUT_ROOT = "outputs/shared_nominal_v2/tst_training_finetuned_multiseed"
DEFAULT_MAIN_RUN = "outputs/shared_nominal_v2/tst_training_finetuned_main"
OBSERVABLE_EFFECT_COLUMNS = ["audit_current_observable_effect", "audit_history_effect_window_count",
                             "audit_context_effect_window_count", "audit_effect_stratum"]
METRIC_COLUMNS = [
    "test_accuracy", "test_macro_f1", "test_weighted_f1",
    "val_accuracy", "val_macro_f1", "val_weighted_f1",
    "test_common_accuracy", "test_common_macro_f1", "test_common_weighted_f1",
    "val_common_accuracy", "val_common_macro_f1", "val_common_weighted_f1",
    "test_non_overlap_accuracy", "test_non_overlap_macro_f1",
    "test_common_non_overlap_accuracy", "test_common_non_overlap_macro_f1",
    "test_full_accuracy", "test_full_macro_f1",
]
PRIMARY_TABLE_COLUMNS = ["test_common_accuracy", "test_common_macro_f1", "test_common_weighted_f1"]
STEP11_IDENTITY_KEYS = ["source_contract", "script_sha256", "settings", "selected_sequence_lengths",
    "source_embedding_dir", "source_sequence_dir", "output_root", "runtime", "seed_policy", "comparison_policy"]
MANIFEST_NAME = "multiseed_run_manifest.json"


def parse_int_list(values: Optional[List[str]], default: List[int]) -> List[int]:
    if values is None:
        return list(default)
    parts = [part.strip() for item in values for part in str(item).split(",") if part.strip()]
    if len(parts) == 1 and parts[0].lower() == "all" and default == DEFAULT_LENGTHS:
        return list(DEFAULT_LENGTHS)
    if not parts or any(part.lower() == "all" for part in parts):
        raise ValueError("Use explicit integer seeds; 'all' is allowed only as the entire lengths list.")
    out = [int(part) for part in parts]
    if len(out) != len(set(out)):
        raise ValueError("Duplicate seeds/lengths are not independent runs and are not allowed.")
    return out


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"Cannot decode source: {path}")


def step11_compatibility_issue(path: Path) -> Optional[str]:
    """Check the delivered interface without importing/executing candidate files.

    This is a compatibility screen, not evidence that the input data are valid.
    Step 11 and completed-run verification still enforce the actual provenance.
    """
    if not path.is_file():
        return "file not found"
    try:
        tree = ast.parse(read_text(path), filename=str(path))
    except (OSError, ValueError, SyntaxError) as exc:
        return f"source cannot be parsed: {exc}"
    flags = {arg.value for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "add_argument"
             for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
    missing = REQUIRED_TRAIN_FLAGS - flags
    if missing:
        return "missing required CLI option(s): " + ", ".join(sorted(missing))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    required = {"load_source_contract", "validate_all_sequence_inputs", "prepare_run", "finish_run",
                "validate_observability_source_contract", "load_sequence_observable_effects"}
    if not required.issubset(functions):
        return "missing shared-nominal revision/audit support; adding --seed alone is insufficient"
    return None


def resolve_train_script(requested: Optional[str], base_dir: Path) -> Path:
    candidates = ([resolve_path(requested, base_dir)] if requested is not None else
                  [resolve_path(name, base_dir) for name in (DEFAULT_TRAIN_SCRIPT, LEGACY_TRAIN_SCRIPT, OLDER_TRAIN_SCRIPT)])
    issues = {path: step11_compatibility_issue(path) for path in candidates}
    compatible = [path for path, issue in issues.items() if issue is None]
    if not compatible:
        details = "\n".join(f"  {path}: {issue}" for path, issue in issues.items())
        raise ValueError("No compatible revised Step 11 was selected.\n" + details +
                         "\nSave the supplied 11.train_tst_classifier.py in the project directory "
                         "and use --train_script with that exact path. Training was not started.")
    if len({sha256_file(path) for path in compatible}) > 1:
        raise ValueError("Different revised Step 11 files are present. Select the intended source "
                         "explicitly with --train_script:\n" + "\n".join(map(str, compatible)))
    if requested is None:
        for path, issue in issues.items():
            if issue is not None and path.exists():
                print(f"[Incompatible Step 11 ignored] {path}: {issue}", flush=True)
    return compatible[0]


def step11_input_paths(original: Path) -> Dict[str, Path]:
    """Read the child's declared paths using a small, non-executing AST parser.

    Step 11 has no input-path CLI flags. Check the actual source constants so a
    main run from other inputs is rejected before launching expensive training.
    """
    tree = ast.parse(read_text(original), filename=str(original))
    known: Dict[str, Any] = {"BASE_DIR": original.resolve().parent}
    targets = {"EMBEDDING_DIR": "embedding_dir", "SEQUENCE_DIR": "sequence_dir",
               "EXPECTED_FINETUNED_MODEL_DIR": "model_dir"}

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in known:
            return known[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return Path(value(node.left)) / value(node.right)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path"
                and len(node.args) == 1 and not node.keywords):
            return Path(value(node.args[0]))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"resolve", "expanduser"} and not node.args and not node.keywords):
            return getattr(Path(value(node.func.value)), node.func.attr)()
        raise ValueError("Step 11 input paths must be explicit Path/string constants derived from BASE_DIR.")

    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in targets:
                known[name] = value(node.value)
                found[targets[name]] = Path(known[name]).expanduser().resolve()
    if set(found) != set(targets.values()):
        raise ValueError("Step 11 does not declare the required embedding/sequence/model input paths.")
    return found


def validate_child_input_paths(run: Dict[str, Any], inputs: Dict[str, Path]) -> None:
    for key, source_key in (("embedding_dir", "source_embedding_dir"), ("sequence_dir", "source_sequence_dir")):
        if Path(run[source_key]).resolve() != inputs[key].resolve():
            raise ValueError(
                "Reference run input paths differ from the selected Step 11 source. "
                "Use the same Step 11 source/project and input artifacts; no new seed was started."
            )


def preflight_train_script(python_exe: str, original: Path) -> str:
    """Check the actual interpreter/CLI before writing a multi-seed job record."""
    digest = sha256_file(original)
    try:
        result = subprocess.run([python_exe, str(original), "--help"], cwd=str(original.parent),
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Step 11 --help timed out before training: {original}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"Step 11 CLI preflight failed before training: {original}\n"
                           f"{(result.stderr or result.stdout)[-4000:]}")
    advertised = set(re.findall(r"--[A-Za-z][A-Za-z0-9_-]*", result.stdout))
    missing = REQUIRED_TRAIN_FLAGS - advertised
    if missing:
        raise ValueError(f"Step 11 --help does not advertise {sorted(missing)}: {original}. Training was not started.")
    if sha256_file(original) != digest:
        raise ValueError("Step 11 source changed during CLI preflight; training was not started.")
    print(f"[Step 11 selected] {original}", flush=True)
    return digest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required completion/provenance file missing: {path}")
    value = json.loads(read_text(path))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Invalid JSON record: {path}")
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path: Path, value: Dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_dataframe(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, "\ufeff" + frame.to_csv(index=False))


def verify_artifacts(root: Path, hashes: Dict[str, str]) -> None:
    if not hashes:
        raise ValueError(f"Empty completed artifact inventory: {root}")
    for name, digest in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Missing/changed completed artifact: {path}")


def integer_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame or not pd.api.types.is_integer_dtype(frame[column].dtype) or frame[column].isna().any():
        raise ValueError(f"Missing/non-integer prediction column: {column}")
    return frame[column].to_numpy(dtype=np.int64)


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, classes: int) -> Dict[str, float]:
    if len(y_true) == 0 or np.any((y_true < 0) | (y_true >= classes) | (y_pred < 0) | (y_pred >= classes)):
        raise ValueError("Empty/invalid prediction labels.")
    matrix = np.bincount(y_true * classes + y_pred, minlength=classes * classes).reshape(classes, classes)
    support, predicted, tp = matrix.sum(1), matrix.sum(0), np.diag(matrix)
    f1 = np.divide(2.0 * tp, support + predicted, out=np.zeros(classes), where=(support + predicted) != 0)
    return {"accuracy": float(tp.sum() / len(y_true)), "macro_f1": float(f1.mean()),
            "weighted_f1": float(np.sum(f1 * support) / len(y_true))}


def require_close(value: Any, expected: float, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value) or not math.isclose(float(value), expected, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(f"Metric/count mismatch or nonfinite value: {context}")


def validate_observability_contract(source: Dict[str, Any]) -> None:
    audit = source.get("observability_audit", {})
    files = {split: f"audit/{split}_window_observable_effects.csv" for split in ("train", "val", "test")}
    if (audit.get("schema_version") != "counterfactual_observable_change_v1"
            or audit.get("dataset_id") != source.get("dataset_id")
            or audit.get("split_files") != files or not audit.get("scope")
            or not audit.get("source_dataset_dir")
            or not isinstance(audit.get("step7_build_manifest_sha256"), str)
            or len(audit["step7_build_manifest_sha256"]) != 64
            or set(audit.get("artifacts", {})) != set(files.values())
            or any(not isinstance(d, str) or len(d) != 64 for d in audit.get("artifacts", {}).values())):
        raise ValueError("Missing or inconsistent shared-nominal observable-effect audit contract. Use the revised Step 11 results.")


def validate_observable_effect_predictions(pred: pd.DataFrame, length: int) -> None:
    current, history, total = [integer_column(pred, name) for name in OBSERVABLE_EFFECT_COLUMNS[:3]]
    if (not np.isin(current, [0, 1]).all() or np.any((history < 0) | (history >= length))
            or not np.array_equal(total, current + history)):
        raise ValueError("Invalid current/history/context observable-effect counts in predictions.")
    expected = np.where(current == 1, "current_changed", np.where(history > 0, "history_only_changed", "no_observable_change"))
    if "audit_effect_stratum" not in pred or not np.array_equal(pred["audit_effect_stratum"].to_numpy(), expected):
        raise ValueError("Prediction observable-effect stratum disagrees with its context counts.")


def observable_effect_key(pred: pd.DataFrame) -> str:
    columns = ["sequence_end_row", "episode_id", "true_label_id"] + OBSERVABLE_EFFECT_COLUMNS
    canonical = pred[columns].sort_values("sequence_end_row").reset_index(drop=True)
    return hashlib.sha256(canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def comparable_protocol(run: Dict[str, Any]) -> Dict[str, Any]:
    # Only the head seed and the physical output directory may differ.
    identity = {key: run[key] for key in STEP11_IDENTITY_KEYS if key != "output_root"}
    identity["settings"] = {key: value for key, value in run["settings"].items() if key != "DEFAULT_RANDOM_SEED"}
    return identity


def step11_identity_hash(run: Dict[str, Any]) -> str:
    # Step 11 hashes these six Python dicts with integer keys, before JSON saves
    # their keys as strings. Restore ONLY those settings: upstream JSON maps have
    # string keys by design and must retain their original lexicographic order.
    identity = {key: run[key] for key in STEP11_IDENTITY_KEYS}
    identity["settings"] = dict(run["settings"])
    for name in ("LENGTH_AWARE_REGULARIZATION", "MISSING_AWARE_REGULARIZATION",
                 "ROBUST_KEEP_WINDOWS_BY_LENGTH", "ROBUST_LAST_WINDOW_KEEP_PROB_BY_LENGTH",
                 "ROBUST_DIM_DROPOUT_PROB_BY_LENGTH", "ROBUST_GAUSSIAN_NOISE_STD_BY_LENGTH"):
        values = identity["settings"].get(name)
        if not isinstance(values, dict):
            raise ValueError(f"Step 11 settings lack required length table: {name}")
        restored = {int(key): value for key, value in values.items()}
        if len(restored) != len(values):
            raise ValueError(f"Ambiguous Step 11 length keys: {name}")
        identity["settings"][name] = restored
    return object_hash(identity)


def validate_seed_manifest(root: Path, seed: int, lengths: List[int],
                           original_script: Optional[Path], missing_aware: Optional[bool],
                           require_complete: bool = True) -> Dict[str, Any]:
    run = load_json(root / "run_manifest.json")
    if require_complete and run.get("status") != "COMPLETE":
        raise ValueError(f"Seed {seed} is not COMPLETE: {root}. Finish the run before aggregation.")
    if any(key not in run for key in STEP11_IDENTITY_KEYS):
        raise ValueError(f"Seed {seed} is not a revised Step 11 run.")
    if step11_identity_hash(run) != run.get("run_id"):
        raise ValueError(f"Seed {seed} run identity is inconsistent.")
    settings = run["settings"]
    if settings.get("DEFAULT_RANDOM_SEED") != seed or run["selected_sequence_lengths"] != lengths:
        raise ValueError(f"Seed/ordered lengths mismatch: {root}")
    if original_script is not None and run["script_sha256"] != sha256_file(original_script):
        raise ValueError(f"Seed {seed} was produced by different Step 11 source code.")
    if missing_aware is not None and settings.get("USE_MISSING_AWARE_TRAINING") != missing_aware:
        raise ValueError(f"Seed {seed} missing-aware protocol mismatch.")
    if (settings.get("SEQUENCE_FIT_SUBSET") != "full" or settings.get("PRIMARY_TEST_SUBSET") != "full"
            or settings.get("PRIMARY_SELECTION_METRIC") != "val_macro_f1"
            or settings.get("USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS") is not False):
        raise ValueError("Step 12 requires the agreed full-fit/val-selected/unperturbed protocol.")
    if Path(run["output_root"]).resolve() != root.resolve():
        raise ValueError("Moved/mislabelled seed output. Use its recorded location, or restore the original layout.")
    source = run["source_contract"]
    if not source.get("dataset_id") or run.get("dataset_id") != source["dataset_id"]:
        raise ValueError("Missing/mismatched dataset identity.")
    if any(not source.get(key) for key in ("step8_run_id", "step9_run_id", "step10_run_id", "common_endpoint_report")):
        raise ValueError("Incomplete upstream provenance.")
    validate_observability_contract(source)
    return run


def load_verified_seed_result(root: Path, seed: int, lengths: List[int],
                              original_script: Optional[Path] = None,
                              missing_aware: Optional[bool] = None) -> Dict[str, Any]:
    run = validate_seed_manifest(root, seed, lengths, original_script, missing_aware)
    required = {"tst_overall_comparison.csv", "tst_overall_comparison.json", "tst_common_endpoint_comparison.csv",
                "input_validation_report.json"}
    for length in lengths:
        required.update({f"L{length}/reports/evaluation_metrics_L{length}.json",
                         f"L{length}/checkpoints/best_tst_model_L{length}.pt"})
        for split in ("val", "test"):
            required.add(f"L{length}/predictions/{split}_predictions_L{length}.csv")
            required.add(f"L{length}/reports/observable_effect_diagnostics_{split}_L{length}.json")
    if not required.issubset(run.get("artifacts", {})):
        raise ValueError(f"Seed {seed} completion inventory lacks required reports/checkpoints/predictions.")
    verify_artifacts(root, run["artifacts"])
    manifest_hash = sha256_file(root / "run_manifest.json")
    frame = pd.read_csv(root / "tst_overall_comparison.csv")
    actual_lengths = integer_column(frame, "sequence_length").tolist()
    if len(actual_lengths) != len(lengths) or sorted(actual_lengths) != sorted(lengths):
        raise ValueError(f"Seed {seed}: missing/duplicate/unexpected L rows.")
    source = run["source_contract"]
    provenance = {key: source[key] for key in ("dataset_id", "step8_run_id", "step9_run_id", "step10_run_id")}
    provenance["step11_run_id"] = run["run_id"]
    overall = load_json(root / "tst_overall_comparison.json")
    if overall.get("provenance") != provenance or overall.get("selected_sequence_lengths") != lengths:
        raise ValueError(f"Seed {seed}: overall JSON provenance/length mismatch.")
    endpoint_report = source["common_endpoint_report"]
    if endpoint_report.get("reference_length") != 10 or overall.get("common_endpoint_report") != endpoint_report:
        raise ValueError("Missing/mixed common endpoint contract.")
    label_order = overall.get("label_order", [])
    if len(label_order) != 6 or overall.get("label_to_id") != dict(zip(label_order, range(6))):
        raise ValueError("Invalid six-class mapping.")
    rows, matched_keys, effect_keys = [], {}, {}
    for _, series in frame.iterrows():
        row = series.to_dict()
        length = int(row["sequence_length"])
        report = load_json(root / f"L{length}/reports/evaluation_metrics_L{length}.json")
        if (any(row.get(key) != value for key, value in provenance.items()) or report.get("provenance") != provenance
                or report.get("sequence_length") != length or report.get("label_order") != label_order
                or report.get("common_endpoint_report") != endpoint_report
                or report.get("train_config", {}).get("random_seed") != seed
                or report.get("test_metric_protocol") != "standard_unperturbed"
                or row.get("test_metric_protocol") != "standard_unperturbed"):
            raise ValueError(f"Seed {seed}/L{length}: report/CSV provenance mismatch.")
        for split in ("val", "test"):
            pred = pd.read_csv(root / f"L{length}/predictions/{split}_predictions_L{length}.csv", keep_default_na=False)
            for key, value in {**provenance, "source_split": split}.items():
                if key not in pred or not pred[key].eq(value).all():
                    raise ValueError(f"Seed {seed}/L{length}/{split}: prediction provenance mismatch.")
            ends = integer_column(pred, "sequence_end_row")
            y_true, y_pred = integer_column(pred, "true_label_id"), integer_column(pred, "pred_label_id")
            flags = integer_column(pred, "common_population")
            population = endpoint_report["splits"][split]
            if (not np.isin(flags, [0, 1]).all() or len(np.unique(ends)) != len(ends) or np.any(ends < 0)
                    or len(pred) != population["available_sequences_by_L"][str(length)]):
                raise ValueError(f"Seed {seed}/L{length}/{split}: endpoint coverage mismatch.")
            mask = flags == 1
            digest = hashlib.sha256(np.sort(ends[mask]).astype("<i8", copy=False).tobytes()).hexdigest()
            if int(mask.sum()) != population["common_endpoint_count"] or digest != population["sorted_end_row_indices_sha256"]:
                raise ValueError(f"Seed {seed}/L{length}/{split}: common endpoints differ.")
            if "episode_id" not in pred or pred["episode_id"].eq("").any():
                raise ValueError("Missing prediction episode identifiers.")
            validate_observable_effect_predictions(pred, length)
            effect_keys[f"{split}/L{length}"] = observable_effect_key(pred)
            effect_path = root / f"L{length}/reports/observable_effect_diagnostics_{split}_L{length}.json"
            diagnostic = load_json(effect_path)
            recorded_path = report.get("observable_effect_diagnostics", {}).get(split)
            if (not recorded_path or Path(recorded_path).resolve() != effect_path.resolve()
                    or diagnostic.get("provenance") != provenance
                    or diagnostic.get("source_audit") != source["observability_audit"]
                    or diagnostic.get("sequence_length") != length or diagnostic.get("split") != split):
                raise ValueError(f"Seed {seed}/L{length}/{split}: observable-effect diagnostic provenance mismatch.")
            keys = pred.loc[mask, ["sequence_end_row", "episode_id", "true_label_id"]].sort_values("sequence_end_row")
            key_digest = hashlib.sha256(keys.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
            if split in matched_keys and matched_keys[split] != key_digest:
                raise ValueError("Common endpoint episode/label identities differ across L.")
            matched_keys[split] = key_digest
            full, common = metric_values(y_true, y_pred, 6), metric_values(y_true[mask], y_pred[mask], 6)
            common_summary = report[f"{split}_subset_summaries"]["common_population"]
            for metric, value in full.items():
                require_close(row.get(f"{split}_{metric}"), value, f"{split}_{metric}")
                require_close(report[f"{split}_summary"].get(metric), value, f"report/{split}/{metric}")
            for metric, value in common.items():
                require_close(common_summary.get(metric), value, f"report/{split}/common/{metric}")
                column = f"{split}_common_{metric}"
                # Step 11 does not put common weighted-F1 in the comparison CSV.
                if column in row:
                    require_close(row[column], value, column)
                row[column] = value
            require_close(common_summary.get("total"), int(mask.sum()), "common summary count")
            require_close(row.get(f"{split}_common_count"), int(mask.sum()), "common CSV count")
        for metric in METRIC_COLUMNS:
            require_close(row.get(metric), float(row.get(metric, float("nan"))), metric)
        row.update(seed=seed, seed_output_root=str(root.resolve()), seed_manifest_sha256=manifest_hash)
        rows.append(row)
    return {"run": run, "rows": rows, "matched_keys": matched_keys,
            "observable_effect_keys": effect_keys, "manifest_sha256": manifest_hash}


def verify_current_sources(run: Dict[str, Any]) -> None:
    """Verify the current on-disk handoff before choosing an existing reference run."""
    source = run["source_contract"]
    root9, root10 = Path(run["source_embedding_dir"]), Path(run["source_sequence_dir"])
    report9 = load_json(root9 / "embedding_extraction_report.json")
    root8 = Path(report9["source_model_dir"]).parent
    for step, root, hashes in (("step8", root8, source["step8_model_sha256"]),
                              ("step9", root9, source["step9_artifacts"]),
                              ("step10", root10, source["step10_artifacts"])):
        if sha256_file(root / "run_manifest.json") != source["manifest_sha256"][step]:
            raise ValueError(f"Current {step} provenance differs from the completed reference run.")
        verify_artifacts(root, hashes)


def run_one_seed(python_exe: str, original_script: Path, output_root: Path, seed: int,
                 lengths: List[int], missing_aware: bool, skip_existing: bool) -> Dict[str, object]:
    seed_root = output_root / f"seed_{seed}"
    if seed_root.exists():
        result = load_verified_seed_result(seed_root, seed, lengths, original_script, missing_aware)
        print(f"[Verified COMPLETE seed reused] {seed}: {seed_root}")
        return {"seed": seed, "status": "reused_complete", "output_root": str(seed_root),
                "step11_run_id": result["run"]["run_id"], "manifest_sha256": result["manifest_sha256"]}
    # The compatibility flag cannot disable provenance checks or permit overwriting.
    cmd = [python_exe, str(original_script), "--seed", str(seed), "--lengths", *map(str, lengths),
           "--output-root", str(seed_root), "--missing-aware" if missing_aware else "--no-missing-aware"]
    print(f"\n[Train seed {seed}] {subprocess.list2cmdline(cmd)}", flush=True)
    started = time.time()
    child = subprocess.Popen(cmd, cwd=str(original_script.parent))
    try:
        returncode = child.wait()
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    if returncode != 0:
        raise RuntimeError(f"Step 11 failed for seed={seed}, exit={returncode}; completed seeds are preserved.")
    result = load_verified_seed_result(seed_root, seed, lengths, original_script, missing_aware)
    return {"seed": seed, "status": "completed", "elapsed_seconds": time.time() - started,
            "output_root": str(seed_root), "step11_run_id": result["run"]["run_id"],
            "manifest_sha256": result["manifest_sha256"]}


def format_mean_std(mean_value: float, std_value: Optional[float], digits: int = 4) -> str:
    if std_value is None or pd.isna(std_value):
        return f"{mean_value:.{digits}f} (SD n/a)"
    return f"{mean_value:.{digits}f} $\\pm$ {std_value:.{digits}f}"


def latex_escape_label(text: str) -> str:
    return str(text).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [r"\begin{table}[t]", r"\centering",
        r"\caption{TST performance on matched endpoints across head-training seeds}",
        r"\label{tab:tst_multiseed}", r"\footnotesize", r"\begin{tabular}{lccc}", r"\hline",
        r"Model & Accuracy & Macro-F1 & Weighted-F1 \\", r"\hline"]
    for _, row in summary.iterrows():
        values = [format_mean_std(row[f"{key}_mean"], row[f"{key}_std"]) for key in PRIMARY_TABLE_COLUMNS]
        lines.append(f"TST, \\(L={int(row['sequence_length'])}\\) & " + " & ".join(values) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\parbox{0.96\columnwidth}{\footnotesize "
        r"All lengths and seeds use the same test endpoints. Values are mean $\pm$ sample SD "
        r"across head-training seeds with a fixed encoder and data split. SD is not a confidence "
        r"interval; it is undefined for a single seed.}", r"\end{table}"]
    atomic_text(path, "\n".join(lines) + "\n")


def write_latex_sentence(summary: pd.DataFrame, path: Path) -> None:
    counts = summary["num_seeds"].unique()
    if len(counts) != 1:
        raise ValueError("Unequal seed counts cannot be described as a complete multi-seed experiment.")
    count = int(counts[0])
    lengths = ", ".join(f"\\(L={int(value)}\\)" for value in summary["sequence_length"])
    text = (f"For {lengths}, Table~\\ref{{tab:tst_multiseed}} reports test performance on the same "
            f"prediction endpoints over {count} head-training seed(s), conditional on the same fine-tuned "
            "encoder and episode split. ")
    text += ("Reported dispersion is sample standard deviation across head seeds, not an episode-bootstrap "
             "confidence interval or a test of statistical significance." if count > 1 else
             "This is a single-seed result; between-seed standard deviation cannot be estimated.")
    atomic_text(path, text + "\n")


def aggregate_results(output_root: Path, seeds: List[int], lengths: List[int],
                      seed_roots: Optional[Dict[int, Path]] = None, original_script: Optional[Path] = None,
                      missing_aware: Optional[bool] = None) -> Dict[str, str]:
    if not seeds or len(set(seeds)) != len(seeds) or not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("Nonempty, unique seeds and lengths are required.")
    roots = seed_roots or {seed: output_root / f"seed_{seed}" for seed in seeds}
    rows, baseline, matched, effect_keys = [], None, None, None
    verified = []
    for seed in seeds:
        result = load_verified_seed_result(Path(roots[seed]), seed, lengths, original_script, missing_aware)
        signature = comparable_protocol(result["run"])
        if baseline is not None and (signature != baseline or result["matched_keys"] != matched
                                     or result["observable_effect_keys"] != effect_keys):
            raise ValueError("Seeds mix source data/encoder/ordered L/settings/runtime or endpoint labels.")
        baseline, matched = signature, result["matched_keys"]
        effect_keys = result["observable_effect_keys"]
        rows.extend(result["rows"])
        verified.append({"seed": seed, "output_root": str(roots[seed]), "run_id": result["run"]["run_id"],
                         "manifest_sha256": result["manifest_sha256"]})
    raw = pd.DataFrame(rows).sort_values(["seed", "sequence_length"]).reset_index(drop=True)
    if len(raw) != len(seeds) * len(lengths) or raw.duplicated(["seed", "sequence_length"]).any():
        raise ValueError("Incomplete or duplicated seed x L grid.")
    if not np.isfinite(raw[METRIC_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite metrics cannot be omitted from seed aggregation.")
    # pandas std uses ddof=1: sample SD, not population SD or standard error.
    summary = raw.groupby("sequence_length")[METRIC_COLUMNS].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary = summary.reset_index().sort_values("sequence_length")
    summary.insert(1, "num_seeds", len(seeds))
    summary["val_common_count"] = int(raw.val_common_count.iloc[0])
    summary["test_common_count"] = int(raw.test_common_count.iloc[0])
    # No user-facing aggregates are written until every source, row and metric passes.
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {"raw_csv": output_root / "multiseed_raw_results.csv", "summary_csv": output_root / "multiseed_summary.csv",
             "table_tex": output_root / "multiseed_tst_table.tex", "sentence_tex": output_root / "multiseed_tst_sentence.tex",
             "validation_json": output_root / "multiseed_validation.json"}
    save_dataframe(paths["raw_csv"], raw)
    save_dataframe(paths["summary_csv"], summary)
    write_latex_table(summary, paths["table_tex"])
    write_latex_sentence(summary, paths["sentence_tex"])
    save_json(paths["validation_json"], {"status": "VALIDATED", "seeds": seeds, "ordered_lengths": lengths,
        "complete_grid": True, "primary_table_columns": PRIMARY_TABLE_COLUMNS, "std_ddof": 1,
        "statistical_scope": "Head seed variability conditional on fixed encoder/data; not bootstrap CI.",
        "source_contract": baseline["source_contract"], "common_prediction_keys_sha256": matched,
        "observable_effect_keys_sha256": effect_keys,
        "observable_effect_scope": "Same audit annotations across head seeds at each L; diagnostic only, never seed selection or model inputs.",
        "runs": verified})
    return {key: str(path) for key, path in paths.items()}


@contextmanager
def job_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".multiseed_runner.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started": time.time()}, stream)
    except FileExistsError as exc:
        raise RuntimeError(f"Runner lock exists: {path}. Another run may be active. After a crash, confirm the old runner/child stopped before removing this lock.") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def write_review_zip(root: Path, artifacts: Dict[str, str], seed_roots: Dict[int, Path]) -> Path:
    output = root / "12_MULTISEED_TST_REVIEW.zip"
    with tempfile.NamedTemporaryFile(dir=root, suffix=".zip.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(Path(__file__), "source/" + Path(__file__).name)
            for path in [root / MANIFEST_NAME, *map(Path, artifacts.values())]:
                archive.write(path, path.name)
            for seed, seed_root in seed_roots.items():
                for name in ("run_manifest.json", "tst_overall_comparison.csv", "tst_overall_comparison.json",
                             "tst_common_endpoint_comparison.csv", "input_validation_report.json"):
                    archive.write(seed_root / name, f"seed_{seed}/{name}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Validated multi-seed TST runner and matched-endpoint aggregation.")
    parser.add_argument("--train_script", "--train-script", default=None,
                        help="Exact revised Step 11 path. Default: inspect delivered and legacy filenames beside this runner.")
    parser.add_argument("--output_root", "--output-root", default=None,
                        help=f"Default: {DEFAULT_OUTPUT_ROOT} beside Step 11. Explicit relative paths use the working directory.")
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--lengths", nargs="*", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--missing-aware", action="store_true")
    parser.add_argument("--skip_existing", "--skip-existing", action="store_true", help="Compatibility flag; all reuse is always verified.")
    parser.add_argument("--aggregate_only", "--aggregate-only", action="store_true")
    parser.add_argument("--main_run", "--main-run", default=None,
                        help=f"Default: {DEFAULT_MAIN_RUN} beside Step 11. Explicit relative paths use the working directory.")
    parser.add_argument("--no_reuse_main", "--no-reuse-main", action="store_true")
    parser.add_argument("--retry_failed", "--retry-failed", action="store_true", help="Archive/restart an incomplete local seed only after its previous child stopped.")
    args = parser.parse_args()
    requested = str(resolve_path(args.train_script, Path.cwd())) if args.train_script is not None else None
    original = resolve_train_script(requested, Path(__file__).resolve().parent)
    inputs = step11_input_paths(original)
    output = resolve_path(args.output_root, Path.cwd()) if args.output_root is not None else resolve_path(DEFAULT_OUTPUT_ROOT, original.parent)
    main_root = None if args.no_reuse_main else (
        resolve_path(args.main_run, Path.cwd()) if args.main_run is not None else resolve_path(DEFAULT_MAIN_RUN, original.parent))
    for source in (inputs["embedding_dir"], inputs["sequence_dir"], inputs["model_dir"].parent, main_root):
        if source is not None and (output.is_relative_to(source.resolve()) or source.resolve().is_relative_to(output)):
            raise ValueError("Multi-seed output must be separate from the main run and upstream input directories.")
    if original.is_relative_to(output):
        raise ValueError("Output cannot contain the training source/project.")
    seeds, lengths = parse_int_list(args.seeds, DEFAULT_SEEDS), parse_int_list(args.lengths, DEFAULT_LENGTHS)
    if any(seed < 0 or seed >= 2**32 for seed in seeds) or any(length not in DEFAULT_LENGTHS for length in lengths):
        raise ValueError("Seeds must be in [0, 2**32); lengths must be selected from 1,2,3,5,10.")
    executable = shutil.which(args.python)
    if not executable:
        raise FileNotFoundError(f"Python executable not found: {args.python}")
    train_script_hash = sha256_file(original) if args.aggregate_only else preflight_train_script(executable, original)
    identity = {"runner_sha256": sha256_file(Path(__file__)), "train_script": str(original),
                "train_script_sha256": train_script_hash, "output_root": str(output),
                "seeds": seeds, "ordered_lengths": lengths, "python": str(Path(executable).resolve()),
                "missing_aware": args.missing_aware, "main_run": str(main_root) if main_root else None}
    with job_lock(output):
        manifest_path = output / MANIFEST_NAME
        existing = load_json(manifest_path) if manifest_path.exists() else None
        if existing and (existing.get("job_id") != object_hash(identity) or existing.get("identity") != identity):
            raise ValueError("Output belongs to another source/seed list/protocol. Use a new --output_root.")
        if existing is None and any(p.name != ".multiseed_runner.lock" for p in output.iterdir()):
            raise FileExistsError("Existing output lacks this runner's manifest; old results are preserved. Use a new --output_root.")
        seed_roots = {seed: output / f"seed_{seed}" for seed in seeds}
        if existing:
            for item in existing.get("runs", []):
                if item["seed"] in seed_roots:
                    seed_roots[item["seed"]] = Path(item["output_root"])
        if main_root is not None and main_root.exists():
            record = load_json(main_root / "run_manifest.json")
            main_seed = record.get("settings", {}).get("DEFAULT_RANDOM_SEED")
            if main_seed in seeds:
                if seed_roots[main_seed].exists() and seed_roots[main_seed] != main_root:
                    raise ValueError("Both a local and main run exist for the same seed. Choose a new root with --no_reuse_main to disambiguate.")
                seed_roots[main_seed] = main_root
        # Validate all existing runs BEFORE launching any new expensive training.
        baseline = None
        incomplete = []
        for seed, root in seed_roots.items():
            if not root.exists():
                if args.aggregate_only:
                    raise FileNotFoundError(f"Missing requested seed {seed}: {root}")
                continue
            run = validate_seed_manifest(root, seed, lengths, original, args.missing_aware, require_complete=False)
            validate_child_input_paths(run, inputs)
            if run.get("status") != "COMPLETE":
                if root == main_root or not args.retry_failed or args.aggregate_only:
                    raise ValueError(f"Seed {seed} is incomplete: {root}. Finish the main run first; use --retry_failed only for a stopped local seed.")
                if root != output / f"seed_{seed}":
                    raise ValueError("Only this job's local seed directory can be archived for retry.")
                incomplete.append((seed, root))
            else:
                load_verified_seed_result(root, seed, lengths, original, args.missing_aware)
            signature = comparable_protocol(run)
            if baseline is not None and signature != baseline:
                raise ValueError("Existing seed runs use different data/encoder/settings/runtime.")
            baseline = signature
        if baseline is not None:
            verify_current_sources(baseline)
        if existing and existing.get("status") == "COMPLETE":
            verify_artifacts(output, existing.get("artifact_sha256", {}))
        manifest = existing or {"job_id": object_hash(identity), "identity": identity, "runs": [], "archived_incomplete": []}
        manifest.update(status="RUNNING", last_error=None)
        save_json(manifest_path, manifest)
        try:
            for seed, root in incomplete:
                archived = output / "_incomplete" / f"seed_{seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}"
                archived.parent.mkdir(parents=True, exist_ok=True)
                root.rename(archived)
                manifest["archived_incomplete"].append({"seed": seed, "path": str(archived)})
                save_json(manifest_path, manifest)
            for seed in seeds:
                root = seed_roots[seed]
                if sha256_file(original) != identity["train_script_sha256"]:
                    raise ValueError("Step 11 source changed while the multi-seed job was running.")
                if baseline is not None:
                    verify_current_sources(baseline)
                if root.exists():
                    result = load_verified_seed_result(root, seed, lengths, original, args.missing_aware)
                    info = {"seed": seed, "status": "reused_main" if root == main_root else "reused_complete",
                            "output_root": str(root), "step11_run_id": result["run"]["run_id"],
                            "manifest_sha256": result["manifest_sha256"]}
                    print(f"[Verified seed {seed} reused] {root}", flush=True)
                elif args.aggregate_only:
                    raise FileNotFoundError(f"Missing requested seed {seed}")
                else:
                    manifest["active_seed"] = seed
                    save_json(manifest_path, manifest)
                    info = run_one_seed(executable, original, output, seed, lengths, args.missing_aware, args.skip_existing)
                    result = load_verified_seed_result(root, seed, lengths, original, args.missing_aware)
                signature = comparable_protocol(result["run"])
                validate_child_input_paths(result["run"], inputs)
                if baseline is not None and signature != baseline:
                    raise ValueError("New seed differs from the reference data/encoder/protocol/runtime; aggregation stopped.")
                baseline = signature
                manifest["runs"] = [item for item in manifest["runs"] if item["seed"] != seed] + [info]
                manifest["active_seed"] = None
                save_json(manifest_path, manifest)
            artifacts = aggregate_results(output, seeds, lengths, seed_roots, original, args.missing_aware)
            verify_current_sources(baseline)
            manifest.update(status="COMPLETE", artifacts=artifacts,
                            artifact_sha256={Path(path).name: sha256_file(Path(path)) for path in artifacts.values()})
            save_json(manifest_path, manifest)
            review = write_review_zip(output, artifacts, seed_roots)
        except BaseException as exc:
            manifest.update(status="FAILED", last_error=f"{type(exc).__name__}: {exc}")
            save_json(manifest_path, manifest)
            raise
    print(f"\nVerified complete grid: {len(seeds)} seeds x {len(lengths)} lengths.")
    summary = pd.read_csv(artifacts["summary_csv"])
    print(summary[["sequence_length", "num_seeds", "test_common_count",
                   "test_common_macro_f1_mean", "test_common_macro_f1_std"]].to_string(index=False))
    print(f"Results: {output}\nReview ZIP: {review}")


if __name__ == "__main__":
    main()
