#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 28 (A): L10 vs L20 matched-test evidence-location audit + paired episode bootstrap.

Purpose
-------
Read ONLY finished Step 27 final-test predictions, the three ORIGINAL Step 11 L10
predictions, and the Step 7/9 test-window audit/metadata. Never fit models or
change any Step 7--27 artifact. This is an EXPLORATORY post-test analysis.

Default (save beside Steps 10--27, use the same .venv):
    python "28.l20_evidence_location_bootstrap.py" --audit-only
    python "28.l20_evidence_location_bootstrap.py"

Optional explicit paths (also work with independent copies of the inputs):
    python "28.l20_evidence_location_bootstrap.py" ^
      --final-zip "27_L20_FINAL_TEST_REVIEW.zip" ^
      --l10-files "test_predictions_L10(1).csv" "test_predictions_L10(2).csv" "test_predictions_L10(3).csv" ^
      --window-meta "test_window_metadata.csv" ^
      --window-audit "test_window_observable_effects.csv"

Results: outputs/shared_nominal_v2/28_l20_evidence_location/run_<UTC>/
    28_L10_L20_EVIDENCE_REVIEW.zip  # send this file for review
    28_input_audit.json, 28_protocol.json, 28_metrics_by_seed.csv,
    28_model_summary.csv, 28_paired_bootstrap.csv, 28_group_summary.csv,
    28_group_class_detail.csv, 28_group_means.csv, 28_group_class_means.csv,
    28_prediction_transitions.csv,
    28_endpoint_evidence.csv, 28_bootstrap_distribution.npz, 28_run_manifest.json

Scientific and statistical contract
-----------------------------------
* Endpoint population: all, and ONLY, the 30,561 actual L20-eligible TEST
  endpoints from the authenticated Step 27 final-test run (derived/verified,
  NOT trusted from a hardcoded count). Episodes, labels, and keys MUST match.
* Models: historical L10 TST (same three seed IDs as Step 27) versus Step 27
  L20 TST; this is not a matched TRAINING-cohort comparison.
* Per-seed six-class macro-F1 is calculated before arithmetic seed averaging.
  Predictions/probabilities are never ensembled. Seed IDs do not imply paired
  model initializations.
* Whole episodes are sampled with replacement WITHIN each of six true-label
  strata, preserving all the episode's evaluated endpoints and multiplicities.
  Exactly the SAME episode multiplicity vector is used for both lengths and all
  three seeds. The reported two-sided percentile 95% confidence interval is
  conditional on these trained models, benchmark, episode classes and generator.
  It is NOT a CI across seeds or independent external validation.
* Single length contrast: L20 minus L10. Secondary accuracy and Normal-episode
  endpoint false-alarm fraction are descriptive with separately labeled CIs.
* Evidence groups for ATTACK-LABELED endpoints:
    A = >=1 paired-nominal changed observable within the last 10 records;
    B = no change in the last 10, >=1 change only in records 11..20 back;
    C = no change in all 20 records.
  The current record IS included in the most recent 10. NORMAL endpoints are
  summarized separately (the absence of an overlay is not a new threat group).
* Group location is a POST-HOC property using true paired-nominal records,
  never a model feature, bound, deployable reference or class-identifiability proof.
  Subgroup analyses/transition counts are exploratory DESCRIPTIVE statistics,
  not preregistered or adjusted hypothesis tests.
* Directional expectations were formulated AFTER observing the Step 27 test
  result: largest benefit expected in B; smaller in A; C may improve due to
  correlated inputs despite the lack of a paired-nominal observable change.
  This is explicitly not independent preregistration.
* Run with no GPU; NumPy/Pandas only. Neither a 63.py nor a Step 16 script is
  needed. The Step 15 code informed the identical bootstrap estimator.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABELS = (
    "NORMAL", "RF_LINK_JAMMING", "GNSS_SPOOFING",
    "UNAUTHORIZED_TELECOMMAND_INJECTION", "TELECOMMAND_REPLAY",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE",
)
SEEDS = (20260610, 20260611, 20260612)
K = len(LABELS)
BASE = Path(__file__).resolve().parent
ROOT_REL = Path("outputs/shared_nominal_v2")
DEFAULT_27 = ROOT_REL / "27_L20_TST_VALIDATION_PILOT" / "27_L20_FINAL_TEST_REVIEW.zip"
DEFAULT_OUTPUT = ROOT_REL / "28_l20_evidence_location"
EXPECTED_DATASET_ID = "478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b"
AUDIT_COLS = (
    "scenario_id", "episode_id", "episode_type", "episode_step", "window_id", "timestamp", "label",
)
META_COLS = AUDIT_COLS + ("label_id", "embedding_row_index")


def require(ok: bool, why: str) -> None:
    if not bool(ok):
        raise ValueError(why)


def digest_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def endpoint_digest(ends: np.ndarray) -> str:
    return digest_bytes(np.sort(np.asarray(ends, dtype="<i8")).tobytes())


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def json_bytes(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, data: Any) -> None:
    write_bytes(path, json_bytes(data))


def write_csv(path: Path, df: pd.DataFrame) -> None:
    write_bytes(path, ("\ufeff" + df.to_csv(index=False)).encode("utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    require(path.is_file(), f"Missing input CSV: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    require(not frame.columns.duplicated().any() and len(frame) > 0, f"Empty/duplicate-column CSV: {path}")
    return frame


def load_zip_csv(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with z.open(name) as stream:
        df = pd.read_csv(stream, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    require(not df.empty and not df.columns.duplicated().any(), f"Empty/invalid ZIP CSV: {name}")
    return df


def json_from_zip(z: zipfile.ZipFile, name: str) -> dict:
    obj = json.loads(z.read(name).decode("utf-8-sig"))
    require(isinstance(obj, dict), f"Expected JSON object in ZIP: {name}")
    return obj


def strict_int(col: pd.Series, context: str, minval: int | None = 0) -> np.ndarray:
    s = col.astype(str)
    require(s.str.fullmatch(r"-?\d+").all(), f"{context}: non-integer values")
    value = pd.to_numeric(s, errors="raise").to_numpy(dtype=np.int64)
    if minval is not None:
        require((value >= minval).all(), f"{context}: values below {minval}")
    return value


def strict_float(col: pd.Series, context: str) -> np.ndarray:
    x = pd.to_numeric(col, errors="raise").to_numpy(dtype=np.float64)
    require(np.isfinite(x).all(), f"{context}: nonfinite")
    return x


def confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    require(y.shape == p.shape and y.ndim == 1 and len(y) > 0,
            "Invalid true/pred length for confusion")
    require(np.isin(y, np.arange(K)).all() and np.isin(p, np.arange(K)).all(),
            "Invalid class ID in predictions")
    return np.bincount(y * K + p, minlength=K*K).reshape(K, K)


def metric(cm: np.ndarray) -> dict[str, float]:
    tp = np.diag(cm).astype(np.float64)
    row = cm.sum(1)
    col = cm.sum(0)
    den = row + col
    class_f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return {
        "macro_f1": float(class_f1.mean()),
        "accuracy": float(tp.sum() / row.sum()),
        "normal_false_alarm_fraction": float((row[0]-tp[0]) / row[0]) if row[0] else float("nan"),
    }


def safe_equal_float(x: Any, y: Any, message: str, atol: float = 2e-9) -> None:
    require(math.isfinite(float(x)) and math.isfinite(float(y))
            and math.isclose(float(x), float(y), rel_tol=0, abs_tol=atol),
            f"{message}: recomputed={x}, reported={y}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=BASE)
    p.add_argument("--final-zip", type=Path, default=None, help="Step 27 FINAL TEST review ZIP")
    p.add_argument("--l10-files", type=Path, nargs=3, default=None,
                   metavar=("SEED_20260610", "SEED_20260611", "SEED_20260612"))
    p.add_argument("--window-meta", type=Path, default=None, help="Step 9 test_window_metadata.csv")
    p.add_argument("--window-audit", type=Path, default=None, help="Step 7 audit/test_window_observable_effects.csv")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--audit-only", action="store_true", help="Input and group audit; no bootstrap resampling")
    p.add_argument("--replicates", type=int, default=20_000,
                   help="Bootstrap replicates; 20,000 is the intended analysis protocol")
    p.add_argument("--bootstrap-seed", type=int, default=20261013)
    p.add_argument("--batch", type=int, default=100)
    a = p.parse_args()
    require(a.replicates >= 20000 or a.audit_only,
            "Full analysis requires >=20,000 replicates. Use --audit-only for a quicker check.")
    require(a.batch >= 1 and 0 <= a.bootstrap_seed < 2**32, "Invalid bootstrap batch/seed")
    a.project_root = a.project_root.expanduser().resolve()
    a.final_zip = (a.final_zip or (a.project_root / DEFAULT_27)).expanduser().resolve()
    a.window_meta = (a.window_meta or
        (a.project_root / ROOT_REL / "tst_embeddings_finetuned_main/test_window_metadata.csv")).expanduser().resolve()
    a.window_audit = (a.window_audit or
        (a.project_root / "dataset/shared_nominal_v2/audit/test_window_observable_effects.csv")).expanduser().resolve()
    a.output_root = (a.output_root or (a.project_root / DEFAULT_OUTPUT)).expanduser().resolve()
    if a.l10_files is not None:
        a.l10_files = [p.expanduser().resolve() for p in a.l10_files]
    for pth in (a.final_zip, a.window_meta, a.window_audit):
        require(pth.is_file(), f"Input not found: {pth}")
    return a


def validate_final_zip(z: zipfile.ZipFile, path: Path):
    required = {"27_run_manifest.json", "27_final_test.json", "27_test_summary.json",
                "27_test_seed_comparison.csv", "27_evidence_availability_test.json",
                "source/27.l20_tst_pilot.py"}
    for seed in SEEDS:
        required.update((f"seed_{seed}/test_predictions.csv", f"seed_{seed}/27_seed_manifest.json",
                         f"seed_{seed}/val_predictions.csv"))
    names = set(z.namelist())
    require(required.issubset(names), f"Step 27 review ZIP missing: {sorted(required-names)}")
    require(len(names) == len(z.namelist()), "Step 27 ZIP contains duplicate member names")
    r = json_from_zip(z, "27_run_manifest.json")
    f = json_from_zip(z, "27_final_test.json")
    summary = json_from_zip(z, "27_test_summary.json")
    evidence = json_from_zip(z, "27_evidence_availability_test.json")
    ident = r.get("identity", {})
    require(r.get("status") == "COMPLETE" and f.get("status") == "COMPLETE",
            "Step 27 validation and final-test must be COMPLETE")
    require(r.get("identity_hash") == f.get("identity_hash") == summary.get("identity_hash"),
            "Step 27 identity mismatch across validation/test records")
    require(ident.get("schema") == "taes_step27_l20_validation_pilot_v1"
            and ident.get("new_length") == 20 and ident.get("reference_length") == 10,
            "Unexpected Step 27 length/protocol")
    require(ident.get("selected_seeds") == list(SEEDS) and r.get("seeds") == list(SEEDS)
            and summary.get("num_head_seeds") == 3,
            "Unexpected Step 27 selected seeds")
    src = ident.get("source_contract", {})
    require(src.get("dataset_id") == EXPECTED_DATASET_ID, "Unexpected dataset ID; refuse to mix benchmarks")
    require(digest_bytes(z.read("source/27.l20_tst_pilot.py")) == ident.get("pilot_source_sha256"),
            "Step 27 source differs from recorded pilot identity")
    for name, prop in [("27_val_summary.json", "val_summary_sha256"),
                       ("27_test_summary.json", "test_summary_sha256"),
                       ("27_test_seed_comparison.csv", "test_comparison_sha256")]:
        expected = r.get(prop) if prop == "val_summary_sha256" else f.get(prop)
        require(digest_bytes(z.read(name)) == expected, f"Step 27 {name} SHA-256 mismatch")
    require(f.get("val_summary_sha256") == r.get("val_summary_sha256"),
            "Test differs from completed validation selection")
    require(evidence == summary.get("evidence_availability"),
            "Step 27 evidence availability JSON mismatch")
    require(summary.get("split") == "test" and summary.get("statistical_scope", "").startswith("Exploratory"),
            "Step 27 result is not the final exploratory test")
    cohort = int(summary["common_endpoints"])
    require(cohort > 0 and summary.get("common_episodes", 0) > 0,
            "Empty Step 27 evaluation cohort")
    for seed in SEEDS:
        m = json_from_zip(z, f"seed_{seed}/27_seed_manifest.json")
        require(m.get("status") == "COMPLETE" and m.get("seed") == seed and
                m.get("identity_hash") == r["identity_hash"],
                f"Step 27 seed manifest invalid: {seed}")
        for member in ("val_predictions.csv",):
            rel = f"seed_{seed}/{member}"
            require(digest_bytes(z.read(rel)) == m.get("artifacts", {}).get(member),
                    f"Step 27 seed validation prediction hash mismatch: {rel}")
    return ident, summary, evidence


def resolve_l10_files(args, identity: dict) -> list[Path]:
    if args.l10_files is not None:
        files = args.l10_files
    else:
        files = []
        for seed in SEEDS:
            recorded = identity.get("historical_L10_seed_roots", {}).get(str(seed), "")
            root = (args.project_root / ROOT_REL /
                    ("tst_training_finetuned_main" if seed == 20260610
                     else f"tst_training_finetuned_multiseed/seed_{seed}"))
            candidates = ([Path(recorded)] if recorded else []) + [root]
            candidates = [c / "L10/predictions/test_predictions_L10.csv" for c in candidates]
            match = next((p for p in candidates if p.is_file()), None)
            if match is None:
                raise FileNotFoundError(
                    f"L10 seed {seed} prediction not found. Checked: {candidates}. "
                    "Supply the three exact paths with --l10-files in seed order.")
            files.append(match)
    require(len(files) == len(SEEDS) and len({p.resolve() for p in files}) == len(SEEDS),
            "Need three distinct L10 prediction CSV files")
    require(all(p.is_file() for p in files), f"Missing L10 files: {files}")
    return files


def authenticate_original_l10_manifest(path: Path, seed: int, identity: dict) -> dict:
    # The pilot independently authenticated each full original Step 11 run.
    # If a matching local manifest remains accessible, verify it again.
    root = path.parent.parent.parent
    record = root / "run_manifest.json"
    expected = identity.get("historical_L10_seed_manifests", {}).get(str(seed))
    if record.is_file():
        require(digest_file(record) == expected, f"Original L10 seed {seed} manifest hash changed")
        r = json.loads(record.read_text(encoding="utf-8-sig"))
        require(r.get("status") == "COMPLETE" and
                r.get("run_id") == identity["historical_L10_runs"][str(seed)],
                f"Original L10 seed {seed} run ID/status mismatch")
        rel = "L10/predictions/test_predictions_L10.csv"
        if rel in r.get("artifacts", {}):
            require(digest_file(path) == r["artifacts"][rel],
                    f"Original Step 11 prediction hash mismatch for seed {seed}")
        return {"available": True, "path": str(record), "sha256": digest_file(record)}
    return {"available": False,
            "note": "CSV identity and metrics cross-checked against completed Step 27; original run manifest not supplied here"}


def load_windows(meta_path: Path, audit_path: Path, identity: dict):
    source = identity["source_contract"]
    expected_meta = source["step9_artifacts"]["test_window_metadata.csv"]
    expected_audit = source["observability_audit"]["artifacts"]["audit/test_window_observable_effects.csv"]
    hashes = {"test_window_metadata.csv": digest_file(meta_path),
              "test_window_observable_effects.csv": digest_file(audit_path)}
    require(hashes["test_window_metadata.csv"] == expected_meta,
            "Step 9 window metadata hash differs from frozen Step 27 input")
    require(hashes["test_window_observable_effects.csv"] == expected_audit,
            "Step 7 observable-effect audit hash differs from frozen Step 27 input")
    m, a = load_csv(meta_path), load_csv(audit_path)
    require(all(c in m for c in META_COLS) and all(c in a for c in AUDIT_COLS),
            "Window metadata/audit required columns are missing")
    require(len(m) == len(a), "Window metadata vs audit row count mismatch")
    for col in AUDIT_COLS:
        require(np.array_equal(m[col].to_numpy(), a[col].to_numpy()),
                f"Step 7/Step 9 window audit metadata mismatch in {col}")
    index = strict_int(m["embedding_row_index"], "window embedding_row_index")
    require(np.array_equal(index, np.arange(len(m))), "Window metadata not in Step 9 embedding order")
    label = strict_int(m["label_id"], "window label ID")
    require(np.isin(label, np.arange(K)).all() and
            np.array_equal(m["label"].to_numpy(), np.asarray(LABELS)[label]),
            "Window labels differ from canonical six classes")
    steps = strict_int(m["episode_step"], "episode_step")
    win_ids = strict_int(m["window_id"], "window_id")
    flags = strict_int(a["audit_observable_effect_present"], "audit effect flag")
    diffs = strict_int(a["audit_observable_difference_count"], "audit difference count")
    require(np.isin(flags, [0, 1]).all() and np.array_equal(flags, (diffs > 0).astype(np.int64)),
            "Paired-nominal effect flag and difference count disagree")
    for extra in ("audit_persistent_effect_present", "audit_attack_active"):
        if extra in a:
            require(np.isin(strict_int(a[extra], extra), [0, 1]).all(), f"Invalid {extra}")
    require(not m.duplicated(["episode_id", "episode_step"]).any(), "Duplicate window step within an episode")
    require(m.groupby("episode_id", sort=False)["label_id"].nunique().eq(1).all(),
            "Mixed scenario labels within one episode")
    timestamps = pd.to_datetime(m["timestamp"], errors="raise", utc=True)

    n = len(m)
    old_count = np.full(n, -1, dtype=np.int64)
    extended_count = np.full(n, -1, dtype=np.int64)
    expected_ends: list[np.ndarray] = []
    eligible_epi: list[str] = []
    for episode_id, group in m.groupby("episode_id", sort=False):
        pos = group.index.to_numpy(dtype=np.int64)
        order = np.argsort(steps[pos], kind="stable")
        pos = pos[order]
        require(np.all(np.diff(steps[pos]) == 1) and np.all(np.diff(win_ids[pos]) == 1),
                f"Non-consecutive episode_step/window_id in episode {episode_id}")
        if len(pos) > 1:
            dt = timestamps.iloc[pos].diff().dropna().dt.total_seconds().to_numpy()
            require(np.all(dt == 1.0), f"Nonconsecutive 1-second timestamps in {episode_id}")
        if len(pos) < 20:
            continue
        eligible_epi.append(str(episode_id))
        selected = pos[19:]
        prefix = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(flags[pos], dtype=np.int64)))
        i = np.arange(19, len(pos), dtype=np.int64)
        old_count[selected] = prefix[i+1] - prefix[i-9]
        extended_count[selected] = prefix[i+1] - prefix[i-19]
        expected_ends.append(selected)
    require(expected_ends, "No L20-eligible endpoint exists")
    ends = np.concatenate(expected_ends)
    require(len(np.unique(ends)) == len(ends) and np.all(old_count[ends] >= 0)
            and np.all(extended_count[ends] >= old_count[ends]),
            "Invalid historical group counts")
    cohort = pd.DataFrame({
        "sequence_end_row": ends,
        "episode_id": m["episode_id"].to_numpy()[ends],
        "episode_step": steps[ends],
        "true_label_id": label[ends],
        "last10_changed_count": old_count[ends],
        "last20_changed_count": extended_count[ends],
        "older10_changed_count": extended_count[ends] - old_count[ends],
    }).sort_values("sequence_end_row").reset_index(drop=True)
    cohort["evidence_group"] = np.where(
        cohort["true_label_id"].to_numpy() == 0, "Normal",
        np.where(cohort["last10_changed_count"].to_numpy() > 0, "A_recent10",
                 np.where(cohort["older10_changed_count"].to_numpy() > 0,
                          "B_only_older10", "C_no_change20")))
    require(len(cohort) == len(ends) and
            cohort.duplicated("sequence_end_row").sum() == 0,
            "Endpoint cohort is incomplete or duplicated")
    return cohort, hashes, {"test_windows": n, "all_test_episodes": int(m.episode_id.nunique()),
                            "L20_eligible_episodes": len(eligible_epi),
                            "normal_window_paired_change_count": int(flags[label == 0].sum()),
                            "audit_changed_window_count": int(flags.sum())}


def validate_cohort(cohort: pd.DataFrame, summary: dict, evidence: dict) -> dict:
    ends = cohort["sequence_end_row"].to_numpy(dtype=np.int64)
    require(len(cohort) == int(summary["common_endpoints"]) and
            cohort.episode_id.nunique() == int(summary["common_episodes"]),
            "Rebuilt L20 endpoint/episode counts differ from Step 27")
    check_digest = endpoint_digest(ends)
    require(check_digest == summary["endpoint_sha256"] == evidence["endpoint_sha256"],
            "Independent endpoint hash differs from Step 27")
    group = cohort["evidence_group"].value_counts().to_dict()
    attack = cohort["true_label_id"].to_numpy(dtype=np.int64) > 0
    old = cohort["last10_changed_count"].to_numpy() > 0
    newer = cohort["last20_changed_count"].to_numpy() > 0
    require(int(attack.sum()) == evidence["attack_labeled_endpoints"], "Attack endpoint count mismatch")
    require(int(np.sum(attack & old)) == evidence["L10_in_context_change_attack"] and
            int(np.sum(attack & newer)) == evidence["L20_in_context_change_attack"] and
            int(group.get("B_only_older10", 0)) == evidence["L20_additional_evidence_attack"],
            "Independently reconstructed evidence cohort differs from Step 27")
    y = cohort["true_label_id"].to_numpy(dtype=np.int64)
    for length, active in ((10, old), (20, newer)):
        reference_pred = np.where(active, y, 0)
        reference = metric(confusion(y, reference_pred))["macro_f1"]
        safe_equal_float(reference, evidence[f"L{length}_nondeployable_reference_macro_f1"],
                         f"Step 27 L{length} descriptive evidence-reference check")
    return {"test_endpoints": len(cohort), "episodes": int(cohort.episode_id.nunique()),
            "attack_endpoints": int(attack.sum()), "group_counts": group,
            "endpoint_digest_sha256": check_digest,
            "paired_nominal_reference_is_non_deployable": True}


def aligned_predictions(d: pd.DataFrame, cohort: pd.DataFrame, context: str,
                        seed: int, length: int, identity: dict) -> np.ndarray:
    src = identity["source_contract"]
    required = {"sequence_end_row", "episode_id", "true_label_id", "pred_label_id", "source_split"}
    require(required.issubset(d.columns), f"{context}: missing {sorted(required-set(d))}")
    require(not d[list(required)].eq("").any().any(), f"{context}: blank keys/labels")
    ends = strict_int(d["sequence_end_row"], f"{context}:endpoint")
    require(len(np.unique(ends)) == len(ends), f"{context}: duplicate endpoint row")
    require((d["source_split"] == "test").all(), f"{context}: non-test predictions")
    actual = d.copy()
    actual["sequence_end_row"] = ends
    actual["true_label_id"] = strict_int(d["true_label_id"], f"{context}:true")
    actual["pred_label_id"] = strict_int(d["pred_label_id"], f"{context}:pred")
    for column, expected in (("dataset_id", src["dataset_id"]),
                             ("step8_run_id", src["step8_run_id"]),
                             ("step9_run_id", src["step9_run_id"])):
        require(column in d and d[column].eq(expected).all(),
                f"{context}: {column} dataset/source mismatch")
    if length == 20:
        require(len(actual) == len(cohort), f"{context}: L20 row count mismatch")
        for column, expected in (("seed", seed), ("sequence_length", 20)):
            require(column in d and np.all(strict_int(d[column], f"{context}:{column}") == expected),
                    f"{context}: {column} mismatch")
        require("original_step10_run_id" in d and
                d["original_step10_run_id"].eq(src["step10_run_id"]).all(),
                f"{context}: original Step 10 source mismatch")
        if "confidence" in d:
            confidence = strict_float(d["confidence"], f"{context}:confidence")
            require(((confidence >= 0) & (confidence <= 1)).all(),
                    f"{context}: invalid confidence")
    else:
        require("step10_run_id" in d and d["step10_run_id"].eq(src["step10_run_id"]).all(),
                f"{context}: Step 10 source mismatch")
        expected_run = identity.get("historical_L10_runs", {}).get(str(seed))
        require("step11_run_id" in d and d["step11_run_id"].eq(expected_run).all(),
                f"{context}: historical seed/run ID mismatch")
        require("temporal_context_seconds" in d and
                np.all(strict_int(d["temporal_context_seconds"], f"{context}:temporal_context") == 10),
                f"{context}: not a L10 prediction")
        if "common_population" in d:
            require(np.isin(strict_int(d["common_population"], context + ':common'), [0,1]).all(),
                    f"{context}: invalid common flag")
    indexed = actual.set_index("sequence_end_row", verify_integrity=True)
    target = cohort["sequence_end_row"].to_numpy(dtype=np.int64)
    missing = np.setdiff1d(target, ends, assume_unique=True)
    require(len(missing) == 0, f"{context}: {len(missing)} L20 endpoints absent from predictions")
    if length == 20:
        extra = np.setdiff1d(ends, target, assume_unique=True)
        require(len(extra) == 0, f"{context}: {len(extra)} unexpected L20 endpoints")
    selected = indexed.loc[target]
    require(np.array_equal(selected["episode_id"].to_numpy(), cohort["episode_id"].to_numpy()),
            f"{context}: episode identity mismatch")
    require(np.array_equal(selected["true_label_id"].to_numpy(dtype=np.int64),
                           cohort["true_label_id"].to_numpy(dtype=np.int64)),
            f"{context}: scenario ground truth mismatch")
    pred = selected["pred_label_id"].to_numpy(dtype=np.int64)
    require(np.isin(pred, range(K)).all(), f"{context}: predicted IDs out of range")
    if length == 10:
        if "true_label" in selected:
            require(np.array_equal(selected["true_label"], np.asarray(LABELS)[cohort["true_label_id"].to_numpy(dtype=np.int64)]),
                    f"{context}: true_label names mismatch")
        if "pred_label" in selected:
            require(np.array_equal(selected["pred_label"], np.asarray(LABELS)[pred]),
                    f"{context}: pred_label names mismatch")
        if "sequence_end_step" in selected:
            require(np.array_equal(strict_int(selected["sequence_end_step"], context+':end_step'),
                                   cohort["episode_step"].to_numpy(dtype=np.int64)),
                    f"{context}: episode step mismatch")
        if "correct" in selected:
            require(np.array_equal(strict_int(selected["correct"], context+':correct'),
                                   (pred == cohort["true_label_id"].to_numpy()).astype(np.int64)),
                    f"{context}: incorrect correctness flag")
        if "audit_context_effect_window_count" in selected:
            require(np.array_equal(strict_int(selected["audit_context_effect_window_count"], context+':audit_count'),
                                   cohort["last10_changed_count"].to_numpy(dtype=np.int64)),
                    f"{context}: L10 audit flags differ from independently rebuilt causal history")
        if "audit_current_observable_effect" in selected:
            a_flag = strict_int(selected["audit_current_observable_effect"], context+':current_audit')
            require(np.all(a_flag <= cohort["last10_changed_count"].to_numpy()),
                    f"{context}: current observable change exceeds L10 count")
    return pred


def validate_report_metrics(seed: int, y: np.ndarray, p10: np.ndarray, p20: np.ndarray,
                            comparison: pd.DataFrame) -> dict:
    row = comparison.loc[comparison["seed"].astype(str) == str(seed)]
    require(len(row) == 1, f"Step 27 test comparison missing seed {seed}")
    row = row.iloc[0]
    m10 = metric(confusion(y, p10)); m20 = metric(confusion(y, p20))
    for length, summary in ((10, m10), (20, m20)):
        for name, column in (("macro_f1", "macro_f1"), ("accuracy", "accuracy")):
            safe_equal_float(summary[name], row[f"L{length}_{column}"],
                             f"Step 27 comparison seed {seed}/L{length}/{name}")
    safe_equal_float(m20["macro_f1"]-m10["macro_f1"], row["delta_L20_minus_L10"],
                     f"Step 27 reported delta seed {seed}")
    return {"seed": seed, "L10": m10, "L20": m20, "delta_macro_f1": m20["macro_f1"]-m10["macro_f1"]}


def draw_weights(strata: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    weights = np.zeros((size, len(strata)), dtype=np.int32)
    for label in range(K):
        pos = np.flatnonzero(strata == label)
        require(len(pos) > 0, f"No episodes of class {label} in bootstrap")
        weights[:, pos] = rng.multinomial(len(pos), np.full(len(pos), 1.0/len(pos)), size=size)
    return weights


def bootstrapped_statistics(counts: np.ndarray, strata: np.ndarray,
                             replicates: int, batch: int, rng_seed: int):
    # counts [episode, length (L10,L20), seed, true, pred]
    require(counts.ndim == 5 and counts.shape[1:] == (2, len(SEEDS), K, K),
            "Invalid paired episode-confusion array")
    n_ep = counts.shape[0]
    tp = np.diagonal(counts, axis1=-2, axis2=-1).astype(np.float64)
    true_support = counts.sum(-1).astype(np.float64)
    pred_support = counts.sum(-2).astype(np.float64)
    sufficient = np.stack((tp, true_support, pred_support), axis=-2)
    flat = sufficient.reshape(n_ep, -1)
    results = np.empty((replicates, 2, 3), dtype=np.float64)  # length x (macro, accuracy, FA_N)
    rng = np.random.default_rng(rng_seed)
    started = time.perf_counter()
    for lo in range(0, replicates, batch):
        hi = min(lo+batch, replicates)
        weights = draw_weights(strata, hi-lo, rng)
        selected = (weights.astype(np.float64) @ flat).reshape(hi-lo, 2, len(SEEDS), 3, K)
        selected_tp, selected_support, selected_pred = (selected[..., 0, :], selected[..., 1, :], selected[..., 2, :])
        den = selected_support + selected_pred
        class_f1 = np.divide(2*selected_tp, den,
                             out=np.zeros_like(den), where=den != 0)
        results[lo:hi, :, 0] = class_f1.mean(-1).mean(-1)
        results[lo:hi, :, 1] = (selected_tp.sum(-1) / selected_support.sum(-1)).mean(-1)
        normal_support = selected_support[..., 0]
        require((normal_support > 0).all(), "Bootstrap yielded zero Normal endpoints")
        results[lo:hi, :, 2] = ((normal_support-selected_tp[..., 0])/normal_support).mean(-1)
        if lo == 0 or hi == replicates or hi % 2000 == 0:
            print(f"Bootstrap {hi:,}/{replicates:,} | {time.perf_counter()-started:.1f} s", flush=True)
    return results


def source_metric_table(y: np.ndarray, predictions: np.ndarray):
    rows = []
    for length_i, length in enumerate((10, 20)):
        for seed_i, seed in enumerate(SEEDS):
            row = {"model": f"TST_L{length}", "length": length, "seed": seed}
            row.update(metric(confusion(y, predictions[length_i, seed_i])))
            rows.append(row)
    return pd.DataFrame(rows)


def subgroup_tables(cohort: pd.DataFrame, predictions: np.ndarray):
    y = cohort["true_label_id"].to_numpy(dtype=np.int64)
    group_name = cohort["evidence_group"].to_numpy(dtype=str)
    sequence = ["A_recent10", "B_only_older10", "C_no_change20", "Normal"]
    groups, classes, transition = [], [], []
    for group in sequence:
        selected = group_name == group
        total = int(selected.sum())
        require(total > 0, f"Evidence group {group} is empty")
        for seed_i, seed in enumerate(SEEDS):
            p10, p20 = predictions[:, seed_i, :]
            a = p10[selected] == y[selected]
            b = p20[selected] == y[selected]
            n00 = int((~a & ~b).sum())
            n01 = int((~a & b).sum())
            n10 = int((a & ~b).sum())
            n11 = int((a & b).sum())
            groups.append({"group": group, "seed": seed, "endpoints": total,
                           "L10_correct_scenario_recall": float(a.mean()),
                           "L20_correct_scenario_recall": float(b.mean()),
                           "delta_recall": float(b.mean()-a.mean()),
                           "L10_wrong_L20_correct": n01,
                           "L10_correct_L20_wrong": n10,
                           "both_correct": n11, "both_wrong": n00,
                           "net_correction": n01-n10})
            transition.append({"group": group, "class_label": "ALL_IN_GROUP", "seed": seed,
                               "endpoints": total, "L10_wrong_L20_correct": n01,
                               "L10_correct_L20_wrong": n10,
                               "both_correct": n11, "both_wrong": n00,
                               "net_correction": n01-n10})
            for class_id in range(K):
                take = selected & (y == class_id)
                n = int(take.sum())
                if not n:
                    continue
                x = p10[take] == class_id
                z = p20[take] == class_id
                aa = int((~x & z).sum()); bb = int((x & ~z).sum())
                classes.append({"group": group, "class_id": class_id, "class_label": LABELS[class_id],
                                "seed": seed, "endpoints": n,
                                "L10_correct_scenario_recall": float(x.mean()),
                                "L20_correct_scenario_recall": float(z.mean()),
                                "delta_recall": float(z.mean()-x.mean()),
                                "L10_wrong_L20_correct": aa,
                                "L10_correct_L20_wrong": bb,
                                "net_correction": aa-bb})
                transition.append({"group": group, "class_label": LABELS[class_id], "seed": seed,
                                   "endpoints": n, "L10_wrong_L20_correct": aa,
                                   "L10_correct_L20_wrong": bb,
                                   "both_correct": int((x & z).sum()),
                                   "both_wrong": int((~x & ~z).sum()),
                                   "net_correction": aa-bb})
    group_details = pd.DataFrame(groups)
    class_details = pd.DataFrame(classes)
    for group in sequence:
        rows = group_details[group_details.group == group]
        require(rows.endpoints.nunique() == 1, "Group sample count differs between seeds")
        require(np.all((rows.L10_wrong_L20_correct+rows.L10_correct_L20_wrong +
                        rows.both_correct+rows.both_wrong) == rows.endpoints),
                "Group transition contingency table incomplete")
    return group_details, class_details, pd.DataFrame(transition)


def generate_review_zip(output: Path) -> Path:
    path = output / "28_L10_L20_EVIDENCE_REVIEW.zip"
    keep = [p for p in sorted(output.iterdir()) if p.is_file() and
            p != path and p.suffix in {".json", ".csv", ".npz"}]
    with tempfile.NamedTemporaryFile(dir=output, suffix=".zip.tmp", delete=False) as f:
        temp = Path(f.name)
    try:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.write(Path(__file__), "source/" + Path(__file__).name)
            for item in keep:
                z.write(item, item.name)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    with zipfile.ZipFile(args.final_zip, "r") as z:
        zip_identity, summary, evidence = validate_final_zip(z, args.final_zip)
        l10_files = resolve_l10_files(args, zip_identity)
        cohort, window_hashes, window_info = load_windows(args.window_meta, args.window_audit, zip_identity)
        cohort_info = validate_cohort(cohort, summary, evidence)
        comparison = load_zip_csv(z, "27_test_seed_comparison.csv")
        require(set(strict_int(comparison["seed"], "27 seed comparison")) == set(SEEDS)
                and len(comparison) == 3, "Step 27 comparison does not contain exactly 3 seeds")
        keys = cohort["sequence_end_row"].to_numpy(dtype=np.int64)
        y = cohort["true_label_id"].to_numpy(dtype=np.int64)
        predictions = np.zeros((2, len(SEEDS), len(cohort)), dtype=np.int64)
        audits: list[dict] = []
        for i, seed in enumerate(SEEDS):
            raw10_path = l10_files[i]
            source10_hash = digest_file(raw10_path)
            run_check = authenticate_original_l10_manifest(raw10_path, seed, zip_identity)
            csv10 = load_csv(raw10_path)
            predictions[0, i] = aligned_predictions(csv10, cohort, f"L10 seed {seed}", seed, 10, zip_identity)
            require(digest_file(raw10_path) == source10_hash,
                    f"L10 file mutated during read: {raw10_path}")
            member = f"seed_{seed}/test_predictions.csv"
            predictions[1, i] = aligned_predictions(load_zip_csv(z, member), cohort,
                                                      f"L20 seed {seed}", seed, 20, zip_identity)
            checked = validate_report_metrics(seed, y, predictions[0, i], predictions[1, i], comparison)
            audits.append({"seed": seed, "L10_file": str(raw10_path),
                           "L10_file_sha256": source10_hash, "L10_rows": len(csv10),
                           "L20_zip_member": member, "L20_member_sha256": digest_bytes(z.read(member)),
                           "L20_rows": len(cohort), "L10_original_manifest": run_check,
                           "metrics_check": checked})
            print(f"[PASS] seed={seed} | L10={checked['L10']['macro_f1']:.6f} "
                  f"L20={checked['L20']['macro_f1']:.6f} | "
                  f"delta={checked['delta_macro_f1']:+.6f}", flush=True)
        require(endpoint_digest(keys) == summary["endpoint_sha256"], "Cohort changed")
        results = source_metric_table(y, predictions)
        means = results.groupby("length", sort=True)[["macro_f1", "accuracy", "normal_false_alarm_fraction"]].agg(["mean", "std"])
        mean_l10 = means.loc[10, ("macro_f1", "mean")]
        mean_l20 = means.loc[20, ("macro_f1", "mean")]
        safe_equal_float(mean_l10, summary["L10_macro_f1_mean"], "Step 27 L10 mean")
        safe_equal_float(mean_l20, summary["L20_macro_f1_mean"], "Step 27 L20 mean")
        safe_equal_float(mean_l20 - mean_l10, summary["delta_L20_minus_L10_mean"], "Step 27 mean difference")
        group_table, class_table, transitions = subgroup_tables(cohort, predictions)
        print("Evidence groups:", cohort.evidence_group.value_counts().to_dict(), flush=True)
        output = args.output_root / ("run_" + utc_stamp())
        require(not output.exists(), f"Output run folder already exists: {output}")
        output.mkdir(parents=True, exist_ok=False)
        protocol = {"analysis": "Step 28-A matched L10/L20 evidence-location and bootstrap",
                    "version": "28_exploratory_post_test_v1", "test_already_inspected_before_design": True,
                    "predeclared_directional_expectations": {
                        "B_only_older10": "L20 additional records may produce the largest gain",
                        "A_recent10": "smaller gains anticipated but neither sign nor magnitude assumed",
                        "C_no_change20": "limited gains expected but normal class prediction is NOT enforced",
                    },
                    "models": ["historical TST L10", "Step27 TST L20"], "seeds": list(SEEDS),
                    "bootstrap_replicates": args.replicates, "bootstrap_seed": args.bootstrap_seed,
                    "bootstrap_batch": args.batch,
                    "resampling": "within-six-class, whole-episode stratified paired multinomial bootstrap",
                    "estimand": "arithmetic mean across 3 fixed per-seed six-class endpoint macro-F1 scores",
                    "contrast": "L20 minus L10", "ci": "two-sided 95% percentile, conditional",
                    "subgroups": "posthoc attack-label A/B/C; subgroup transition and recall descriptive only",
                    "no_training": True, "baseline_training_cohorts_are_unmatched": True,
                    "standard_error_or_seed_resampling": False,
                    "scope": "controlled shared_nominal_v2 benchmark; no operational detection guarantee",
                    "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                    "code_sha256": digest_file(Path(__file__)),
                    "audit_only": bool(args.audit_only)}
        write_json(output / "28_protocol.json", protocol)
        manifest = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "output": str(output), "script_sha256": protocol["code_sha256"]}
        write_json(output / "28_run_manifest.json", manifest)
        try:
            episode_codes, episode_order = pd.factorize(cohort["episode_id"], sort=True)
            inventory = cohort.groupby("episode_id", sort=True).agg(
                class_id=("true_label_id", "first"), endpoints=("true_label_id", "size"))
            require(np.array_equal(inventory.index.to_numpy(), np.asarray(episode_order)) and
                    cohort.groupby("episode_id")["true_label_id"].nunique().eq(1).all(),
                    "Inconsistent episode label across evaluated endpoints")
            np_epi = len(inventory)
            counts = np.empty((np_epi, 2, len(SEEDS), K, K), dtype=np.int64)
            for l in range(2):
                for s in range(len(SEEDS)):
                    flat = ((episode_codes * K + y) * K + predictions[l, s])
                    counts[:, l, s] = np.bincount(flat, minlength=np_epi*K*K).reshape(np_epi, K, K)
            require(np.all(counts.sum(axis=(0, 3, 4)) == len(cohort)),
                    "Episode confusion counts do not reconstruct the full endpoint population")
            write_csv(output / "28_endpoint_evidence.csv", cohort)
            write_csv(output / "28_metrics_by_seed.csv", results)
            write_csv(output / "28_group_summary.csv", group_table)
            write_csv(output / "28_group_class_detail.csv", class_table)
            group_mean = group_table.groupby("group", sort=False).agg(
                endpoints=("endpoints", "first"),
                L10_recall_mean=("L10_correct_scenario_recall", "mean"),
                L20_recall_mean=("L20_correct_scenario_recall", "mean"),
                delta_recall_mean=("delta_recall", "mean"),
                L10_wrong_L20_correct_seed_mean=("L10_wrong_L20_correct", "mean"),
                L10_correct_L20_wrong_seed_mean=("L10_correct_L20_wrong", "mean"),
                net_correction_seed_mean=("net_correction", "mean"),
            ).reset_index()
            class_mean = class_table.groupby(["group", "class_id", "class_label"], sort=False).agg(
                endpoints=("endpoints", "first"),
                L10_recall_mean=("L10_correct_scenario_recall", "mean"),
                L20_recall_mean=("L20_correct_scenario_recall", "mean"),
                delta_recall_mean=("delta_recall", "mean"),
                net_correction_seed_mean=("net_correction", "mean"),
            ).reset_index()
            write_csv(output / "28_group_means.csv", group_mean)
            write_csv(output / "28_group_class_means.csv", class_mean)
            write_csv(output / "28_prediction_transitions.csv", transitions)
            input_record = {"dataset_id": zip_identity["source_contract"]["dataset_id"],
                            "step27_zip": str(args.final_zip), "step27_zip_sha256": digest_file(args.final_zip),
                            "step27_identity_hash": summary["identity_hash"],
                            "window_metadata_path": str(args.window_meta),
                            "window_audit_path": str(args.window_audit),
                            "window_hashes": window_hashes, "window_validation": window_info,
                            "cohort_validation": cohort_info,
                            "historical_L10_source_run_ids": zip_identity["historical_L10_runs"],
                            "prediction_audits": audits,
                            "source_27_reported_evidence": evidence,
                            "original_step15_bootstrap_estimator": "whole-episode class stratification; same draw for all models/seeds"}
            write_json(output / "28_input_audit.json", input_record)
            summary_rows = []
            for length in (10, 20):
                row = {"length": length, "model": f"TST_L{length}", "n_seeds": len(SEEDS),
                       "test_endpoints": len(cohort), "test_episodes": np_epi}
                for met in ("macro_f1", "accuracy", "normal_false_alarm_fraction"):
                    sample = results.loc[results.length == length, met].to_numpy(dtype=float)
                    row[f"{met}_seed_mean"] = float(sample.mean())
                    row[f"{met}_seed_sd_ddof1"] = float(sample.std(ddof=1))
                summary_rows.append(row)
            write_csv(output / "28_model_summary.csv", pd.DataFrame(summary_rows))
            manifest["status"] = "AUDIT_ONLY_COMPLETE" if args.audit_only else "RUNNING_BOOTSTRAP"
            write_json(output / "28_run_manifest.json", manifest)
            if not args.audit_only:
                boot = bootstrapped_statistics(counts, inventory["class_id"].to_numpy(dtype=np.int64),
                                               args.replicates, args.batch, args.bootstrap_seed)
                columns = ["macro_f1", "accuracy", "normal_false_alarm_fraction"]
                comparison_rows = []
                for j, name in enumerate(columns):
                    d = boot[:, 1, j] - boot[:, 0, j]
                    ci_l10 = np.quantile(boot[:, 0, j], [.025, .975])
                    ci_l20 = np.quantile(boot[:, 1, j], [.025, .975])
                    ci = np.quantile(d, [.025, .975])
                    v10 = float(results.loc[results.length == 10, name].mean())
                    v20 = float(results.loc[results.length == 20, name].mean())
                    comparison_rows.append({"metric": name, "contrast": "TST_L20_minus_TST_L10",
                        "L10_seed_mean": v10, "L20_seed_mean": v20, "delta": v20-v10,
                        "ci95_delta_low": float(ci[0]), "ci95_delta_high": float(ci[1]),
                        "L10_ci95_low": float(ci_l10[0]), "L10_ci95_high": float(ci_l10[1]),
                        "L20_ci95_low": float(ci_l20[0]), "L20_ci95_high": float(ci_l20[1]),
                        "ci_basis": "paired six-class-stratified whole-episode percentile bootstrap",
                        "exploratory_post_test": True})
                write_csv(output / "28_paired_bootstrap.csv", pd.DataFrame(comparison_rows))
                write_json(output / "28_findings.json", {
                    "status": "EXPLORATORY_POST_TEST", "dataset_id": EXPECTED_DATASET_ID,
                    "matched_test_endpoints": len(cohort), "test_episodes": np_epi,
                    "group_counts": cohort.evidence_group.value_counts().to_dict(),
                    "historical_L10_training_cohort_not_matched_to_L20": True,
                    "sampled_units": "class-stratified whole episodes",
                    "seed_handling": "three fixed model runs; macro-F1 computed per seed then averaged",
                    "model_seed_means": summary_rows,
                    "paired_bootstrap": comparison_rows,
                    "group_seed_means": group_mean.to_dict(orient="records"),
                    "limitations": ["subgroups identified using nondeployable paired nominal audit",
                                    "design followed inspection of Step 27 held-out test",
                                    "not a TST-specific advantage or operational attack-state detection result"],
                })
                with tempfile.NamedTemporaryFile(dir=output, suffix=".npz.tmp", delete=False) as temp:
                    temp_path = Path(temp.name)
                try:
                    with temp_path.open("wb") as stream:
                        np.savez_compressed(stream, boot_metrics=boot, metric_names=np.asarray(columns),
                                            model_names=np.asarray(["TST_L10", "TST_L20"]),
                                            head_seeds=np.asarray(SEEDS))
                    temp_path.replace(output / "28_bootstrap_distribution.npz")
                finally:
                    temp_path.unlink(missing_ok=True)
                manifest["status"] = "COMPLETE"
                print(pd.DataFrame(comparison_rows)[["metric", "delta", "ci95_delta_low", "ci95_delta_high"]].to_string(index=False),
                      flush=True)
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["elapsed_seconds"] = round(time.perf_counter()-started, 2)
            manifest["artifacts"] = {p.name: digest_file(p) for p in sorted(output.iterdir())
                                      if p.is_file() and p.name != "28_run_manifest.json"}
            write_json(output / "28_run_manifest.json", manifest)
            review = generate_review_zip(output)
            print(f"\n{manifest['status']} | {len(cohort):,} endpoints / {np_epi} episodes",
                  f"\nReview ZIP: {review}", flush=True)
        except BaseException as exc:
            manifest.update(status="FAILED", failure=f"{type(exc).__name__}: {exc}",
                            finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "28_run_manifest.json", manifest)
            raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nSTEP 28 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
