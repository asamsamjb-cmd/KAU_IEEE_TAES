# -*- coding: utf-8 -*-
"""Step 31 (experiment D) -- same-population context-length sweep, L = 1, 2, 3, 5, 10, 20.

No training. Re-scores existing predictions on the L20-eligible TEST endpoints
(30,561 endpoints / 751 episodes, the Step 27/28 cohort) and places them beside
the evidence-availability reference computed with the same paired-nominal
flags at each L.

Inputs (all read-only, all hash-checked against Step 27 / Step 28 records):
  * Step 12 historical TST heads, seeds 20260610-20260612:
        outputs/shared_nominal_v2/tst_training_finetuned_main/L{L}/predictions/test_predictions_L{L}.csv   (20260610)
        outputs/shared_nominal_v2/tst_training_finetuned_multiseed/seed_{seed}/L{L}/predictions/test_predictions_L{L}.csv
    for L in 1, 2, 3, 5, 10
  * Step 27 final test review ZIP (L20 predictions, run manifest, test reference values):
        outputs/shared_nominal_v2/27_L20_TST_VALIDATION_PILOT/27_L20_FINAL_TEST_REVIEW.zip
  * Step 28 run folder (cohort file, metrics by seed, input audit):
        outputs/shared_nominal_v2/28_l20_evidence_location/run_*/
  * Step 9 test window metadata and Step 7 paired-nominal audit:
        outputs/shared_nominal_v2/tst_embeddings_finetuned_main/test_window_metadata.csv
        dataset/shared_nominal_v2/audit/test_window_observable_effects.csv

Gates (hard stop on failure):
  G1 input hashes equal the values recorded by Step 27 (manifest) and Step 28 (input audit)
  G2 every L covers all 30,561 cohort endpoints with identical labels and episode IDs
  G3 L10/L20 per-seed macro-F1 reproduce 28_metrics_by_seed.csv (abs 1e-9)
  G4 last-10 / last-20 change counts reproduce 28_endpoint_evidence.csv exactly
  G5 L10/L20 reference macro-F1 reproduce 27_evidence_availability_test.json (abs 2e-9)

Findings are recorded, never enforced: TST below reference at each L, monotonicity in L.

Run from the code directory (PowerShell, .venv active):
    py "31.l20_population_lsweep.py"
Optional:  --final-zip <zip>  --step28-run <folder>  --output-root <dir>  --graph-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = (
    "NORMAL", "RF_LINK_JAMMING", "GNSS_SPOOFING",
    "UNAUTHORIZED_TELECOMMAND_INJECTION", "TELECOMMAND_REPLAY",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE",
)
K = len(LABELS)
SEEDS = (20260610, 20260611, 20260612)
LENGTHS = (1, 2, 3, 5, 10, 20)
HISTORICAL_LENGTHS = (1, 2, 3, 5, 10)
NEW_LENGTH = 20
COHORT_MIN_STEP = NEW_LENGTH - 1

BASE = Path(__file__).resolve().parent
ROOT_REL = Path("outputs/shared_nominal_v2")
DEFAULT_27 = ROOT_REL / "27_L20_TST_VALIDATION_PILOT" / "27_L20_FINAL_TEST_REVIEW.zip"
DEFAULT_28 = ROOT_REL / "28_l20_evidence_location"
DEFAULT_OUTPUT = ROOT_REL / "31_l20_population_lsweep"
DEFAULT_GRAPH = Path("graph") / "S6_context_length_sweep_L20"
EXPECTED_DATASET_ID = "478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b"
FIG_STEM = "fig_context_length_sweep_L20pop"
ANNOTATE_L10_TO_L20_GAIN = True


# ----------------------------------------------------------------------------- helpers
class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for part in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_ends(a: np.ndarray) -> str:
    # identical to Step 27 / 28 digest_ends
    return hashlib.sha256(np.sort(np.asarray(a, dtype="<i8")).tobytes()).hexdigest()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", suffix=".tmp", delete=False) as fp:
        tmp = Path(fp.name)
        json.dump(obj, fp, indent=2, ensure_ascii=False, allow_nan=False)
    tmp.replace(path)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8-sig", suffix=".tmp",
                                     delete=False, newline="") as fp:
        tmp = Path(fp.name)
        df.to_csv(fp, index=False)
    tmp.replace(path)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def load_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", keep_default_na=False)


def strict_int(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    require(values.notna().all(), f"Non-numeric value in {name}")
    arr = values.to_numpy(dtype=np.float64)
    require(np.all(arr == np.round(arr)), f"Non-integer value in {name}")
    return arr.astype(np.int64)


def as_str(series: pd.Series) -> np.ndarray:
    return np.asarray(series.astype("string").to_numpy(), dtype=object)


def confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    require(y.shape == p.shape and y.ndim == 1 and len(y) > 0, "Invalid true/pred arrays")
    require(np.isin(y, np.arange(K)).all() and np.isin(p, np.arange(K)).all(), "Invalid class ID")
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)


def metric(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(np.float64)
    row = cm.sum(1).astype(np.float64)
    col = cm.sum(0).astype(np.float64)
    den = row + col
    class_f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return {
        "macro_f1": float(class_f1.mean()),
        "accuracy": float(tp.sum() / row.sum()),
        "normal_false_alarm_fraction": float((row[0] - tp[0]) / row[0]) if row[0] else float("nan"),
        "class_f1": [float(v) for v in class_f1],
    }


def close(a: float, b: float, atol: float) -> bool:
    return np.isfinite(a) and np.isfinite(b) and abs(float(a) - float(b)) <= atol


# ----------------------------------------------------------------------------- arguments
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=BASE)
    p.add_argument("--final-zip", type=Path, default=None, help="Step 27 FINAL TEST review ZIP")
    p.add_argument("--step28-run", type=Path, default=None, help="Step 28 run folder (run_*)")
    p.add_argument("--window-meta", type=Path, default=None)
    p.add_argument("--window-audit", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--graph-dir", type=Path, default=None)
    a = p.parse_args()
    a.project_root = a.project_root.expanduser().resolve()
    a.final_zip = (a.final_zip or (a.project_root / DEFAULT_27)).expanduser().resolve()
    a.window_meta = (a.window_meta or (a.project_root / ROOT_REL /
                     "tst_embeddings_finetuned_main" / "test_window_metadata.csv")).expanduser().resolve()
    a.window_audit = (a.window_audit or (a.project_root / "dataset" / "shared_nominal_v2" /
                      "audit" / "test_window_observable_effects.csv")).expanduser().resolve()
    a.output_root = (a.output_root or (a.project_root / DEFAULT_OUTPUT)).expanduser().resolve()
    a.graph_dir = (a.graph_dir or (a.project_root / DEFAULT_GRAPH)).expanduser().resolve()
    if a.step28_run is None:
        runs = sorted((a.project_root / DEFAULT_28).glob("run_*"))
        complete = []
        for r in runs:
            mf = r / "28_run_manifest.json"
            if mf.is_file():
                try:
                    if json.loads(mf.read_text(encoding="utf-8")).get("status") == "COMPLETE":
                        complete.append(r)
                except json.JSONDecodeError:
                    pass
        require(len(complete) == 1,
                f"Expected exactly one COMPLETE Step 28 run under {a.project_root / DEFAULT_28}; "
                f"found {len(complete)}. Pass --step28-run explicitly.")
        a.step28_run = complete[0]
    a.step28_run = a.step28_run.expanduser().resolve()
    for pth in (a.final_zip, a.window_meta, a.window_audit):
        require(pth.is_file(), f"Input not found: {pth}")
    require(a.step28_run.is_dir(), f"Step 28 run folder not found: {a.step28_run}")
    return a


# ----------------------------------------------------------------------------- inputs
def load_step28(run: Path) -> dict:
    files = {name: run / name for name in ("28_run_manifest.json", "28_input_audit.json",
                                            "28_endpoint_evidence.csv", "28_metrics_by_seed.csv")}
    for name, path in files.items():
        require(path.is_file(), f"Step 28 file missing: {path}")
    manifest = json.loads(files["28_run_manifest.json"].read_text(encoding="utf-8"))
    require(manifest.get("status") == "COMPLETE", "Step 28 run is not COMPLETE")
    recorded = manifest.get("artifacts", {})
    for name in ("28_input_audit.json", "28_endpoint_evidence.csv", "28_metrics_by_seed.csv"):
        if name in recorded:
            require(digest_file(files[name]) == recorded[name], f"Step 28 artifact hash mismatch: {name}")
    audit = json.loads(files["28_input_audit.json"].read_text(encoding="utf-8"))
    cohort = load_csv(files["28_endpoint_evidence.csv"])
    for col in ("sequence_end_row", "episode_id", "episode_step", "true_label_id",
                "last10_changed_count", "last20_changed_count", "older10_changed_count"):
        require(col in cohort, f"28_endpoint_evidence.csv lacks column {col}")
    cohort = cohort.sort_values("sequence_end_row").reset_index(drop=True)
    metrics = load_csv(files["28_metrics_by_seed.csv"])
    return {"run": run, "manifest": manifest, "audit": audit, "cohort": cohort, "metrics": metrics,
            "hashes": {name: digest_file(path) for name, path in files.items()}}


def load_step27(zip_path: Path, audit28: dict) -> dict:
    zip_sha = digest_file(zip_path)
    require(zip_sha == audit28.get("step27_zip_sha256"),
            "Step 27 final ZIP hash differs from the ZIP that Step 28 analysed")
    with zipfile.ZipFile(zip_path, "r") as z:
        names = set(z.namelist())
        for member in ("27_run_manifest.json", "27_test_summary.json", "27_evidence_availability_test.json"):
            require(member in names, f"Step 27 ZIP missing {member}")
        manifest = json.loads(z.read("27_run_manifest.json").decode("utf-8"))
        summary = json.loads(z.read("27_test_summary.json").decode("utf-8"))
        evidence = json.loads(z.read("27_evidence_availability_test.json").decode("utf-8"))
        preds, member_sha = {}, {}
        by_seed = {int(e["seed"]): e for e in audit28.get("prediction_audits", [])}
        for seed in SEEDS:
            member = f"seed_{seed}/test_predictions.csv"
            require(member in names, f"Step 27 ZIP missing {member}")
            data = z.read(member)
            member_sha[seed] = digest_bytes(data)
            require(seed in by_seed and member_sha[seed] == by_seed[seed].get("L20_member_sha256"),
                    f"Step 27 L20 prediction member hash differs from Step 28 record (seed {seed})")
            preds[seed] = load_csv_bytes(data)
    source = manifest["identity"]["source_contract"]
    require(source["dataset_id"] == EXPECTED_DATASET_ID, "Step 27 dataset_id is not shared_nominal_v2")
    require(evidence.get("split") == "test", "27_evidence_availability_test.json is not the test split")
    return {"path": zip_path, "sha256": zip_sha, "manifest": manifest, "summary": summary,
            "evidence": evidence, "predictions": preds, "member_sha256": member_sha, "source": source}


def load_windows(meta_path: Path, audit_path: Path, source: dict) -> dict:
    meta_sha, audit_sha = digest_file(meta_path), digest_file(audit_path)
    require(meta_sha == source["step9_artifacts"]["test_window_metadata.csv"],
            "Step 9 test_window_metadata.csv hash differs from Step 27 record")
    require(audit_sha == source["observability_audit"]["artifacts"]["audit/test_window_observable_effects.csv"],
            "Step 7 test_window_observable_effects.csv hash differs from Step 27 record")
    m, a = load_csv(meta_path), load_csv(audit_path)
    for col in ("embedding_row_index", "episode_id", "episode_step", "label_id"):
        require(col in m, f"window metadata lacks column {col}")
    require("audit_observable_effect_present" in a, "window audit lacks audit_observable_effect_present")
    require(len(m) == len(a), "window metadata vs audit row count mismatch")
    for col in ("episode_id", "episode_step"):
        if col in a:
            require(np.array_equal(as_str(m[col]), as_str(a[col])) if col == "episode_id"
                    else np.array_equal(strict_int(m[col], col), strict_int(a[col], col)),
                    f"window metadata vs audit mismatch in {col}")
    index = strict_int(m["embedding_row_index"], "embedding_row_index")
    require(np.array_equal(index, np.arange(len(m))), "window metadata is not in embedding row order")
    flags = strict_int(a["audit_observable_effect_present"], "audit_observable_effect_present")
    require(np.isin(flags, [0, 1]).all(), "audit flag must be 0/1")
    return {"meta": m, "flags": flags, "episode": as_str(m["episode_id"]),
            "step": strict_int(m["episode_step"], "episode_step"),
            "label": strict_int(m["label_id"], "label_id"),
            "hashes": {"test_window_metadata.csv": meta_sha, "test_window_observable_effects.csv": audit_sha}}


def historical_prediction_path(root: Path, seed: int, length: int) -> Path:
    base = (root / ROOT_REL / "tst_training_finetuned_main" if seed == 20260610
            else root / ROOT_REL / "tst_training_finetuned_multiseed" / f"seed_{seed}")
    return base / f"L{length}" / "predictions" / f"test_predictions_L{length}.csv"


def load_historical(root: Path, audit28: dict, source: dict) -> dict:
    by_seed = {int(e["seed"]): e for e in audit28.get("prediction_audits", [])}
    out, hashes = {}, {}
    for seed in SEEDS:
        for length in HISTORICAL_LENGTHS:
            path = historical_prediction_path(root, seed, length)
            require(path.is_file(), f"Historical prediction file not found: {path}")
            sha = digest_file(path)
            hashes[(seed, length)] = (path, sha)
            if length == 10:
                require(seed in by_seed and sha == by_seed[seed].get("L10_file_sha256"),
                        f"Historical L10 prediction hash differs from Step 28 record (seed {seed}): {path}")
            df = load_csv(path)
            for col in ("sequence_end_row", "episode_id", "true_label_id", "pred_label_id"):
                require(col in df, f"{path} lacks column {col}")
            for col, expected in (("dataset_id", source["dataset_id"]), ("step8_run_id", source["step8_run_id"]),
                                  ("step9_run_id", source["step9_run_id"]), ("step10_run_id", source["step10_run_id"]),
                                  ("source_split", "test")):
                if col in df:
                    require(bool(df[col].astype("string").eq(expected).all()),
                            f"{path}: provenance column {col} differs from Step 27 source contract")
            out[(seed, length)] = df
    return {"frames": out, "hashes": hashes}


# ----------------------------------------------------------------------------- cohort
def build_cohort_counts(windows: dict, cohort: pd.DataFrame) -> dict:
    """Per-endpoint in-context change counts for every L, from the window flags; checks against Step 28."""
    flags, episode, step = windows["flags"], windows["episode"], windows["step"]
    n = len(flags)
    ends = strict_int(cohort["sequence_end_row"], "sequence_end_row")
    require(len(np.unique(ends)) == len(ends) and ends.min() >= 0 and ends.max() < n, "Invalid cohort end rows")
    counts = {L: np.full(n, -1, dtype=np.int64) for L in LENGTHS}
    codes, _ = pd.factorize(pd.Series(episode), sort=False)
    order = np.lexsort((step, codes))  # group by episode (first-appearance order), then step
    ep_sorted = episode[order]
    boundaries = np.flatnonzero(np.concatenate(([True], ep_sorted[1:] != ep_sorted[:-1], [True])))
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        pos = order[s:e]
        require(np.all(np.diff(step[pos]) == 1), f"Non-consecutive episode_step in episode {ep_sorted[s]}")
        prefix = np.concatenate(([0], np.cumsum(flags[pos], dtype=np.int64)))
        i = np.arange(len(pos), dtype=np.int64)
        for L in LENGTHS:
            valid = i >= L - 1
            counts[L][pos[valid]] = prefix[i[valid] + 1] - prefix[i[valid] - L + 1]
    for L in LENGTHS:
        require(np.all(counts[L][ends] >= 0), f"Cohort endpoint lacks L={L} context (episode too short)")
    # G4: reproduce Step 28 counts exactly
    require(np.array_equal(counts[10][ends], strict_int(cohort["last10_changed_count"], "last10"))
            and np.array_equal(counts[20][ends], strict_int(cohort["last20_changed_count"], "last20")),
            "G4 failed: last-10 / last-20 change counts do not reproduce 28_endpoint_evidence.csv")
    require(np.array_equal(windows["label"][ends], strict_int(cohort["true_label_id"], "true_label_id"))
            and np.array_equal(episode[ends], as_str(cohort["episode_id"]))
            and np.all(step[ends] >= COHORT_MIN_STEP),
            "Cohort labels/episodes/steps disagree with window metadata")
    return {"ends": ends, "y": windows["label"][ends], "episode": episode[ends],
            "hit": {L: counts[L][ends] > 0 for L in LENGTHS}}


def match_predictions(df: pd.DataFrame, cohort: dict, what: str) -> np.ndarray:
    ends = strict_int(df["sequence_end_row"], f"{what} sequence_end_row")
    require(len(np.unique(ends)) == len(ends), f"{what}: duplicate sequence_end_row")
    frame = pd.DataFrame({"end": ends, "y": strict_int(df["true_label_id"], f"{what} true_label_id"),
                          "p": strict_int(df["pred_label_id"], f"{what} pred_label_id"),
                          "ep": as_str(df["episode_id"])}).set_index("end")
    present = frame.index.isin(cohort["ends"])
    missing = ~pd.Index(cohort["ends"]).isin(frame.index)
    require(not missing.any(), f"G2 failed: {what} does not cover {int(missing.sum())} cohort endpoints")
    sel = frame.loc[cohort["ends"]]
    require(np.array_equal(sel["y"].to_numpy(), cohort["y"]), f"G2 failed: {what} labels differ from cohort")
    require(np.array_equal(sel["ep"].to_numpy(), cohort["episode"]), f"G2 failed: {what} episode IDs differ")
    _ = present
    return sel["p"].to_numpy(dtype=np.int64)


# ----------------------------------------------------------------------------- analysis
def run_analysis(cohort: dict, hist: dict, s27: dict, s28: dict) -> dict:
    y = cohort["y"]
    attack = y != 0
    rows_seed, rows_L, rows_class = [], [], []
    reference = {}
    for L in LENGTHS:
        hit = cohort["hit"][L]
        ref_pred = np.where(hit & attack, y, 0).astype(np.int64)
        ref = metric(confusion(y, ref_pred))
        reference[L] = {"macro_f1": ref["macro_f1"], "accuracy": ref["accuracy"],
                        "attack_coverage": float(hit[attack].mean()),
                        "attack_endpoints_with_change": int(np.count_nonzero(hit & attack))}
    # G5
    ev = s27["evidence"]
    for L, key in ((10, "L10_nondeployable_reference_macro_f1"), (20, "L20_nondeployable_reference_macro_f1")):
        if key in ev:
            require(close(reference[L]["macro_f1"], float(ev[key]), 2e-9),
                    f"G5 failed: L{L} reference macro-F1 {reference[L]['macro_f1']} != Step 27 {ev[key]}")
    if "endpoint_sha256" in ev:
        require(digest_ends(cohort["ends"]) == ev["endpoint_sha256"], "Cohort endpoint digest differs from Step 27")

    per_seed = {}
    for seed in SEEDS:
        for L in LENGTHS:
            df = s27["predictions"][seed] if L == NEW_LENGTH else hist["frames"][(seed, L)]
            what = f"seed {seed} L={L}"
            pred = match_predictions(df, cohort, what)
            m = metric(confusion(y, pred))
            per_seed[(seed, L)] = m
            rows_seed.append({"model": "TST", "length": L, "seed": seed, "macro_f1": m["macro_f1"],
                              "accuracy": m["accuracy"],
                              "normal_false_alarm_fraction": m["normal_false_alarm_fraction"],
                              "reference_macro_f1": reference[L]["macro_f1"],
                              "gap_to_reference": m["macro_f1"] - reference[L]["macro_f1"]})
            for c, f1 in enumerate(m["class_f1"]):
                rows_class.append({"length": L, "seed": seed, "class_id": c, "class_label": LABELS[c], "f1": f1})
    # G3
    m28 = s28["metrics"]
    for _, r in m28.iterrows():
        L, seed = int(r["length"]), int(r["seed"])
        require((seed, L) in per_seed, f"Step 28 metrics row not reproduced: {seed} L={L}")
        for col in ("macro_f1", "accuracy", "normal_false_alarm_fraction"):
            require(close(per_seed[(seed, L)][col], float(r[col]), 1e-9),
                    f"G3 failed: seed {seed} L={L} {col} {per_seed[(seed, L)][col]} != Step 28 {r[col]}")

    seed_frame = pd.DataFrame(rows_seed)
    class_frame = pd.DataFrame(rows_class)
    for L in LENGTHS:
        sub = seed_frame[seed_frame.length == L]
        rows_L.append({
            "length": L, "n_seeds": len(sub), "test_endpoints": int(len(y)),
            "test_episodes": int(len(np.unique(cohort["episode"]))),
            "tst_macro_f1_mean": float(sub.macro_f1.mean()), "tst_macro_f1_sd_ddof1": float(sub.macro_f1.std(ddof=1)),
            "tst_accuracy_mean": float(sub.accuracy.mean()),
            "tst_normal_false_alarm_fraction_mean": float(sub.normal_false_alarm_fraction.mean()),
            "reference_macro_f1": reference[L]["macro_f1"],
            "reference_attack_coverage": reference[L]["attack_coverage"],
            "attack_endpoints_with_change": reference[L]["attack_endpoints_with_change"],
            "gap_tst_minus_reference": float(sub.macro_f1.mean()) - reference[L]["macro_f1"],
        })
    sweep = pd.DataFrame(rows_L)
    tst = sweep.tst_macro_f1_mean.to_numpy()
    findings = {
        "tst_below_reference_at_every_L": bool(np.all(sweep.gap_tst_minus_reference.to_numpy() < 0)),
        "lengths_where_tst_not_below_reference": [int(v) for v in sweep.length[sweep.gap_tst_minus_reference >= 0]],
        "tst_monotone_nondecreasing_in_L": bool(np.all(np.diff(tst) >= 0)),
        "reference_monotone_nondecreasing_in_L": bool(np.all(np.diff(sweep.reference_macro_f1.to_numpy()) >= 0)),
        "gain_L10_to_L20_tst": float(tst[LENGTHS.index(20)] - tst[LENGTHS.index(10)]),
        "gain_L10_to_L20_reference": float(reference[20]["macro_f1"] - reference[10]["macro_f1"]),
        "gain_L1_to_L20_tst": float(tst[LENGTHS.index(20)] - tst[LENGTHS.index(1)]),
        "gain_L1_to_L20_reference": float(reference[20]["macro_f1"] - reference[1]["macro_f1"]),
    }
    return {"sweep": sweep, "per_seed": seed_frame, "per_class": class_frame, "findings": findings,
            "reference": reference}


# ----------------------------------------------------------------------------- figure
def draw_figure(sweep: pd.DataFrame, findings: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                         "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    x = np.arange(len(LENGTHS))
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.errorbar(x, sweep.tst_macro_f1_mean, yerr=sweep.tst_macro_f1_sd_ddof1, marker="o", markersize=3.5,
                linewidth=1.2, capsize=2, color="C0", label="TST", zorder=3)
    ax.plot(x, sweep.reference_macro_f1, linestyle="--", marker="s", markersize=3.5, linewidth=1.0,
            color="0.45", label="Availability reference", zorder=2)
    if ANNOTATE_L10_TO_L20_GAIN:
        i10, i20 = LENGTHS.index(10), LENGTHS.index(20)
        ax.annotate(f"{findings['gain_L10_to_L20_tst']:+.3f}",
                    xy=(x[i20], sweep.tst_macro_f1_mean.iloc[i20]),
                    xytext=((x[i10] + x[i20]) / 2, sweep.tst_macro_f1_mean.iloc[i10] - 0.12),
                    ha="center", va="top", fontsize=7, color="C0",
                    arrowprops={"arrowstyle": "-", "color": "C0", "linewidth": 0.6})
    ax.set_xticks(x, [str(L) for L in LENGTHS])
    ax.set_xlabel("Context length L (windows)")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png", "svg"):
        p = out_dir / f"{FIG_STEM}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- main
def main() -> int:
    args = parse_args()
    started = now_utc()
    s28 = load_step28(args.step28_run)
    s27 = load_step27(args.final_zip, s28["audit"])
    windows = load_windows(args.window_meta, args.window_audit, s27["source"])
    hist = load_historical(args.project_root, s28["audit"], s27["source"])
    cohort = build_cohort_counts(windows, s28["cohort"])
    require(int(len(cohort["ends"])) == int(s28["audit"]["cohort_validation"]["test_endpoints"]),
            "Cohort size differs from Step 28 record")
    result = run_analysis(cohort, hist, s27, s28)

    run_dir = args.output_root / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    run_dir.mkdir(parents=True, exist_ok=False)
    sweep, per_seed, per_class = result["sweep"], result["per_seed"], result["per_class"]
    save_csv(run_dir / f"{FIG_STEM}_values.csv", sweep)
    save_csv(run_dir / "31_per_seed_metrics.csv", per_seed)
    class_mean = (per_class.groupby(["length", "class_id", "class_label"], sort=True)["f1"]
                  .agg(f1_mean="mean", f1_seed_sd_ddof1=lambda v: float(np.std(v, ddof=1))).reset_index())
    save_csv(run_dir / "31_per_class_f1.csv", class_mean)
    save_csv(run_dir / "31_per_class_f1_by_seed.csv", per_class)
    fig_paths = draw_figure(sweep, result["findings"], run_dir)

    sources = [f"# Step 31 inputs ({started})",
               f"step27_final_zip\t{s27['path']}\t{s27['sha256']}",
               f"step28_run\t{s28['run']}"]
    sources += [f"step28/{name}\t{sha}" for name, sha in s28["hashes"].items()]
    sources += [f"step27_member seed_{seed}/test_predictions.csv\t{sha}" for seed, sha in s27["member_sha256"].items()]
    sources += [f"window/{name}\t{sha}" for name, sha in windows["hashes"].items()]
    sources += [f"historical seed={seed} L={L}\t{path}\t{sha}" for (seed, L), (path, sha) in sorted(hist["hashes"].items())]
    (run_dir / "sources.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")

    gate_report = {
        "status": "GATES_PASSED",
        "G1_input_hashes": "matched Step 27 manifest and Step 28 input audit",
        "G2_cohort_coverage": {"test_endpoints": int(len(cohort["ends"])),
                               "test_episodes": int(len(np.unique(cohort["episode"]))),
                               "endpoint_sha256": digest_ends(cohort["ends"])},
        "G3_step28_metrics_reproduced": True,
        "G4_step28_change_counts_reproduced": True,
        "G5_step27_reference_reproduced": True,
        "findings_not_enforced": result["findings"],
        "reference_by_L": {str(L): v for L, v in result["reference"].items()},
    }
    save_json(run_dir / "31_gate_report.json", gate_report)
    manifest = {
        "schema": "taes_step31_same_population_lsweep_v1",
        "status": "COMPLETE", "started_utc": started, "finished_utc": now_utc(),
        "dataset_id": EXPECTED_DATASET_ID, "seeds": list(SEEDS), "lengths": list(LENGTHS),
        "population": "L20-eligible test endpoints (Step 27/28 cohort); all lengths scored on the same endpoints",
        "head_provenance": {"L1-L10": "Step 12 historical heads, each trained on its own L-eligible training cohort",
                            "L20": "Step 27 heads trained on the L20-eligible training cohort"},
        "statistical_scope": "3 head seeds, mean and ddof=1 SD; no bootstrap; exploratory extension designed after the L20 test was inspected",
        "reference_note": "Evidence-availability reference: ground-truth label where any paired-nominal change lies within the L-window context, else NORMAL; nondeployable, not a bound",
        "script_sha256": digest_file(Path(__file__).resolve()),
        "artifacts": {},
    }
    for p in sorted(run_dir.iterdir()):
        if p.is_file():
            manifest["artifacts"][p.name] = digest_file(p)
    save_json(run_dir / "31_run_manifest.json", manifest)

    args.graph_dir.mkdir(parents=True, exist_ok=True)
    for p in fig_paths + [run_dir / f"{FIG_STEM}_values.csv", run_dir / "sources.txt"]:
        shutil.copy2(p, args.graph_dir / p.name)

    review = run_dir / "31_L20_POPULATION_LSWEEP_REVIEW.zip"
    with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(run_dir.iterdir()):
            if p.is_file() and p != review:
                z.write(p, p.name)
        z.write(Path(__file__).resolve(), "source/" + Path(__file__).name)

    print(f"[Step 31] cohort {len(cohort['ends']):,} endpoints / {len(np.unique(cohort['episode']))} episodes; gates passed")
    for _, r in sweep.iterrows():
        print(f"  L={int(r.length):>2}  TST {r.tst_macro_f1_mean:.4f} +/- {r.tst_macro_f1_sd_ddof1:.4f}"
              f"  reference {r.reference_macro_f1:.4f}  gap {r.gap_tst_minus_reference:+.4f}"
              f"  coverage {r.reference_attack_coverage:.3f}")
    f = result["findings"]
    print(f"  L10->L20 gain: TST {f['gain_L10_to_L20_tst']:+.4f}  reference {f['gain_L10_to_L20_reference']:+.4f};"
          f" below reference at every L: {f['tst_below_reference_at_every_L']};"
          f" TST monotone in L: {f['tst_monotone_nondecreasing_in_L']}")
    print(f"  run folder: {run_dir}")
    print(f"  figure:     {args.graph_dir / (FIG_STEM + '.pdf')}")
    print(f"  review zip: {review}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as exc:
        print(f"[Step 31] STOPPED: {exc}", file=sys.stderr)
        sys.exit(2)
