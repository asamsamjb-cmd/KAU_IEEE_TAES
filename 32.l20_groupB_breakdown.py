# -*- coding: utf-8 -*-
"""Step 32 (experiment E) -- group-B breakdown by the number of changed windows 11-20 back.

No training. Takes the Step 28 group B (attack-labelled L20-eligible TEST endpoints whose
paired-nominal change lies only in windows 11-20 before the endpoint; 1,961 endpoints),
splits it by older10_changed_count (1 / 2 / 3 or more), and reports, per bin and per seed,
the correct-scenario recall of the historical L10 head and the Step 27 L20 head, the
L10->L20 prediction transitions, the fraction predicted NORMAL, and the per-class recall.
A full per-count table (1, 2, 3, ..., max) is written as well.

Inputs (read-only, hash-checked against Step 27 / Step 28 records):
  * Step 28 run folder: 28_endpoint_evidence.csv, 28_group_summary.csv, 28_input_audit.json, 28_run_manifest.json
  * Step 27 final test review ZIP: seed_{seed}/test_predictions.csv (L20), 27_run_manifest.json
  * Step 12 historical L10 predictions, seeds 20260610-20260612:
        outputs/shared_nominal_v2/tst_training_finetuned_main/L10/predictions/test_predictions_L10.csv
        outputs/shared_nominal_v2/tst_training_finetuned_multiseed/seed_{seed}/L10/predictions/test_predictions_L10.csv

Gates (hard stop on failure):
  G1 input hashes equal the values recorded by Step 27 (manifest) and Step 28 (input audit / artifacts)
  G2 L10 and L20 predictions cover every group-B endpoint with identical labels and episode IDs
  G3 summing the bins reproduces the Step 28 group-B row for every seed exactly
     (endpoints, recall, L10_wrong_L20_correct, L10_correct_L20_wrong, both_correct, both_wrong)

Run from the code directory (PowerShell, .venv active):
    py "32.l20_groupB_breakdown.py"
Optional:  --final-zip <zip>  --step28-run <folder>  --output-root <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
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
GROUP_B = "B_only_older10"
BINS = (("1", 1, 1), ("2", 2, 2), ("3+", 3, None))  # (label, min count, max count or None)

BASE = Path(__file__).resolve().parent
ROOT_REL = Path("outputs/shared_nominal_v2")
DEFAULT_27 = ROOT_REL / "27_L20_TST_VALIDATION_PILOT" / "27_L20_FINAL_TEST_REVIEW.zip"
DEFAULT_28 = ROOT_REL / "28_l20_evidence_location"
DEFAULT_OUTPUT = ROOT_REL / "32_l20_groupB_breakdown"
EXPECTED_DATASET_ID = "478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b"


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


def close(a: float, b: float, atol: float) -> bool:
    return np.isfinite(a) and np.isfinite(b) and abs(float(a) - float(b)) <= atol


def sd1(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")


# ----------------------------------------------------------------------------- arguments
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=BASE)
    p.add_argument("--final-zip", type=Path, default=None, help="Step 27 FINAL TEST review ZIP")
    p.add_argument("--step28-run", type=Path, default=None, help="Step 28 run folder (run_*)")
    p.add_argument("--output-root", type=Path, default=None)
    a = p.parse_args()
    a.project_root = a.project_root.expanduser().resolve()
    a.final_zip = (a.final_zip or (a.project_root / DEFAULT_27)).expanduser().resolve()
    a.output_root = (a.output_root or (a.project_root / DEFAULT_OUTPUT)).expanduser().resolve()
    if a.step28_run is None:
        complete = []
        for r in sorted((a.project_root / DEFAULT_28).glob("run_*")):
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
    require(a.final_zip.is_file(), f"Input not found: {a.final_zip}")
    require(a.step28_run.is_dir(), f"Step 28 run folder not found: {a.step28_run}")
    return a


# ----------------------------------------------------------------------------- inputs
def load_step28(run: Path) -> dict:
    names = ("28_run_manifest.json", "28_input_audit.json", "28_endpoint_evidence.csv", "28_group_summary.csv")
    files = {name: run / name for name in names}
    for name, path in files.items():
        require(path.is_file(), f"Step 28 file missing: {path}")
    manifest = json.loads(files["28_run_manifest.json"].read_text(encoding="utf-8"))
    require(manifest.get("status") == "COMPLETE", "Step 28 run is not COMPLETE")
    recorded = manifest.get("artifacts", {})
    hashes = {}
    for name in names[1:]:
        hashes[name] = digest_file(files[name])
        if name in recorded:
            require(hashes[name] == recorded[name], f"Step 28 artifact hash mismatch: {name}")
    audit = json.loads(files["28_input_audit.json"].read_text(encoding="utf-8"))
    cohort = load_csv(files["28_endpoint_evidence.csv"])
    for col in ("sequence_end_row", "episode_id", "episode_step", "true_label_id",
                "last10_changed_count", "last20_changed_count", "older10_changed_count", "evidence_group"):
        require(col in cohort, f"28_endpoint_evidence.csv lacks column {col}")
    group_summary = load_csv(files["28_group_summary.csv"])
    for col in ("group", "seed", "endpoints", "L10_correct_scenario_recall", "L20_correct_scenario_recall",
                "L10_wrong_L20_correct", "L10_correct_L20_wrong", "both_correct", "both_wrong"):
        require(col in group_summary, f"28_group_summary.csv lacks column {col}")
    return {"run": run, "manifest": manifest, "audit": audit, "cohort": cohort,
            "group_summary": group_summary, "hashes": hashes}


def load_step27(zip_path: Path, audit28: dict) -> dict:
    zip_sha = digest_file(zip_path)
    require(zip_sha == audit28.get("step27_zip_sha256"),
            "Step 27 final ZIP hash differs from the ZIP that Step 28 analysed")
    by_seed = {int(e["seed"]): e for e in audit28.get("prediction_audits", [])}
    with zipfile.ZipFile(zip_path, "r") as z:
        names = set(z.namelist())
        require("27_run_manifest.json" in names, "Step 27 ZIP missing 27_run_manifest.json")
        manifest = json.loads(z.read("27_run_manifest.json").decode("utf-8"))
        preds, member_sha = {}, {}
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
    return {"path": zip_path, "sha256": zip_sha, "source": source, "predictions": preds, "member_sha256": member_sha}


def historical_l10_path(root: Path, seed: int) -> Path:
    base = (root / ROOT_REL / "tst_training_finetuned_main" if seed == 20260610
            else root / ROOT_REL / "tst_training_finetuned_multiseed" / f"seed_{seed}")
    return base / "L10" / "predictions" / "test_predictions_L10.csv"


def load_historical_l10(root: Path, audit28: dict, source: dict) -> dict:
    by_seed = {int(e["seed"]): e for e in audit28.get("prediction_audits", [])}
    frames, hashes = {}, {}
    for seed in SEEDS:
        path = historical_l10_path(root, seed)
        require(path.is_file(), f"Historical L10 prediction file not found: {path}")
        sha = digest_file(path)
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
        frames[seed] = df
        hashes[seed] = (path, sha)
    return {"frames": frames, "hashes": hashes}


# ----------------------------------------------------------------------------- matching
def match_predictions(df: pd.DataFrame, ends: np.ndarray, y: np.ndarray, episode: np.ndarray, what: str) -> np.ndarray:
    rows = strict_int(df["sequence_end_row"], f"{what} sequence_end_row")
    require(len(np.unique(rows)) == len(rows), f"{what}: duplicate sequence_end_row")
    frame = pd.DataFrame({"end": rows, "y": strict_int(df["true_label_id"], f"{what} true_label_id"),
                          "p": strict_int(df["pred_label_id"], f"{what} pred_label_id"),
                          "ep": as_str(df["episode_id"])}).set_index("end")
    missing = ~pd.Index(ends).isin(frame.index)
    require(not missing.any(), f"G2 failed: {what} does not cover {int(missing.sum())} group-B endpoints")
    sel = frame.loc[ends]
    require(np.array_equal(sel["y"].to_numpy(), y), f"G2 failed: {what} labels differ from group B")
    require(np.array_equal(sel["ep"].to_numpy(), episode), f"G2 failed: {what} episode IDs differ from group B")
    p = sel["p"].to_numpy(dtype=np.int64)
    require(np.isin(p, np.arange(K)).all(), f"{what}: invalid class ID")
    return p


# ----------------------------------------------------------------------------- analysis
def bin_label(count: np.ndarray) -> np.ndarray:
    out = np.empty(len(count), dtype=object)
    for label, lo, hi in BINS:
        mask = (count >= lo) & ((count <= hi) if hi is not None else True)
        out[mask] = label
    require(all(v is not None for v in out), "older10_changed_count outside the bin definition")
    return out


def summarise(mask: np.ndarray, y: np.ndarray, p10: np.ndarray, p20: np.ndarray, episode: np.ndarray) -> dict:
    yy, a, b = y[mask], p10[mask], p20[mask]
    c10, c20 = a == yy, b == yy
    return {
        "endpoints": int(mask.sum()),
        "episodes": int(len(np.unique(episode[mask]))),
        "L10_correct_scenario_recall": float(c10.mean()),
        "L20_correct_scenario_recall": float(c20.mean()),
        "delta_recall": float(c20.mean() - c10.mean()),
        "L10_wrong_L20_correct": int(np.count_nonzero(~c10 & c20)),
        "L10_correct_L20_wrong": int(np.count_nonzero(c10 & ~c20)),
        "both_correct": int(np.count_nonzero(c10 & c20)),
        "both_wrong": int(np.count_nonzero(~c10 & ~c20)),
        "L10_pred_normal_fraction": float((a == 0).mean()),
        "L20_pred_normal_fraction": float((b == 0).mean()),
        "L20_wrong_pred_normal_fraction": float((b[~c20] == 0).mean()) if (~c20).any() else float("nan"),
    }


def run_analysis(s28: dict, s27: dict, hist: dict) -> dict:
    cohort = s28["cohort"]
    gb = cohort[cohort["evidence_group"].astype("string") == GROUP_B].sort_values("sequence_end_row").reset_index(drop=True)
    require(len(gb) > 0, "Group B is empty in 28_endpoint_evidence.csv")
    ends = strict_int(gb["sequence_end_row"], "sequence_end_row")
    y = strict_int(gb["true_label_id"], "true_label_id")
    episode = as_str(gb["episode_id"])
    older = strict_int(gb["older10_changed_count"], "older10_changed_count")
    last10 = strict_int(gb["last10_changed_count"], "last10_changed_count")
    require(np.all(y != 0) and np.all(last10 == 0) and np.all(older > 0),
            "Group B rows violate the definition (attack label, no change in last 10, change in windows 11-20)")
    bins = bin_label(older)

    per_seed_bin, per_seed_class, per_seed_count = [], [], []
    endpoint_table = gb[["sequence_end_row", "episode_id", "episode_step", "true_label_id",
                         "older10_changed_count"]].copy()
    endpoint_table["bin"] = bins
    group_rows = s28["group_summary"]
    group_rows = group_rows[group_rows["group"].astype("string") == GROUP_B]
    require(len(group_rows) == len(SEEDS), "28_group_summary.csv lacks a group-B row per seed")

    for seed in SEEDS:
        p10 = match_predictions(hist["frames"][seed], ends, y, episode, f"historical L10 seed {seed}")
        p20 = match_predictions(s27["predictions"][seed], ends, y, episode, f"Step 27 L20 seed {seed}")
        endpoint_table[f"L10_pred_{seed}"] = p10
        endpoint_table[f"L20_pred_{seed}"] = p20
        # G3: whole group B must reproduce the Step 28 row
        whole = summarise(np.ones(len(y), dtype=bool), y, p10, p20, episode)
        ref = group_rows[strict_int(group_rows["seed"], "seed") == seed]
        require(len(ref) == 1, f"28_group_summary.csv group-B row for seed {seed} not unique")
        ref = ref.iloc[0]
        for col, tol in (("endpoints", 0), ("L10_wrong_L20_correct", 0), ("L10_correct_L20_wrong", 0),
                         ("both_correct", 0), ("both_wrong", 0),
                         ("L10_correct_scenario_recall", 1e-12), ("L20_correct_scenario_recall", 1e-12)):
            require(close(float(whole[col]), float(ref[col]), tol),
                    f"G3 failed: seed {seed} group-B {col} {whole[col]} != Step 28 {ref[col]}")
        for label, _, _ in BINS:
            mask = bins == label
            row = {"bin_older10_changed_count": label, "seed": seed}
            row.update(summarise(mask, y, p10, p20, episode))
            per_seed_bin.append(row)
            for c in range(1, K):
                cm = mask & (y == c)
                if cm.any():
                    r = {"bin_older10_changed_count": label, "seed": seed, "class_id": c, "class_label": LABELS[c]}
                    r.update(summarise(cm, y, p10, p20, episode))
                    per_seed_class.append(r)
        for count in np.unique(older):
            mask = older == count
            row = {"older10_changed_count": int(count), "seed": seed}
            row.update(summarise(mask, y, p10, p20, episode))
            per_seed_count.append(row)

    bin_seed = pd.DataFrame(per_seed_bin)
    class_seed = pd.DataFrame(per_seed_class)
    count_seed = pd.DataFrame(per_seed_count)

    def seed_mean(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        rows = []
        for key_vals, sub in frame.groupby(keys, sort=False):
            key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
            row = dict(zip(keys, key_vals))
            row["n_seeds"] = int(len(sub))
            row["endpoints"] = int(sub["endpoints"].iloc[0])
            row["episodes"] = int(sub["episodes"].iloc[0])
            for col in ("L10_correct_scenario_recall", "L20_correct_scenario_recall", "delta_recall",
                        "L10_pred_normal_fraction", "L20_pred_normal_fraction", "L20_wrong_pred_normal_fraction"):
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_sd_ddof1"] = sd1(sub[col])
            for col in ("L10_wrong_L20_correct", "L10_correct_L20_wrong", "both_correct", "both_wrong"):
                row[f"{col}_seed_mean"] = float(sub[col].mean())
            rows.append(row)
        return pd.DataFrame(rows)

    bin_mean = seed_mean(bin_seed, ["bin_older10_changed_count"])
    class_mean = seed_mean(class_seed, ["bin_older10_changed_count", "class_id", "class_label"])
    count_mean = seed_mean(count_seed, ["older10_changed_count"])
    bin_order = {label: i for i, (label, _, _) in enumerate(BINS)}
    bin_mean = bin_mean.sort_values("bin_older10_changed_count", key=lambda s: s.map(bin_order)).reset_index(drop=True)
    class_mean = class_mean.sort_values(["bin_older10_changed_count", "class_id"],
                                        key=lambda s: s.map(bin_order) if s.name == "bin_older10_changed_count" else s
                                        ).reset_index(drop=True)
    count_mean = count_mean.sort_values("older10_changed_count").reset_index(drop=True)
    return {"bin_seed": bin_seed, "bin_mean": bin_mean, "class_seed": class_seed, "class_mean": class_mean,
            "count_seed": count_seed, "count_mean": count_mean, "endpoints": endpoint_table,
            "groupB_endpoints": int(len(y)), "groupB_episodes": int(len(np.unique(episode))),
            "bin_counts": {label: int((bins == label).sum()) for label, _, _ in BINS},
            "count_distribution": {int(c): int((older == c).sum()) for c in np.unique(older)}}


# ----------------------------------------------------------------------------- main
def main() -> int:
    args = parse_args()
    started = now_utc()
    s28 = load_step28(args.step28_run)
    s27 = load_step27(args.final_zip, s28["audit"])
    hist = load_historical_l10(args.project_root, s28["audit"], s27["source"])
    result = run_analysis(s28, s27, hist)

    run_dir = args.output_root / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    run_dir.mkdir(parents=True, exist_ok=False)
    save_csv(run_dir / "32_groupB_by_bin.csv", result["bin_mean"])
    save_csv(run_dir / "32_groupB_by_bin_by_seed.csv", result["bin_seed"])
    save_csv(run_dir / "32_groupB_by_bin_by_class.csv", result["class_mean"])
    save_csv(run_dir / "32_groupB_by_bin_by_class_by_seed.csv", result["class_seed"])
    save_csv(run_dir / "32_groupB_by_count_full.csv", result["count_mean"])
    save_csv(run_dir / "32_groupB_by_count_full_by_seed.csv", result["count_seed"])
    save_csv(run_dir / "32_groupB_endpoints.csv", result["endpoints"])

    sources = [f"# Step 32 inputs ({started})",
               f"step27_final_zip\t{s27['path']}\t{s27['sha256']}",
               f"step28_run\t{s28['run']}"]
    sources += [f"step28/{name}\t{sha}" for name, sha in s28["hashes"].items()]
    sources += [f"step27_member seed_{seed}/test_predictions.csv\t{sha}" for seed, sha in s27["member_sha256"].items()]
    sources += [f"historical L10 seed={seed}\t{path}\t{sha}" for seed, (path, sha) in sorted(hist["hashes"].items())]
    (run_dir / "sources.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")

    gate_report = {
        "status": "GATES_PASSED",
        "G1_input_hashes": "matched Step 27 manifest and Step 28 input audit / artifacts",
        "G2_groupB_coverage": {"endpoints": result["groupB_endpoints"], "episodes": result["groupB_episodes"]},
        "G3_step28_groupB_rows_reproduced": True,
        "bin_definition": {label: f"older10_changed_count in [{lo}, {hi if hi is not None else 'max'}]"
                           for label, lo, hi in BINS},
        "bin_counts": result["bin_counts"],
        "older10_changed_count_distribution": result["count_distribution"],
    }
    save_json(run_dir / "32_gate_report.json", gate_report)
    manifest = {
        "schema": "taes_step32_groupB_breakdown_v1",
        "status": "COMPLETE", "started_utc": started, "finished_utc": now_utc(),
        "dataset_id": EXPECTED_DATASET_ID, "seeds": list(SEEDS),
        "population": "Step 28 group B: attack-labelled L20-eligible test endpoints with paired-nominal change only in windows 11-20 before the endpoint",
        "models": {"L10": "Step 12 historical heads (3 of 10 seeds)", "L20": "Step 27 heads"},
        "statistical_scope": "descriptive; 3 head seeds, mean and ddof=1 SD; subgroup identified with the nondeployable paired-nominal audit; exploratory extension designed after the L20 test was inspected",
        "script_sha256": digest_file(Path(__file__).resolve()),
        "artifacts": {},
    }
    for p in sorted(run_dir.iterdir()):
        if p.is_file():
            manifest["artifacts"][p.name] = digest_file(p)
    save_json(run_dir / "32_run_manifest.json", manifest)

    review = run_dir / "32_L20_GROUPB_BREAKDOWN_REVIEW.zip"
    with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(run_dir.iterdir()):
            if p.is_file() and p != review:
                z.write(p, p.name)
        z.write(Path(__file__).resolve(), "source/" + Path(__file__).name)

    print(f"[Step 32] group B {result['groupB_endpoints']:,} endpoints / {result['groupB_episodes']} episodes; gates passed")
    for _, r in result["bin_mean"].iterrows():
        print(f"  changed windows 11-20 back = {r.bin_older10_changed_count:>2}: n={int(r.endpoints):>5} ({int(r.episodes)} ep)"
              f"  L10 recall {r.L10_correct_scenario_recall_mean:.4f}  L20 recall {r.L20_correct_scenario_recall_mean:.4f}"
              f"  L10 wrong->L20 right {r.L10_wrong_L20_correct_seed_mean:.1f}  L10 right->L20 wrong {r.L10_correct_L20_wrong_seed_mean:.1f}")
    print(f"  run folder: {run_dir}")
    print(f"  review zip: {review}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as exc:
        print(f"[Step 32] STOPPED: {exc}", file=sys.stderr)
        sys.exit(2)
