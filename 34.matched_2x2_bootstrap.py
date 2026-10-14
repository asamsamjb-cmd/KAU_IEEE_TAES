# -*- coding: utf-8 -*-
"""Step 34 -- official paired episode bootstrap for the matched 2x2 (TST / Mean+Max  x  L10 / L20).

No training, no new inference. Reads the twelve saved TEST prediction files of the matched cohort
(30,561 L20-eligible endpoints / 751 episodes) and computes, with ONE fixed set of resampling draws
shared by every model and seed:

  three pre-specified contrasts (one family):
    C1  TST L20 (Step 27)        - matched TST L10 (Step 29)
    C2  Mean+Max L20 (Step 30)   - Mean+Max L10 (Step 30)
    C3  TST L20 (Step 27)        - Mean+Max L20 (Step 30)
  plus the four cell values themselves.

Procedure (identical in form to Steps 15 / 28 / 46): six-class-stratified whole-episode multinomial
resampling (each class stratum resampled to its own episode count), 20,000 draws, fixed RNG seed,
per draw: macro-F1 per head seed -> mean over the three seeds -> contrast; percentile intervals at
95% and at the Bonferroni-adjusted 98.333% level for the three-contrast family. Seeds are fixed,
not resampled; identical seed numbers across models are NOT paired initialisations.

Inputs (read-only, located automatically under outputs/shared_nominal_v2/ unless passed explicitly):
  --zip27  27_L20_FINAL_TEST_REVIEW.zip          seed_<seed>/test_predictions.csv  (TST L20)
  --zip29  29_L10_MATCHED_FINAL_TEST_REVIEW.zip  seed_<seed>/test_predictions.csv  (matched TST L10)
  --zip30  30_MATCHED_MEANMAX_FINAL_TEST_REVIEW.zip  L10|L20/seed_<seed>/test_predictions.csv (Mean+Max)
  optional: Step 28 run folder (28_input_audit.json) to cross-check the Step 27 member hashes

Gates (hard stop on failure):
  G1 prediction member hashes equal the hashes recorded in 29_final_test.json / 30_final_test.json
     (and, when the Step 28 input audit is found, the Step 27 member hashes recorded there)
  G2 all twelve files cover the same 30,561 endpoints with identical labels and episode IDs;
     every episode carries a single class; 751 episodes
  G3 the four cell means reproduce 30_test_2x2.csv and 29_test_summary.json (abs 1e-9)

Run from the code directory (PowerShell, .venv active):
    py "34.matched_2x2_bootstrap.py"
Optional: --draws 20000 --rng-seed 20261013 --output-root <dir> --zip27/--zip29/--zip30 <path>
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
MODELS = (("TST", 10), ("TST", 20), ("MeanMax_MLP", 10), ("MeanMax_MLP", 20))
CONTRASTS = (
    ("C1_TST_L20_minus_matched_TST_L10", ("TST", 20), ("TST", 10)),
    ("C2_MeanMax_L20_minus_MeanMax_L10", ("MeanMax_MLP", 20), ("MeanMax_MLP", 10)),
    ("C3_TST_L20_minus_MeanMax_L20", ("TST", 20), ("MeanMax_MLP", 20)),
)
FAMILY_SIZE = len(CONTRASTS)
DEFAULT_DRAWS = 20000
DEFAULT_RNG_SEED = 20261013

BASE = Path(__file__).resolve().parent
ROOT_REL = Path("outputs/shared_nominal_v2")
ZIP_NAMES = {"27": "27_L20_FINAL_TEST_REVIEW.zip", "29": "29_L10_MATCHED_FINAL_TEST_REVIEW.zip",
             "30": "30_MATCHED_MEANMAX_FINAL_TEST_REVIEW.zip"}
DEFAULT_OUTPUT = ROOT_REL / "34_matched_2x2_bootstrap"
EXPECTED_DATASET_ID = "478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b"
EXPECTED_ENDPOINTS, EXPECTED_EPISODES = 30561, 751


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


def macro_f1_from_cm(cm: np.ndarray) -> np.ndarray:
    """cm: (..., K, K) confusion matrices (rows true, cols predicted) -> macro-F1 (...)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(np.float64)
    row = cm.sum(-1).astype(np.float64)
    col = cm.sum(-2).astype(np.float64)
    den = row + col
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return f1.mean(-1)


# ----------------------------------------------------------------------------- arguments
def find_unique(root: Path, name: str, what: str) -> Path:
    hits = sorted(p for p in (root / ROOT_REL).rglob(name) if p.is_file())
    require(len(hits) == 1, f"Expected exactly one {name} under {root / ROOT_REL}; found {len(hits)}. Pass --{what}.")
    return hits[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=BASE)
    p.add_argument("--zip27", type=Path, default=None)
    p.add_argument("--zip29", type=Path, default=None)
    p.add_argument("--zip30", type=Path, default=None)
    p.add_argument("--step28-run", type=Path, default=None, help="optional Step 28 run folder for the Step 27 member hash check")
    p.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    p.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    p.add_argument("--output-root", type=Path, default=None)
    a = p.parse_args()
    a.project_root = a.project_root.expanduser().resolve()
    a.zip27 = (a.zip27 or find_unique(a.project_root, ZIP_NAMES["27"], "zip27")).expanduser().resolve()
    a.zip29 = (a.zip29 or find_unique(a.project_root, ZIP_NAMES["29"], "zip29")).expanduser().resolve()
    a.zip30 = (a.zip30 or find_unique(a.project_root, ZIP_NAMES["30"], "zip30")).expanduser().resolve()
    a.output_root = (a.output_root or (a.project_root / DEFAULT_OUTPUT)).expanduser().resolve()
    if a.step28_run is None:
        audits = sorted(p for p in (a.project_root / ROOT_REL).rglob("28_input_audit.json") if p.is_file())
        a.step28_run = audits[0].parent if len(audits) == 1 else None
    for z in (a.zip27, a.zip29, a.zip30):
        require(z.is_file(), f"ZIP not found: {z}")
    require(a.draws >= 1000, "--draws must be at least 1000")
    return a


# ----------------------------------------------------------------------------- inputs
def read_zip(path: Path) -> tuple[dict, str]:
    with zipfile.ZipFile(path, "r") as z:
        members = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    return members, digest_file(path)


def load_predictions(members: dict, member: str, what: str) -> pd.DataFrame:
    require(member in members, f"{what}: member {member} missing")
    df = load_csv_bytes(members[member])
    for col in ("sequence_end_row", "episode_id", "true_label_id", "pred_label_id"):
        require(col in df, f"{what}: {member} lacks column {col}")
    return df


def load_inputs(args) -> dict:
    m27, sha27 = read_zip(args.zip27)
    m29, sha29 = read_zip(args.zip29)
    m30, sha30 = read_zip(args.zip30)
    hashes = {"zip27": sha27, "zip29": sha29, "zip30": sha30}
    recorded = {}
    # Step 29 / Step 30 record their own test prediction hashes
    for key, members, final_name in (("29", m29, "29_final_test.json"), ("30", m30, "30_final_test.json")):
        require(final_name in members, f"ZIP {key} lacks {final_name}")
        ft = json.loads(members[final_name].decode("utf-8"))
        require(ft.get("status") == "COMPLETE", f"Step {key} final test is not COMPLETE")
        recorded[key] = ft
    # optional Step 28 audit for the Step 27 members
    audit28 = None
    if args.step28_run is not None and (args.step28_run / "28_input_audit.json").is_file():
        audit28 = json.loads((args.step28_run / "28_input_audit.json").read_text(encoding="utf-8"))
        require(audit28.get("step27_zip_sha256") == sha27,
                "G1 failed: Step 27 final ZIP differs from the ZIP recorded in the Step 28 input audit")
    by_seed28 = {int(e["seed"]): e for e in (audit28 or {}).get("prediction_audits", [])}

    preds, member_hashes, identity = {}, {}, {}
    for seed in SEEDS:
        # TST L20 (Step 27)
        member = f"seed_{seed}/test_predictions.csv"
        df = load_predictions(m27, member, "Step 27")
        sha = digest_bytes(m27[member])
        if seed in by_seed28:
            require(sha == by_seed28[seed].get("L20_member_sha256"), f"G1 failed: Step 27 member hash differs from Step 28 record (seed {seed})")
        preds[("TST", 20, seed)] = df
        member_hashes[f"27:{member}"] = sha
        # matched TST L10 (Step 29)
        member = f"seed_{seed}/test_predictions.csv"
        df = load_predictions(m29, member, "Step 29")
        sha = digest_bytes(m29[member])
        require(sha == recorded["29"]["test_prediction_sha256"].get(str(seed)), f"G1 failed: Step 29 member hash differs from 29_final_test.json (seed {seed})")
        if "step29_identity_hash" in df:
            require(bool(df["step29_identity_hash"].astype("string").eq(recorded["29"]["identity_hash"]).all()), f"Step 29 identity hash mismatch (seed {seed})")
        preds[("TST", 10, seed)] = df
        member_hashes[f"29:{member}"] = sha
        # Mean+Max L10 / L20 (Step 30)
        for L in (10, 20):
            member = f"L{L}/seed_{seed}/test_predictions.csv"
            df = load_predictions(m30, member, "Step 30")
            sha = digest_bytes(m30[member])
            require(sha == recorded["30"]["test_prediction_sha256"].get(f"L{L}/seed_{seed}"), f"G1 failed: Step 30 member hash differs from 30_final_test.json (L{L}, seed {seed})")
            if "step30_identity_hash" in df:
                require(bool(df["step30_identity_hash"].astype("string").eq(recorded["30"]["identity_hash"]).all()), f"Step 30 identity hash mismatch (L{L}, seed {seed})")
            if "dataset_id" in df:
                require(bool(df["dataset_id"].astype("string").eq(EXPECTED_DATASET_ID).all()), f"Step 30 dataset_id mismatch (L{L}, seed {seed})")
            preds[("MeanMax_MLP", L, seed)] = df
            member_hashes[f"30:{member}"] = sha
    identity["29"] = recorded["29"]["identity_hash"]
    identity["30"] = recorded["30"]["identity_hash"]
    # recorded point estimates for G3
    require("30_test_2x2.csv" in m30, "ZIP 30 lacks 30_test_2x2.csv")
    table_2x2 = load_csv_bytes(m30["30_test_2x2.csv"])
    require("29_test_summary.json" in m29, "ZIP 29 lacks 29_test_summary.json")
    summary29 = json.loads(m29["29_test_summary.json"].decode("utf-8"))
    return {"preds": preds, "hashes": hashes, "member_hashes": member_hashes, "identity": identity,
            "table_2x2": table_2x2, "summary29": summary29, "audit28_used": audit28 is not None}


# ----------------------------------------------------------------------------- alignment
def align(preds: dict) -> dict:
    ref_key = ("TST", 20, SEEDS[0])
    ref = preds[ref_key].sort_values("sequence_end_row").reset_index(drop=True)
    ends = strict_int(ref["sequence_end_row"], "sequence_end_row")
    require(len(np.unique(ends)) == len(ends), "duplicate sequence_end_row in reference file")
    y = strict_int(ref["true_label_id"], "true_label_id")
    episode = as_str(ref["episode_id"])
    require(len(ends) == EXPECTED_ENDPOINTS, f"G2 failed: {len(ends)} endpoints, expected {EXPECTED_ENDPOINTS}")
    ep_codes, ep_names = pd.factorize(pd.Series(episode), sort=True)
    require(len(ep_names) == EXPECTED_EPISODES, f"G2 failed: {len(ep_names)} episodes, expected {EXPECTED_EPISODES}")
    ep_class = np.full(len(ep_names), -1, dtype=np.int64)
    for c, lab in zip(ep_codes, y):
        require(ep_class[c] in (-1, lab), "G2 failed: an episode carries more than one class label")
        ep_class[c] = lab
    require(np.isin(ep_class, np.arange(K)).all(), "G2 failed: episode class outside 0..5")
    pred = {}
    for key, df in preds.items():
        d = df.sort_values("sequence_end_row").reset_index(drop=True)
        require(np.array_equal(strict_int(d["sequence_end_row"], "sequence_end_row"), ends), f"G2 failed: {key} endpoint set differs")
        require(np.array_equal(strict_int(d["true_label_id"], "true_label_id"), y), f"G2 failed: {key} labels differ")
        require(np.array_equal(as_str(d["episode_id"]), episode), f"G2 failed: {key} episode IDs differ")
        p = strict_int(d["pred_label_id"], "pred_label_id")
        require(np.isin(p, np.arange(K)).all(), f"{key}: invalid class ID")
        pred[key] = p
    return {"ends": ends, "y": y, "episode": episode, "ep_codes": ep_codes, "ep_names": np.asarray(ep_names, dtype=object),
            "ep_class": ep_class, "pred": pred}


# ----------------------------------------------------------------------------- bootstrap
def per_episode_confusion(y: np.ndarray, p: np.ndarray, ep_codes: np.ndarray, n_ep: int) -> np.ndarray:
    cell = y * K + p
    cm = np.zeros((n_ep, K * K), dtype=np.int64)
    np.add.at(cm, (ep_codes, cell), 1)
    return cm


def run_bootstrap(al: dict, draws: int, rng_seed: int) -> dict:
    n_ep = len(al["ep_names"])
    cms = {key: per_episode_confusion(al["y"], p, al["ep_codes"], n_ep) for key, p in al["pred"].items()}
    # point estimates on the full cohort
    point = {key: float(macro_f1_from_cm(cm.sum(0).reshape(K, K))) for key, cm in cms.items()}
    # shared stratified draws: weights[draw, episode]
    rng = np.random.default_rng(rng_seed)
    weights = np.zeros((draws, n_ep), dtype=np.int32)
    strata = {}
    for c in range(K):
        idx = np.flatnonzero(al["ep_class"] == c)
        strata[c] = int(len(idx))
        if len(idx) == 0:
            continue
        counts = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=draws)
        weights[:, idx] = counts
    require(np.all(weights.sum(1) == n_ep), "bootstrap draw sizes differ from the episode count")
    # macro-F1 per draw for every model x seed
    boot = {}
    w = weights.astype(np.float64)
    for key, cm in cms.items():
        agg = w @ cm.astype(np.float64)          # (draws, K*K)
        boot[key] = macro_f1_from_cm(agg.reshape(draws, K, K))
    cell = {m: np.mean([boot[(m[0], m[1], s)] for s in SEEDS], axis=0) for m in MODELS}
    cell_point = {m: float(np.mean([point[(m[0], m[1], s)] for s in SEEDS])) for m in MODELS}
    cell_sd = {m: float(np.std([point[(m[0], m[1], s)] for s in SEEDS], ddof=1)) for m in MODELS}
    alpha95 = (2.5, 97.5)
    alphaB = (100 * 0.05 / (2 * FAMILY_SIZE), 100 * (1 - 0.05 / (2 * FAMILY_SIZE)))
    rows_c = []
    for name, a, b in CONTRASTS:
        d = cell[a] - cell[b]
        rows_c.append({
            "contrast": name, "minuend": f"{a[0]}_L{a[1]}", "subtrahend": f"{b[0]}_L{b[1]}",
            "point_estimate": cell_point[a] - cell_point[b],
            "bootstrap_mean": float(d.mean()),
            "ci95_low": float(np.percentile(d, alpha95[0])), "ci95_high": float(np.percentile(d, alpha95[1])),
            "bonferroni_level_pct": round(100 - 100 * 0.05 / FAMILY_SIZE, 3),
            "bonf_low": float(np.percentile(d, alphaB[0])), "bonf_high": float(np.percentile(d, alphaB[1])),
            "share_of_draws_le_zero": float((d <= 0).mean()),
        })
    rows_m = []
    for m in MODELS:
        v = cell[m]
        rows_m.append({"model": m[0], "length": m[1], "n_seeds": len(SEEDS),
                       "macro_f1_mean": cell_point[m], "macro_f1_seed_sd_ddof1": cell_sd[m],
                       "bootstrap_mean": float(v.mean()),
                       "ci95_low": float(np.percentile(v, alpha95[0])), "ci95_high": float(np.percentile(v, alpha95[1]))})
    rows_s = [{"model": k[0], "length": k[1], "seed": k[2], "macro_f1": v} for k, v in sorted(point.items())]
    return {"contrasts": pd.DataFrame(rows_c), "cells": pd.DataFrame(rows_m), "per_seed": pd.DataFrame(rows_s),
            "boot": boot, "strata": strata, "cell_point": cell_point, "point": point, "weights_digest": digest_bytes(weights.tobytes())}


# ----------------------------------------------------------------------------- main
def main() -> int:
    args = parse_args()
    started = now_utc()
    inp = load_inputs(args)
    al = align(inp["preds"])
    res = run_bootstrap(al, args.draws, args.rng_seed)
    # G3: reproduce recorded point estimates
    t = inp["table_2x2"]
    for m in MODELS:
        row = t[(t["model"].astype("string") == m[0]) & (strict_int(t["length"], "length") == m[1])]
        require(len(row) == 1, f"30_test_2x2.csv lacks {m}")
        rec = float(row["macro_f1_mean"].iloc[0])
        require(close(res["cell_point"][m], rec, 1e-9), f"G3 failed: {m} mean {res['cell_point'][m]} != recorded {rec}")
    require(close(res["cell_point"][("TST", 10)], float(inp["summary29"]["matched_L10_macro_f1_mean"]), 1e-9)
            and close(res["cell_point"][("TST", 20)], float(inp["summary29"]["original_L20_macro_f1_mean"]), 1e-9),
            "G3 failed: TST cells do not reproduce 29_test_summary.json")

    run_dir = args.output_root / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    run_dir.mkdir(parents=True, exist_ok=False)
    save_csv(run_dir / "34_contrasts.csv", res["contrasts"])
    save_csv(run_dir / "34_cells.csv", res["cells"])
    save_csv(run_dir / "34_per_seed_macro_f1.csv", res["per_seed"])
    np.savez_compressed(run_dir / "34_bootstrap_distribution.npz",
                        draws=np.stack([np.stack([res["boot"][(m[0], m[1], s)] for s in SEEDS], axis=1) for m in MODELS], axis=1),
                        models=np.array([f"{m[0]}_L{m[1]}" for m in MODELS]), seeds=np.array(SEEDS),
                        episode_class=al["ep_class"], episode_names=al["ep_names"].astype(str))
    sources = [f"# Step 34 inputs ({started})",
               f"zip27\t{args.zip27}\t{inp['hashes']['zip27']}", f"zip29\t{args.zip29}\t{inp['hashes']['zip29']}",
               f"zip30\t{args.zip30}\t{inp['hashes']['zip30']}",
               f"step28_audit_used_for_27_members\t{inp['audit28_used']}\t{args.step28_run}"]
    sources += [f"member {k}\t{v}" for k, v in sorted(inp["member_hashes"].items())]
    (run_dir / "sources.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")
    gate_report = {
        "status": "GATES_PASSED",
        "G1_member_hashes": "29/30 members matched their final_test records" + ("; 27 members matched the Step 28 input audit" if inp["audit28_used"] else "; 27 members recorded (no Step 28 audit found)"),
        "G2_alignment": {"endpoints": int(len(al["ends"])), "episodes": int(len(al["ep_names"])),
                         "endpoint_sha256": digest_ends(al["ends"]), "episodes_per_class": {LABELS[c]: n for c, n in res["strata"].items()}},
        "G3_point_estimates_reproduced": {f"{m[0]}_L{m[1]}": res["cell_point"][m] for m in MODELS},
        "bootstrap": {"draws": args.draws, "rng": f"numpy default_rng({args.rng_seed})", "stratification": "six-class whole-episode multinomial, shared draws",
                      "seed_handling": "fixed head seeds; per-draw macro-F1 per seed averaged over 3 seeds; seeds not resampled",
                      "family": [c[0] for c in CONTRASTS], "bonferroni_marginal_level_pct": round(100 - 100 * 0.05 / FAMILY_SIZE, 3),
                      "weights_sha256": res["weights_digest"]},
        "identity": inp["identity"],
    }
    save_json(run_dir / "34_gate_report.json", gate_report)
    manifest = {"schema": "taes_step34_matched_2x2_bootstrap_v1", "status": "COMPLETE", "started_utc": started, "finished_utc": now_utc(),
                "dataset_id": EXPECTED_DATASET_ID, "seeds": list(SEEDS), "models": [f"{m[0]}_L{m[1]}" for m in MODELS],
                "population": "matched L20-eligible test cohort, 30,561 endpoints / 751 episodes (Steps 27/29/30)",
                "statistical_scope": "conditional on fixed encoder, benchmark and head seeds; post-test exploratory extension; CI crossing zero is not evidence of equivalence",
                "script_sha256": digest_file(Path(__file__).resolve()), "artifacts": {}}
    for p in sorted(run_dir.iterdir()):
        if p.is_file():
            manifest["artifacts"][p.name] = digest_file(p)
    save_json(run_dir / "34_run_manifest.json", manifest)
    review = run_dir / "34_MATCHED_2X2_BOOTSTRAP_REVIEW.zip"
    with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(run_dir.iterdir()):
            if p.is_file() and p != review:
                z.write(p, p.name)
        z.write(Path(__file__).resolve(), "source/" + Path(__file__).name)

    print(f"[Step 34] gates passed; {len(al['ends']):,} endpoints / {len(al['ep_names'])} episodes; {args.draws} draws, rng {args.rng_seed}")
    for _, r in res["cells"].iterrows():
        print(f"  {r.model:>11} L{int(r.length):<2} macro-F1 {r.macro_f1_mean:.4f} (seed SD {r.macro_f1_seed_sd_ddof1:.4f})  95% CI [{r.ci95_low:.4f}, {r.ci95_high:.4f}]")
    for _, r in res["contrasts"].iterrows():
        print(f"  {r.contrast}: {r.point_estimate:+.4f}  95% [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]  "
              f"{r.bonferroni_level_pct}% [{r.bonf_low:+.4f}, {r.bonf_high:+.4f}]  P(draw<=0) {r.share_of_draws_le_zero:.4f}")
    print(f"  run folder: {run_dir}")
    print(f"  review zip: {review}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as exc:
        print(f"[Step 34] STOPPED: {exc}", file=sys.stderr)
        sys.exit(2)
