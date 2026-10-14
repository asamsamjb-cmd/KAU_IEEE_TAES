# -*- coding: utf-8 -*-
"""Step 33 (experiment F) -- Step 72 evidence-sparsity levels x context length L = 20.

No training. For each Step 72 realization (R100 / R060 / R035 / R020 = attack event
retention 1.0 / 0.6 / 0.35 / 0.2) this script
  1. builds the L20-eligible endpoint cohort (episode-bounded, stride 1) from the Step 72
     evaluation records -- a subset of the Step 72 L10 endpoints;
  2. runs the three frozen Step 27 L20 TST heads (seeds 20260610-20260612) on the cached
     Step 72 window embeddings (inference only);
  3. re-scores the frozen Step 12 TST heads at L = 1, 2, 3, 5, 10 (same three seeds) on
     the same cohort from the Step 72 saved predictions;
  4. computes the evidence-availability reference at L = 1, 2, 3, 5, 10, 20 on the same
     cohort from the Step 72 paired-nominal window flags;
  5. splits the attack endpoints into A (change within the last 10 windows), B (change only
     in windows 11-20 back) and C (no change in 20 windows), as in Step 28, per level.

Inputs (read-only, hashed into sources.txt and gated):
  * Step 72 run folder (default outputs/shared_nominal_v2/72_evidence_sparsity/run_evidence_sparsity_v1; every file
    below, the four embeddings.npy caches included, is SHA-256 hashed into sources.txt and the gate report):
      generation/<level>/evaluation_records.csv, evaluation_window_observable_effects.csv,
      evaluation_endpoints_L10.csv; cache/<level>/embeddings.npy;
      evaluation/<level>/combined_predictions.npz, lsweep_predictions.npz;
      sweep_statistics/family_summary.csv
  * Step 27 pilot folder (default outputs/shared_nominal_v2/27_L20_TST_VALIDATION_PILOT):
      seed_<seed>/best_L20.pt (+ 27_seed_manifest.json), sequence_cache/test_tst_sequences_L20.npz,
      27_L20_FINAL_TEST_REVIEW.zip (saved L20 test predictions for the inference canary)
  * Step 9 test window embeddings (default outputs/shared_nominal_v2/tst_embeddings_finetuned_main/test_window_embeddings.npz)
  * Step 11 source (11.train_tst_classifier.py beside this script) -- model class and evaluation helpers

Gates (hard stop on failure):
  G1 Step 27 checkpoint / cache / ZIP hashes match the Step 27 manifests; Step 9 embeddings hash matches the Step 27 record
  G2 Step 72 records, window flags and L10 endpoints are mutually aligned; every L10 endpoint is a contiguous
     10-window block inside one episode; every L20 endpoint is a contiguous 20-window block
  G3 reproduction of Step 72: TST_L1..L10 ten-seed means and Reference_L1..L10 on the full L10 population
     reproduce sweep_statistics/family_summary.csv (abs 1e-6)
  G4 inference canary: the Step 27 L20 heads evaluated here on the real test split reproduce the saved
     Step 27 test decisions exactly (per seed)
Findings are recorded, never enforced (TST below reference at each level/L, monotonicity, gains).

Run from the code directory (PowerShell, .venv active, same GPU environment as Steps 27/72):
    py "33.evidence_sparsity_L20.py"
Optional: --step72-run <dir> --step27-root <dir> --final-zip <zip> --test-embeddings <npz>
          --step11 <py> --output-root <dir> --canary-max-mismatch N (default 0)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
SUB_LENGTHS = (1, 2, 3, 5)
LENGTHS = (1, 2, 3, 5, 10, 20)
L10, L20 = 10, 20
META = ("scenario_id", "episode_id", "episode_type", "episode_step", "window_id", "timestamp", "label")

BASE = Path(__file__).resolve().parent
ROOT_REL = Path("outputs/shared_nominal_v2")
DEFAULT_72 = ROOT_REL / "72_evidence_sparsity" / "run_evidence_sparsity_v1"
DEFAULT_27 = ROOT_REL / "27_L20_TST_VALIDATION_PILOT"
DEFAULT_TEST_EMB = ROOT_REL / "tst_embeddings_finetuned_main" / "test_window_embeddings.npz"
DEFAULT_OUTPUT = ROOT_REL / "33_evidence_sparsity_L20"
STEP11_PATTERNS = ("11.train_tst_classifier.py", "11. train_tst_classifier.py", "11_train_tst_classifier.py")
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


def load_csv(path: Path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, usecols=usecols)


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
    attack = slice(1, K)
    return {
        "macro_f1": float(class_f1.mean()),
        "accuracy": float(tp.sum() / row.sum()),
        "attack_macro_recall": float(np.divide(tp[attack], row[attack], out=np.zeros(K - 1), where=row[attack] > 0).mean()),
        "normal_false_alarm_fraction": float((row[0] - tp[0]) / row[0]) if row[0] else float("nan"),
        "class_f1": [float(v) for v in class_f1],
    }


def close(a: float, b: float, atol: float) -> bool:
    return np.isfinite(a) and np.isfinite(b) and abs(float(a) - float(b)) <= atol


def sd1(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- arguments
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=BASE)
    p.add_argument("--step72-run", type=Path, default=None)
    p.add_argument("--step27-root", type=Path, default=None)
    p.add_argument("--final-zip", type=Path, default=None, help="Step 27 FINAL TEST review ZIP (canary targets)")
    p.add_argument("--test-embeddings", type=Path, default=None, help="Step 9 test_window_embeddings.npz (canary)")
    p.add_argument("--step11", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--canary-max-mismatch", type=int, default=0,
                   help="Allowed decision mismatches per seed in the inference canary (default 0 = exact)")
    a = p.parse_args()
    a.project_root = a.project_root.expanduser().resolve()
    a.step72_run = (a.step72_run or (a.project_root / DEFAULT_72)).expanduser().resolve()
    a.step27_root = (a.step27_root or (a.project_root / DEFAULT_27)).expanduser().resolve()
    a.final_zip = (a.final_zip or (a.step27_root / "27_L20_FINAL_TEST_REVIEW.zip")).expanduser().resolve()
    a.test_embeddings = (a.test_embeddings or (a.project_root / DEFAULT_TEST_EMB)).expanduser().resolve()
    a.output_root = (a.output_root or (a.project_root / DEFAULT_OUTPUT)).expanduser().resolve()
    if a.step11 is None:
        candidates = [a.project_root / s for s in STEP11_PATTERNS if (a.project_root / s).is_file()]
        require(len({digest_file(c) for c in candidates}) == 1,
                f"Expected one Step 11 source beside this script ({STEP11_PATTERNS}); found {len(candidates)}. Pass --step11.")
        a.step11 = candidates[0]
    a.step11 = a.step11.expanduser().resolve()
    for pth in (a.final_zip, a.test_embeddings, a.step11):
        require(pth.is_file(), f"Input not found: {pth}")
    require(a.step72_run.is_dir(), f"Step 72 run folder not found: {a.step72_run}")
    require(a.step27_root.is_dir(), f"Step 27 folder not found: {a.step27_root}")
    return a


# ----------------------------------------------------------------------------- Step 27 inputs
def load_step27(root: Path, final_zip: Path) -> dict:
    manifest = json.loads((root / "27_run_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("status") == "COMPLETE", "Step 27 pilot manifest is not COMPLETE")
    source = manifest["identity"]["source_contract"]
    require(source["dataset_id"] == EXPECTED_DATASET_ID, "Step 27 dataset_id is not shared_nominal_v2")
    checkpoints, hashes = {}, {}
    for seed in SEEDS:
        seed_manifest = json.loads((root / f"seed_{seed}" / "27_seed_manifest.json").read_text(encoding="utf-8"))
        require(seed_manifest.get("status") == "COMPLETE" and int(seed_manifest.get("seed")) == seed,
                f"Step 27 seed {seed} manifest is not COMPLETE")
        ckpt = root / f"seed_{seed}" / "best_L20.pt"
        require(ckpt.is_file(), f"Step 27 checkpoint missing: {ckpt}")
        sha = digest_file(ckpt)
        require(sha == seed_manifest["artifacts"]["best_L20.pt"], f"G1 failed: Step 27 checkpoint hash differs (seed {seed})")
        checkpoints[seed] = ckpt
        hashes[f"seed_{seed}/best_L20.pt"] = sha
    cache_manifest = json.loads((root / "sequence_cache" / "27_L20_sequence_manifest.json").read_text(encoding="utf-8"))
    cache_npz = root / "sequence_cache" / "test_tst_sequences_L20.npz"
    sha = digest_file(cache_npz)
    require(cache_manifest.get("status") == "COMPLETE" and sha == cache_manifest["artifacts"]["test_tst_sequences_L20.npz"],
            "G1 failed: Step 27 test L20 sequence cache hash differs from its manifest")
    hashes["sequence_cache/test_tst_sequences_L20.npz"] = sha
    zip_sha = digest_file(final_zip)
    hashes["27_L20_FINAL_TEST_REVIEW.zip"] = zip_sha
    preds = {}
    with zipfile.ZipFile(final_zip, "r") as z:
        names = set(z.namelist())
        require("27_run_manifest.json" in names, "Step 27 final ZIP lacks 27_run_manifest.json")
        zmanifest = json.loads(z.read("27_run_manifest.json").decode("utf-8"))
        require(zmanifest.get("identity_hash") == manifest.get("identity_hash"),
                "G1 failed: Step 27 final ZIP belongs to a different pilot identity")
        for seed in SEEDS:
            member = f"seed_{seed}/test_predictions.csv"
            require(member in names, f"Step 27 final ZIP lacks {member}")
            data = z.read(member)
            hashes[f"zip:{member}"] = digest_bytes(data)
            preds[seed] = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", keep_default_na=False)
    return {"root": root, "manifest": manifest, "source": source, "checkpoints": checkpoints,
            "cache_npz": cache_npz, "test_predictions": preds, "hashes": hashes}


def load_test_embeddings(path: Path, s27: dict) -> np.ndarray:
    sha = digest_file(path)
    require(sha == s27["source"]["step9_artifacts"]["test_window_embeddings.npz"],
            "G1 failed: Step 9 test_window_embeddings.npz hash differs from the Step 27 record")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    candidates = [v for v in arrays.values() if v.ndim == 2 and v.shape[1] == 768 and np.issubdtype(v.dtype, np.floating)]
    require(len(candidates) == 1, f"Step 9 test embeddings: expected one (N, 768) float array, found {len(candidates)} in {list(arrays)}")
    return np.ascontiguousarray(candidates[0], dtype=np.float32), sha


# ----------------------------------------------------------------------------- Step 72 inputs
def load_level(run: Path, level: str) -> dict:
    gen, ev = run / "generation" / level, run / "evaluation" / level
    files = {
        "records": gen / "evaluation_records.csv",
        "effects": gen / "evaluation_window_observable_effects.csv",
        "endpoints": gen / "evaluation_endpoints_L10.csv",
        "embeddings": run / "cache" / level / "embeddings.npy",
        "combined": ev / "combined_predictions.npz",
        "lsweep": ev / "lsweep_predictions.npz",
    }
    for name, path in files.items():
        require(path.is_file(), f"Step 72 {level}: missing {path}")
    records = load_csv(files["records"], usecols=list(META))
    effects = load_csv(files["effects"])
    require(len(records) == len(effects), f"{level}: records vs window effects row count mismatch")
    for col in META:
        require(np.array_equal(as_str(records[col]), as_str(effects[col])), f"G2 failed: {level}: records vs effects differ in {col}")
    episode = as_str(records["episode_id"])
    step = strict_int(records["episode_step"], "episode_step")
    label_str = as_str(records["label"])
    label_index = {name: i for i, name in enumerate(LABELS)}
    require(all(v in label_index for v in np.unique(label_str)), f"{level}: unknown label in records")
    window_label = np.array([label_index[v] for v in label_str], dtype=np.int64)
    flags = strict_int(effects["audit_observable_effect_present"], "audit_observable_effect_present")
    require(np.isin(flags, [0, 1]).all(), f"{level}: audit flag must be 0/1")
    require(not np.any(flags[window_label == 0]), f"{level}: NORMAL windows carry paired-nominal changes")

    endpoints = load_csv(files["endpoints"])
    for col in ("sequence_id", "episode_id", "sequence_start_row", "sequence_end_row", "sequence_end_step", "label", "label_id", "sequence_length"):
        require(col in endpoints, f"{level}: evaluation_endpoints_L10.csv lacks {col}")
    endpoints = endpoints.sort_values("sequence_id").reset_index(drop=True)
    sid = strict_int(endpoints["sequence_id"], "sequence_id")
    require(np.array_equal(sid, np.arange(len(sid))), f"{level}: sequence_id is not 0..N-1")
    end = strict_int(endpoints["sequence_end_row"], "sequence_end_row")
    start = strict_int(endpoints["sequence_start_row"], "sequence_start_row")
    y = strict_int(endpoints["label_id"], "label_id")
    require(np.all(strict_int(endpoints["sequence_length"], "sequence_length") == L10), f"{level}: endpoints are not L10 endpoints")
    require(np.array_equal(start, end - (L10 - 1)), f"G2 failed: {level}: L10 endpoints are not contiguous 10-window blocks")
    require(np.array_equal(y, window_label[end]) and np.array_equal(as_str(endpoints["episode_id"]), episode[end]),
            f"G2 failed: {level}: endpoint labels/episodes differ from records at the end row")
    idx10 = end[:, None] - np.arange(L10 - 1, -1, -1)[None, :]
    require(idx10.min() >= 0 and np.all(episode[idx10] == episode[end][:, None])
            and np.all(step[idx10] == step[end][:, None] - np.arange(L10 - 1, -1, -1)[None, :]),
            f"G2 failed: {level}: an L10 block crosses an episode boundary or skips a step")

    with np.load(files["combined"], allow_pickle=False) as z:
        require(np.array_equal(np.asarray(z["y"], dtype=np.int64), y) and np.array_equal(np.asarray(z["sequence_id"], dtype=np.int64), sid),
                f"{level}: combined_predictions.npz does not align with evaluation_endpoints_L10.csv")
        comb_fam = np.asarray(z["families"]).astype(str)
        comb_seed = np.asarray(z["seeds"]).astype(np.int64)
        comb_pred = np.asarray(z["predictions"], dtype=np.int64)
    with np.load(files["lsweep"], allow_pickle=False) as z:
        require(np.array_equal(np.asarray(z["y"], dtype=np.int64), y) and np.array_equal(np.asarray(z["sequence_id"], dtype=np.int64), sid),
                f"{level}: lsweep_predictions.npz does not align with evaluation_endpoints_L10.csv")
        ls_fam = np.asarray(z["families"]).astype(str)
        ls_seed = np.asarray(z["seeds"]).astype(np.int64)
        ls_pred = np.asarray(z["predictions"], dtype=np.int64)
    preds = {}
    for fam, seed, p in zip(comb_fam, comb_seed, comb_pred):
        if fam == "TST":
            preds[("TST_L10", int(seed))] = p
    for fam, seed, p in zip(ls_fam, ls_seed, ls_pred):
        preds[(str(fam), int(seed))] = p
    for L in LENGTHS[:-1]:
        seeds_here = sorted(s for (f, s) in preds if f == f"TST_L{L}")
        require(len(seeds_here) >= len(SEEDS) and all(s in seeds_here for s in SEEDS),
                f"{level}: TST_L{L} predictions for seeds {SEEDS} not all present")
    return {"level": level, "files": files, "episode": episode, "step": step, "window_label": window_label,
            "flags": flags, "end": end, "y": y, "idx10": idx10, "preds": preds,
            "hashes": {name: digest_file(path) for name, path in files.items()}}


def level_cohort_l20(lv: dict) -> dict:
    episode, step, end = lv["episode"], lv["step"], lv["end"]
    offsets = np.arange(L20 - 1, -1, -1)[None, :]
    candidate = end[:, None] - offsets
    ok = candidate.min(1) >= 0
    sel = np.flatnonzero(ok)
    blocks = candidate[sel]
    same_ep = np.all(episode[blocks] == episode[end[sel]][:, None], axis=1)
    consecutive = np.all(step[blocks] == step[end[sel]][:, None] - offsets, axis=1)
    keep = sel[same_ep & consecutive]
    require(len(keep) > 0, f"{lv['level']}: no L20-eligible endpoint")
    idx20 = candidate[keep]
    require(np.all(step[end[keep]] >= L20 - 1), f"{lv['level']}: L20 cohort contains an endpoint with fewer than 19 predecessors")
    return {"positions": keep, "idx20": idx20, "end": end[keep], "y": lv["y"][keep], "episode": episode[end[keep]]}


# ----------------------------------------------------------------------------- reproduction gate (G3)
def reproduce_family_summary(run: Path, levels: dict) -> pd.DataFrame:
    fs = load_csv(run / "sweep_statistics" / "family_summary.csv")
    rows = []
    for level, lv in levels.items():
        sub = fs[fs["level"].astype("string") == level]
        require(len(sub) > 0, f"family_summary.csv lacks level {level}")
        y, flags, idx10 = lv["y"], lv["flags"], lv["idx10"]
        for L in LENGTHS[:-1]:
            fam = f"TST_L{L}"
            seeds_all = sorted(s for (f, s) in lv["preds"] if f == fam)
            values = [metric(confusion(y, lv["preds"][(fam, s)]))["macro_f1"] for s in seeds_all]
            mine = float(np.mean(values))
            recorded = sub[sub["family"].astype("string") == fam]
            require(len(recorded) == 1, f"family_summary.csv lacks {level}/{fam}")
            rec = float(recorded["macro_f1"].iloc[0])
            require(close(mine, rec, 1e-6), f"G3 failed: {level} {fam} mean macro-F1 {mine} != Step 72 {rec}")
            rows.append({"level": level, "family": fam, "runs": len(seeds_all), "recomputed_macro_f1": mine, "step72_macro_f1": rec})
            ref_fam = f"Reference_L{L}"
            ref = metric(confusion(y, np.where(flags[idx10[:, -L:]].any(1) & (y != 0), y, 0)))["macro_f1"]
            recorded = sub[sub["family"].astype("string") == ref_fam]
            require(len(recorded) == 1, f"family_summary.csv lacks {level}/{ref_fam}")
            rec = float(recorded["macro_f1"].iloc[0])
            require(close(ref, rec, 1e-6), f"G3 failed: {level} {ref_fam} macro-F1 {ref} != Step 72 {rec}")
            rows.append({"level": level, "family": ref_fam, "runs": 1, "recomputed_macro_f1": ref, "step72_macro_f1": rec})
    retention = {str(r["level"]): float(r["retention"]) for _, r in fs[["level", "retention"]].drop_duplicates().iterrows()}
    return pd.DataFrame(rows), retention


# ----------------------------------------------------------------------------- inference
class L20Scorer:
    """Runs the frozen Step 27 L20 heads through the Step 11 helpers (same route as Step 27 itself)."""

    def __init__(self, s11, checkpoints: dict, cache_npz: Path):
        import torch
        self.s11, self.torch = s11, torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with np.load(cache_npz, allow_pickle=False) as z:
            self.template = {k: z[k] for k in z.files}
        for key in ("y", "label_ids", "sequence_indices", "start_row_indices", "end_row_indices"):
            require(key in self.template, f"Step 27 sequence cache lacks array {key}")
        self.models = {}
        for seed, path in checkpoints.items():
            ckpt = s11.load_checkpoint_safely(path, self.device)
            require(int(ckpt.get("seed")) == seed, f"Step 27 checkpoint seed mismatch: {path}")
            cfg = s11.ModelConfig(**ckpt["model_config"])
            require(cfg.sequence_length == L20 and cfg.num_classes == K, "Step 27 checkpoint is not an L20 six-class head")
            model = s11.TimeSeriesTransformerClassifier(cfg).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            for prm in model.parameters():
                prm.requires_grad_(False)
            self.models[seed] = model
        self.route = None

    def _seq_dict(self, idx: np.ndarray, y: np.ndarray, window_labels: np.ndarray) -> dict:
        t = self.template
        per_sequence = len(t["label_ids"]) == len(t["y"])
        return {
            "y": y.astype(t["y"].dtype),
            "label_ids": (y if per_sequence else window_labels).astype(t["label_ids"].dtype),
            "sequence_indices": idx.astype(t["sequence_indices"].dtype),
            "start_row_indices": idx[:, 0].astype(t["start_row_indices"].dtype),
            "end_row_indices": idx[:, -1].astype(t["end_row_indices"].dtype),
        }

    def _direct(self, model, embeddings: np.ndarray, idx: np.ndarray) -> np.ndarray:
        torch = self.torch
        out = np.empty((len(idx), K), dtype=np.float32)
        batch = int(getattr(self.s11, "BATCH_SIZE", 256))
        with torch.no_grad():
            for lo in range(0, len(idx), batch):
                x = torch.from_numpy(np.ascontiguousarray(embeddings[idx[lo:lo + batch]], dtype=np.float32)).to(self.device)
                logits = model(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                elif isinstance(logits, dict):
                    logits = logits.get("logits", next(iter(logits.values())))
                out[lo:lo + batch] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return out

    def predict(self, seed: int, embeddings: np.ndarray, idx: np.ndarray, y: np.ndarray, window_labels: np.ndarray) -> np.ndarray:
        model = self.models[seed]
        s11 = self.s11
        if self.route != "direct":
            try:
                seq = self._seq_dict(idx, y, window_labels)
                loader = s11.make_loader(embeddings, seq, s11.BATCH_SIZE, False, self.device)
                ev = s11.evaluate_model(model, loader, s11.make_cross_entropy_loss(None), self.device, K,
                                        {i: str(i) for i in range(K)}, desc="step33")
                order = np.asarray(ev["sequence_ids"], dtype=np.int64)
                pred = np.asarray(ev["y_pred"], dtype=np.int64)
                require(np.array_equal(order, np.arange(len(idx))) and np.array_equal(np.asarray(ev["y_true"], dtype=np.int64), y),
                        "Step 11 evaluation order/labels differ from the submitted sequences")
                self.route = "step11_evaluate_model"
                return pred
            except GateError:
                raise
            except Exception as exc:  # fall back once to a direct forward pass; the canary decides whether it is valid
                if self.route is None:
                    print(f"[Step 33] Step 11 loader route unavailable ({type(exc).__name__}: {exc}); using direct forward pass", flush=True)
                    self.route = "direct"
                else:
                    raise
        return self._direct(model, embeddings, idx).argmax(1).astype(np.int64)

    def release(self):
        self.models.clear()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def inference_canary(scorer: L20Scorer, s27: dict, test_embeddings: np.ndarray, max_mismatch: int) -> dict:
    t = scorer.template
    idx = np.asarray(t["sequence_indices"], dtype=np.int64)
    y = np.asarray(t["y"], dtype=np.int64)
    end = np.asarray(t["end_row_indices"], dtype=np.int64)
    require(idx.shape[1] == L20 and idx.max() < len(test_embeddings) and np.array_equal(idx[:, -1], end),
            "Step 27 test L20 cache does not match the Step 9 test embeddings")
    cached_labels = np.asarray(t["label_ids"], dtype=np.int64)
    if len(cached_labels) == len(test_embeddings):      # Step 10 cache stores per-window labels
        window_labels = cached_labels
    else:                                               # per-sequence labels; per-window array is not used
        window_labels = np.full(len(test_embeddings), -1, dtype=np.int64)
        window_labels[end] = y
    rows = []
    for seed in SEEDS:
        saved = s27["test_predictions"][seed].sort_values("sequence_end_row").reset_index(drop=True)
        saved_end = strict_int(saved["sequence_end_row"], "sequence_end_row")
        order = np.argsort(end, kind="stable")
        require(np.array_equal(saved_end, end[order]) and np.array_equal(strict_int(saved["true_label_id"], "true_label_id"), y[order]),
                f"Step 27 saved test predictions do not align with the L20 cache (seed {seed})")
        pred = scorer.predict(seed, test_embeddings, idx, y, window_labels)[order]
        mism = int(np.count_nonzero(pred != strict_int(saved["pred_label_id"], "pred_label_id")))
        require(mism <= max_mismatch,
                f"G4 failed: inference canary seed {seed}: {mism} decisions differ from the saved Step 27 test predictions "
                f"(route={scorer.route}); run in the same environment as Step 27 or inspect the Step 11 route")
        rows.append({"seed": seed, "endpoints": int(len(pred)), "decision_mismatches": mism, "route": scorer.route})
    return {"status": "PASS", "rows": rows}


# ----------------------------------------------------------------------------- analysis
def analyse_level(lv: dict, cohort: dict, l20_pred: dict, retention: float) -> dict:
    y, flags, idx20, episode = cohort["y"], lv["flags"], cohort["idx20"], cohort["episode"]
    pos = cohort["positions"]
    attack = y != 0
    n_ep = int(len(np.unique(episode)))
    grid, per_seed = [], []
    hit = {L: flags[idx20[:, -L:]].any(1) for L in LENGTHS}
    for L in LENGTHS:
        ref = metric(confusion(y, np.where(hit[L] & attack, y, 0)))
        values = []
        for seed in SEEDS:
            p = l20_pred[seed] if L == L20 else lv["preds"][(f"TST_L{L}", seed)][pos]
            m = metric(confusion(y, p))
            values.append(m)
            per_seed.append({"level": lv["level"], "retention": retention, "length": L, "seed": seed,
                             "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                             "attack_macro_recall": m["attack_macro_recall"],
                             "normal_false_alarm_fraction": m["normal_false_alarm_fraction"],
                             "reference_macro_f1": ref["macro_f1"], "gap_to_reference": m["macro_f1"] - ref["macro_f1"]})
        f1 = [v["macro_f1"] for v in values]
        grid.append({"level": lv["level"], "retention": retention, "length": L, "n_seeds": len(SEEDS),
                     "cohort_endpoints": int(len(y)), "cohort_episodes": n_ep,
                     "tst_macro_f1_mean": float(np.mean(f1)), "tst_macro_f1_sd_ddof1": sd1(f1),
                     "tst_accuracy_mean": float(np.mean([v["accuracy"] for v in values])),
                     "tst_attack_macro_recall_mean": float(np.mean([v["attack_macro_recall"] for v in values])),
                     "tst_normal_false_alarm_fraction_mean": float(np.mean([v["normal_false_alarm_fraction"] for v in values])),
                     "reference_macro_f1": ref["macro_f1"], "reference_attack_coverage": float(hit[L][attack].mean()),
                     "attack_endpoints_with_change": int(np.count_nonzero(hit[L] & attack)),
                     "gap_tst_minus_reference": float(np.mean(f1)) - ref["macro_f1"]})
    # Step 28-style groups on this cohort
    groups = np.where(~attack, "Normal", np.where(hit[L10], "A_recent10", np.where(hit[L20], "B_only_older10", "C_no_change20")))
    group_rows = []
    for name in ("A_recent10", "B_only_older10", "C_no_change20", "Normal"):
        mask = groups == name
        if not mask.any():
            continue
        r10 = [float((lv["preds"][("TST_L10", s)][pos][mask] == y[mask]).mean()) for s in SEEDS]
        r20 = [float((l20_pred[s][mask] == y[mask]).mean()) for s in SEEDS]
        flip_up = [int(np.count_nonzero((lv["preds"][("TST_L10", s)][pos][mask] != y[mask]) & (l20_pred[s][mask] == y[mask]))) for s in SEEDS]
        flip_down = [int(np.count_nonzero((lv["preds"][("TST_L10", s)][pos][mask] == y[mask]) & (l20_pred[s][mask] != y[mask]))) for s in SEEDS]
        group_rows.append({"level": lv["level"], "retention": retention, "group": name, "endpoints": int(mask.sum()),
                           "episodes": int(len(np.unique(episode[mask]))),
                           "share_of_attack_endpoints": float(mask.sum() / attack.sum()) if name != "Normal" else float("nan"),
                           "L10_correct_scenario_recall_mean": float(np.mean(r10)), "L20_correct_scenario_recall_mean": float(np.mean(r20)),
                           "L20_correct_scenario_recall_sd_ddof1": sd1(r20),
                           "L10_wrong_L20_correct_seed_mean": float(np.mean(flip_up)),
                           "L10_correct_L20_wrong_seed_mean": float(np.mean(flip_down))})
    return {"grid": grid, "per_seed": per_seed, "groups": group_rows}


# ----------------------------------------------------------------------------- main
def main() -> int:
    args = parse_args()
    started = now_utc()
    s27 = load_step27(args.step27_root, args.final_zip)
    test_embeddings, test_emb_sha = load_test_embeddings(args.test_embeddings, s27)
    s11 = import_file("taes33_step11", args.step11)
    for name in ("ModelConfig", "TimeSeriesTransformerClassifier", "load_checkpoint_safely", "make_loader",
                 "evaluate_model", "make_cross_entropy_loss", "BATCH_SIZE"):
        require(hasattr(s11, name), f"Step 11 source lacks {name}")

    fs_path = args.step72_run / "sweep_statistics" / "family_summary.csv"
    require(fs_path.is_file(), f"Step 72 family_summary.csv not found: {fs_path}")
    level_names = [str(v) for v in load_csv(fs_path)["level"].drop_duplicates().tolist()]
    require(len(level_names) >= 2, "Step 72 family_summary.csv lists fewer than two levels")
    print(f"[Step 33] levels: {level_names}; loading Step 72 inputs", flush=True)
    levels = {name: load_level(args.step72_run, name) for name in level_names}
    repro, retention = reproduce_family_summary(args.step72_run, levels)

    scorer = L20Scorer(s11, s27["checkpoints"], s27["cache_npz"])
    canary = inference_canary(scorer, s27, test_embeddings, args.canary_max_mismatch)
    del test_embeddings
    print(f"[Step 33] canary passed via {scorer.route}", flush=True)

    results, l20_store = [], {}
    for name, lv in levels.items():
        cohort = level_cohort_l20(lv)
        embeddings = np.load(lv["files"]["embeddings"], mmap_mode="r", allow_pickle=False)
        require(embeddings.ndim == 2 and embeddings.shape[0] == len(lv["flags"]) and embeddings.shape[1] == 768,
                f"{name}: cached embeddings shape {embeddings.shape} does not match {len(lv['flags'])} windows")
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        l20_pred = {}
        for seed in SEEDS:
            print(f"[Step 33] {name}: L20 inference seed {seed} on {len(cohort['y']):,} endpoints", flush=True)
            l20_pred[seed] = scorer.predict(seed, emb, cohort["idx20"], cohort["y"], lv["window_label"])
        del emb
        results.append(analyse_level(lv, cohort, l20_pred, retention[name]))
        l20_store[name] = {"sequence_end_row": cohort["end"], "y": cohort["y"],
                           **{f"pred_seed_{s}": l20_pred[s] for s in SEEDS}}
    scorer.release()

    grid = pd.DataFrame([r for res in results for r in res["grid"]])
    per_seed = pd.DataFrame([r for res in results for r in res["per_seed"]])
    groups = pd.DataFrame([r for res in results for r in res["groups"]])

    run_dir = args.output_root / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    run_dir.mkdir(parents=True, exist_ok=False)
    save_csv(run_dir / "33_level_length_grid.csv", grid)
    save_csv(run_dir / "33_per_seed_metrics.csv", per_seed)
    save_csv(run_dir / "33_level_groups.csv", groups)
    save_csv(run_dir / "33_step72_reproduction.csv", repro)
    for name, arrays in l20_store.items():
        np.savez_compressed(run_dir / f"33_L20_predictions_{name}.npz", **arrays)

    findings = {}
    for name in level_names:
        g = grid[grid.level == name].sort_values("length")
        tst = g.tst_macro_f1_mean.to_numpy()
        ref = g.reference_macro_f1.to_numpy()
        findings[name] = {
            "tst_below_reference_at_every_L": bool(np.all(tst < ref)),
            "tst_monotone_nondecreasing_in_L": bool(np.all(np.diff(tst) >= 0)),
            "gain_L10_to_L20_tst": float(tst[-1] - tst[-2]), "gain_L10_to_L20_reference": float(ref[-1] - ref[-2]),
            "gain_L1_to_L20_tst": float(tst[-1] - tst[0]), "gain_L1_to_L20_reference": float(ref[-1] - ref[0]),
            "cohort_endpoints": int(g.cohort_endpoints.iloc[0]), "cohort_episodes": int(g.cohort_episodes.iloc[0]),
        }
    sources = [f"# Step 33 inputs ({started})", f"step72_run\t{args.step72_run}", f"step27_root\t{args.step27_root}",
               f"step11\t{args.step11}\t{digest_file(args.step11)}", f"test_embeddings\t{args.test_embeddings}\t{test_emb_sha}"]
    sources += [f"step27/{k}\t{v}" for k, v in s27["hashes"].items()]
    sources += [f"step72/sweep_statistics/family_summary.csv\t{digest_file(fs_path)}"]
    for name, lv in levels.items():
        sources += [f"step72/{name}/{k}\t{lv['files'][k]}\t{v}" for k, v in lv["hashes"].items()]
    (run_dir / "sources.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")
    gate_report = {
        "status": "GATES_PASSED",
        "G1_input_hashes": "Step 27 checkpoints/cache/ZIP and Step 9 test embeddings matched their manifests",
        "step72_input_sha256": {name: {k: v for k, v in lv["hashes"].items()} for name, lv in levels.items()},
        "G2_step72_alignment": "records, window flags and L10 endpoints aligned; L10 and L20 blocks contiguous within episodes",
        "G3_step72_reproduced": {"rows": int(len(repro)), "tolerance": 1e-6},
        "G4_inference_canary": canary,
        "levels": {name: {"retention": retention[name], **findings[name]} for name in level_names},
    }
    save_json(run_dir / "33_gate_report.json", gate_report)
    manifest = {
        "schema": "taes_step33_evidence_sparsity_L20_v1", "status": "COMPLETE",
        "started_utc": started, "finished_utc": now_utc(), "dataset_id": EXPECTED_DATASET_ID,
        "seeds": list(SEEDS), "lengths": list(LENGTHS), "levels": level_names,
        "population": "per level: L20-eligible endpoints of the Step 72 L10 endpoint set; all lengths scored on the same endpoints",
        "heads": {"L1-L10": "Step 12 frozen heads (Step 72 saved predictions, three of ten seeds)", "L20": "Step 27 frozen heads"},
        "inference_route": scorer.route, "device": str(scorer.device),
        "statistical_scope": "descriptive; 3 head seeds, mean and ddof=1 SD; within-generator sensitivity analysis; exploratory extension designed after the L20 test was inspected",
        "script_sha256": digest_file(Path(__file__).resolve()), "artifacts": {},
    }
    for p in sorted(run_dir.iterdir()):
        if p.is_file():
            manifest["artifacts"][p.name] = digest_file(p)
    save_json(run_dir / "33_run_manifest.json", manifest)
    review = run_dir / "33_EVIDENCE_SPARSITY_L20_REVIEW.zip"
    with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(run_dir.iterdir()):
            if p.is_file() and p != review:
                z.write(p, p.name)
        z.write(Path(__file__).resolve(), "source/" + Path(__file__).name)

    print(f"[Step 33] gates passed; inference route = {scorer.route}")
    for name in level_names:
        g = grid[grid.level == name].set_index("length")
        f = findings[name]
        print(f"  {name} (retention {retention[name]:.2f}; {f['cohort_endpoints']:,} endpoints / {f['cohort_episodes']} ep): "
              f"TST L10 {g.loc[L10, 'tst_macro_f1_mean']:.4f} -> L20 {g.loc[L20, 'tst_macro_f1_mean']:.4f} ({f['gain_L10_to_L20_tst']:+.4f}); "
              f"reference {g.loc[L10, 'reference_macro_f1']:.4f} -> {g.loc[L20, 'reference_macro_f1']:.4f} ({f['gain_L10_to_L20_reference']:+.4f}); "
              f"below reference at every L: {f['tst_below_reference_at_every_L']}")
        gb = groups[(groups.level == name) & (groups.group == "B_only_older10")]
        if len(gb):
            r = gb.iloc[0]
            print(f"      group B {int(r.endpoints):,} endpoints ({100 * r.share_of_attack_endpoints:.1f}% of attack): "
                  f"L10 recall {r.L10_correct_scenario_recall_mean:.3f} -> L20 {r.L20_correct_scenario_recall_mean:.3f}")
    print(f"  run folder: {run_dir}")
    print(f"  review zip: {review}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as exc:
        print(f"[Step 33] STOPPED: {exc}", file=sys.stderr)
        sys.exit(2)
