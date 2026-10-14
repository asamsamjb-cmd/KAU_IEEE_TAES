import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ============================================================
# Figure 02 (Section VI-B): macro-F1 versus temporal context length L
# on the common 38,078 L=10-eligible test endpoints.
#   line 1: DistilBERT + TST, mean over ten head seeds (Step 12)
#   line 2: evidence-availability reference (nondeployable) on the same
#           endpoints, computed with the rule of fingerprint_audit.py:
#           any observed attack evidence inside [start, end] -> true label,
#           otherwise NORMAL.
# Replaces the submitted Fig. 4 (single-run test accuracy).
# ============================================================

CODE_ROOT = Path(r"C:\Users\User\Desktop\2. 연구\2. 26년 이후 연구\1. IEEE TAES 저널\코드 작성")
ROOT = CODE_ROOT / "outputs" / "shared_nominal_v2"

SEED_DIRS = [ROOT / "tst_training_finetuned_main"] + [
    ROOT / "tst_training_finetuned_multiseed" / f"seed_{seed}" for seed in range(20260611, 20260620)
]
L_VALUES = [1, 2, 3, 5, 10]

TST_CM_FILE = "confusion_matrix_test_common_population_L{L}.csv"   # per seed, per L
PRED_FILE = "test_predictions_L{L}.csv"                             # main run, per L

# Maker outputs with raw audit flags (same files fingerprint_audit.py reads).
DATASET_DIRS = [CODE_ROOT / "dataset", CODE_ROOT / "dataset" / "shared_nominal_v2"]
MAKER_FILES = {
    "NORMAL": "normal_observable_dataset_with_raw.csv",
    "RF": "rf_link_jamming_observable_dataset_with_raw.csv",
    "GNSS": "gnss_spoofing_observable_dataset_with_raw.csv",
    "INJ": "unauthorized_telecommand_injection_observable_dataset_with_raw.csv",
    "REPLAY": "telecommand_replay_observable_dataset_with_raw.csv",
    "GND": "ground_segment_ttc_control_path_compromise_observable_dataset_with_raw.csv",
}
# Evidence definition.
#   "paired_nominal": manuscript definition (Section V-A, d_t): the 22 normalized
#       fields of the record differ from the paired-nominal record stored in the
#       baseline_observable_<field> columns. This is what Table IX(a) uses.
#   "raw_flags": class-specific raw attack flags as in fingerprint_audit.py
#       (kept only for comparison; gives slightly lower values).
EVIDENCE_MODE = "paired_nominal"

OBS_FIELDS = (
    "observation_stage", "path_type", "direction", "source_role", "destination_role", "protocol_context",
    "operational_phase", "message_type", "service_context", "command_intent", "payload_or_parameter_context",
    "sequence_freshness_status", "control_security_state", "signal_quality_state", "tracking_stability",
    "navigation_consistency", "timing_consistency", "decoder_status", "command_execution_result",
    "retry_timeout_state", "session_state", "state_transition_consistency",
)
REQUIRED_FLAGS = {
    "NORMAL": [],
    "RF": ["raw_attack_active", "raw_observation_on_target"],
    "GNSS": ["raw_effect_applied", "raw_residual_effect_active"],
    "INJ": ["raw_observed_injection"],
    "REPLAY": ["raw_observed_replay"],
    "GND": ["raw_observed_compromise"],
}

EXPECTED_TOTAL = 38078
EXPECTED_NORMAL = 5762

# Manuscript Table IX(a) values, used only as a consistency gate.
EXPECTED_TST = {1: 0.437, 2: 0.553, 3: 0.634, 5: 0.736, 10: 0.847}
EXPECTED_REF = {1: 0.446, 2: 0.582, 3: 0.665, 5: 0.766, 10: 0.870}
GATE_TOL = 0.0015

GRAPH_ROOT = CODE_ROOT / "graph"
OUT_DIR = GRAPH_ROOT / "02_context_length_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_STEM = "fig_context_length_sweep"

target_order = [
    "NORMAL",
    "RF_LINK_JAMMING",
    "GNSS_SPOOFING",
    "UNAUTHORIZED_TELECOMMAND_INJECTION",
    "TELECOMMAND_REPLAY",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE",
]
N_CLASSES = 6


# ============================================================
# Helpers
# ============================================================

def _strip(label: str) -> str:
    label = str(label)
    for prefix in ("true_", "pred_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
    return label


def load_counts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first = df.columns[0]
    other = [c for c in df.columns if c != first]
    other_stripped = [_strip(c) for c in other]
    if pd.api.types.is_numeric_dtype(df[first]) or not set(target_order) <= set(other_stripped):
        raise ValueError(f"{path}: unexpected confusion-matrix layout.")
    cm = pd.DataFrame(df[other].to_numpy(dtype=float), index=[_strip(v) for v in df[first]], columns=other_stripped)
    missing_rows = [lab for lab in target_order if lab not in cm.index]
    if missing_rows:
        raise ValueError(f"{path}: true-label rows missing: {missing_rows}")
    return cm.reindex(index=target_order, columns=target_order).fillna(0).astype(np.int64)


def macro_f1_from_cm(cm: pd.DataFrame) -> float:
    m = cm.to_numpy(dtype=float)
    tp = np.diag(m)
    fp = m.sum(axis=0) - tp
    fn = m.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0).mean())


def macro_f1_from_labels(y: np.ndarray, p: np.ndarray) -> float:
    f = []
    for c in range(N_CLASSES):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def check_population(cm: pd.DataFrame, path: Path) -> None:
    total = int(cm.to_numpy().sum())
    normal = int(cm.loc["NORMAL"].sum())
    if total != EXPECTED_TOTAL or normal != EXPECTED_NORMAL:
        sys.exit(f"{path}: {total} endpoints / {normal} Normal, expected {EXPECTED_TOTAL} / {EXPECTED_NORMAL}.")


def find_maker_file(name: str) -> Path:
    for d in DATASET_DIRS:
        p = d / name
        if p.is_file():
            return p
    hits = sorted((CODE_ROOT / "dataset").rglob(name))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sys.exit(f"Maker output not found under {CODE_ROOT / 'dataset'}: {name}")
    sys.exit(f"Several copies of {name} found; set DATASET_DIRS explicitly:\n" + "\n".join(str(h) for h in hits))


# ============================================================
# 1. Observed-evidence flags per record (rule of fingerprint_audit.py)
# ============================================================

sources = []
evidence_parts = []
for cls, fname in MAKER_FILES.items():
    path = find_maker_file(fname)
    sources.append(str(path))
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    header.columns = [c.lstrip("\ufeff").strip() for c in header.columns]
    if EVIDENCE_MODE == "paired_nominal":
        baseline_cols = [f"baseline_observable_{f}" for f in OBS_FIELDS]
        needed = ["episode_id", "episode_step"] + list(OBS_FIELDS) + baseline_cols
    else:
        needed = ["episode_id", "episode_step"] + REQUIRED_FLAGS[cls]
    missing = [c for c in needed if c not in header.columns]
    if missing:
        sys.exit(f"{path}: columns missing {missing}")
    df = pd.read_csv(path, usecols=needed, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    if EVIDENCE_MODE == "paired_nominal":
        actual = df[list(OBS_FIELDS)].to_numpy()
        baseline = df[baseline_cols].to_numpy()
        ev = (actual != baseline).any(axis=1)
        if cls == "NORMAL" and ev.any():
            print(f"[warn] NORMAL has {int(ev.sum())} records differing from baseline; expected 0.")
    else:
        g = lambda c: (pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy() > 0)
        if cls == "NORMAL":
            ev = np.zeros(len(df), bool)
        elif cls == "RF":
            ev = g("raw_attack_active") & g("raw_observation_on_target")
        elif cls == "GNSS":
            ev = g("raw_effect_applied") | g("raw_residual_effect_active")
        elif cls == "INJ":
            ev = g("raw_observed_injection")
        elif cls == "REPLAY":
            ev = g("raw_observed_replay")
        else:
            ev = g("raw_observed_compromise")
    evidence_parts.append(pd.DataFrame({
        "episode_id": df["episode_id"].astype(str),
        "step": df["episode_step"].astype(int),
        "evidence": ev,
    }))
    print(f"[evidence] {cls:7s} rows={len(df):6d} observed-evidence rows={int(ev.sum()):6d}")

w = pd.concat(evidence_parts, ignore_index=True).sort_values(["episode_id", "step"])
w["cum"] = w.groupby("episode_id")["evidence"].cumsum()
cum = {(e, s): int(c) for e, s, c in zip(w["episode_id"].to_numpy(), w["step"].to_numpy(), w["cum"].to_numpy())}


def context_has_evidence(episode: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    out = np.zeros(len(episode), bool)
    for i, (e, s0, s1) in enumerate(zip(episode, start, end)):
        before = cum.get((e, s0 - 1), 0) if s0 > 0 else 0
        out[i] = (cum.get((e, s1), 0) - before) > 0
    return out


# ============================================================
# 2. Common endpoints and per-L reference from the main run's predictions
# ============================================================

def load_predictions(L: int) -> pd.DataFrame:
    path = SEED_DIRS[0] / f"L{L}" / "predictions" / PRED_FILE.format(L=L)
    if not path.exists():
        sys.exit(f"Missing file: {path}")
    need = ["episode_id", "sequence_start_step", "sequence_end_step", "true_label_id", "pred_label_id"]
    pr = pd.read_csv(path, encoding="utf-8-sig", dtype={"episode_id": str})
    miss = [c for c in need if c not in pr.columns]
    if miss:
        sys.exit(f"{path}: prediction file lacks columns {miss}")
    sources.append(str(path))
    return pr


pred = {L: load_predictions(L) for L in L_VALUES}
common = set(zip(pred[10]["episode_id"].to_numpy(), pred[10]["sequence_end_step"].astype(int).to_numpy()))
if len(common) != EXPECTED_TOTAL:
    sys.exit(f"L=10 prediction file defines {len(common)} endpoints, expected {EXPECTED_TOTAL}.")

ref_mean = []
tst_main = []
for L in L_VALUES:
    pr = pred[L]
    keep = np.array([(e, s) in common for e, s in zip(pr["episode_id"].to_numpy(), pr["sequence_end_step"].astype(int).to_numpy())])
    pr = pr[keep].reset_index(drop=True)
    if len(pr) != EXPECTED_TOTAL:
        sys.exit(f"L={L}: {len(pr)} common endpoints after filtering, expected {EXPECTED_TOTAL}.")
    known = pr["episode_id"].isin(set(w["episode_id"].unique())).mean()
    if known < 0.99:
        sys.exit(f"L={L}: only {known:.1%} of prediction episode_ids exist in the maker outputs.")
    ctx = context_has_evidence(
        pr["episode_id"].to_numpy(),
        pr["sequence_start_step"].astype(int).to_numpy(),
        pr["sequence_end_step"].astype(int).to_numpy(),
    )
    y = pr["true_label_id"].astype(int).to_numpy()
    yh = pr["pred_label_id"].astype(int).to_numpy()
    oracle = np.where(ctx, y, 0)
    ref_mean.append(macro_f1_from_labels(y, oracle))
    tst_main.append(macro_f1_from_labels(y, yh))
    n_attack_nochange = int(((y != 0) & ~ctx).sum())
    print(f"[context ] L={L:2d}: attack endpoints without in-context change = {n_attack_nochange:6d} "
          f"of {int((y != 0).sum())} (expected 5,339 at L=10)")
ref_mean = np.array(ref_mean)

# ============================================================
# 3. TST: ten seeds x five L from the per-seed confusion matrices
# ============================================================

tst_scores = {L: [] for L in L_VALUES}
for seed_dir in SEED_DIRS:
    for L in L_VALUES:
        path = seed_dir / f"L{L}" / "reports" / TST_CM_FILE.format(L=L)
        if not path.exists():
            sys.exit(f"Missing file: {path}")
        cm = load_counts(path)
        check_population(cm, path)
        tst_scores[L].append(macro_f1_from_cm(cm))
        sources.append(str(path))
tst_mean = np.array([np.mean(tst_scores[L]) for L in L_VALUES])
tst_sd = np.array([np.std(tst_scores[L], ddof=1) for L in L_VALUES])

# ============================================================
# 4. Consistency gate against the manuscript values
# ============================================================

summary = pd.DataFrame({
    "L": L_VALUES,
    "tst_macro_f1_mean": tst_mean.round(4),
    "tst_macro_f1_seed_sd": tst_sd.round(4),
    "tst_macro_f1_main_seed_from_predictions": np.round(tst_main, 4),
    "reference_macro_f1": ref_mean.round(4),
    "n_seeds": [len(tst_scores[L]) for L in L_VALUES],
})
print()
print(summary.to_string(index=False))

bad = []
for i, L in enumerate(L_VALUES):
    if abs(tst_mean[i] - EXPECTED_TST[L]) > GATE_TOL:
        bad.append(f"TST L={L}: computed {tst_mean[i]:.4f} vs manuscript {EXPECTED_TST[L]:.3f}")
    if abs(ref_mean[i] - EXPECTED_REF[L]) > GATE_TOL:
        bad.append(f"Reference L={L}: computed {ref_mean[i]:.4f} vs manuscript {EXPECTED_REF[L]:.3f}")
if bad:
    print("\n[STOP] Computed values do not match the manuscript Table IX(a):")
    print("\n".join("  " + b for b in bad))
    print("Nothing was saved. Send this printout back for diagnosis.")
    sys.exit(1)

summary.to_csv(OUT_DIR / f"{FIG_STEM}_values.csv", index=False)
with open(OUT_DIR / f"{FIG_STEM}_sources.txt", "w", encoding="utf-8") as fh:
    fh.write(f"Evidence definition: {EVIDENCE_MODE} "
             "(paired_nominal = any of the 22 observable fields differs from baseline_observable_<field>).\n")
    fh.write("Reference: oracle(true label if any evidence in [start, end] else NORMAL) on common endpoints.\n")
    fh.write("TST: macro-F1 from per-seed confusion matrices, averaged over ten seeds.\n\n")
    fh.write("\n".join(sources))
    fh.write("\n")

# ============================================================
# 5. Figure — rendered at single-column size (3.5 in), fraction axis
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

L = np.array(L_VALUES)

fig, ax = plt.subplots(figsize=(3.5, 2.7), dpi=300)

ax.plot(
    L, ref_mean,
    color="0.45", marker="s", markersize=4.5, markerfacecolor="white",
    markeredgecolor="0.45", markeredgewidth=1.0, linewidth=1.3, linestyle="--",
    label="Availability reference (nondeployable)", zorder=3,
)
ax.plot(
    L, tst_mean,
    color="#1f77b4", marker="o", markersize=5, markeredgecolor="black",
    markeredgewidth=0.5, linewidth=1.8,
    label="TST (ten-seed mean)", zorder=4,
)

ax.set_xlabel(r"Temporal context length $L$")
ax.set_ylabel("Macro-F1")
ax.set_xticks(L)
ax.set_xlim(0.7, 10.3)
ax.set_ylim(0.40, 0.92)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.55)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.75", fancybox=False)

ax.annotate(
    rf"$L=10$: {tst_mean[-1]:.3f}",
    xy=(9.9, tst_mean[-1] - 0.008),
    xytext=(6.4, 0.62),
    arrowprops=dict(arrowstyle="-|>", linewidth=1.2, color="black", mutation_scale=12, shrinkA=6, shrinkB=0),
    fontsize=8.5,
)

plt.tight_layout()
plt.savefig(OUT_DIR / f"{FIG_STEM}.pdf", bbox_inches="tight")
plt.savefig(OUT_DIR / f"{FIG_STEM}.png", bbox_inches="tight", dpi=600)
plt.savefig(OUT_DIR / f"{FIG_STEM}.svg", bbox_inches="tight")
print(f"\nSaved figure and values to: {OUT_DIR}")

plt.show()