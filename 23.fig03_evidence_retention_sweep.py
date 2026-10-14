import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ============================================================
# Figure 03 (Section VI-B): evidence-retention sweep (Step 72).
#   x    : realized ten-record attack-evidence coverage (%) per setting
#   lines: DistilBERT + TST L=10 (ten-seed mean), Mean+Max L=10 (ten-seed mean),
#          evidence-availability reference L=10 (nondeployable), TST L=1 (ten-seed mean)
# New figure; no counterpart in the submitted manuscript.
# ============================================================

CODE_ROOT = Path(r"C:\Users\User\Desktop\2. 연구\2. 26년 이후 연구\1. IEEE TAES 저널\코드 작성")
RUN = CODE_ROOT / "outputs" / "shared_nominal_v2" / "72_evidence_sparsity" / "run_evidence_sparsity_v1"

SETTINGS = ["R100", "R060", "R035", "R020"]
NOMINAL_RETENTION = {"R100": 1.00, "R060": 0.60, "R035": 0.35, "R020": 0.20}
EXPECTED_ENDPOINTS = 41028

# Manuscript values (VI-B evidence-retention paragraph); consistency gate only.
EXPECTED = {
    "coverage_pct": {"R100": 83.01, "R060": 72.14, "R035": 57.45, "R020": 41.96},
    "tst_l10":      {"R100": 0.848, "R060": 0.761, "R035": 0.630, "R020": 0.487},
    "reference_l10": {"R100": 0.870, "R060": 0.791, "R035": 0.677, "R020": 0.545},
    "tst_l1":       {"R100": 0.447, "R060": 0.333, "R035": 0.249, "R020": 0.191},
}
GATE_TOL_F1 = 0.0015
GATE_TOL_PCT = 0.05

GRAPH_ROOT = CODE_ROOT / "graph"
OUT_DIR = GRAPH_ROOT / "03_evidence_retention_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_STEM = "fig_evidence_retention_sweep"

N_CLASSES = 6


def macro_f1_from_labels(y: np.ndarray, p: np.ndarray) -> float:
    f = []
    for c in range(N_CLASSES):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def pick_family(families, pattern: str, label: str) -> str:
    hits = [f for f in families if re.search(pattern, str(f), flags=re.IGNORECASE)]
    if len(hits) != 1:
        sys.exit(f"Could not identify the {label} family uniquely. Candidates matching '{pattern}': {hits}. "
                 f"Available families: {sorted(map(str, families))}")
    return hits[0]


rows = []
sources = []
for s in SETTINGS:
    m_path = RUN / "evaluation" / s / "metrics_by_run.csv"
    l_path = RUN / "evaluation" / s / "lsweep_metrics_by_run.csv"
    c_path = RUN / "generation" / s / "evaluation_context_observable_effects_L10.csv"
    for p in (m_path, l_path, c_path):
        if not p.exists():
            sys.exit(f"Missing file: {p}")
        sources.append(str(p))

    # --- learned heads at L=10 (ten seeds each)
    m = pd.read_csv(m_path)
    fam = m["family"].astype(str)
    tst_name = pick_family(fam.unique(), r"^TST$", "TST L=10")
    meanmax_name = pick_family(fam.unique(), r"meanmax", "Mean+Max L=10")
    tst10 = m.loc[fam == tst_name, "macro_f1"].astype(float).to_numpy()
    mm10 = m.loc[fam == meanmax_name, "macro_f1"].astype(float).to_numpy()

    # --- TST L=1 from the L-sweep heads
    lsw = pd.read_csv(l_path)
    lfam = lsw["family"].astype(str)
    tst1 = lsw.loc[lfam == "TST_L1", "macro_f1"].astype(float).to_numpy()
    if len(tst1) == 0:
        sys.exit(f"{l_path}: no rows with family == 'TST_L1'. Families: {sorted(lfam.unique())}")

    # --- reference and realized coverage from the paired-nominal audit of the context
    ctx = pd.read_csv(c_path)
    n_end = len(ctx)
    if n_end != EXPECTED_ENDPOINTS:
        sys.exit(f"{c_path}: {n_end} endpoints, expected {EXPECTED_ENDPOINTS}.")
    y = ctx["label_id"].astype(int).to_numpy()
    has_ctx = ctx["audit_context_effect_window_count"].astype(int).to_numpy() > 0
    oracle = np.where(has_ctx, y, 0)
    reference = macro_f1_from_labels(y, oracle)
    attack = y != 0
    coverage_pct = 100.0 * has_ctx[attack].mean()

    rows.append({
        "setting": s,
        "nominal_retention": NOMINAL_RETENTION[s],
        "n_endpoints": n_end,
        "n_attack_endpoints": int(attack.sum()),
        "coverage_attack_pct": coverage_pct,
        "tst_l10_mean": tst10.mean(), "tst_l10_sd": tst10.std(ddof=1), "n_seeds_tst_l10": len(tst10),
        "meanmax_l10_mean": mm10.mean(), "meanmax_l10_sd": mm10.std(ddof=1), "n_seeds_meanmax": len(mm10),
        "tst_l1_mean": tst1.mean(), "tst_l1_sd": tst1.std(ddof=1), "n_seeds_tst_l1": len(tst1),
        "reference_l10": reference,
        "gain_l10_minus_l1": tst10.mean() - tst1.mean(),
        "reference_minus_tst_l10": reference - tst10.mean(),
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(4).to_string(index=False))

# ============================================================
# Consistency gate against the manuscript values
# ============================================================

bad = []
for _, r in df.iterrows():
    s = r["setting"]
    checks = [
        ("coverage %", r["coverage_attack_pct"], EXPECTED["coverage_pct"][s], GATE_TOL_PCT),
        ("TST L10", r["tst_l10_mean"], EXPECTED["tst_l10"][s], GATE_TOL_F1),
        ("reference L10", r["reference_l10"], EXPECTED["reference_l10"][s], GATE_TOL_F1),
        ("TST L1", r["tst_l1_mean"], EXPECTED["tst_l1"][s], GATE_TOL_F1),
    ]
    for name, got, exp, tol in checks:
        if abs(got - exp) > tol:
            bad.append(f"{s} {name}: computed {got:.4f} vs manuscript {exp}")
if bad:
    print("\n[STOP] Computed values do not match the manuscript:")
    print("\n".join("  " + b for b in bad))
    print("Nothing was saved. Upload graph\\03_evidence_retention_sweep\\gate_report.txt if present.")
    (OUT_DIR / "gate_report.txt").write_text("\n".join(bad), encoding="utf-8")
    sys.exit(1)

df.round(5).to_csv(OUT_DIR / f"{FIG_STEM}_values.csv", index=False)
with open(OUT_DIR / f"{FIG_STEM}_sources.txt", "w", encoding="utf-8") as fh:
    fh.write("Step 72 evidence-retention sweep. Learned heads: ten seeds per setting (metrics_by_run / lsweep_metrics_by_run).\n")
    fh.write("Reference: true label if audit_context_effect_window_count > 0 else NORMAL, over all L=10 endpoints.\n")
    fh.write("Coverage: share of attack-labeled L=10 endpoints with audit_context_effect_window_count > 0.\n\n")
    fh.write("\n".join(sources))
    fh.write("\n")

# ============================================================
# Figure — rendered at single-column size (3.5 in), fraction axis
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

x = df["coverage_attack_pct"].to_numpy()
ref_y = df["reference_l10"].to_numpy()
tst10_y = df["tst_l10_mean"].to_numpy()
mm10_y = df["meanmax_l10_mean"].to_numpy()
tst1_y = df["tst_l1_mean"].to_numpy()
gain = df["gain_l10_minus_l1"].to_numpy()

fig, ax = plt.subplots(figsize=(3.5, 3.15), dpi=300)

ax.plot(x, ref_y, color="0.45", marker="s", markersize=4.5, markerfacecolor="white",
        markeredgecolor="0.45", markeredgewidth=1.0, linewidth=1.3, linestyle="--",
        label="Availability reference, $L=10$", zorder=3)
ax.plot(x, tst10_y, color="#1f77b4", marker="o", markersize=5.5, markeredgecolor="black",
        markeredgewidth=0.5, linewidth=2.0, label="TST, $L=10$", zorder=4)
# Mean+Max nearly overlaps the TST line; draw it thinner and on top so both remain visible.
ax.plot(x, mm10_y, color="#009E73", marker="^", markersize=4, markerfacecolor="white",
        markeredgecolor="#009E73", markeredgewidth=1.0, linewidth=1.0, linestyle="-.",
        label="Mean+Max, $L=10$", zorder=6)
ax.plot(x, tst1_y, color="#D55E00", marker="D", markersize=4, markeredgecolor="black",
        markeredgewidth=0.5, linewidth=1.4, linestyle=":", label="TST, $L=1$", zorder=4)

# L=10 minus L=1 gain (macro-F1 units), in the empty band below the TST L=10 points
for xi, yi, g in zip(x, tst10_y, gain):
    ax.annotate(f"+{g:.3f}", xy=(xi, yi), xytext=(0, -12), textcoords="offset points",
                ha="center", va="top", fontsize=7.5, color="#1f77b4")

ax.set_xlabel("Ten-record attack-evidence coverage (%)")
ax.set_ylabel("Macro-F1")
ax.set_xlim(x.max() + 4, x.min() - 4)   # sparser evidence to the right
ax.set_ylim(0.10, 1.00)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.55)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

top = ax.secondary_xaxis("top")
top.set_xticks(x)
top.set_xticklabels([f"{s}" for s in df["setting"]])
top.tick_params(labelsize=7.5)
top.spines["top"].set_visible(False)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True,
          framealpha=0.95, edgecolor="0.75", fancybox=False, columnspacing=1.2, handlelength=2.2)

plt.tight_layout()
plt.savefig(OUT_DIR / f"{FIG_STEM}.pdf", bbox_inches="tight")
plt.savefig(OUT_DIR / f"{FIG_STEM}.png", bbox_inches="tight", dpi=600)
plt.savefig(OUT_DIR / f"{FIG_STEM}.svg", bbox_inches="tight")
print(f"\nSaved figure and values to: {OUT_DIR}")

plt.show()