import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ----------------------------------------------------------------------
# Data: evidence-group removal diagnostic, L=10 (Step 53, shared_nominal_v2)
# Replaces the hardcoded lists of the submitted Fig. 5 script with the run files.
# ----------------------------------------------------------------------
CODE_ROOT = Path(r"C:\Users\User\Desktop\2. 연구\2. 26년 이후 연구\1. IEEE TAES 저널\코드 작성")
RUN = CODE_ROOT / "outputs" / "shared_nominal_v2" / "53_field_group_removal" / "run_group_removal_v2"
SUMMARY_PATH = RUN / "model_summary.csv"
PAIRED_PATH = RUN / "paired_comparisons.csv"

GRAPH_ROOT = CODE_ROOT / "graph"
OUT_DIR = GRAPH_ROOT / "05_field_group_removal"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_STEM = "fig_field_group_removal"

CONDITIONS = ["clean", "remove_context", "remove_message", "remove_receiver", "remove_trace"]
groups = [
    "No removal",
    "Context fields",
    "Message and control\nsemantics",
    "RF/GNSS \nreceiver evidence",
    "Operational and\ncommunication trace",
]

# Manuscript values (Table XIV(b)); consistency gate only.
EXPECTED_F1 = {"clean": 0.847, "remove_context": 0.802, "remove_message": 0.489,
               "remove_receiver": 0.614, "remove_trace": 0.707}
EXPECTED_DELTA = {"remove_context": -0.045, "remove_message": -0.358,
                  "remove_receiver": -0.232, "remove_trace": -0.140}
GATE_TOL = 0.0015

for p in (SUMMARY_PATH, PAIRED_PATH):
    if not p.exists():
        sys.exit(f"Missing file: {p}")

summary = pd.read_csv(SUMMARY_PATH).set_index("condition")
paired = pd.read_csv(PAIRED_PATH).set_index("contrast")

missing = [c for c in CONDITIONS if c not in summary.index]
if missing:
    sys.exit(f"{SUMMARY_PATH}: conditions missing {missing}; found {list(summary.index)}")

accuracy = [float(summary.loc[c, "accuracy_mean"]) for c in CONDITIONS]
macro_f1 = [float(summary.loc[c, "macro_f1_mean"]) for c in CONDITIONS]
normal_fpr = [float(summary.loc[c, "normal_fpr_mean"]) for c in CONDITIONS]
f1_sd = [float(summary.loc[c, "macro_f1_head_seed_sd_ddof1"]) for c in CONDITIONS]

# Delta shown beside the bar = condition minus clean (negative), as in the original figure.
delta_f1 = [None] + [-float(summary.loc[c, "delta_macro_f1_clean_minus_condition"]) for c in CONDITIONS[1:]]

# Bonferroni-adjusted paired intervals (clean minus condition in the file; sign flipped here).
ci_low, ci_high = [None], [None]
for c in CONDITIONS[1:]:
    key = f"clean_minus_{c}"
    if key not in paired.index:
        sys.exit(f"{PAIRED_PATH}: contrast {key} not found; found {list(paired.index)}")
    ci_low.append(-float(paired.loc[key, "bonferroni_high"]))
    ci_high.append(-float(paired.loc[key, "bonferroni_low"]))

baseline_f1 = macro_f1[0]

# ----------------------------------------------------------------------
# Consistency gate against the manuscript values
# ----------------------------------------------------------------------
bad = []
for c, f1 in zip(CONDITIONS, macro_f1):
    if abs(f1 - EXPECTED_F1[c]) > GATE_TOL:
        bad.append(f"{c} macro-F1: file {f1:.4f} vs manuscript {EXPECTED_F1[c]}")
for c, d in zip(CONDITIONS[1:], delta_f1[1:]):
    if abs(d - EXPECTED_DELTA[c]) > GATE_TOL:
        bad.append(f"{c} delta: file {d:+.4f} vs manuscript {EXPECTED_DELTA[c]:+.3f}")
if bad:
    print("[STOP] Run files do not match the manuscript Table XIV(b):")
    print("\n".join("  " + b for b in bad))
    (OUT_DIR / "gate_report.txt").write_text("\n".join(bad), encoding="utf-8")
    sys.exit(1)

values = pd.DataFrame({
    "condition": CONDITIONS,
    "label": [g.replace("\n", " ") for g in groups],
    "accuracy_mean": accuracy,
    "macro_f1_mean": macro_f1,
    "macro_f1_head_seed_sd": f1_sd,
    "normal_fpr_mean": normal_fpr,
    "delta_f1_condition_minus_clean": delta_f1,
    "bonferroni_ci_low": ci_low,
    "bonferroni_ci_high": ci_high,
})
print(values.round(4).to_string(index=False))
values.round(5).to_csv(OUT_DIR / f"{FIG_STEM}_values.csv", index=False)
with open(OUT_DIR / f"{FIG_STEM}_sources.txt", "w", encoding="utf-8") as fh:
    fh.write("Step 53 test-time field-group removal on the fixed Step 47 TST heads (ten seeds), "
             "38,078 common test endpoints.\n")
    fh.write(f"{SUMMARY_PATH}\n{PAIRED_PATH}\n")
    fh.write("Delta = condition minus clean; CI = 4-contrast Bonferroni (98.75%) paired episode bootstrap.\n")

# ----------------------------------------------------------------------
# Colorblind-safe palette (unchanged)
# ----------------------------------------------------------------------
C_ACC = "#0072B2"   # blue
C_F1 = "#E69F00"    # orange
C_REF = "#666666"   # gray

# ----------------------------------------------------------------------
# Publication style (unchanged)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.6, 3.1))

y = np.arange(len(groups))
h = 0.36

b_acc = ax.barh(
    y + h / 2,
    accuracy,
    height=h,
    color=C_ACC,
    edgecolor="black",
    linewidth=0.4,
    label="Accuracy",
    zorder=3,
)

b_f1 = ax.barh(
    y - h / 2,
    macro_f1,
    height=h,
    color=C_F1,
    edgecolor="black",
    linewidth=0.4,
    label="Macro-F1",
    zorder=3,
)

# Baseline reference line
ax.axvline(
    baseline_f1,
    color=C_REF,
    linestyle="--",
    linewidth=0.8,
    label="No-removal Macro-F1",
    zorder=2,
)

# ----------------------------------------------------------------------
# Value labels (unchanged format)
# ----------------------------------------------------------------------
for r, v in zip(b_acc, accuracy):
    ax.text(
        v + 0.012,
        r.get_y() + r.get_height() / 2,
        f"{v:.3f}",
        va="center",
        ha="left",
        fontsize=9.8,
    )

for r, v, d in zip(b_f1, macro_f1, delta_f1):
    if d is None:
        txt = f"{v:.3f}"
    else:
        txt = f"{v:.3f}  ($\\Delta${d:+.3f})"

    ax.text(
        v + 0.012,
        r.get_y() + r.get_height() / 2,
        txt,
        va="center",
        ha="left",
        fontsize=9.8,
    )

# ----------------------------------------------------------------------
# Axes (unchanged)
# ----------------------------------------------------------------------
ax.set_yticks(y)
ax.set_yticklabels(groups)
ax.invert_yaxis()

ax.set_xlim(0, 1.18)
ax.set_ylim(len(groups) - 0.5, -0.6)
ax.set_xlabel("Score")

ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.tick_params(axis="both", length=3)

ax.grid(axis="x", linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)
ax.set_axisbelow(True)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# Legend above the axes: with the new values the bars reach ~0.85, so the
# lower-right corner used by the submitted figure is no longer empty.
leg = ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=3,
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=1.0,
    handlelength=1.4,
    borderpad=0.35,
    labelspacing=0.3,
    columnspacing=0.9,
    borderaxespad=0.2,
)
leg.get_frame().set_linewidth(0.5)

fig.tight_layout(pad=0.45)

# ----------------------------------------------------------------------
# Save into graph\05_field_group_removal\
# ----------------------------------------------------------------------
fig.savefig(OUT_DIR / f"{FIG_STEM}.pdf", bbox_inches="tight", pad_inches=0)
fig.savefig(OUT_DIR / f"{FIG_STEM}.png", dpi=600, bbox_inches="tight")
fig.savefig(OUT_DIR / f"{FIG_STEM}.svg", bbox_inches="tight")
print(f"\nSaved figure and values to: {OUT_DIR}")
plt.show()
