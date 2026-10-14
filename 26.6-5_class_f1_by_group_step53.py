import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Figure 05 (Section VI-D): class-wise F1 under test-time removal of one
# field group (Step 53). Reuses the submitted Fig. 8 heatmap layout; the
# columns are now removed groups instead of Missing-k conditions.
# ============================================================

CODE_ROOT = Path(r"C:\Users\User\Desktop\2. 연구\2. 26년 이후 연구\1. IEEE TAES 저널\코드 작성")
RUN = CODE_ROOT / "outputs" / "shared_nominal_v2" / "53_field_group_removal" / "run_group_removal_v2"
PER_CLASS_PATH = RUN / "per_class_metrics.csv"

GRAPH_ROOT = CODE_ROOT / "graph"
OUT_DIR = GRAPH_ROOT / "05_class_f1_by_group"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_STEM = "fig_class_f1_by_group"

# ============================================================
# Label order and display names (unchanged)
# ============================================================

target_order = [
    "NORMAL",
    "RF_LINK_JAMMING",
    "GNSS_SPOOFING",
    "UNAUTHORIZED_TELECOMMAND_INJECTION",
    "TELECOMMAND_REPLAY",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE",
]

display_labels = {
    "NORMAL": "Normal",
    "RF_LINK_JAMMING": "RF Link\nJamming",
    "GNSS_SPOOFING": "GNSS\nSpoofing",
    "UNAUTHORIZED_TELECOMMAND_INJECTION": "TC\nInjection",
    "TELECOMMAND_REPLAY": "TC\nReplay",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE": "TT&C Path\nCompromise",
}

# Columns: removed field group (file condition -> display label)
condition_order = ["clean", "remove_context", "remove_message", "remove_receiver", "remove_trace"]
condition_labels = {
    "clean": "None",
    "remove_context": "Context",
    "remove_message": "Message/control",
    "remove_receiver": "Receiver",
    "remove_trace": "Trace",
}

# Manuscript values (VI-D text and Step 53 record); consistency gate only.
EXPECTED = {
    ("clean", "RF_LINK_JAMMING"): 0.863,
    ("clean", "GNSS_SPOOFING"): 0.956,
    ("clean", "UNAUTHORIZED_TELECOMMAND_INJECTION"): 0.948,
    ("clean", "TELECOMMAND_REPLAY"): 0.847,
    ("clean", "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE"): 0.818,
    ("remove_receiver", "GNSS_SPOOFING"): 0.000,
    ("remove_receiver", "RF_LINK_JAMMING"): 0.668,
    ("remove_message", "UNAUTHORIZED_TELECOMMAND_INJECTION"): 0.397,
    ("remove_message", "TELECOMMAND_REPLAY"): 0.197,
    ("remove_message", "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE"): 0.063,
    ("remove_trace", "RF_LINK_JAMMING"): 0.295,
}
GATE_TOL = 0.0015

# ============================================================
# Build scenario-wise F1 matrix
# Rows: threat scenarios; Columns: removed field group; Values: F1 (%)
# ============================================================

if not PER_CLASS_PATH.exists():
    sys.exit(f"Missing file: {PER_CLASS_PATH}")

df = pd.read_csv(PER_CLASS_PATH)
required_columns = {"condition", "label", "f1_mean"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    sys.exit(f"{PER_CLASS_PATH.name} is missing columns: {missing_columns}")

df["condition"] = df["condition"].astype(str)
df["label"] = df["label"].astype(str)

heatmap_data = pd.DataFrame(index=target_order, columns=condition_order, dtype=float)
sd_data = pd.DataFrame(index=target_order, columns=condition_order, dtype=float)
for cond in condition_order:
    sub = df[df["condition"] == cond].set_index("label")
    missing = [lab for lab in target_order if lab not in sub.index]
    if missing:
        sys.exit(f"{PER_CLASS_PATH.name}: condition '{cond}' lacks labels {missing}; "
                 f"conditions found: {sorted(df['condition'].unique())}")
    heatmap_data[cond] = sub.loc[target_order, "f1_mean"].astype(float).to_numpy() * 100.0
    if "f1_head_seed_sd_ddof1" in sub.columns:
        sd_data[cond] = sub.loc[target_order, "f1_head_seed_sd_ddof1"].astype(float).to_numpy() * 100.0

# ------------------------------------------------------------
# Consistency gate against the manuscript values
# ------------------------------------------------------------
bad = []
for (cond, lab), exp in EXPECTED.items():
    got = heatmap_data.loc[lab, cond] / 100.0
    if abs(got - exp) > GATE_TOL:
        bad.append(f"{cond} / {lab}: file {got:.4f} vs manuscript {exp:.3f}")
if bad:
    print("[STOP] Per-class values do not match the manuscript:")
    print("\n".join("  " + b for b in bad))
    (OUT_DIR / "gate_report.txt").write_text("\n".join(bad), encoding="utf-8")
    sys.exit(1)

print("Class F1 (%) by removed group:")
print(heatmap_data.round(1).to_string())

values = heatmap_data.copy()
values.index = [display_labels[l].replace("\n", " ") for l in target_order]
values.columns = [condition_labels[c].replace("\n", " ") for c in condition_order]
values.round(2).to_csv(OUT_DIR / f"{FIG_STEM}_values_percent.csv")
sd_out = sd_data.copy()
sd_out.index = values.index
sd_out.columns = values.columns
sd_out.round(2).to_csv(OUT_DIR / f"{FIG_STEM}_seed_sd_percent.csv")
with open(OUT_DIR / f"{FIG_STEM}_sources.txt", "w", encoding="utf-8") as fh:
    fh.write("Step 53 test-time field-group removal, fixed Step 47 TST heads (ten seeds), 38,078 common test endpoints.\n")
    fh.write("Cell = mean class F1 over ten head seeds (percent).\n")
    fh.write(f"{PER_CLASS_PATH}\n")

heatmap_data.index = [display_labels[label] for label in target_order]
x_labels = [condition_labels[c] for c in condition_order]

# ============================================================
# Plot settings — single-column size (3.5 in); style of the submitted Fig. 8
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.5, 2.7), dpi=300)

# ============================================================
# Heatmap (vmin/vmax cover the full 0--100 range of the new values)
# ============================================================

im = ax.imshow(
    heatmap_data.to_numpy(),
    cmap="YlGnBu",
    vmin=0,
    vmax=100,
    aspect="auto",
    interpolation="nearest",
)

ax.set_xticks(np.arange(len(condition_order)))
ax.set_xticklabels(x_labels, rotation=30, ha="right", rotation_mode="anchor")
ax.set_yticks(np.arange(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)
ax.set_xlabel("Removed field group", labelpad=6)
ax.set_ylabel("Threat scenario", labelpad=6)

# ============================================================
# Cell annotations (unchanged format: one decimal, bold for collapsed cells)
# ============================================================

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        value = heatmap_data.iloc[i, j]
        text_color = "white" if value >= 75 else "black"
        ax.text(
            j,
            i,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=7.5,
            color=text_color,
            fontweight="bold" if value < 50 else "normal",
        )

# Grid lines between cells (unchanged)
ax.set_xticks(np.arange(-0.5, len(condition_order), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(heatmap_data.index), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
ax.tick_params(which="minor", bottom=False, left=False)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
cbar.set_label("F1-score (%)", rotation=270, labelpad=12)
cbar.ax.tick_params(labelsize=7)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()

# ============================================================
# Save into graph\05_class_f1_by_group\
# ============================================================

plt.savefig(OUT_DIR / f"{FIG_STEM}.pdf", bbox_inches="tight", pad_inches=0)
plt.savefig(OUT_DIR / f"{FIG_STEM}.png", bbox_inches="tight", dpi=600)
plt.savefig(OUT_DIR / f"{FIG_STEM}.svg", bbox_inches="tight")
print(f"\nSaved figure and values to: {OUT_DIR}")

plt.show()
