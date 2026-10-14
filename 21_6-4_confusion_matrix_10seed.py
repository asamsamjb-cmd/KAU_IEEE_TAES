import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Input: TEN per-seed test confusion matrices of the shared_nominal_v2
# Step 12 TST L=10 run, evaluated on the common 38,078 test endpoints.
#   seed 20260610 -> tst_training_finetuned_main
#   seeds 20260611..20260619 -> tst_training_finetuned_multiseed\seed_<seed>
# ============================================================

ROOT = Path(
    r"C:\Users\User\Desktop\2. 연구\2. 26년 이후 연구\1. IEEE TAES 저널\코드 작성\outputs\shared_nominal_v2"
)

# Report file to read in each seed's L10\reports folder.
# common_population = the common L=10 test endpoints used throughout Section VI.
# (If needed, "confusion_matrix_test_standard_L10.csv" can be substituted; the
#  support check below will tell you whether the population is the same.)
FILE_NAME = "confusion_matrix_test_common_population_L10.csv"

CSV_PATHS = [ROOT / "tst_training_finetuned_main" / "L10" / "reports" / FILE_NAME] + [
    ROOT / "tst_training_finetuned_multiseed" / f"seed_{seed}" / "L10" / "reports" / FILE_NAME
    for seed in range(20260611, 20260620)
]

# Pipeline-side aggregate (Step 15) used as an independent cross-check.
CROSSCHECK_PATH = (
    ROOT / "15_paired_episode_bootstrap" / "run_20261004T102551_378013Z"
    / "seed_average_row_normalized_confusion.csv"
)

N_SEEDS = 10
EXPECTED_TOTAL = 38078    # common L=10 test endpoints
EXPECTED_NORMAL = 5762    # Normal-labeled endpoints (32,316 attack-labeled)

# ============================================================
# Output: one numbered folder per figure under the manuscript graph root.
# 01 = Section VI-A confusion matrix.
# ============================================================

GRAPH_ROOT = Path(
    r"C:\Users\User\Desktop\2. 연구\2. 26년 이후 연구\1. IEEE TAES 저널\코드 작성\graph"
)
OUT_DIR = GRAPH_ROOT / "01_confusion_matrix_L10"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_STEM = "fig_confusion_matrix_L10_normalized"   # name used by \includegraphics in the tex

# ============================================================
# Unified label order and display names (unchanged)
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

labels_short = [display_labels[label] for label in target_order]

# ============================================================
# Loader: confusion-matrix CSV (first column = true labels, possibly
# prefixed "true_"; other columns = predicted labels, possibly prefixed
# "pred_") or, as a fallback, a prediction file with true/pred columns.
# ============================================================

TRUE_COL_CANDIDATES = ["true_label", "y_true", "label", "true"]
PRED_COL_CANDIDATES = ["pred_label", "y_pred", "prediction", "pred"]


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

    if (not pd.api.types.is_numeric_dtype(df[first])) and set(target_order) <= set(other_stripped):
        cm = pd.DataFrame(
            df[other].to_numpy(dtype=float),
            index=[_strip(v) for v in df[first]],
            columns=other_stripped,
        )
    else:
        tcol = next((c for c in TRUE_COL_CANDIDATES if c in df.columns), None)
        pcol = next((c for c in PRED_COL_CANDIDATES if c in df.columns), None)
        if tcol is None or pcol is None:
            raise ValueError(
                f"{path}: not a confusion-matrix CSV and no true/pred label columns found."
            )
        cm = pd.crosstab(df[tcol].astype(str), df[pcol].astype(str))

    missing_rows = [lab for lab in target_order if lab not in cm.index]
    if missing_rows:
        raise ValueError(f"{path}: true-label rows missing: {missing_rows}")
    missing_cols = [lab for lab in target_order if lab not in cm.columns]
    if missing_cols:
        print(f"[warn] {path.parent.parent.parent.name}: never-predicted classes filled with 0: {missing_cols}")

    cm = cm.reindex(index=target_order, columns=target_order).fillna(0)
    if not np.allclose(cm.to_numpy(), np.round(cm.to_numpy())):
        raise ValueError(f"{path}: expected integer counts, found non-integer values.")
    return cm.astype(np.int64)


# ============================================================
# Collect and validate the per-seed files
# ============================================================

if len(CSV_PATHS) != N_SEEDS:
    sys.exit(f"Expected {N_SEEDS} per-seed files, configured {len(CSV_PATHS)}.")

missing = [p for p in CSV_PATHS if not p.exists()]
if missing:
    sys.exit("Missing file(s):\n" + "\n".join(str(p) for p in missing))

mats = [load_counts(p) for p in CSV_PATHS]

supports = [m.sum(axis=1) for m in mats]
for k, s in enumerate(supports[1:], start=2):
    if not s.equals(supports[0]):
        sys.exit(f"Row supports differ between file 1 and file {k}; seeds are not on the same endpoints.")

support = supports[0]
if int(support.sum()) != EXPECTED_TOTAL:
    sys.exit(f"Total endpoints {int(support.sum())} != expected {EXPECTED_TOTAL} (wrong population or file).")
if int(support["NORMAL"]) != EXPECTED_NORMAL:
    sys.exit(f"Normal endpoints {int(support['NORMAL'])} != expected {EXPECTED_NORMAL}.")

# ============================================================
# Row-normalize PER SEED, then average (metric-level averaging).
# Counts are NOT pooled across seeds.
# ============================================================

norm_stack = np.stack(
    [(m.div(m.sum(axis=1), axis=0) * 100.0).to_numpy(dtype=float) for m in mats]
)
cm_mean = pd.DataFrame(norm_stack.mean(axis=0), index=target_order, columns=target_order)
cm_sd = pd.DataFrame(norm_stack.std(axis=0, ddof=1), index=target_order, columns=target_order)

# Ten-seed mean counts (for the parenthesized annotations, as in the original figure).
cm_count_mean = sum(m.astype(float) for m in mats) / len(mats)

prec_stack = np.stack(
    [(np.diag(m.to_numpy()) / np.maximum(m.sum(axis=0).to_numpy(), 1)) for m in mats]
)
precision_mean = pd.Series(prec_stack.mean(axis=0), index=target_order)
recall_mean = pd.Series(np.diag(cm_mean.to_numpy()) / 100.0, index=target_order)

# ============================================================
# Printout for cross-checking against the manuscript (VI-A):
#   GNSS recall 0.938, compromise recall 0.700,
#   Normal recall 0.960, Normal precision 0.490;
#   attack rows predicted Normal: RF 20.9, GNSS 5.7, INJ 8.9,
#   REPLAY 25.0, COMPROMISE 28.8 (%).
# ============================================================

print(f"Seeds averaged: {len(mats)}")
print("Row support per class:")
print(support.to_string())
print("\nMean row-normalized confusion matrix (%):")
print(cm_mean.round(2).to_string())
print("\nSeed SD of row-normalized cells (%):")
print(cm_sd.round(2).to_string())
print("\nClass recall (from mean matrix) / precision (mean of per-seed):")
print(pd.DataFrame({"recall": recall_mean.round(3), "precision": precision_mean.round(3)}).to_string())
print("\nShare of attack rows predicted Normal (%):")
print(cm_mean.loc[target_order[1:], "NORMAL"].round(1).to_string())

# Independent cross-check against the Step 15 aggregate, if present.
if CROSSCHECK_PATH.exists():
    try:
        ref_long = pd.read_csv(CROSSCHECK_PATH)
        if "model" in ref_long.columns:
            ref_long = ref_long[ref_long["model"].astype(str) == "TST"]
        ref = ref_long.pivot(index="true_label", columns="pred_label", values="seed_mean_row_normalized")
        ref.index = [_strip(v) for v in ref.index]
        ref.columns = [_strip(c) for c in ref.columns]
        ref = ref.reindex(index=target_order, columns=target_order).astype(float)
        if np.nanmax(ref.to_numpy()) <= 1.0 + 1e-9:
            ref = ref * 100.0
        diff = np.nanmax(np.abs(ref.to_numpy() - cm_mean.to_numpy()))
        print(f"\nCross-check vs Step 15 seed_average_row_normalized_confusion.csv: max |diff| = {diff:.4f} pp")
        if diff > 0.05:
            print("[warn] Difference exceeds 0.05 pp; confirm both use the same endpoints and averaging.")
    except Exception as exc:  # format mismatch only warns
        print(f"[warn] Could not parse cross-check file: {exc}")
else:
    print("\n[info] Step 15 cross-check file not found; skipped.")

cm_mean.round(3).to_csv(OUT_DIR / f"{FIG_STEM}_mean_percent.csv")
cm_sd.round(3).to_csv(OUT_DIR / f"{FIG_STEM}_seed_sd_percent.csv")
cm_count_mean.round(1).to_csv(OUT_DIR / f"{FIG_STEM}_mean_counts.csv")
with open(OUT_DIR / f"{FIG_STEM}_sources.txt", "w", encoding="utf-8") as fh:
    fh.write("Per-seed confusion CSVs averaged (row-normalized per seed, then mean):\n")
    fh.write("\n".join(str(p) for p in CSV_PATHS))
    fh.write(f"\n\nCross-check file: {CROSSCHECK_PATH}\n")

# ============================================================
# Journal-style plot settings (unchanged)
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=300)

im = ax.imshow(
    cm_mean.to_numpy(),
    cmap="Blues",
    vmin=0,
    vmax=100,
    interpolation="nearest",
)

ax.set_xticks(np.arange(len(labels_short)))
ax.set_yticks(np.arange(len(labels_short)))
ax.set_xticklabels(labels_short, rotation=0, ha="center")
ax.set_yticklabels(labels_short)
ax.set_xlabel("Predicted Threat Scenario", labelpad=8)
ax.set_ylabel("True Threat Scenario", labelpad=8)

# ============================================================
# Cell annotations: same format as the original figure —
# percentage (two decimals) + count in parentheses.
# Percentage = mean of per-seed row-normalized rates;
# count = ten-seed mean count rounded to an integer.
# ============================================================

for i in range(cm_mean.shape[0]):
    for j in range(cm_mean.shape[1]):
        percent = cm_mean.iloc[i, j]
        count = int(round(cm_count_mean.iloc[i, j]))

        if count == 0:
            text = "0"
        else:
            text = f"{percent:.2f}\n({count})"

        text_color = "white" if percent >= 50 else "black"

        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            color=text_color,
            fontsize=12
        )

ax.set_xticks(np.arange(-0.5, len(labels_short), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(labels_short), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
ax.tick_params(which="minor", bottom=False, left=False)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Row-normalized rate (%)", rotation=270, labelpad=16)
cbar.ax.tick_params(labelsize=10)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()

# ============================================================
# Save into graph\01_confusion_matrix_L10\ (PDF is the one the tex includes).
# ============================================================

plt.savefig(OUT_DIR / f"{FIG_STEM}.pdf", bbox_inches="tight", pad_inches=0)
plt.savefig(OUT_DIR / f"{FIG_STEM}.png", bbox_inches="tight", dpi=600)
plt.savefig(OUT_DIR / f"{FIG_STEM}.svg", bbox_inches="tight")
print(f"\nSaved figure and source CSVs to: {OUT_DIR}")

plt.show()