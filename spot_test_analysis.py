"""
Capture ELISA spot-test bar chart analysis.

Reads Bar graph.xlsx, recomputes all signals/S:N ratios/paired SDs from
Sheet1 raw OD readings, validates against Figures sheet values, and
produces Figure Z.

Usage:
    python spot_test_analysis.py

Outputs:
    figure_z.png   grouped bar chart, 300 dpi
    Console:       bg_subtracted and sn_ratios DataFrames + validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# File path (Bar graph.xlsx lives one directory above this script)
# ---------------------------------------------------------------------------
XLSX = Path(__file__).parent.parent / "Bar graph.xlsx"

# ---------------------------------------------------------------------------
# Plate layout parameters
# ---------------------------------------------------------------------------

# First data row (plate row A) for each timepoint in Sheet1 (0-based pandas index).
# Sheet1 structure per timepoint:
#   row 0  — title line ("Capture ELISA Xmins ...")
#   row 1  — column-number header (1 2 ... 12)
#   rows 2–9 — OD data for plate rows A–H
BLOCK_DATA_START = {20: 2, 40: 14, 60: 26}

# (capture, detector) → (first_plate_row, first_plate_col) within the 8×12 block.
# All indices are 0-based.
QUADRANTS = {
    ("11F4", "11F4"): (0, 0),   # rows A–D, cols 1–6
    ("11F4", "B10"):  (0, 6),   # rows A–D, cols 7–12
    ("B10",  "11F4"): (4, 0),   # rows E–H, cols 1–6
    ("B10",  "B10"):  (4, 6),   # rows E–H, cols 7–12
}

# Within each 4-row × 6-col quadrant, sample groups occupy 2-row × 3-col sub-blocks.
SAMPLES_IN_QUADRANT = {
    "BFV WT":  (slice(0, 2), slice(0, 3)),   # rows 1–2, cols 1–3
    "bJEV":    (slice(0, 2), slice(3, 6)),   # rows 1–2, cols 4–6
    "GETV WT": (slice(2, 4), slice(0, 3)),   # rows 3–4, cols 1–3
    "Mock":    (slice(2, 4), slice(3, 6)),   # rows 3–4, cols 4–6
}

SAMPLE_ORDER   = ["BFV WT", "GETV WT", "bJEV"]
TIMEPOINTS     = [20, 40, 60]
QUADRANT_ORDER = [("11F4", "11F4"), ("11F4", "B10"), ("B10", "11F4"), ("B10", "B10")]

# ---------------------------------------------------------------------------
# Load raw OD data
# ---------------------------------------------------------------------------
raw = pd.read_excel(XLSX, sheet_name="Sheet1", header=None)


def get_block(tp: int) -> np.ndarray:
    """Return the 8×12 OD array for the given timepoint (float64)."""
    start = BLOCK_DATA_START[tp]
    # Sheet1 columns 1–12 (pandas indices 1–12) hold the OD values
    return raw.iloc[start : start + 8, 1:13].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Compute background-subtracted signals, S/N ratios, and paired SDs
# ---------------------------------------------------------------------------
records_bg = []
records_sn = []

for tp in TIMEPOINTS:
    block = get_block(tp)
    for (cap, det), (qr, qc) in QUADRANTS.items():
        quadrant = block[qr : qr + 4, qc : qc + 6]

        mock_vals   = quadrant[SAMPLES_IN_QUADRANT["Mock"]].flatten()
        mock_mean   = mock_vals.mean()

        for sample in SAMPLE_ORDER:
            sample_vals = quadrant[SAMPLES_IN_QUADRANT[sample]].flatten()
            sample_mean = sample_vals.mean()

            mean_signal = sample_mean - mock_mean
            sn          = sample_mean / mock_mean

            # Paired differences: position i of sample minus position i of mock.
            # Both sub-blocks are 2×3, flattened in row-major order.
            paired_diffs = sample_vals - mock_vals
            sd_paired    = float(np.std(paired_diffs, ddof=1))

            records_bg.append(dict(
                capture=cap, detector=det, sample=sample, timepoint=tp,
                mean_signal=mean_signal, sd_paired=sd_paired,
            ))
            records_sn.append(dict(
                capture=cap, detector=det, sample=sample, timepoint=tp,
                s_to_n=sn,
            ))

bg_subtracted = pd.DataFrame(records_bg)
sn_ratios     = pd.DataFrame(records_sn)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
pd.set_option("display.float_format", "{:.6f}".format)
pd.set_option("display.max_rows", 100)
print("\n=== Background-subtracted signals ===")
print(bg_subtracted.to_string(index=False))
print("\n=== S/N ratios ===")
print(sn_ratios.to_string(index=False))


# ---------------------------------------------------------------------------
# Validation against full-precision values from the Figures sheet
# ---------------------------------------------------------------------------
def lookup(cap, det, samp, tp):
    row = bg_subtracted.query(
        "capture == @cap and detector == @det "
        "and sample == @samp and timepoint == @tp"
    )
    return float(row["mean_signal"].iloc[0])


# Reference values read at full precision from the Figures sheet
REFERENCES = [
    ("11F4", "11F4", "BFV WT",  20,  1.6565),
    ("11F4", "11F4", "BFV WT",  40,  2.8273333333333337),
    ("11F4", "11F4", "BFV WT",  60,  3.6376666666666666),
    ("11F4", "11F4", "GETV WT", 20,  0.26266666666666666),
    ("11F4", "11F4", "GETV WT", 40,  0.47466666666666674),
    ("11F4", "11F4", "GETV WT", 60,  0.6399999999999999),
    ("11F4", "11F4", "bJEV",    20, -0.004166666666666659),
    ("11F4", "11F4", "bJEV",    40, -0.0015000000000000083),
    ("11F4", "11F4", "bJEV",    60, -0.0006666666666666765),
    ("B10",  "B10",  "bJEV",    20, -0.004166666666666666),
    ("B10",  "B10",  "bJEV",    40, -0.002166666666666664),
    ("B10",  "B10",  "bJEV",    60, -0.0016666666666666705),
]

TOL = 1e-6
print("\n=== Validation ===")
all_ok = True
for cap, det, samp, tp, expected in REFERENCES:
    actual = lookup(cap, det, samp, tp)
    ok = abs(actual - expected) < TOL
    all_ok = all_ok and ok
    status = "PASS" if ok else f"FAIL  got={actual:.10f}  expected={expected:.10f}"
    print(f"  {cap}/{det}  {samp:<8s}  {tp} min  {status}")

for cap, det, samp, tp, expected in REFERENCES:
    actual = lookup(cap, det, samp, tp)
    assert abs(actual - expected) < TOL, (
        f"Mismatch {cap}/{det}/{samp}/{tp}min: got {actual:.10f}, expected {expected:.10f}"
    )
print("\nAll validation checks passed.")

# ---------------------------------------------------------------------------
# Figure Z — grouped bar chart
# ---------------------------------------------------------------------------
COLORS = {20: "#1f77b4", 40: "#ff7f0e", 60: "#2ca02c"}
N_TP      = len(TIMEPOINTS)
N_SAMPLES = len(SAMPLE_ORDER)

BAR_W     = 0.20   # width of a single bar
INNER_GAP = 0.05   # gap between sample sub-groups within a quadrant group
GROUP_GAP = 0.55   # extra gap between quadrant groups

# Pre-compute x positions
# Each quadrant group spans: N_SAMPLES * (N_TP * BAR_W + INNER_GAP) wide
sample_group_w = N_TP * BAR_W + INNER_GAP
quadrant_group_w = N_SAMPLES * sample_group_w

fig, ax = plt.subplots(figsize=(16, 6))

sample_label_positions = []
sample_label_texts     = []
group_label_positions  = []

for g_idx, (cap, det) in enumerate(QUADRANT_ORDER):
    group_origin = g_idx * (quadrant_group_w + GROUP_GAP)
    group_label_positions.append(group_origin + quadrant_group_w / 2)

    for s_idx, sample in enumerate(SAMPLE_ORDER):
        sample_origin = group_origin + s_idx * sample_group_w
        sample_label_positions.append(sample_origin + (N_TP * BAR_W) / 2)
        sample_label_texts.append(sample)

        for t_idx, tp in enumerate(TIMEPOINTS):
            row = bg_subtracted.query(
                "capture == @cap and detector == @det "
                "and sample == @sample and timepoint == @tp"
            )
            mean_val = float(row["mean_signal"].iloc[0])
            sd_val   = float(row["sd_paired"].iloc[0])
            x = sample_origin + t_idx * BAR_W

            ax.bar(
                x, mean_val, BAR_W * 0.88,
                color=COLORS[tp],
                yerr=sd_val, capsize=3,
                error_kw={"elinewidth": 1.2, "capthick": 1.2},
                label=f"{tp} min" if (g_idx == 0 and s_idx == 0) else "",
            )

# Sample sub-group labels (below bars, row 1)
ax.set_xticks(sample_label_positions)
ax.set_xticklabels(sample_label_texts, fontsize=8, rotation=30, ha="right")

# Capture/detector group labels (below bars, row 2) using annotation
y_annot = -0.28  # axes fraction
for g_idx, (cap, det) in enumerate(QUADRANT_ORDER):
    ax.annotate(
        f"capture {cap}\ndetector {det}",
        xy=(group_label_positions[g_idx], 0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -46),
        textcoords="offset points",
        ha="center", va="top",
        fontsize=9, fontweight="bold",
    )

# Dividers between quadrant groups
for g_idx in range(1, len(QUADRANT_ORDER)):
    divider_x = g_idx * (quadrant_group_w + GROUP_GAP) - GROUP_GAP / 2
    ax.axvline(divider_x, color="grey", linewidth=0.7, linestyle="--", alpha=0.6)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Mean background-subtracted OD (sample − mock)", fontsize=11)
ax.set_title(
    "Capture ELISA signal across antibody pairings, samples, and timepoints",
    fontsize=12, fontweight="bold", pad=12,
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels,
    title="Development time", fontsize=10, title_fontsize=10,
    loc="upper right", framealpha=0.9,
)

ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
out_path = Path(__file__).parent / "figure_z.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
