"""Create a publication-quality figure of ELISA results with sample names and legend."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_standard_curves import sample_results

# UQ palette
UQ_PURPLE = "#51247A"
UQ_MAGENTA = "#962A8B"
UQ_GREEN = "#2EA836"
UQ_GREY = "#97999B"

# Short names for x-axis
short_names = [
    "Buffer exchanged\nintracellular",
    "SN after PEG &\ncushion",
    "Total intracellular\nbefore cushion",
    "Total intracellular\nafter cushion",
    "FortiCHO\nlong slow spin",
    "SF900\nlong slow spin",
]

# Plate assignment
plates = ["P2", "P2", "P2", "P2", "P3", "P3"]

# Unpack computed concentrations — fill SAMPLES OD values in plot_standard_curves.py
def _val(x):
    return 0.0 if (x is None or np.isnan(x)) else x

timepoints = ["20 min", "40 min", "60 min"]
data_20 = [_val(r['mean_20min']) for r in sample_results]
err_20  = [_val(r['sd_20min'])   for r in sample_results]
data_40 = [_val(r['mean_40min']) for r in sample_results]
err_40  = [_val(r['sd_40min'])   for r in sample_results]
data_60 = [_val(r['mean_60min']) for r in sample_results]
err_60  = [_val(r['sd_60min'])   for r in sample_results]

# Create figure with more height for x-axis labels
fig, ax = plt.subplots(figsize=(14, 8))

# Bar positions
x = np.arange(len(short_names))
width = 0.25

# Colours for timepoints - matching the user's chart
# Blue, Orange, Green
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# Plot bars
bars_20 = ax.bar(x - width, data_20, width, label="20 min", 
                  color=colors[0], edgecolor='white', linewidth=1.5,
                  yerr=err_20, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 2})
bars_40 = ax.bar(x, data_40, width, label="40 min",
                  color=colors[1], edgecolor='white', linewidth=1.5,
                  yerr=err_40, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 2})
bars_60 = ax.bar(x + width, data_60, width, label="60 min",
                  color=colors[2], edgecolor='white', linewidth=1.5,
                  yerr=err_60, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 2})

# Formatting
ax.set_ylabel("Antigen concentration (µg/mL, undiluted)", fontsize=13, fontweight='bold')
ax.set_xlabel("Sample", fontsize=13, fontweight='bold')
ax.set_title("BFV Antigen Quantification by Capture ELISA — All Samples and Timepoints", 
             fontsize=14, fontweight='bold', pad=16)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10, rotation=45, ha='right')
ax.set_yscale('log')
ax.set_ylim(0.01, 100)

# Add plate labels below sample names
for i, (name, plate) in enumerate(zip(short_names, plates)):
    ax.text(i, 0.003, f"({plate})", ha='center', fontsize=8, 
            style='italic', color=UQ_GREY, weight='normal')

# Grid and legend
ax.grid(True, axis='y', alpha=0.3, which='both', linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)
legend = ax.legend(title="ELISA development time", loc='upper left', 
                   fontsize=11, title_fontsize=11, framealpha=0.95)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.7)

# Annotate BLOQ samples (auto-detected: any timepoint where OD was below the standard curve range)
bloq_positions = [
    (i, j)
    for i, r in enumerate(sample_results)
    for j, key in enumerate(['mean_20min', 'mean_40min', 'mean_60min'])
    if np.isnan(r[key])
]
for sample_idx, bar_idx in bloq_positions:
    ax.text(sample_idx + (bar_idx - 1) * width, 0.015, "BLOQ", 
            ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

# Tight layout with more bottom space
plt.tight_layout(pad=1.0)
plt.subplots_adjust(bottom=0.15)

# Save figure
plt.savefig('all_samples_all_timepoints.png', dpi=300, bbox_inches='tight')
print("✓ Saved: all_samples_all_timepoints.png")
plt.show()

# ============================================================================
# Create a second figure for just the 60 minute timepoint
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Bar positions
x = np.arange(len(short_names))
width = 0.6

# Plot bars for 60 min only
bars_60 = ax.bar(x, data_60, width, label="60 min",
                  color=colors[2], edgecolor='white', linewidth=1.5,
                  yerr=err_60, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 2})

# Formatting
ax.set_ylabel("Antigen concentration (µg/mL, undiluted)", fontsize=13, fontweight='bold')
ax.set_xlabel("Sample", fontsize=13, fontweight='bold')
ax.set_title("BFV Antigen Quantification by Capture ELISA — 60 Minute Timepoint", 
             fontsize=14, fontweight='bold', pad=16)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10, rotation=45, ha='right')
ax.set_yscale('log')
ax.set_ylim(0.01, 100)

# Add plate labels below sample names
for i, (name, plate) in enumerate(zip(short_names, plates)):
    ax.text(i, 0.003, f"({plate})", ha='center', fontsize=8, 
            style='italic', color=UQ_GREY, weight='normal')

# Grid
ax.grid(True, axis='y', alpha=0.3, which='both', linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Auto-detect BLOQ at 60 min
bloq_60_positions = [i for i, r in enumerate(sample_results) if np.isnan(r['mean_60min'])]
for sample_idx in bloq_60_positions:
    ax.text(sample_idx, 0.015, "BLOQ", 
            ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

# Tight layout
plt.tight_layout(pad=1.0)
plt.subplots_adjust(bottom=0.15)

# Save figure
plt.savefig('all_samples_60min.png', dpi=300, bbox_inches='tight')
print("✓ Saved: all_samples_60min.png")
plt.show()
