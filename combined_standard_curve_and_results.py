"""
Create a combined publication-quality figure with:
1. Left panel: 60-minute standard curve with 4PL fit
2. Right panel: 60-minute ELISA results by sample
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# ============================================================================
# STANDARD CURVE DATA (60 MIN)
# ============================================================================

# Standard dilution series (concentration in µg/mL)
STD1_CONCENTRATIONS = np.array([10.0, 3.333, 1.111, 0.370, 0.123, 0.041, 
                                0.0137, 0.00457, 0.00152, 0.000507, 0.000169, 0.0000563])

STD2_CONCENTRATIONS = np.array([8.0, 2.667, 0.889, 0.296, 0.0987, 0.0329, 
                                0.0110, 0.00366, 0.00122, 0.000407, 0.000136, 0.0000453])

# Raw OD values (blank-subtracted) - 60 min timepoint
raw_data_60min = {
    'std1_row_a': [3.278, 3.453, 3.303, 2.915, 1.8, 0.869, 0.372, 0.211, 0.121, 0.086, 0.075, 0.072],
    'std1_row_b': [3.275, 3.393, 3.215, 2.589, 1.53, 0.571, 0.248, 0.145, 0.101, 0.081, 0.072, 0.073],
    'std1_row_c': [3.297, 3.281, 3.14, 2.157, 1.217, 0.542, 0.21, 0.131, 0.088, 0.074, 0.071, 0.073],
    'std2_row_d': [3.247, 3.16, 2.637, 1.656, 0.81, 0.372, 0.258, 0.229, 0.105, 0.067, 0.066, 0.069],
    'std2_row_e': [3.167, 3.123, 2.516, 1.564, 0.847, 0.378, 0.166, 0.245, 0.11, 0.069, 0.062, 0.06],
    'std2_row_f': [2.901, 3.082, 2.745, 1.949, 0.761, 0.38, 0.154, 0.106, 0.077, 0.091, 0.057, 0.059],
}

# ============================================================================
# 4PL CURVE FITTING
# ============================================================================

def fourpl(conc, A, B, C, D):
    """
    4-parameter logistic equation.
    OD = D + (A - D) / (1 + (conc/C)^B)
    """
    return D + (A - D) / (1 + np.power(conc / C, B))


def fit_standard_curve(conc_std1, od_std1, conc_std2, od_std2):
    """Fit 4PL curve to pooled standard data."""
    all_conc = np.concatenate([conc_std1, conc_std2])
    all_od = np.concatenate([od_std1, od_std2])
    
    od_min = np.min(all_od)
    od_max = np.max(all_od)
    ec50_guess = np.median(all_conc)
    
    p0 = [od_min, 1.0, ec50_guess, od_max]
    bounds = ([od_min - 0.5, 0.1, 1e-6, od_max - 0.5],
              [od_max + 0.5, 5.0, 100, od_max + 0.5])
    
    try:
        popt, _ = curve_fit(fourpl, all_conc, all_od, p0=p0, bounds=bounds, maxfev=5000)
        A, B, C, D = popt
        
        residuals = all_od - fourpl(all_conc, A, B, C, D)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((all_od - np.mean(all_od))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return A, B, C, D, r_squared
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None, None, None, None


# Process 60 min data
std1_rows = [np.array(raw_data_60min['std1_row_a']),
             np.array(raw_data_60min['std1_row_b']),
             np.array(raw_data_60min['std1_row_c'])]

std2_rows = [np.array(raw_data_60min['std2_row_d']),
             np.array(raw_data_60min['std2_row_e']),
             np.array(raw_data_60min['std2_row_f'])]

od_std1 = np.concatenate(std1_rows)
od_std2 = np.concatenate(std2_rows)

A, B, C, D, r2 = fit_standard_curve(
    np.tile(STD1_CONCENTRATIONS, 3), od_std1,
    np.tile(STD2_CONCENTRATIONS, 3), od_std2
)

# ============================================================================
# ELISA RESULTS DATA (60 MIN)
# ============================================================================

UQ_PURPLE = "#51247A"
UQ_MAGENTA = "#962A8B"
UQ_GREEN = "#2EA836"
UQ_GREY = "#97999B"

short_names = [
    "Buffer exchanged\nintracellular",
    "SN after PEG &\ncushion",
    "Total intracellular\n(pre-cushion)",
    "Total intracellular\n(post-cushion,\ndebris removed)*",
    "FortiCHO\nlong slow spin",
    "SF900\nlong slow spin",
]

plates = ["P2", "P2", "P2", "P2", "P3", "P3"]

data_60 = [0, 39.6691, 0.6097, 0.2187, 2.4929, 8.8231]
err_60 = [0, 4.6399, 0.0294, 0.0302, 0.5370, 1.3943]

# ============================================================================
# CREATE COMBINED FIGURE
# ============================================================================

fig, (ax_curve, ax_results) = plt.subplots(1, 2, figsize=(16, 6))

# ========== LEFT PANEL: STANDARD CURVE ==========

# Scatter: Standard 1 (blue circles)
for i, row_od in enumerate(std1_rows):
    ax_curve.scatter(STD1_CONCENTRATIONS, row_od, 
                     color="#1f77b4", s=80, alpha=0.7, 
                     label="Std 1 (10 µg/mL)" if i == 0 else "")

# Scatter: Standard 2 (green squares)
for i, row_od in enumerate(std2_rows):
    ax_curve.scatter(STD2_CONCENTRATIONS, row_od, 
                     color="#2ca02c", s=80, alpha=0.7, marker="s",
                     label="Std 2 (8 µg/mL)" if i == 0 else "")

# Fitted 4PL curve
xx = np.logspace(-5, 1.2, 200)
yy = fourpl(xx, A, B, C, D)
ax_curve.plot(xx, yy, 'k-', lw=2.5, label=f"4PL fit (R²={r2:.4f})")

# Usable OD range (15-80% of span)
span = D - A
usable_low = A + 0.15 * span
usable_high = A + 0.80 * span
ax_curve.axhspan(usable_low, usable_high, alpha=0.15, color="green", 
                 label="Usable range (15-80%)")

# Formatting
ax_curve.set_xscale('log')
ax_curve.set_xlabel('Concentration (µg/mL)', fontsize=13, fontweight='bold')
ax_curve.set_ylabel('OD₄₀₅ (blank-subtracted)', fontsize=13, fontweight='bold')
ax_curve.set_title(f"Standard curve — 60 minute timepoint\nEC₅₀ = {C:.3f} µg/mL", 
                   fontsize=14, fontweight='bold', pad=12)
ax_curve.grid(True, alpha=0.3, which='both', linestyle='--')

# Add 4PL equation text box
equation_text = f"$OD = {D:.3f} + \\frac{{{A:.3f} - {D:.3f}}}{{1 + \\left(\\frac{{c}}{{{C:.3f}}}\\right)^{{{B:.2f}}}}}$"
ax_curve.text(0.08, 0.92, equation_text, transform=ax_curve.transAxes,
              fontsize=10, verticalalignment='top', horizontalalignment='left',
              bbox=dict(boxstyle='round,pad=0.7', facecolor='white', 
                       edgecolor='gray', linewidth=1.2, alpha=0.95))

ax_curve.set_xlim(1e-5, 10)

# ========== RIGHT PANEL: ELISA RESULTS ==========

x = np.arange(len(short_names))
width = 0.6
colors = ["#2ca02c"]  # Green for 60 min

# Plot bars for 60 min
bars_60 = ax_results.bar(x, data_60, width, label="60 min",
                         color=colors[0], edgecolor='white', linewidth=1.5,
                         yerr=err_60, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 2})

# Formatting
ax_results.set_ylabel("Antigen concentration (µg/mL, undiluted)", fontsize=13, fontweight='bold')
ax_results.set_xlabel("Sample", fontsize=13, fontweight='bold')
ax_results.set_title("BFV Antigen Quantification by Capture ELISA — 60 Minute Timepoint", 
                     fontsize=14, fontweight='bold', pad=12)
ax_results.set_xticks(x)
ax_results.set_xticklabels(short_names, fontsize=10, rotation=90, ha='right')
ax_results.set_yscale('log')
ax_results.set_ylim(0.01, 100)

# Add plate labels below sample names
for i, (name, plate) in enumerate(zip(short_names, plates)):
    ax_results.text(i, 0.003, f"({plate})", ha='center', fontsize=8, 
                    style='italic', color=UQ_GREY, weight='normal')

# Grid
ax_results.grid(True, axis='y', alpha=0.3, which='both', linestyle='-', linewidth=0.5)
ax_results.set_axisbelow(True)

# Annotate BLOQ sample
ax_results.text(0, 0.015, "BLOQ", 
                ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

plt.tight_layout()

# Save figure
plt.savefig('combined_60min_figure.png', dpi=300, bbox_inches='tight')
print("✓ Saved: combined_60min_figure.png")

plt.show()

# ============================================================================
# FIGURE CAPTION
# ============================================================================

caption = """
BFV Antigen Quantification by Capture ELISA — 60 Minute Timepoint. BFV antigen 
concentrations determined by capture ELISA at the 60-minute development timepoint. 
Concentrations are reported as µg/mL of undiluted sample. Left panel: Standard curve 
with 4-parameter logistic (4PL) fit, showing Standard 1 (Std 1, blue circles; 10 µg/mL 
starting concentration, 1–3 dilution series) and Standard 2 (Std 2, green squares; 8 µg/mL 
starting concentration, same dilution series). Both standards were pooled for curve fitting. 
Light green shaded region indicates the usable OD range (15–80% of the standard curve span). 
Right panel: Bars represent the mean antigen concentration calculated from all in-range 
replicates across all assayed dilutions for each sample. Error bars denote the standard 
deviation of replicates. BLOQ indicates results below the limit of quantification (OD readings 
fell outside the assay's usable range of 0.515–2.608 OD₄₀₅, corresponding to 15–80% of the 
standard curve span). Plate designations (P2, P3) are shown below each sample group. The log 
scale on the y-axis accommodates the ~200-fold range in antigen levels across samples.
"""

print("\n" + "="*80)
print("FIGURE CAPTION")
print("="*80)
print(caption)
