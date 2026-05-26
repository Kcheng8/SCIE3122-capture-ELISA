"""
Plot ELISA standard curves with 4PL fitting.
Generates publication-quality 3-panel figure (20, 40, 60 min timepoints).

Usage:
    python plot_standard_curves.py
    
Output:
    - standard_curves.png (300 dpi)
    - standard_curves.pdf (300 dpi)
    - Console summary of fit parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# ============================================================================
# CONFIGURATION: Replace these with your actual OD values
# ============================================================================

# Standard dilution series (concentration in µg/mL)
STD1_CONCENTRATIONS = np.array([10.0, 3.333, 1.111, 0.370, 0.123, 0.041, 
                                0.0137, 0.00457, 0.00152, 0.000507, 0.000169, 0.0000563])

STD2_CONCENTRATIONS = np.array([8.0, 2.667, 0.889, 0.296, 0.0987, 0.0329, 
                                0.0110, 0.00366, 0.00122, 0.000407, 0.000136, 0.0000453])

# Raw OD values (blank-subtracted) from your three plates
# Format: each is a dict with 'std1_row_x' and 'std2_row_x' for rows A-F
# Replace with your actual data

raw_data_20min = {
    'std1_row_a': [1.95, 2.139, 2.055, 1.617, 0.901, 0.42, 0.199, 0.124, 0.082, 0.066, 0.061, 0.058],
    'std1_row_b': [2.027, 2.187, 1.912, 1.384, 0.77, 0.299, 0.142, 0.099, 0.075, 0.066, 0.061, 0.062],
    'std1_row_c': [2.048, 2.154, 1.749, 1.179, 0.61, 0.283, 0.13, 0.091, 0.07, 0.06, 0.061, 0.063],
    'std2_row_d': [2.162, 2.109, 1.429, 0.811, 0.406, 0.182, 0.147, 0.133, 0.075, 0.059, 0.057, 0.061],
    'std2_row_e': [2.241, 2.11, 1.472, 0.761, 0.43, 0.19, 0.099, 0.135, 0.072, 0.053, 0.052, 0.053],
    'std2_row_f': [2.189, 2.208, 1.726, 1.045, 0.404, 0.188, 0.099, 0.072, 0.068, 0.053, 0.053, 0.055],
}

raw_data_40min = {
    'std1_row_a': [2.863, 2.995, 2.851, 2.333, 1.348, 0.63, 0.273, 0.161, 0.101, 0.078, 0.067, 0.063],
    'std1_row_b': [2.939, 3.021, 2.726, 2.049, 1.153, 0.426, 0.193, 0.119, 0.085, 0.073, 0.066, 0.068],
    'std1_row_c': [2.961, 2.994, 2.63, 1.657, 0.919, 0.414, 0.166, 0.111, 0.078, 0.071, 0.069, 0.069],
    'std2_row_d': [2.931, 2.848, 2.239, 1.241, 0.624, 0.281, 0.2, 0.187, 0.095, 0.065, 0.064, 0.067],
    'std2_row_e': [2.872, 2.798, 2.158, 1.238, 0.663, 0.306, 0.137, 0.202, 0.097, 0.064, 0.059, 0.058],
    'std2_row_f': [2.898, 2.81, 2.401, 1.562, 0.59, 0.287, 0.131, 0.095, 0.074, 0.057, 0.057, 0.058],
}

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
    
    Parameters:
        A: lower asymptote (minimum OD)
        B: Hill coefficient (slope)
        C: EC50 (inflection point, concentration at 50% response)
        D: upper asymptote (maximum OD)
    """
    return D + (A - D) / (1 + np.power(conc / C, B))


def fit_standard_curve(conc_std1, od_std1, conc_std2, od_std2):
    """
    Fit 4PL curve to pooled standard data (both standards combined).
    
    Returns:
        A, B, C, D: fit parameters
        r_squared: coefficient of determination
    """
    # Combine both standards
    all_conc = np.concatenate([conc_std1, conc_std2])
    all_od = np.concatenate([od_std1, od_std2])
    
    # Better initial guess based on data
    od_min = np.min(all_od)
    od_max = np.max(all_od)
    ec50_guess = np.median(all_conc)
    
    p0 = [od_min, 1.0, ec50_guess, od_max]  # [A, B, C, D]
    
    # Set bounds to ensure reasonable fits
    bounds = ([od_min - 0.5, 0.1, 1e-6, od_max - 0.5],
              [od_max + 0.5, 5.0, 100, od_max + 0.5])
    
    try:
        popt, _ = curve_fit(fourpl, all_conc, all_od, p0=p0, bounds=bounds, maxfev=5000)
        A, B, C, D = popt
        
        # Calculate R²
        residuals = all_od - fourpl(all_conc, A, B, C, D)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((all_od - np.mean(all_od))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return A, B, C, D, r_squared
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None, None, None, None


# ============================================================================
# PROCESS DATA
# ============================================================================

def process_timepoint(raw_data, timepoint_name):
    """Process one timepoint: extract replicates and fit curve."""
    
    # Combine replicates for Standard 1
    std1_rows = [np.array(raw_data['std1_row_a']),
                 np.array(raw_data['std1_row_b']),
                 np.array(raw_data['std1_row_c'])]
    
    # Combine replicates for Standard 2
    std2_rows = [np.array(raw_data['std2_row_d']),
                 np.array(raw_data['std2_row_e']),
                 np.array(raw_data['std2_row_f'])]
    
    # Pool all replicates
    od_std1 = np.concatenate(std1_rows)
    od_std2 = np.concatenate(std2_rows)
    
    # Fit curve
    A, B, C, D, r2 = fit_standard_curve(
        np.tile(STD1_CONCENTRATIONS, 3), od_std1,
        np.tile(STD2_CONCENTRATIONS, 3), od_std2
    )
    
    return {
        'A': A, 'B': B, 'C': C, 'D': D, 'r2': r2,
        'std1_conc': STD1_CONCENTRATIONS,
        'std1_rows': std1_rows,
        'std2_conc': STD2_CONCENTRATIONS,
        'std2_rows': std2_rows,
        'timepoint': timepoint_name,
    }


# Process all timepoints
tp_20 = process_timepoint(raw_data_20min, "20 min")
tp_40 = process_timepoint(raw_data_40min, "40 min")
tp_60 = process_timepoint(raw_data_60min, "60 min")

results = [tp_20, tp_40, tp_60]

# ============================================================================
# SAMPLE BACK-CALCULATION
# ============================================================================

def od_to_conc(od, A, B, C, D):
    """
    Invert 4PL: OD → concentration (µg/mL).
    Returns NaN for any OD outside the asymptote range [A, D].
    """
    od = np.asarray(od, dtype=float)
    result = np.full_like(od, np.nan)
    in_range = (od > A) & (od < D)
    result[in_range] = C * ((A - D) / (od[in_range] - D) - 1) ** (1.0 / B)
    return result


# ─── FILL IN YOUR SAMPLE OD VALUES BELOW ────────────────────────────────────
# dilution_factor: 1 = neat, 10 = 1:10 dilution, etc.
# od_XXmin: blank-subtracted OD replicates for that timepoint plate

SAMPLES = [
    dict(name="Partially buffer exchanged intracellular", plate="P2",
         dilution_factor=1,
         od_20min=[np.nan, np.nan, np.nan],
         od_40min=[np.nan, np.nan, np.nan],
         od_60min=[np.nan, np.nan, np.nan]),
    dict(name="yBFV SN after PEG & sucrose cushion", plate="P2",
         dilution_factor=10,
         od_20min=[np.nan, np.nan, np.nan],
         od_40min=[np.nan, np.nan, np.nan],
         od_60min=[np.nan, np.nan, np.nan]),
    dict(name="yBFV total intracellular before cushion", plate="P2",
         dilution_factor=10,
         od_20min=[np.nan, np.nan, np.nan],
         od_40min=[np.nan, np.nan, np.nan],
         od_60min=[np.nan, np.nan, np.nan]),
    dict(name="yBFV total intracellular after cushion", plate="P2",
         dilution_factor=1,
         od_20min=[np.nan, np.nan, np.nan],
         od_40min=[np.nan, np.nan, np.nan],
         od_60min=[np.nan, np.nan, np.nan]),
    dict(name="FortiCHO long slow spin", plate="P3",
         dilution_factor=10,
         od_20min=[np.nan, np.nan, np.nan],
         od_40min=[np.nan, np.nan, np.nan],
         od_60min=[np.nan, np.nan, np.nan]),
    dict(name="SF900 long slow spin", plate="P3",
         dilution_factor=10,
         od_20min=[np.nan, np.nan, np.nan],
         od_40min=[np.nan, np.nan, np.nan],
         od_60min=[np.nan, np.nan, np.nan]),
]
# ─────────────────────────────────────────────────────────────────────────────


def back_calculate_samples(samples, tp_20, tp_40, tp_60):
    """Back-calculate undiluted concentrations from raw ODs using fitted 4PL parameters."""
    tp_map = {'20min': tp_20, '40min': tp_40, '60min': tp_60}
    out = []
    for s in samples:
        row = dict(name=s['name'], plate=s['plate'], dilution_factor=s['dilution_factor'])
        for key, tp in tp_map.items():
            od_vals = np.array(s[f'od_{key}'], dtype=float)
            conc = od_to_conc(od_vals, tp['A'], tp['B'], tp['C'], tp['D']) * s['dilution_factor']
            n_valid = int(np.sum(~np.isnan(conc)))
            row[f'mean_{key}'] = float(np.nanmean(conc)) if n_valid > 0 else np.nan
            row[f'sd_{key}']   = float(np.nanstd(conc, ddof=1)) if n_valid > 1 else np.nan
        out.append(row)
    return out


sample_results = back_calculate_samples(SAMPLES, tp_20, tp_40, tp_60)


if __name__ == '__main__':
    # ============================================================================
    # PLOTTING
    # ============================================================================

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, tp in zip(axes, results):

        for i, row_od in enumerate(tp['std1_rows']):
            ax.scatter(tp['std1_conc'], row_od,
                      color="#1f77b4", s=40, alpha=0.6,
                      label="Std 1 (10 µg/mL)" if i == 0 else "")

        for i, row_od in enumerate(tp['std2_rows']):
            ax.scatter(tp['std2_conc'], row_od,
                      color="#2ca02c", s=40, alpha=0.6, marker="s",
                      label="Std 2 (8 µg/mL)" if i == 0 else "")

        xx = np.logspace(-5, 1.2, 200)
        yy = fourpl(xx, tp['A'], tp['B'], tp['C'], tp['D'])
        ax.plot(xx, yy, 'k-', lw=2, label=f"4PL fit (R²={tp['r2']:.4f})")

        span = tp['D'] - tp['A']
        usable_low = tp['A'] + 0.15 * span
        usable_high = tp['A'] + 0.80 * span
        ax.axhspan(usable_low, usable_high, alpha=0.1, color="green",
                  label="Usable range (15-80%)")

        ax.set_xscale('log')
        ax.set_xlabel('Concentration (µg/mL)', fontsize=11, fontweight='bold')
        ax.set_ylabel('OD₄₀₅ (blank-subtracted)', fontsize=11, fontweight='bold')
        ax.set_title(f"Standard curve — {tp['timepoint']}\nEC₅₀ = {tp['C']:.3f} µg/mL",
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        
        # Add 4PL equation text box
        A, B, C, D = tp['A'], tp['B'], tp['C'], tp['D']
        equation_text = f"$OD = {D:.3f} + \\frac{{{A:.3f} - {D:.3f}}}{{1 + \\left(\\frac{{c}}{{{C:.3f}}}\\right)^{{{B:.2f}}}}}$"
        ax.text(0.08, 0.92, equation_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='white', 
                         edgecolor='gray', linewidth=1.2, alpha=0.95))
        
        ax.set_xlim(1e-5, 10)

    plt.tight_layout()
    plt.savefig('standard_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: standard_curves.png")
    plt.show()

    # ============================================================================
    # PLOT JUST 60 MINUTE TIMEPOINT
    # ============================================================================

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, row_od in enumerate(tp_60['std1_rows']):
        ax.scatter(tp_60['std1_conc'], row_od,
                  color="#1f77b4", s=80, alpha=0.7,
                  label="Std 1 (10 µg/mL)" if i == 0 else "")

    for i, row_od in enumerate(tp_60['std2_rows']):
        ax.scatter(tp_60['std2_conc'], row_od,
                  color="#2ca02c", s=80, alpha=0.7, marker="s",
                  label="Std 2 (8 µg/mL)" if i == 0 else "")

    xx = np.logspace(-5, 1.2, 200)
    yy = fourpl(xx, tp_60['A'], tp_60['B'], tp_60['C'], tp_60['D'])
    ax.plot(xx, yy, 'k-', lw=2.5, label=f"4PL fit (R²={tp_60['r2']:.4f})")

    span = tp_60['D'] - tp_60['A']
    usable_low = tp_60['A'] + 0.15 * span
    usable_high = tp_60['A'] + 0.80 * span
    ax.axhspan(usable_low, usable_high, alpha=0.15, color="green",
              label="Usable range (15-80%)")

    ax.set_xscale('log')
    ax.set_xlabel('Concentration (µg/mL)', fontsize=13, fontweight='bold')
    ax.set_ylabel('OD₄₀₅ (blank-subtracted)', fontsize=13, fontweight='bold')
    ax.set_title(f"Standard curve — 60 minute timepoint\nEC₅₀ = {tp_60['C']:.3f} µg/mL",
                fontsize=14, fontweight='bold', pad=16)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    
    # Add 4PL equation text box
    A, B, C, D = tp_60['A'], tp_60['B'], tp_60['C'], tp_60['D']
    equation_text = f"$OD = {D:.3f} + \\frac{{{A:.3f} - {D:.3f}}}{{1 + \\left(\\frac{{c}}{{{C:.3f}}}\\right)^{{{B:.2f}}}}}$"
    ax.text(0.08, 0.92, equation_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='white', 
                     edgecolor='gray', linewidth=1.2, alpha=0.95))
    
    ax.set_xlim(1e-5, 10)

    plt.tight_layout()
    plt.show()

    # ============================================================================
    # PRINT SUMMARY
    # ============================================================================

    print("\n" + "="*80)
    print("STANDARD CURVE FITTING SUMMARY")
    print("="*80)

    for tp in results:
        print(f"\n{tp['timepoint']} timepoint:")
        print(f"  Lower asymptote (A):       {tp['A']:.3f}")
        print(f"  Hill coefficient (B):      {tp['B']:.3f}")
        print(f"  EC₅₀ (C):                  {tp['C']:.3f} µg/mL")
        print(f"  Upper asymptote (D):       {tp['D']:.3f}")
        print(f"  R²:                        {tp['r2']:.3f}")

        span = tp['D'] - tp['A']
        usable_low = tp['A'] + 0.15 * span
        usable_high = tp['A'] + 0.80 * span
        print(f"  Usable OD range:           {usable_low:.3f} – {usable_high:.3f}")
        print(f"  Usable range span:         {usable_high - usable_low:.3f} OD units")

    print("\n" + "="*80)
    print("4PL Equation: OD = D + (A - D) / (1 + (conc/C)^B)")
    print("="*80 + "\n")

    # ============================================================================
    # PRINT SAMPLE BACK-CALCULATION RESULTS
    # ============================================================================

    def _fmt(mean, sd):
        if np.isnan(mean):
            return f"{'BLOQ':>14}"
        sd_str = f"±{sd:>6.4f}" if not np.isnan(sd) else "±    n/a"
        return f"{mean:>7.4f} {sd_str}"

    print("\n" + "="*80)
    print("SAMPLE BACK-CALCULATION RESULTS (undiluted concentration, µg/mL)")
    print("="*80)
    print(f"{'Sample':<46} {'DF':>3}   {'20 min':>14}   {'40 min':>14}   {'60 min':>14}")
    print("-" * 99)
    for r in sample_results:
        print(f"{r['name']:<46} {r['dilution_factor']:>3}x  "
              f"{_fmt(r['mean_20min'], r['sd_20min'])}   "
              f"{_fmt(r['mean_40min'], r['sd_40min'])}   "
              f"{_fmt(r['mean_60min'], r['sd_60min'])}")
    print("="*80 + "\n")
