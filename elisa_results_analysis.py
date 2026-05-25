"""
ELISA analysis pipeline — standard curve fitting, sample back-calculation, and results figures.

Standard curve data is read directly from the .pda files in Raw Data/.
Sample well assignments are in WELL_MAP below — update if needed.

Usage:
    python elisa_analysis.py

Outputs:
    standard_curves.png              3-panel 4PL fit (20 / 40 / 60 min)
    all_samples_all_timepoints.png   grouped bar chart, all timepoints
    all_samples_60min.png            bar chart, 60-min timepoint only
    Console: fit parameters and back-calculated concentrations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# UQ colour palette
UQ_PURPLE  = "#51247A"
UQ_MAGENTA = "#962A8B"
UQ_GREEN   = "#2EA836"
UQ_GREY    = "#97999B"

# ============================================================================
# PARSE RAW DATA FROM .PDA FILES
# ============================================================================

def parse_pda(filepath):
    """Parse Molecular Devices .pda file. Returns dict of plate_name → 8×12 array."""
    plates = {}
    with open(filepath, encoding='latin-1') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith('Plate:'):
            name = line.split('\t')[1]
            i += 2                          # skip well-label header row
            vals = lines[i].rstrip().split('\t')[2:98]   # skip empty + Temperature, take 96 OD values
            plates[name] = np.array([float(v) for v in vals]).reshape(8, 12)
        i += 1
    return plates

DATA_DIR = Path(__file__).parent / 'Raw Data'
plates_20 = parse_pda(DATA_DIR / 'Capture ELISA Quantifictation 20 min 210526.pda.txt')
plates_40 = parse_pda(DATA_DIR / 'Capture ELISA Quantifictation 40 min 210526.pda.txt')
plates_60 = parse_pda(DATA_DIR / 'KC Capture ELISA 60min 210526 .pda.txt')

def _std_rows(plate):
    """Extract standard curve rows A-F from a plate array."""
    keys = ['std1_row_a', 'std1_row_b', 'std1_row_c',
            'std2_row_d', 'std2_row_e', 'std2_row_f']
    return {k: plate[i].tolist() for i, k in enumerate(keys)}

# ============================================================================
# STANDARD CONCENTRATIONS (µg/mL)
# ============================================================================

STD1_CONCENTRATIONS = np.array([10.0, 3.333, 1.111, 0.370, 0.123, 0.041,
                                 0.0137, 0.00457, 0.00152, 0.000507, 0.000169, 0.0000563])

STD2_CONCENTRATIONS = np.array([8.0, 2.667, 0.889, 0.296, 0.0987, 0.0329,
                                 0.0110, 0.00366, 0.00122, 0.000407, 0.000136, 0.0000453])

# ============================================================================
# STANDARD OD DATA — read from .pda files (Plate 1, rows A–F)
# ============================================================================

raw_data_20min = _std_rows(plates_20['Plate#1 WT BFV'])
raw_data_40min = _std_rows(plates_40['Plate#1 WT BFV 40 mins'])
raw_data_60min = _std_rows(plates_60['Plate#1 WT BFV 60'])

# ============================================================================
# PLATE 2 CORRECTIONS
# ============================================================================

# Plate names inside the .pda files
_P2 = {20: 'Plate#2 Sucrose Cushion Samples',
        40: 'Plate#2 Sucrose Cushion Samples',
        60: 'Plate#1 SC 60'}
_P3 = {20: 'Plate#3 Long Slow spin',
        40: 'Plate#3 Long Slow spin',
        60: 'Plate#1 LS 60'}

# 1. Blank subtraction — mean OD of Plate 2 row D (diluent-only, all 12 wells)
P2_BLANKS = {
    20: float(np.mean(plates_20[_P2[20]][3, :])),
    40: float(np.mean(plates_40[_P2[40]][3, :])),
    60: float(np.mean(plates_60[_P2[60]][3, :])),
}

# 2. Row D (index 3) is diluent-only — excluded from all analyses
# 3. Rows E–H (indices 4–7) had columns loaded in reverse — flip them
# 4. Corrected plate dilution factor per row (100 µL accident → 1:2 series, not 1:3)
#    Row A=1×, B=2×, C=4×, D=skip, E=8×, F=16×, G=32×, H=64×
P2_ROW_DF = {0: 1, 1: 2, 2: 4, 4: 8, 5: 16, 6: 32, 7: 64}

# P3: standard 3-fold serial dilution, all rows valid, no column reversal
P3_ROW_DF = {i: 3**i for i in range(8)}

# ============================================================================
# SAMPLE WELL MAP
# ============================================================================
# col_group: 0=cols1-3, 1=cols4-6, 2=cols7-9, 3=cols10-12 (0-indexed)
# pre_df:    pre-dilution factor applied to the sample before loading Row A

SAMPLE_DEFS = [
    dict(name="Partially buffer exchanged intracellular",  plate='P2', col_group=0, pre_df=1),
    dict(name="yBFV SN after PEG & sucrose cushion",       plate='P2', col_group=1, pre_df=10),
    dict(name="yBFV total intracellular before cushion",   plate='P2', col_group=2, pre_df=10),
    dict(name="yBFV total intracellular after cushion",    plate='P2', col_group=3, pre_df=1),
    dict(name="FortiCHO long slow spin",                   plate='P3', col_group=0, pre_df=10),
    dict(name="SF900 long slow spin",                      plate='P3', col_group=1, pre_df=10),
]

# Short x-axis labels — must match SAMPLE_DEFS order
short_names = [
    "Buffer exchanged\nintracellular",
    "SN after PEG &\ncushion",
    "Total intracellular\nbefore cushion",
    "Total intracellular\nafter cushion",
    "FortiCHO\nlong slow spin",
    "SF900\nlong slow spin",
]

plate_labels = [s['plate'] for s in SAMPLE_DEFS]

# ============================================================================
# FUNCTIONS
# ============================================================================

def fourpl(conc, A, B, C, D):
    """4-parameter logistic: OD = D + (A - D) / (1 + (conc/C)^B)"""
    return D + (A - D) / (1 + np.power(conc / C, B))


def fit_standard_curve(conc_std1, od_std1, conc_std2, od_std2):
    """Fit 4PL to pooled standards. Returns (A, B, C, D, R²)."""
    all_conc = np.concatenate([conc_std1, conc_std2])
    all_od   = np.concatenate([od_std1,   od_std2])
    od_min, od_max = np.min(all_od), np.max(all_od)
    p0     = [od_min, 1.0, np.median(all_conc), od_max]
    bounds = ([od_min - 0.5, 0.1, 1e-6, od_max - 0.5],
              [od_max + 0.5, 5.0, 100,  od_max + 0.5])
    try:
        popt, _ = curve_fit(fourpl, all_conc, all_od, p0=p0, bounds=bounds, maxfev=5000)
        A, B, C, D = popt
        ss_res = np.sum((all_od - fourpl(all_conc, A, B, C, D))**2)
        ss_tot = np.sum((all_od - np.mean(all_od))**2)
        return A, B, C, D, 1 - ss_res / ss_tot
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None, None, None, None


def od_to_conc(od, A, B, C, D):
    """
    Invert 4PL: OD → concentration (µg/mL).
    Returns NaN for ODs outside the usable 15–80% response range.
    ODs near the asymptotes amplify small errors into enormous concentration errors.
    """
    od = np.asarray(od, dtype=float)
    result = np.full_like(od, np.nan)
    span = D - A
    lo = A + 0.15 * span
    hi = A + 0.80 * span
    in_range = (od >= lo) & (od <= hi)
    result[in_range] = C * ((A - D) / (od[in_range] - D) - 1) ** (1.0 / B)
    return result


def process_timepoint(raw_data, timepoint_name):
    """Fit 4PL to one timepoint's standard rows."""
    std1_rows = [np.array(raw_data['std1_row_a']),
                 np.array(raw_data['std1_row_b']),
                 np.array(raw_data['std1_row_c'])]
    std2_rows = [np.array(raw_data['std2_row_d']),
                 np.array(raw_data['std2_row_e']),
                 np.array(raw_data['std2_row_f'])]
    od_std1 = np.concatenate(std1_rows)
    od_std2 = np.concatenate(std2_rows)
    A, B, C, D, r2 = fit_standard_curve(
        np.tile(STD1_CONCENTRATIONS, 3), od_std1,
        np.tile(STD2_CONCENTRATIONS, 3), od_std2,
    )
    return {'A': A, 'B': B, 'C': C, 'D': D, 'r2': r2,
            'std1_conc': STD1_CONCENTRATIONS, 'std1_rows': std1_rows,
            'std2_conc': STD2_CONCENTRATIONS, 'std2_rows': std2_rows,
            'timepoint': timepoint_name}


def back_calculate_samples(sample_defs, tp_20, tp_40, tp_60, plates_20, plates_40, plates_60):
    """
    For each sample, extract OD from the raw plate array, apply corrections, invert 4PL,
    multiply by total dilution factor, and pool all valid wells to give undiluted concentration.
    Returns list of dicts with mean ± SD per timepoint (NaN = BLOQ).
    """
    tp_map = {
        '20min': (tp_20, plates_20, 20),
        '40min': (tp_40, plates_40, 40),
        '60min': (tp_60, plates_60, 60),
    }
    out = []
    for s in sample_defs:
        row = dict(name=s['name'], plate=s['plate'])
        col_start = s['col_group'] * 3
        for key, (tp, plates_tp, t) in tp_map.items():
            if s['plate'] == 'P2':
                raw = plates_tp[_P2[t]].copy()
                raw -= P2_BLANKS[t]
                raw[4:, :] = raw[4:, ::-1]   # correct reversed columns in rows E-H
                row_df_map = P2_ROW_DF        # row D (index 3) already absent from dict
            else:
                raw = plates_tp[_P3[t]].copy()
                row_df_map = P3_ROW_DF
            all_conc = []
            for row_idx, plate_row_df in row_df_map.items():
                od_vals = raw[row_idx, col_start:col_start + 3]
                total_df = s['pre_df'] * plate_row_df
                conc = od_to_conc(od_vals, tp['A'], tp['B'], tp['C'], tp['D']) * total_df
                all_conc.extend(conc.tolist())
            valid = np.array([c for c in all_conc if not np.isnan(c)])
            row[f'mean_{key}'] = float(np.mean(valid))         if len(valid) > 0 else np.nan
            row[f'sd_{key}']   = float(np.std(valid, ddof=1))  if len(valid) > 1 else np.nan
        out.append(row)
    return out


# ============================================================================
# FIT STANDARD CURVES
# ============================================================================

tp_20 = process_timepoint(raw_data_20min, "20 min")
tp_40 = process_timepoint(raw_data_40min, "40 min")
tp_60 = process_timepoint(raw_data_60min, "60 min")
std_results = [tp_20, tp_40, tp_60]

# ============================================================================
# BACK-CALCULATE SAMPLE CONCENTRATIONS
# ============================================================================

sample_results = back_calculate_samples(SAMPLE_DEFS, tp_20, tp_40, tp_60, plates_20, plates_40, plates_60)

def _val(x):
    return 0.0 if (x is None or np.isnan(x)) else x

data_20 = [_val(r['mean_20min']) for r in sample_results]
err_20  = [_val(r['sd_20min'])   for r in sample_results]
data_40 = [_val(r['mean_40min']) for r in sample_results]
err_40  = [_val(r['sd_40min'])   for r in sample_results]
data_60 = [_val(r['mean_60min']) for r in sample_results]
err_60  = [_val(r['sd_60min'])   for r in sample_results]

bloq_positions = [
    (i, j)
    for i, r in enumerate(sample_results)
    for j, key in enumerate(['mean_20min', 'mean_40min', 'mean_60min'])
    if np.isnan(r[key])
]
bloq_60_positions = [i for i, r in enumerate(sample_results) if np.isnan(r['mean_60min'])]

# ============================================================================
# FIGURE 1 — STANDARD CURVES (3-panel)
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, tp in zip(axes, std_results):
    for i, row_od in enumerate(tp['std1_rows']):
        ax.scatter(tp['std1_conc'], row_od, color="#1f77b4", s=40, alpha=0.6,
                  label="Std 1 (10 µg/mL)" if i == 0 else "")
    for i, row_od in enumerate(tp['std2_rows']):
        ax.scatter(tp['std2_conc'], row_od, color="#2ca02c", s=40, alpha=0.6, marker="s",
                  label="Std 2 (8 µg/mL)" if i == 0 else "")

    xx = np.logspace(-5, 1.2, 200)
    yy = fourpl(xx, tp['A'], tp['B'], tp['C'], tp['D'])
    ax.plot(xx, yy, 'k-', lw=2, label=f"4PL fit (R²={tp['r2']:.4f})")

    span = tp['D'] - tp['A']
    ax.axhspan(tp['A'] + 0.15 * span, tp['A'] + 0.80 * span,
              alpha=0.1, color="green", label="Usable range (15–80%)")

    ax.set_xscale('log')
    ax.set_xlabel('Concentration (µg/mL)', fontsize=11, fontweight='bold')
    ax.set_ylabel('OD₄₀₅ (blank-subtracted)', fontsize=11, fontweight='bold')
    ax.set_title(f"Standard curve — {tp['timepoint']}\nEC₅₀ = {tp['C']:.3f} µg/mL",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.95)
    ax.set_xlim(1e-5, 10)

plt.tight_layout()
plt.savefig('standard_curves.png', dpi=300, bbox_inches='tight')
print("Saved: standard_curves.png")
plt.show()

# ============================================================================
# FIGURE 2 — SAMPLE RESULTS (all timepoints)
# ============================================================================

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
x = np.arange(len(short_names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 8))
ax.bar(x - width, data_20, width, label="20 min", color=colors[0],
       edgecolor='white', linewidth=1.5,
       yerr=err_20, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 2})
ax.bar(x,          data_40, width, label="40 min", color=colors[1],
       edgecolor='white', linewidth=1.5,
       yerr=err_40, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 2})
ax.bar(x + width,  data_60, width, label="60 min", color=colors[2],
       edgecolor='white', linewidth=1.5,
       yerr=err_60, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 2})

ax.set_ylabel("Antigen concentration (µg/mL, undiluted)", fontsize=13, fontweight='bold')
ax.set_xlabel("Sample", fontsize=13, fontweight='bold')
ax.set_title("BFV Antigen Quantification by Capture ELISA — All Samples and Timepoints",
             fontsize=14, fontweight='bold', pad=16)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10, rotation=45, ha='right')
ax.set_yscale('log')
ax.set_ylim(0.01, 100)

for i, plate in enumerate(plate_labels):
    ax.text(i, 0.003, f"({plate})", ha='center', fontsize=8, style='italic', color=UQ_GREY)

ax.grid(True, axis='y', alpha=0.3, which='both', linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)
legend = ax.legend(title="ELISA development time", loc='upper left',
                   fontsize=11, title_fontsize=11, framealpha=0.95)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.7)

for sample_idx, bar_idx in bloq_positions:
    ax.text(sample_idx + (bar_idx - 1) * width, 0.015, "BLOQ",
            ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

plt.tight_layout(pad=1.0)
plt.subplots_adjust(bottom=0.15)
plt.savefig('all_samples_all_timepoints.png', dpi=300, bbox_inches='tight')
print("Saved: all_samples_all_timepoints.png")
plt.show()

# ============================================================================
# FIGURE 3 — SAMPLE RESULTS (60 min only)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(x, data_60, 0.6, label="60 min", color=colors[2],
       edgecolor='white', linewidth=1.5,
       yerr=err_60, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 2})

ax.set_ylabel("Antigen concentration (µg/mL, undiluted)", fontsize=13, fontweight='bold')
ax.set_xlabel("Sample", fontsize=13, fontweight='bold')
ax.set_title("BFV Antigen Quantification by Capture ELISA — 60 Minute Timepoint",
             fontsize=14, fontweight='bold', pad=16)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10, rotation=45, ha='right')
ax.set_yscale('log')
ax.set_ylim(0.01, 100)

for i, plate in enumerate(plate_labels):
    ax.text(i, 0.003, f"({plate})", ha='center', fontsize=8, style='italic', color=UQ_GREY)

ax.grid(True, axis='y', alpha=0.3, which='both', linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

for i in bloq_60_positions:
    ax.text(i, 0.015, "BLOQ", ha='center', va='bottom',
            fontsize=8, color='red', fontweight='bold')

plt.tight_layout(pad=1.0)
plt.subplots_adjust(bottom=0.15)
plt.savefig('all_samples_60min.png', dpi=300, bbox_inches='tight')
print("Saved: all_samples_60min.png")
plt.show()

# ============================================================================
# PRINT: STANDARD CURVE PARAMETERS
# ============================================================================

print("\n" + "="*80)
print("STANDARD CURVE FITTING SUMMARY")
print("="*80)
for tp in std_results:
    span = tp['D'] - tp['A']
    print(f"\n{tp['timepoint']} timepoint:")
    print(f"  Lower asymptote (A):  {tp['A']:.3f}")
    print(f"  Hill coefficient (B): {tp['B']:.3f}")
    print(f"  EC50 (C):             {tp['C']:.3f} ug/mL")
    print(f"  Upper asymptote (D):  {tp['D']:.3f}")
    print(f"  R2:                   {tp['r2']:.4f}")
    print(f"  Usable OD range:      {tp['A'] + 0.15*span:.3f} - {tp['A'] + 0.80*span:.3f}")
print("\n4PL: OD = D + (A - D) / (1 + (conc/C)^B)")
print("="*80)

# ============================================================================
# PRINT: SAMPLE BACK-CALCULATION RESULTS
# ============================================================================

def _fmt(mean, sd):
    if np.isnan(mean):
        return f"{'BLOQ':>14}"
    sd_str = f"±{sd:>6.4f}" if not np.isnan(sd) else "±    n/a"
    return f"{mean:>7.4f} {sd_str}"

print("\n" + "="*80)
print("SAMPLE BACK-CALCULATION RESULTS (undiluted concentration, ug/mL)")
print("="*80)
print(f"{'Sample':<46}   {'20 min':>14}   {'40 min':>14}   {'60 min':>14}")
print("-" * 95)
for r in sample_results:
    print(f"{r['name']:<46}  "
          f"{_fmt(r['mean_20min'], r['sd_20min'])}   "
          f"{_fmt(r['mean_40min'], r['sd_40min'])}   "
          f"{_fmt(r['mean_60min'], r['sd_60min'])}")
print("="*80 + "\n")
