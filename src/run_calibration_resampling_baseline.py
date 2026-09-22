"""
src/run_calibration_resampling_baseline.py

Canonical generator for Table 1 and data/calibration_resampling_baseline.csv:
Calibration resampling analysis across calibration set sizes (n_cal).
Reports:
  - Mean interval width
  - Mean standard deviation of point predictions across resamples
  - Correlation between width and resampling SD
  - Fresh Brier score on a fresh test set
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingClassifier

# Add current/parent directory to import path
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils import clip01, true_probability_1d, set_seed, make_base_model
from fast_venn_abers import exact_va_probs

# Note: computes Venn-Abers directly on the fixed base classifier's scores via
# exact_va_probs rather than routing through VennAbersCalibrator(cal_size=None),
# which internally refits a clone of the base estimator on whatever is passed
# to .fit() and additionally splits it 75/25 (sklearn's train_test_split
# default), so it silently used only ~25% of each calibration resample --
# see Request 1 code review, Priority 1.


def run_calibration_resampling_baseline(
    n_cals=(100, 250, 500, 1000, 2000),
    n_train_base: int = 3000,
    n_test: int = 500,
    M: int = 100,
    seed: int = 42,
):
    print(f"Running calibration resampling baseline experiment (n_cals={n_cals}, M={M}, seed={seed})...")
    rng = set_seed(seed)
    
    # 1. Generate Training Data
    x_train = rng.uniform(-2.0, 2.0, size=n_train_base)
    p_train = true_probability_1d(x_train)
    y_train = rng.binomial(1, clip01(p_train))
    
    # 2. Generate Fresh Test Data
    x_test = np.linspace(-2.0, 2.0, n_test)
    p_test_true = true_probability_1d(x_test)
    y_test = rng.binomial(1, clip01(p_test_true))
    
    X_train = x_train.reshape(-1, 1)
    X_test = x_test.reshape(-1, 1)
    
    # 3. Fit Fixed Base Classifier
    base_clf = make_base_model(seed=seed)
    base_clf.fit(X_train, y_train)
    s_test = base_clf.predict_proba(X_test)[:, 1]

    records = []
    
    for n_cal in n_cals:
        print(f"Processing n_cal = {n_cal} ({M} resamples)...")
        all_pmids = np.zeros((M, n_test))
        all_widths = np.zeros((M, n_test))
        briers = []
        
        for m in range(M):
            # Draw independent calibration sample from DGP
            resample_rng = set_seed(seed * 10000 + n_cal * 100 + m)
            x_cal = resample_rng.uniform(-2.0, 2.0, size=n_cal)
            p_cal = true_probability_1d(x_cal)
            y_cal = resample_rng.binomial(1, clip01(p_cal))

            s_cal = base_clf.predict_proba(x_cal.reshape(-1, 1))[:, 1]
            p0, p1, p_mid, w = exact_va_probs(s_cal, y_cal, s_test)

            all_pmids[m, :] = p_mid
            all_widths[m, :] = w
            briers.append(float(np.mean((p_mid - y_test) ** 2)))
            
        mean_w_point = np.mean(all_widths, axis=0)
        sd_point = np.std(all_pmids, axis=0, ddof=1)
        
        mean_width = float(np.mean(mean_w_point))
        mean_sd = float(np.mean(sd_point))
        corr_val, _ = pearsonr(mean_w_point, sd_point)
        fresh_brier = float(np.mean(briers))
        
        records.append({
            "n_cal": int(n_cal),
            "mean_width": mean_width,
            "mean_sd": mean_sd,
            "corr": float(corr_val),
            "fresh_brier": fresh_brier,
        })
        print(f"  n_cal={n_cal:4d} | Mean Width={mean_width:.4f} | Mean SD={mean_sd:.4f} | Corr={corr_val:.4f} | Fresh Brier={fresh_brier:.4f}")
        
    df = pd.DataFrame(records)
    
    # Save canonical CSV
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "calibration_resampling_baseline.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved canonical data to: {csv_path}")
    
    # Save LaTeX table
    generate_latex_table(df)
    
    return df


def generate_latex_table(df: pd.DataFrame):
    """Generates Table 1 (paper/table_calibration_resampling.tex)."""
    paper_dir = os.path.join(root_dir, "paper")
    os.makedirs(paper_dir, exist_ok=True)
    
    rows = []
    for _, row in df.iterrows():
        n = int(row["n_cal"])
        w = f"{row['mean_width']:.4f}"
        sd = f"{row['mean_sd']:.4f}"
        corr = f"{row['corr']:.4f}"
        brier = f"{row['fresh_brier']:.4f}"
        rows.append(f"\t\t\t{n} & {w} & {sd} & {corr} & {brier} \\\\")
        
    rows_tex = "\n".join(rows)
    table_tex = f"""\\begin{{table}}[ht]
\t\\centering
\t\\caption{{Calibration resampling analysis across calibration set sizes ($n_{{\\text{{cal}}}}$). We report the mean interval width, the standard deviation of point predictions across resamples, their correlation, and the Brier score on a fresh test set.}}
\t\\label{{tab:calibration_resampling}}
\t\\vspace{{8pt}}
\t\\begin{{tabular}}{{rcccc}}
\t\t\\toprule
\t\t$n_{{\\text{{cal}}}}$ & Mean Width & Mean SD Across Resamples & Corr(width, SD) & Fresh Brier \\\\
\t\t\\midrule
{rows_tex}
\t\t\\bottomrule
\t\\end{{tabular}}
\\end{{table}}
"""
    tex_path = os.path.join(paper_dir, "table_calibration_resampling.tex")
    with open(tex_path, "w") as f:
        f.write(table_tex)
    print(f"Saved Table 1 LaTeX to: {tex_path}")


if __name__ == "__main__":
    run_calibration_resampling_baseline()
