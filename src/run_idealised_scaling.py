"""
src/run_idealised_scaling.py

Canonical generator for Figure 3 and data/SCALING_IDEALISED.csv:
Empirically verifying the sample-size scaling laws of Venn-Abers interval width,
pointwise calibration instability, and U_cal in a classifier-free 1D synthetic experiment.

DGP:
  S ~ Uniform(0, 1)
  Y | S=s ~ Bernoulli(0.2 + 0.6 * s)
  Evaluated at fixed interior score s_0 = 0.5.

Theoretical scaling rates:
  mean_width ~ n^{-2/3}   (slope = -0.667)
  sd_p_mid   ~ n^{-1/3}   (slope = -0.333)
  mean_ucal  ~ n^{-1/3}   (slope = -0.333)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import linregress, t
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add current/parent directory to import path
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils import set_seed
from fast_venn_abers import exact_va_probs

# Note: this experiment's "base model" is the identity map (the calibration
# score IS s, clipped to [0, 1]), so we compute Venn-Abers directly on the
# raw scores via exact_va_probs rather than routing through
# VennAbersCalibrator(cal_size=None), which previously split the n_cal
# calibration draw 75/25 (via sklearn's train_test_split default) and used
# only the 25% "cal" portion -- see Request 1 code review, Priority 1. That
# bug only mislabelled n_cal here (true n was n_cal/4), since the identity
# estimator has no state to corrupt by refitting, but it's fixed anyway for
# correctness of the reported n_cal values.


def run_idealised_scaling_experiment(
    n_cal_list=(500, 1000, 2000, 4000, 8000, 16000, 32000),
    n_reps: int = 200,
    s0: float = 0.5,
    seed: int = 42,
):
    """
    Run Monte Carlo simulation across sample sizes and evaluate at s0 = 0.5.
    """
    print(f"Running idealised scaling experiment (R={n_reps}, s0={s0}, seed={seed})...")
    rng = set_seed(seed)

    records = []
    
    for n_cal in n_cal_list:
        widths = []
        p_mids = []
        u_cals = []
        
        for r in range(n_reps):
            # Sample calibration data from DGP
            s_cal = rng.uniform(0.0, 1.0, size=n_cal)
            p_true = 0.2 + 0.6 * s_cal
            y_cal = rng.binomial(1, p_true)

            p0, p1, _, w = exact_va_probs(s_cal, y_cal, np.array([s0]))

            pm = 0.5 * (p0[0] + p1[0])
            width_val = w[0]
            ucal_val = np.sqrt(pm * (1.0 - pm) * width_val)
            
            widths.append(width_val)
            p_mids.append(pm)
            u_cals.append(ucal_val)
            
        mean_width = float(np.mean(widths))
        sd_p_mid = float(np.std(p_mids, ddof=1))
        mean_ucal = float(np.mean(u_cals))
        
        records.append({
            "n_cal": int(n_cal),
            "mean_width": mean_width,
            "sd_p_mid": sd_p_mid,
            "mean_ucal": mean_ucal,
        })
        print(f"  n_cal={n_cal:5d}: mean_width={mean_width:.6f}, sd_p_mid={sd_p_mid:.6f}, mean_ucal={mean_ucal:.6f}")
        
    df = pd.DataFrame(records)
    
    # Save canonical CSV
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "SCALING_IDEALISED.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved canonical data to: {csv_path}")
    
    # Fit scaling laws
    log_n = np.log(df["n_cal"])
    dof = len(df) - 2
    t_crit = t.ppf(0.975, dof)
    
    results = {}
    for col, name in [("mean_width", "Width"), ("sd_p_mid", "Bootstrap SD"), ("mean_ucal", "U_cal")]:
        log_y = np.log(df[col])
        res = linregress(log_n, log_y)
        ci_lower = res.slope - t_crit * res.stderr
        ci_upper = res.slope + t_crit * res.stderr
        results[col] = {
            "slope": float(res.slope),
            "intercept": float(res.intercept),
            "stderr": float(res.stderr),
            "rvalue": float(res.rvalue),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }
        print(f"{name} scaling: slope = {res.slope:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}], R^2 = {res.rvalue**2:.4f})")
        
    # Generate publication plot
    generate_scaling_plot(df, results)
    
    return df, results


def generate_scaling_plot(df: pd.DataFrame, fits: dict = None):
    """Generates Figure 3 (paper/idealised_scaling_laws.png and .pdf)."""
    paper_dir = os.path.join(root_dir, "paper")
    os.makedirs(paper_dir, exist_ok=True)
    
    n = df["n_cal"].values
    w = df["mean_width"].values
    sd = df["sd_p_mid"].values
    ucal = df["mean_ucal"].values
    
    log_n = np.log(n)
    if fits is None:
        fits = {}
        dof = len(df) - 2
        t_crit = t.ppf(0.975, dof)
        for col in ["mean_width", "sd_p_mid", "mean_ucal"]:
            res = linregress(log_n, np.log(df[col]))
            fits[col] = {
                "slope": float(res.slope),
                "intercept": float(res.intercept),
                "ci_lower": float(res.slope - t_crit * res.stderr),
                "ci_upper": float(res.slope + t_crit * res.stderr),
            }
            
    res_w_slope = fits["mean_width"]["slope"]
    res_w_intercept = fits["mean_width"]["intercept"]
    res_sd_slope = fits["sd_p_mid"]["slope"]
    res_sd_intercept = fits["sd_p_mid"]["intercept"]
    res_ucal_slope = fits["mean_ucal"]["slope"]
    res_ucal_intercept = fits["mean_ucal"]["intercept"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    
    # Panel (a): Width vs n
    ax1.loglog(n, w, "o-", color="#1f77b4", lw=1.8, label="Empirical width")
    fit_w = np.exp(res_w_intercept) * (n ** res_w_slope)
    ax1.loglog(n, fit_w, ":", color="#1f77b4", lw=1.4, label=f"Fit (slope = {res_w_slope:.3f})")
    anchor_idx = 2
    ref_w = w[anchor_idx] * ((n / n[anchor_idx]) ** (-2.0 / 3.0))
    ax1.loglog(n, ref_w, "k--", lw=1.4, label=r"Theory $n^{-2/3}$ ($-0.667$)")
    
    ax1.set_xlabel(r"Calibration size $n_{\mathrm{cal}}$")
    ax1.set_ylabel(r"Mean interval width $\bar{w}$")
    ax1.set_title(r"(a) Venn--Abers width scaling")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.9)
    ax1.set_xticks(n)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    
    # Panel (b): Bootstrap SD and U_cal vs n
    ax2.loglog(n, sd, "s-", color="#2ca02c", lw=1.8, label="Bootstrap SD")
    fit_sd = np.exp(res_sd_intercept) * (n ** res_sd_slope)
    ax2.loglog(n, fit_sd, ":", color="#2ca02c", lw=1.2, label=f"SD fit (slope = {res_sd_slope:.3f})")
    
    ax2.loglog(n, ucal, "^-", color="#d62728", lw=1.8, label=r"$U_{\mathrm{cal}}=\sqrt{\hat{p}(1-\hat{p})w}$")
    fit_ucal = np.exp(res_ucal_intercept) * (n ** res_ucal_slope)
    ax2.loglog(n, fit_ucal, ":", color="#d62728", lw=1.2, label=f"$U_{{\mathrm{{cal}}}}$ fit (slope = {res_ucal_slope:.3f})")
    
    ref_sd = sd[anchor_idx] * ((n / n[anchor_idx]) ** (-1.0 / 3.0))
    ax2.loglog(n, ref_sd, "k--", lw=1.4, label=r"Theory $n^{-1/3}$ ($-0.333$)")
    
    ax2.set_xlabel(r"Calibration size $n_{\mathrm{cal}}$")
    ax2.set_ylabel(r"Instability index")
    ax2.set_title(r"(b) Probability-scale instability scaling")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.9)
    ax2.set_xticks(n)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    
    plt.tight_layout()
    png_path = os.path.join(paper_dir, "idealised_scaling_laws.png")
    pdf_path = os.path.join(paper_dir, "idealised_scaling_laws.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved Figure 3 to:\n  - {png_path}\n  - {pdf_path}")


if __name__ == "__main__":
    run_idealised_scaling_experiment()
