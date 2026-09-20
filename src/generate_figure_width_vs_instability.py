"""
src/generate_figure_width_vs_instability.py

Generates publication-quality 2-panel figure:
  - Panel (a): Raw Venn--Abers interval width w vs Bootstrap calibration SD
  - Panel (b): Instability index U_cal vs Bootstrap calibration SD

Outputs:
  - paper/width_vs_bootstrap_instability.png (300 DPI)
  - paper/width_vs_bootstrap_instability.pdf
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress
import matplotlib.pyplot as plt

# Configure Matplotlib for academic publication style (identical to generate_final_figures_tables.py)
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

def generate_figure(csv_path: str = None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if csv_path is None:
        csv_path = os.path.join(project_root, "data", "classifier_bootstrap_n500.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(project_root, "classifier_bootstrap_n500.csv")
            
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find data CSV at {csv_path}")
        
    df = pd.read_csv(csv_path)
    w = df["width"].values
    u_cal = df["u_cal"].values
    sd = df["bootstrap_sd"].values
    
    # Compute correlations
    r_w, _ = pearsonr(w, sd)
    rho_w, _ = spearmanr(w, sd)
    
    r_u, _ = pearsonr(u_cal, sd)
    rho_u, _ = spearmanr(u_cal, sd)
    
    # Linear trend lines
    reg_w = linregress(w, sd)
    reg_u = linregress(u_cal, sd)
    
    # Held-out multiplicative fit MAE
    n = len(df)
    fit_idx = np.arange(0, n, 2)
    eval_idx = np.arange(1, n, 2)
    c_w = np.sum(sd[fit_idx] * w[fit_idx]) / np.sum(w[fit_idx] ** 2)
    c_u = np.sum(sd[fit_idx] * u_cal[fit_idx]) / np.sum(u_cal[fit_idx] ** 2)
    mae_w = float(np.mean(np.abs(sd[eval_idx] - c_w * w[eval_idx])))
    mae_u = float(np.mean(np.abs(sd[eval_idx] - c_u * u_cal[eval_idx])))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    
    # Panel (a): Raw width vs Bootstrap SD
    ax1.scatter(w, sd, color="#1f77b4", alpha=0.55, edgecolors="none", s=22, label="Test instances")
    w_grid = np.linspace(w.min(), w.max(), 100)
    ax1.plot(w_grid, reg_w.intercept + reg_w.slope * w_grid, color="#0b3c5d", lw=2.0,
             label=f"Linear trend")
    ax1.set_xlabel(r"Venn--Abers interval width $w$")
    ax1.set_ylabel(r"Bootstrap standard deviation $\sigma_{\mathrm{boot}}$")
    ax1.set_title(r"(a) Raw interval width vs. instability")
    
    # Info box for panel (a)
    text_a = (f"Pearson $r = {r_w:.4f}$\n"
              f"Spearman $\\rho = {rho_w:.4f}$\n"
              f"Held-out MAE $= {mae_w:.4f}$")
    ax1.text(0.05, 0.95, text_a, transform=ax1.transAxes, verticalalignment="top",
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax1.legend(loc="lower right", framealpha=0.9)
    
    # Panel (b): U_cal vs Bootstrap SD
    ax2.scatter(u_cal, sd, color="#d62728", alpha=0.55, edgecolors="none", s=22, label="Test instances")
    u_grid = np.linspace(u_cal.min(), u_cal.max(), 100)
    ax2.plot(u_grid, reg_u.intercept + reg_u.slope * u_grid, color="#8b0000", lw=2.0,
             label=f"Linear trend")
    ax2.set_xlabel(r"Calibration instability index $U_{\mathrm{cal}} = \sqrt{\hat{p}(1-\hat{p})w}$")
    ax2.set_title(r"(b) $U_{\mathrm{cal}}$ index vs. instability")
    
    # Info box for panel (b)
    text_b = (f"Pearson $r = {r_u:.4f}$\n"
              f"Spearman $\\rho = {rho_u:.4f}$\n"
              f"Held-out MAE $= {mae_u:.4f}$")
    ax2.text(0.05, 0.95, text_b, transform=ax2.transAxes, verticalalignment="top",
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax2.legend(loc="lower right", framealpha=0.9)
    
    plt.tight_layout()
    
    paper_dir = os.path.join(project_root, "paper")
    out_png1 = os.path.join(paper_dir, "width_vs_bootstrap_instability.png")
    out_pdf1 = os.path.join(paper_dir, "width_vs_bootstrap_instability.pdf")
    out_png2 = os.path.join(project_root, "width_vs_bootstrap_instability.png")
    
    plt.savefig(out_png1, dpi=300)
    plt.savefig(out_pdf1)
    plt.savefig(out_png2, dpi=300)
    plt.close()
    
    print(f"Figure successfully generated and saved to:")
    print(f"  - {out_png1}")
    print(f"  - {out_pdf1}")
    print(f"  - {out_png2}")
    
    return {
        "r_w": r_w, "rho_w": rho_w, "mae_w": mae_w,
        "r_u": r_u, "rho_u": rho_u, "mae_u": mae_u,
    }

if __name__ == "__main__":
    generate_figure()
