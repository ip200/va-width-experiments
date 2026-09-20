"""
src/plot_figures_6_7.py

Generates publication-quality Figures 6 and 7 from canonical data:
  - Figure 6: Controlled local calibration support thinning (UCI Adult, Bank, Spambase)
  - Figure 7: Reverse training support intervention (Base-model uncertainty vs width)

Features:
  - North-East legends with generous y-axis headroom to eliminate any overlap with curve data.
  - Generates both high-resolution PNG (300 DPI) and vector PDF assets.
  - Reads directly from canonical data/ directory relative to project root.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "lines.linewidth": 1.6,
    "lines.markersize": 5.5,
})


def plot_figure_6(df_thinning, paper_dir):
    """
    Plots Figure 6: Controlled local calibration support thinning.
    Panel (a): Mean Venn--Abers width vs. retained local support.
    Panel (b): Calibration bootstrap SD vs. retained local support.
    Panel (c): Base-model epistemic SD (invariant by construction).
    """
    print("Plotting Figure 6: Real-Data Local Support Thinning...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.0, 3.8))
    
    palette = {"adult": "#1f77b4", "bank": "#2ca02c", "spambase": "#d62728"}
    markers = {"adult": "o", "bank": "s", "spambase": "^"}
    ds_labels = {"adult": "UCI Adult", "bank": "UCI Bank", "spambase": "UCI Spambase"}
    datasets = ["adult", "bank", "spambase"]

    for ds in datasets:
        sub = df_thinning[df_thinning["dataset"] == ds]
        mean_by_frac = sub.groupby("retained_fraction").agg({
            "mean_width": "mean",
            "mean_sigma_cal": "mean",
            "e_model": "mean",
        }).reset_index().sort_values("retained_fraction")

        fracs = mean_by_frac["retained_fraction"].values * 100.0

        # Panel A: Width vs retained fraction
        ax1.plot(fracs, mean_by_frac["mean_width"], f"{markers[ds]}-", color=palette[ds],
                 lw=1.8, label=ds_labels[ds])
        
        # Panel B: Calibration Bootstrap SD vs retained fraction
        ax2.plot(fracs, mean_by_frac["mean_sigma_cal"], f"{markers[ds]}-", color=palette[ds],
                 lw=1.8, label=ds_labels[ds])

        # Panel C: Model Epistemic SD vs retained fraction
        ax3.plot(fracs, mean_by_frac["e_model"], f"{markers[ds]}--", color=palette[ds],
                 lw=1.8, label=ds_labels[ds])

    # Format Panel A - Headroom for North-East legend
    ax1.set_xlabel("Retained local calibration support (%)")
    ax1.set_ylabel(r"Mean Venn--Abers width $\bar{w}$")
    ax1.set_title(r"(a) Calibration width $w$ vs. local support")
    ax1.invert_xaxis()
    ax1.set_ylim(-0.005, 0.27)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Format Panel B - Headroom for North-East legend
    ax2.set_xlabel("Retained local calibration support (%)")
    ax2.set_ylabel(r"Calibration bootstrap SD $\sigma_{\mathrm{cal}}$")
    ax2.set_title(r"(b) Calibration instability vs. local support")
    ax2.invert_xaxis()
    ax2.set_ylim(0.015, 0.12)
    ax2.legend(loc="upper right", framealpha=0.9)

    # Format Panel C
    ax3.set_xlabel("Retained local calibration support (%)")
    ax3.set_ylabel(r"Base model epistemic SD $E_{\mathrm{model}}$")
    ax3.set_title(r"(c) Base-model uncertainty (invariant)")
    ax3.invert_xaxis()
    ax3.set_ylim(0.035, 0.13)
    ax3.legend(loc="center right", framealpha=0.9)

    plt.tight_layout()
    png_path = os.path.join(paper_dir, "real_data_local_support.png")
    pdf_path = os.path.join(paper_dir, "real_data_local_support.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  Saved Figure 6 to {png_path} and {pdf_path}")


def plot_figure_7(df_reverse, paper_dir):
    """
    Plots Figure 7: Reverse training support intervention.
    Panel (a): Base model epistemic SD vs. retained training support.
    Panel (b): Calibration width under fixed calibration support.
    """
    print("Plotting Figure 7: Reverse Training Intervention...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    
    palette = {"adult": "#1f77b4", "bank": "#2ca02c", "spambase": "#d62728"}
    markers = {"adult": "o", "bank": "s", "spambase": "^"}
    ds_labels = {"adult": "UCI Adult", "bank": "UCI Bank", "spambase": "UCI Spambase"}
    datasets = ["adult", "bank", "spambase"]

    for ds in datasets:
        sub_rev = df_reverse[df_reverse["dataset"] == ds]
        rev_agg = sub_rev.groupby("train_fraction").agg({
            "e_model": "mean",
            "mean_width": "mean",
        }).reset_index().sort_values("train_fraction")

        tr_pct = rev_agg["train_fraction"].values * 100.0
        ax1.plot(tr_pct, rev_agg["e_model"], f"{markers[ds]}-", color=palette[ds],
                 lw=1.8, label=ds_labels[ds])
        ax2.plot(tr_pct, rev_agg["mean_width"], f"{markers[ds]}--", color=palette[ds],
                 lw=1.8, label=ds_labels[ds])

    # Format Panel A - Headroom for North-East legend
    ax1.set_xlabel("Retained training support (%)")
    ax1.set_ylabel(r"Model epistemic SD $E_{\mathrm{model}}$")
    ax1.set_title(r"(a) Base model epistemic SD vs. training support")
    ax1.invert_xaxis()
    ax1.set_ylim(-0.005, 0.185)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Format Panel B - North-East legend
    ax2.set_xlabel("Retained training support (%)")
    ax2.set_ylabel(r"Mean Venn--Abers width $\bar{w}$")
    ax2.set_title(r"(b) Calibration width under fixed cal support")
    ax2.invert_xaxis()
    ax2.set_ylim(0.0, 0.075)
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    png_path = os.path.join(paper_dir, "training_support_epistemic.png")
    pdf_path = os.path.join(paper_dir, "training_support_epistemic.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  Saved Figure 7 to {png_path} and {pdf_path}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    paper_dir = os.path.join(root, "paper")
    os.makedirs(paper_dir, exist_ok=True)

    thin_csv = os.path.join(data_dir, "REAL_DATA_RESULTS.csv")
    df_thinning = pd.read_csv(thin_csv)
    plot_figure_6(df_thinning, paper_dir)

    rev_csv = os.path.join(data_dir, "REVERSE_INTERVENTION_RESULTS.csv")
    df_reverse = pd.read_csv(rev_csv)
    plot_figure_7(df_reverse, paper_dir)


if __name__ == "__main__":
    main()
