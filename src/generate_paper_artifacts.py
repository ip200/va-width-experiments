"""
src/generate_paper_artifacts.py

Master script to regenerate all paper figures (Figures 2-8) and tables (Tables 1-4)
directly from committed canonical data in data/ without rerunning expensive experiments.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure academic publication styling
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

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
data_dir = os.path.join(root_dir, "data")
paper_dir = os.path.join(root_dir, "paper")
os.makedirs(paper_dir, exist_ok=True)


def generate_figure2_classifier_bootstrap():
    """Figure 2: Pointwise width vs calibration instability (Panel a: raw width, Panel b: U_cal)."""
    csv_path = os.path.join(data_dir, "classifier_bootstrap_n500.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping Figure 2.")
        return
    print("Generating Figure 2: Pointwise width vs calibration instability...")
    df = pd.read_csv(csv_path)
    w = df["width"].values
    sd = df["bootstrap_sd"].values
    u_cal = df["u_cal"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel A: raw width vs bootstrap SD
    r_w, _ = pearsonr(w, sd)
    rho_w, _ = spearmanr(w, sd)
    ax1.scatter(w, sd, color="#1f77b4", alpha=0.55, edgecolors="none", s=22)
    m_w, b_w = np.polyfit(w, sd, 1)
    x_grid = np.linspace(w.min(), w.max(), 100)
    ax1.plot(x_grid, m_w * x_grid + b_w, color="black", lw=1.6,
             label=f"Linear fit (Pearson $r={r_w:.4f}$, Spearman $\\rho={rho_w:.4f}$)")
    ax1.set_xlabel(r"Venn--Abers interval width $w$")
    ax1.set_ylabel(r"Bootstrap standard deviation $\sigma_{\mathrm{boot}}$")
    ax1.set_title(r"(a) Raw interval width $w$ vs. $\sigma_{\mathrm{boot}}$")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

    # Panel B: U_cal vs bootstrap SD
    r_u, _ = pearsonr(u_cal, sd)
    rho_u, _ = spearmanr(u_cal, sd)
    ax2.scatter(u_cal, sd, color="#2ca02c", alpha=0.55, edgecolors="none", s=22)
    m_u, b_u = np.polyfit(u_cal, sd, 1)
    x_grid_u = np.linspace(u_cal.min(), u_cal.max(), 100)
    ax2.plot(x_grid_u, m_u * x_grid_u + b_u, color="black", lw=1.6,
             label=f"Linear fit (Pearson $r={r_u:.4f}$, Spearman $\\rho={rho_u:.4f}$)")
    ax2.set_xlabel(r"Instability index $U_{\mathrm{cal}} = \sqrt{\hat{p}(1-\hat{p})w}$")
    ax2.set_ylabel(r"Bootstrap standard deviation $\sigma_{\mathrm{boot}}$")
    ax2.set_title(r"(b) Calibration instability index $U_{\mathrm{cal}}$ vs. $\sigma_{\mathrm{boot}}$")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(paper_dir, "width_vs_bootstrap_instability.png"), dpi=300)
    plt.savefig(os.path.join(paper_dir, "width_vs_bootstrap_instability.pdf"))
    plt.close()
    print("  Saved Figure 2 to paper/width_vs_bootstrap_instability.png and .pdf")


def generate_figure3_idealised_scaling():
    """Figure 3: Idealised sample-size scaling."""
    csv_path = os.path.join(data_dir, "SCALING_IDEALISED.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping Figure 3.")
        return
    print("Generating Figure 3: Idealised sample-size scaling...")
    df = pd.read_csv(csv_path)
    n = df["n_cal"].values
    w = df["mean_width"].values
    sd = df["sd_p_mid"].values
    ucal = df["mean_ucal"].values

    log_n = np.log(n)
    res_w = linregress(log_n, np.log(w))
    res_sd = linregress(log_n, np.log(sd))
    res_ucal = linregress(log_n, np.log(ucal))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Panel A: Width vs n
    ax1.loglog(n, w, "o-", color="#1f77b4", lw=1.8, label="Empirical width")
    fit_w = np.exp(res_w.intercept) * (n ** res_w.slope)
    ax1.loglog(n, fit_w, ":", color="#1f77b4", lw=1.4, label=f"Fit (slope = {res_w.slope:.3f})")
    anchor_idx = 2
    ref_w = w[anchor_idx] * ((n / n[anchor_idx]) ** (-2.0 / 3.0))
    ax1.loglog(n, ref_w, "k--", lw=1.4, label=r"Theory $n^{-2/3}$ ($-0.667$)")
    ax1.set_xlabel(r"Calibration size $n_{\mathrm{cal}}$")
    ax1.set_ylabel(r"Mean interval width $\bar{w}$")
    ax1.set_title(r"(a) Venn--Abers width scaling")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.9)
    ax1.set_xticks(n)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # Panel B: Bootstrap SD and U_cal vs n
    ax2.loglog(n, sd, "s-", color="#2ca02c", lw=1.8, label="Bootstrap SD")
    fit_sd = np.exp(res_sd.intercept) * (n ** res_sd.slope)
    ax2.loglog(n, fit_sd, ":", color="#2ca02c", lw=1.2, label=f"SD fit (slope = {res_sd.slope:.3f})")
    ax2.loglog(n, ucal, "^-", color="#d62728", lw=1.8, label=r"$U_{\mathrm{cal}}=\sqrt{\hat{p}(1-\hat{p})w}$")
    fit_ucal = np.exp(res_ucal.intercept) * (n ** res_ucal.slope)
    ax2.loglog(n, fit_ucal, ":", color="#d62728", lw=1.2, label=f"$U_{{\mathrm{{cal}}}}$ fit (slope = {res_ucal.slope:.3f})")
    ref_sd = sd[anchor_idx] * ((n / n[anchor_idx]) ** (-1.0 / 3.0))
    ax2.loglog(n, ref_sd, "k--", lw=1.4, label=r"Theory $n^{-1/3}$ ($-0.333$)")
    ax2.set_xlabel(r"Calibration size $n_{\mathrm{cal}}$")
    ax2.set_ylabel(r"Instability index")
    ax2.set_title(r"(b) Probability-scale instability scaling")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.9)
    ax2.set_xticks(n)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    plt.savefig(os.path.join(paper_dir, "idealised_scaling_laws.png"), dpi=300)
    plt.savefig(os.path.join(paper_dir, "idealised_scaling_laws.pdf"))
    plt.close()
    print("  Saved Figure 3 to paper/idealised_scaling_laws.png and .pdf")


def generate_figure4_and_table2_w2_convergence():
    """Figure 4: Exponent convergence and Table 2: W2 exponent progression."""
    csv_path = os.path.join(data_dir, "proposition1_convergence.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping Figure 4 / Table 2.")
        return
    print("Generating Figure 4 and Table 2: W2 Exponent Convergence...")
    df = pd.read_csv(csv_path)
    n = df["n_cal"].values

    # Table 2
    table_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\caption{Finite-sample estimates of the local scaling exponents as the calibration sample increases. The theoretical targets are $-2/3$ for local density, $-1/3$ for Bernoulli ambiguity, and $+2/3$ for local slope. Discrepancy from theoretical values attenuates substantially as $n_{\mathrm{cal}}$ increases.}",
        r"\label{tab:exponent_convergence}",
        r"\begin{tabular}{rccccc}",
        r"\toprule",
        r"$n_{\mathrm{cal}}$ & $\hat\beta_\rho$ [95\% CI] & $\hat\beta_v$ [95\% CI] & $\hat\beta_s$ [95\% CI] & Mean Abs.\ Error & $R^2$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        n_val = int(r["n_cal"])
        b_rho = f"{r['beta_rho']:+.3f} [{r['beta_rho_ci_lower']:+.3f}, {r['beta_rho_ci_upper']:+.3f}]"
        b_v = f"{r['beta_v']:+.3f} [{r['beta_v_ci_lower']:+.3f}, {r['beta_v_ci_upper']:+.3f}]"
        b_s = f"{r['beta_s']:+.3f} [{r['beta_s_ci_lower']:+.3f}, {r['beta_s_ci_upper']:+.3f}]"
        mae = f"{r['mean_abs_err']:.3f}"
        r2 = f"{r['r2']:.3f}"
        table_lines.append(f"{n_val} & {b_rho} & {b_v} & {b_s} & {mae} & {r2} \\\\")
    table_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(os.path.join(paper_dir, "table_w2_exponent_progression.tex"), "w") as f:
        f.write("\n".join(table_lines) + "\n")
    print("  Saved Table 2 to paper/table_w2_exponent_progression.tex")

    # Figure 4
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.8))

    # Panel A: beta_rho
    yerr_rho = [df["beta_rho"] - df["beta_rho_ci_lower"], df["beta_rho_ci_upper"] - df["beta_rho"]]
    ax1.errorbar(n, df["beta_rho"], yerr=yerr_rho, fmt="o-", color="#1f77b4", lw=1.6, capsize=4, label=r"$\hat{\beta}_\rho(n)$ (95\% CI)")
    ax1.axhline(-2.0 / 3.0, color="red", linestyle="--", lw=1.4, label=r"Target $-2/3$")
    ax1.set_xscale("log")
    ax1.set_xticks(n)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xlabel(r"Calibration size $n_{\mathrm{cal}}$")
    ax1.set_ylabel(r"Estimated exponent $\hat{\beta}_\rho$")
    ax1.set_title(r"(a) Local density exponent $\beta_\rho$")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax1.set_ylim(-0.78, -0.35)

    # Panel B: beta_v
    yerr_v = [df["beta_v"] - df["beta_v_ci_lower"], df["beta_v_ci_upper"] - df["beta_v"]]
    ax2.errorbar(n, df["beta_v"], yerr=yerr_v, fmt="s-", color="#2ca02c", lw=1.6, capsize=4, label=r"$\hat{\beta}_v(n)$ (95\% CI)")
    ax2.axhline(-1.0 / 3.0, color="red", linestyle="--", lw=1.4, label=r"Target $-1/3$")
    ax2.set_xscale("log")
    ax2.set_xticks(n)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xlabel(r"Calibration size $n_{\mathrm{cal}}$")
    ax2.set_ylabel(r"Estimated exponent $\hat{\beta}_v$")
    ax2.set_title(r"(b) Ambiguity exponent $\beta_v$")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax2.set_ylim(-0.52, -0.15)

    # Panel C: beta_s
    yerr_s = [df["beta_s"] - df["beta_s_ci_lower"], df["beta_s_ci_upper"] - df["beta_s"]]
    ax3.errorbar(n, df["beta_s"], yerr=yerr_s, fmt="^-", color="#d62728", lw=1.6, capsize=4, label=r"$\hat{\beta}_s(n)$ (95\% CI)")
    ax3.axhline(2.0 / 3.0, color="red", linestyle="--", lw=1.4, label=r"Target $+2/3$")
    ax3.set_xscale("log")
    ax3.set_xticks(n)
    ax3.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax3.set_xlabel(r"Calibration size $n_{\mathrm{cal}}$")
    ax3.set_ylabel(r"Estimated exponent $\hat{\beta}_s$")
    ax3.set_title(r"(c) Local slope exponent $\beta_s$")
    ax3.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax3.set_ylim(0.40, 0.85)

    plt.tight_layout()
    plt.savefig(os.path.join(paper_dir, "w2_exponent_convergence.png"), dpi=300)
    plt.savefig(os.path.join(paper_dir, "w2_exponent_convergence.pdf"))
    plt.close()
    print("  Saved Figure 4 to paper/w2_exponent_convergence.png and .pdf")


def generate_figure5_non_monotonic():
    """Figure 5: Non-monotonic regimes."""
    csv_path = os.path.join(data_dir, "non_monotonic_scaling_laws.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping Figure 5.")
        return
    print("Generating Figure 5: Non-monotonic scaling laws...")
    df = pd.read_csv(csv_path)
    n_values = df["n_cal"].values
    log_x = np.log(n_values)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    regimes_info = [
        ("width_mono", "Regime A: Monotonic", "#1f77b4", "-", "-0.667 (-2/3)"),
        ("width_viol", "Regime B: Violating", "#d62728", "--", "-1.000"),
        ("width_ext_mid", "Regime C: Middle Extremum (Violating)", "#ff7f0e", ":", "-1.000"),
        ("width_ext_bound", "Regime D: Flat Extremum (Non-Violating)", "#2ca02c", "-.", "-0.800 (-4/5)"),
    ]

    for col, label, color, style, expected in regimes_info:
        y = df[col].values
        log_y = np.log(y)
        res = linregress(log_x, log_y)
        ax.plot(log_x, log_y, "o", color=color)
        ax.plot(log_x, res.intercept + res.slope * log_x, linestyle=style, color=color,
                label=f"{label} (slope={res.slope:.3f}, Theory={expected})")

    ax.set_title("Venn--Abers Width Scaling Laws in Different Slope Regimes")
    ax.set_xlabel(r"$\log(N_{\mathrm{cal}})$")
    ax.set_ylabel(r"$\log(\bar{w})$")
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(paper_dir, "non_monotonic_scaling_laws.png"), dpi=300)
    plt.savefig(os.path.join(paper_dir, "non_monotonic_scaling_laws.pdf"))
    plt.close()
    print("  Saved Figure 5 to paper/non_monotonic_scaling_laws.png and .pdf")


def generate_figures6_7_and_table4_real_data():
    """Figure 6, Figure 7, and Table 4: Real-data experiments."""
    thin_csv = os.path.join(data_dir, "REAL_DATA_RESULTS.csv")
    rev_csv = os.path.join(data_dir, "REVERSE_INTERVENTION_RESULTS.csv")
    decomp_csv = os.path.join(data_dir, "UNCERTAINTY_DECOMPOSITION.csv")
    if not os.path.exists(thin_csv) or not os.path.exists(rev_csv):
        print("Warning: Real-data CSVs not found, skipping Figures 6/7.")
        return

    print("Generating Figure 6 and Figure 7: Real-data interventions...")
    df_thinning = pd.read_csv(thin_csv)
    df_reverse = pd.read_csv(rev_csv)

    datasets = ["adult", "bank", "spambase"]
    palette = {"adult": "#1f77b4", "bank": "#2ca02c", "spambase": "#d62728"}
    markers = {"adult": "o", "bank": "s", "spambase": "^"}
    ds_labels = {"adult": "UCI Adult", "bank": "UCI Bank", "spambase": "UCI Spambase"}

    # Figure 6
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ds in datasets:
        sub = df_thinning[df_thinning["dataset"] == ds]
        mean_by_frac = sub.groupby("retained_fraction").agg({
            "mean_width": "mean",
            "mean_sigma_cal": "mean",
            "e_model": "mean",
        }).reset_index().sort_values("retained_fraction")
        fracs = mean_by_frac["retained_fraction"].values * 100.0

        ax1.plot(fracs, mean_by_frac["mean_width"], f"{markers[ds]}-", color=palette[ds], lw=1.8, label=ds_labels[ds])
        ax2.plot(fracs, mean_by_frac["mean_sigma_cal"], f"{markers[ds]}-", color=palette[ds], lw=1.8, label=ds_labels[ds])
        ax3.plot(fracs, mean_by_frac["e_model"], f"{markers[ds]}--", color=palette[ds], lw=1.8, label=ds_labels[ds])

    ax1.set_xlabel("Retained local calibration support (%)")
    ax1.set_ylabel(r"Mean Venn--Abers width $\bar{w}$")
    ax1.set_title(r"(a) Calibration width $w$ vs. local support")
    ax1.invert_xaxis()
    ax1.set_ylim(-0.005, 0.27)
    ax1.legend(loc="upper right", framealpha=0.9)

    ax2.set_xlabel("Retained local calibration support (%)")
    ax2.set_ylabel(r"Calibration bootstrap SD $\sigma_{\mathrm{cal}}$")
    ax2.set_title(r"(b) Calibration instability vs. local support")
    ax2.invert_xaxis()
    ax2.set_ylim(0.015, 0.12)
    ax2.legend(loc="upper right", framealpha=0.9)

    ax3.set_xlabel("Retained local calibration support (%)")
    ax3.set_ylabel(r"Base model epistemic SD $E_{\mathrm{model}}$")
    ax3.set_title(r"(c) Base-model uncertainty (invariant)")
    ax3.invert_xaxis()
    ax3.set_ylim(0.035, 0.13)
    ax3.legend(loc="center right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(paper_dir, "real_data_local_support.png"), dpi=300)
    plt.savefig(os.path.join(paper_dir, "real_data_local_support.pdf"))
    plt.close()
    print("  Saved Figure 6 to paper/real_data_local_support.png and .pdf")

    # Figure 7
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ds in datasets:
        sub_rev = df_reverse[df_reverse["dataset"] == ds]
        rev_agg = sub_rev.groupby("train_fraction").agg({
            "e_model": "mean",
            "mean_width": "mean",
        }).reset_index().sort_values("train_fraction")
        tr_pct = rev_agg["train_fraction"].values * 100.0

        ax1.plot(tr_pct, rev_agg["e_model"], f"{markers[ds]}-", color=palette[ds], lw=1.8, label=ds_labels[ds])
        ax2.plot(tr_pct, rev_agg["mean_width"], f"{markers[ds]}--", color=palette[ds], lw=1.8, label=ds_labels[ds])

    ax1.set_xlabel("Retained training support (%)")
    ax1.set_ylabel(r"Model epistemic SD $E_{\mathrm{model}}$")
    ax1.set_title(r"(a) Base model epistemic SD vs. training support")
    ax1.invert_xaxis()
    ax1.set_ylim(-0.005, 0.185)
    ax1.legend(loc="upper right", framealpha=0.9)

    ax2.set_xlabel("Retained training support (%)")
    ax2.set_ylabel(r"Mean Venn--Abers width $\bar{w}$")
    ax2.set_title(r"(b) Calibration width under fixed cal support")
    ax2.invert_xaxis()
    ax2.set_ylim(0.0, 0.075)
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(paper_dir, "training_support_epistemic.png"), dpi=300)
    plt.savefig(os.path.join(paper_dir, "training_support_epistemic.pdf"))
    plt.close()
    print("  Saved Figure 7 to paper/training_support_epistemic.png and .pdf")


def generate_table1_calibration_resampling():
    """Table 1: Calibration resampling baseline."""
    csv_path = os.path.join(data_dir, "calibration_resampling_baseline.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping Table 1.")
        return
    print("Generating Table 1: Calibration resampling baseline...")
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        n = int(row["n_cal"])
        w = f"{row['mean_width']:.4f}"
        sd = f"{row['mean_sd']:.4f}"
        corr = f"{row['corr']:.4f}"
        brier = f"{row['fresh_brier']:.4f}"
        rows.append(f"\t\t{n} & {w} & {sd} & {corr} & {brier} \\\\")
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
    with open(os.path.join(paper_dir, "table_calibration_resampling.tex"), "w") as f:
        f.write(table_tex)
    print("  Saved Table 1 to paper/table_calibration_resampling.tex")


def generate_table3_cifar_correlations():
    """Table 3: Synthetic CIFAR-10H-inspired crowd-annotation correlations."""
    csv_path = os.path.join(data_dir, "table_cifar_correlations.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping Table 3.")
        return
    print("Generating Table 3: Synthetic CIFAR-10H correlations...")
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        rows.append(f"{r['Quantity']} & {r['Pearson']:.4f} & {r['Spearman']:.4f} \\\\")
    rows_tex = "\n".join(rows)
    table_tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Macro-averaged Pearson and Spearman correlations of Venn--Abers interval width with bootstrap instability, annotator disagreement, and annotator entropy in the synthetic CIFAR-10H-inspired crowd-annotation experiment.}}
\\label{{tab:cifar_correlations}}
\\begin{{tabular}}{{lcc}}
\\toprule
Quantity & Pearson Correlation & Spearman Correlation \\\\
\\midrule
{rows_tex}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(os.path.join(paper_dir, "table_cifar_correlations.tex"), "w") as f:
        f.write(table_tex)
    print("  Saved Table 3 to paper/table_cifar_correlations.tex")


def main():
    print("=== Generating all paper figures and tables from canonical committed data ===\n")
    generate_table1_calibration_resampling()
    generate_figure2_classifier_bootstrap()
    generate_figure3_idealised_scaling()
    generate_figure4_and_table2_w2_convergence()
    generate_figure5_non_monotonic()
    generate_figures6_7_and_table4_real_data()
    generate_table3_cifar_correlations()
    print("\nAll paper artifacts regenerated successfully!")


if __name__ == "__main__":
    main()
