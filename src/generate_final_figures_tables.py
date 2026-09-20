"""
src/generate_final_figures_tables.py

Generates the exact final figures and tables for:
arxiv_va_interval_width_ivan_style_final_width.tex

Deliverables:
1. Figure 1: idealised_scaling_laws.png (and .pdf)
2. Table 1:  table_w2_exponent_progression.tex
3. Figure 2: w2_exponent_convergence.png (and .pdf)
4. Table 2:  table_w1_real_agent_nested.tex
5. Figure 3: w1_real_agent_nested.png (and .pdf)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure Matplotlib for academic publication style
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
    "lines.markersize": 6,
})

def generate_figure1_idealised_scaling():
    print("Generating Figure 1: Idealised Scaling Laws...")
    df = pd.read_csv("SCALING_IDEALISED.csv")
    n = df["n_cal"].values
    w = df["mean_width"].values
    sd = df["sd_p_mid"].values
    ucal = df["mean_ucal"].values
    
    log_n = np.log(n)
    res_w = linregress(log_n, np.log(w))
    res_sd = linregress(log_n, np.log(sd))
    res_ucal = linregress(log_n, np.log(ucal))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    
    # Panel (a): Width vs n
    ax1.loglog(n, w, 'o-', color='#1f77b4', lw=1.8, label='Empirical width')
    fit_w = np.exp(res_w.intercept) * (n ** res_w.slope)
    ax1.loglog(n, fit_w, ':', color='#1f77b4', lw=1.4, label=f'Fit (slope = {res_w.slope:.3f})')
    anchor_idx = 2
    ref_w = w[anchor_idx] * ((n / n[anchor_idx]) ** (-2.0 / 3.0))
    ax1.loglog(n, ref_w, 'k--', lw=1.4, label=r'Theory $n^{-2/3}$ ($-0.667$)')
    
    ax1.set_xlabel(r'Calibration size $n_{\mathrm{cal}}$')
    ax1.set_ylabel(r'Mean interval width $\bar{w}$')
    ax1.set_title(r'(a) Venn--Abers width scaling')
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9)
    ax1.set_xticks(n)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    
    # Panel (b): Bootstrap SD and U_cal vs n
    ax2.loglog(n, sd, 's-', color='#2ca02c', lw=1.8, label='Bootstrap SD')
    fit_sd = np.exp(res_sd.intercept) * (n ** res_sd.slope)
    ax2.loglog(n, fit_sd, ':', color='#2ca02c', lw=1.2, label=f'SD fit (slope = {res_sd.slope:.3f})')
    
    ax2.loglog(n, ucal, '^-', color='#d62728', lw=1.8, label=r'$U_{\mathrm{cal}}=\sqrt{\hat{p}(1-\hat{p})w}$')
    fit_ucal = np.exp(res_ucal.intercept) * (n ** res_ucal.slope)
    ax2.loglog(n, fit_ucal, ':', color='#d62728', lw=1.2, label=f'$U_{{\\mathrm{{cal}}}}$ fit (slope = {res_ucal.slope:.3f})')
    
    ref_sd = sd[anchor_idx] * ((n / n[anchor_idx]) ** (-1.0 / 3.0))
    ax2.loglog(n, ref_sd, 'k--', lw=1.4, label=r'Theory $n^{-1/3}$ ($-0.333$)')
    
    ax2.set_xlabel(r'Calibration size $n_{\mathrm{cal}}$')
    ax2.set_ylabel(r'Instability index')
    ax2.set_title(r'(b) Probability-scale instability scaling')
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9)
    ax2.set_xticks(n)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig("idealised_scaling_laws.png", dpi=300)
    plt.savefig("idealised_scaling_laws.pdf")
    plt.savefig("paper/idealised_scaling_laws.png", dpi=300)
    plt.savefig("paper/idealised_scaling_laws.pdf")
    plt.close()
    print("Figure 1 saved to idealised_scaling_laws.png and paper/idealised_scaling_laws.png")

def generate_table1_w2_exponent_progression():
    print("Generating Table 1: W2 Exponent Progression...")
    table_tex = r"""\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{2.5pt}
\caption{Finite-sample estimates of the local scaling exponents as the calibration sample increases. The theoretical targets are $-2/3$ for local density, $-1/3$ for Bernoulli ambiguity, and $+2/3$ for local slope. The discrepancy from the theoretical values falls substantially as $n_{\mathrm{cal}}$ increases, with some sampling variability at the largest calibration size.}
\label{tab:exponent_convergence}
\begin{tabular}{rccccc}
\toprule
$n_{\mathrm{cal}}$ & $\hat\beta_\rho$ [95\% CI] & $\hat\beta_v$ [95\% CI] & $\hat\beta_s$ [95\% CI] & Mean Abs.\ Error & $R^2$ \\
\midrule
500   & $-0.425$ [$-0.453, -0.398$] & $-0.129$ [$-0.178, -0.081$] & $+0.390$ [$+0.344, +0.442$] & 0.241 & 0.319 \\
1000  & $-0.523$ [$-0.551, -0.499$] & $-0.243$ [$-0.291, -0.194$] & $+0.536$ [$+0.483, +0.590$] & 0.121 & 0.406 \\
2000  & $-0.539$ [$-0.570, -0.508$] & $-0.237$ [$-0.288, -0.185$] & $+0.554$ [$+0.497, +0.609$] & 0.112 & 0.433 \\
4000  & $-0.602$ [$-0.630, -0.573$] & $-0.313$ [$-0.366, -0.267$] & $+0.607$ [$+0.553, +0.669$] & 0.048 & 0.492 \\
8000  & $-0.657$ [$-0.693, -0.620$] & $-0.296$ [$-0.368, -0.226$] & $+0.589$ [$+0.512, +0.672$] & 0.042 & 0.490 \\
16000 & $-0.660$ [$-0.693, -0.625$] & $-0.388$ [$-0.470, -0.309$] & $+0.690$ [$+0.592, +0.789$] & 0.028 & 0.537 \\
32000 & $-0.615$ [$-0.670, -0.566$] & $-0.303$ [$-0.404, -0.196$] & $+0.614$ [$+0.480, +0.745$] & 0.045 & 0.479 \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open("table_w2_exponent_progression.tex", "w") as f:
        f.write(table_tex)
    with open("paper/table_w2_exponent_progression.tex", "w") as f:
        f.write(table_tex)
    print("Table 1 saved to table_w2_exponent_progression.tex and paper/table_w2_exponent_progression.tex")

def generate_figure2_w2_exponent_convergence():
    print("Generating Figure 2: W2 Exponent Convergence...")
    df = pd.read_csv("proposition1_convergence.csv")
    n = df["n_cal"].values
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.8))
    
    # Panel (a): beta_rho
    yerr_rho = [df["beta_rho"] - df["beta_rho_ci_lower"], df["beta_rho_ci_upper"] - df["beta_rho"]]
    ax1.errorbar(n, df["beta_rho"], yerr=yerr_rho, fmt='o-', color='#1f77b4', lw=1.6, capsize=4, label=r'$\hat{\beta}_\rho(n)$ (95\% CI)')
    ax1.axhline(-2.0/3.0, color='red', linestyle='--', lw=1.4, label=r'Target $-2/3$')
    ax1.set_xscale('log')
    ax1.set_xticks(n)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xlabel(r'Calibration size $n_{\mathrm{cal}}$')
    ax1.set_ylabel(r'Estimated exponent $\hat{\beta}_\rho$')
    ax1.set_title(r'(a) Local density exponent $\beta_\rho$')
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax1.set_ylim(-0.75, -0.35)
    
    # Panel (b): beta_v
    yerr_v = [df["beta_v"] - df["beta_v_ci_lower"], df["beta_v_ci_upper"] - df["beta_v"]]
    ax2.errorbar(n, df["beta_v"], yerr=yerr_v, fmt='s-', color='#2ca02c', lw=1.6, capsize=4, label=r'$\hat{\beta}_v(n)$ (95\% CI)')
    ax2.axhline(-1.0/3.0, color='red', linestyle='--', lw=1.4, label=r'Target $-1/3$')
    ax2.set_xscale('log')
    ax2.set_xticks(n)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xlabel(r'Calibration size $n_{\mathrm{cal}}$')
    ax2.set_ylabel(r'Estimated exponent $\hat{\beta}_v$')
    ax2.set_title(r'(b) Ambiguity exponent $\beta_v$')
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax2.set_ylim(-0.52, -0.05)
    
    # Panel (c): beta_s
    yerr_s = [df["beta_s"] - df["beta_s_ci_lower"], df["beta_s_ci_upper"] - df["beta_s"]]
    ax3.errorbar(n, df["beta_s"], yerr=yerr_s, fmt='^-', color='#d62728', lw=1.6, capsize=4, label=r'$\hat{\beta}_s(n)$ (95\% CI)')
    ax3.axhline(2.0/3.0, color='red', linestyle='--', lw=1.4, label=r'Target $+2/3$')
    ax3.set_xscale('log')
    ax3.set_xticks(n)
    ax3.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax3.set_xlabel(r'Calibration size $n_{\mathrm{cal}}$')
    ax3.set_ylabel(r'Estimated exponent $\hat{\beta}_s$')
    ax3.set_title(r'(c) Local slope exponent $\beta_s$')
    ax3.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax3.set_ylim(0.30, 0.85)
    
    plt.tight_layout()
    plt.savefig("w2_exponent_convergence.png", dpi=300)
    plt.savefig("w2_exponent_convergence.pdf")
    plt.savefig("paper/w2_exponent_convergence.png", dpi=300)
    plt.savefig("paper/w2_exponent_convergence.pdf")
    plt.close()
    print("Figure 2 saved to w2_exponent_convergence.png and paper/w2_exponent_convergence.png")

def generate_table2_w1_real_agent_nested():
    print("Generating Table 2: W1 Real Agent Nested Decomposition...")
    table_tex = r"""\begin{table}[htbp]
\centering
\small
\caption{Held-out real-agent comparison of bootstrap calibration-instability models on 600 unique tasks across reasoning, question-answering, and tool-use benchmarks (GSM8K, MATH, HotpotQA, TriviaQA, BFCL). Adding Venn--Abers width provides a substantial additional explanatory contribution ($\Delta R^2 = +0.1119$, 95\% bootstrap CI $[0.0845, 0.1495]$). Differences are evaluated using 5,000 task-level bootstrap resamples.}
\label{tab:agent_nested}
\begin{tabular}{llccc}
\toprule
Model & Predictors & $R^2$ & MAE & RMSE \\
\midrule
Model A & Ambiguity $p(1-p)$ & 0.3902 & 0.0047 & 0.0057 \\
Model B & Ambiguity $p(1-p)$ + Width $w$ & 0.5021 & 0.0042 & 0.0051 \\
\midrule
$\Delta$ (B $-$ A) & Incremental gain / error reduction & $+0.1119$ & $+0.0005$ & $+0.0005$ \\
\multicolumn{2}{l}{95\% Bootstrap Confidence Interval} & [0.0845, 0.1495] & [0.0004, 0.0007] & [0.0004, 0.0007] \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open("table_w1_real_agent_nested.tex", "w") as f:
        f.write(table_tex)
    with open("paper/table_w1_real_agent_nested.tex", "w") as f:
        f.write(table_tex)
    print("Table 2 saved to table_w1_real_agent_nested.tex and paper/table_w1_real_agent_nested.tex")

def generate_figure3_w1_real_agent_nested():
    print("Generating Figure 3: W1 Real Agent Nested Comparison...")
    df_boot = pd.read_csv("width_real_agent_nested_bootstrap.csv")
    delta_r2 = df_boot["delta_r2"].values
    
    ci_lower = np.percentile(delta_r2, 2.5)
    ci_upper = np.percentile(delta_r2, 97.5)
    obs_delta = 0.1119
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
    
    # Panel (a): Model Performance (Bar plot from 0 to avoid exaggeration)
    models = ["Model A\n(Ambiguity only)", "Model B\n(Ambiguity + Width)"]
    r2_vals = [0.3902, 0.5021]
    bars = ax1.bar(models, r2_vals, width=0.45, color=['#7f7f7f', '#1f77b4'], edgecolor='black', lw=0.8, alpha=0.85)
    ax1.set_ylim(0, 0.7)
    ax1.set_ylabel(r'Held-out $R^2$')
    ax1.set_title(r'(a) Model explanatory power')
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'$R^2 = {height:.4f}$',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5)
    ax1.axhline(0.3902, color='gray', linestyle=':', lw=1.0)
    
    # Panel (b): Bootstrap distribution of Delta R2
    counts, bins, patches = ax2.hist(delta_r2, bins=45, density=True, color='#1f77b4', edgecolor='black', lw=0.4, alpha=0.6, label=r'Bootstrap $\Delta R^2$')
    
    ax2.axvline(0, color='red', linestyle='--', lw=1.5, label='Zero (No gain)')
    ax2.axvline(obs_delta, color='black', linestyle='-', lw=1.8, label=f'Observed $\\Delta R^2 = +{obs_delta:.4f}$')
    ax2.axvline(ci_lower, color='green', linestyle=':', lw=1.4, label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    ax2.axvline(ci_upper, color='green', linestyle=':', lw=1.4)
    
    ax2.set_xlabel(r'$\Delta R^2$ (Model B $-$ Model A)')
    ax2.set_ylabel(r'Bootstrap density')
    ax2.set_title(r'(b) Task-level bootstrap distribution ($B=5{,}000$)')
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig("w1_real_agent_nested.png", dpi=300)
    plt.savefig("w1_real_agent_nested.pdf")
    plt.savefig("paper/w1_real_agent_nested.png", dpi=300)
    plt.savefig("paper/w1_real_agent_nested.pdf")
    plt.close()
    print("Figure 3 saved to w1_real_agent_nested.png and paper/w1_real_agent_nested.png")

if __name__ == "__main__":
    generate_figure1_idealised_scaling()
    generate_table1_w2_exponent_progression()
    generate_figure2_w2_exponent_convergence()
    generate_table2_w1_real_agent_nested()
    generate_figure3_w1_real_agent_nested()
    print("\nAll figures and tables generated successfully.")
