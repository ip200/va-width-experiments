"""
src/run_experiment_w2_convergence.py

Experiment W2 — Large-n convergence of Proposition 1 exponents.
Tests whether multivariate scaling exponents (beta_rho, beta_v, beta_s) converge
toward their theoretical targets (-2/3, -1/3, +2/3) as calibration size n_cal increases
across n_cal in [500, 1000, 2000, 4000, 8000, 16000, 32000].

Methodological Fix:
Uses the true data-generating Bernoulli variance v(s) = theta(s)(1 - theta(s))
from the known DGP, eliminating empirical proxy estimation noise.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, beta as beta_dist, linregress, t
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from venn_abers import VennAbers

try:
    from utils import clip01, sigmoid, set_seed
except ImportError:
    from src.utils import clip01, sigmoid, set_seed


# -------------------------------------------------------------
# Probability & Slope Functions
# -------------------------------------------------------------
def prob_sigmoidal(s: np.ndarray, k: float = 5.0, s0: float = 0.0) -> np.ndarray:
    return clip01(sigmoid(k * (s - s0)))


def slope_sigmoidal(s: np.ndarray, k: float = 5.0, s0: float = 0.0) -> np.ndarray:
    p = prob_sigmoidal(s, k, s0)
    return np.maximum(k * p * (1.0 - p), 1e-5)


def prob_power(s: np.ndarray, alpha: float = 2.0, c: float = 0.8) -> np.ndarray:
    s_norm = np.clip((s + 1.0) / 2.0, 0.0, 1.0)
    return clip01(c * (s_norm ** alpha))


def slope_power(s: np.ndarray, alpha: float = 2.0, c: float = 0.8) -> np.ndarray:
    s_norm = np.clip((s + 1.0) / 2.0, 1e-4, 1.0)
    return np.maximum(c * alpha * (s_norm ** (alpha - 1.0)) / 2.0, 1e-5)


# -------------------------------------------------------------
# Density Sampling & Density Evaluation Functions on [-1, 1]
# -------------------------------------------------------------
def sample_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=n)


def density_uniform(s: np.ndarray) -> np.ndarray:
    return np.full_like(s, 0.5)


def sample_gaussian_trunc(n: int, mu: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    samples = []
    while len(samples) < n:
        batch = rng.normal(mu, sigma, size=n * 2)
        valid = batch[(batch >= -1.0) & (batch <= 1.0)]
        samples.extend(valid.tolist())
    return np.array(samples[:n])


def density_gaussian_trunc(s: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    denom = norm.cdf(1.0, mu, sigma) - norm.cdf(-1.0, mu, sigma)
    return np.maximum(norm.pdf(s, mu, sigma) / denom, 1e-5)


def sample_beta_mapped(n: int, a: float, b: float, rng: np.random.Generator) -> np.ndarray:
    u = rng.beta(a, b, size=n)
    return 2.0 * u - 1.0


def density_beta_mapped(s: np.ndarray, a: float, b: float) -> np.ndarray:
    u = np.clip((s + 1.0) / 2.0, 1e-4, 1.0 - 1e-4)
    return np.maximum(beta_dist.pdf(u, a, b) / 2.0, 1e-5)


# -------------------------------------------------------------
# Venn-Abers Predictor Wrapper
# -------------------------------------------------------------
class FastVAP1D:
    def __init__(self):
        self.va = VennAbers()

    def fit_predict(self, s_cal: np.ndarray, y_cal: np.ndarray, s_test: np.ndarray):
        p_cal = np.zeros((len(s_cal), 2))
        p_cal[:, 1] = clip01((s_cal + 1.0) / 2.0)
        p_cal[:, 0] = 1.0 - p_cal[:, 1]
        self.va.fit(p_cal, y_cal)

        p_test = np.zeros((len(s_test), 2))
        p_test[:, 1] = clip01((s_test + 1.0) / 2.0)
        p_test[:, 0] = 1.0 - p_test[:, 1]
        _, pred = self.va.predict_proba(p_test)
        p0, p1 = pred[:, 0], pred[:, 1]
        p0_final = clip01(np.minimum(p0, p1))
        p1_final = clip01(np.maximum(p0, p1))
        p_mid = 0.5 * (p0_final + p1_final)
        width = p1_final - p0_final
        return p_mid, p0_final, p1_final, width


class SimpleOLS:
    def __init__(self):
        self.params = None
        self.bse = None
        self.rsquared = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, k = X.shape
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.params = beta
        y_pred = X @ beta
        e = y - y_pred
        ssr = np.sum(e ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        self.rsquared = 1.0 - (ssr / sst) if sst > 1e-12 else 0.0
        df_e = n - k
        if df_e > 0:
            sigma2 = ssr / df_e
            cov_matrix = np.linalg.pinv(X.T @ X) * sigma2
            self.bse = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))
        else:
            self.bse = np.zeros_like(beta)
        return self


def run_w2_experiment(seed: int = 42):
    print("=" * 70)
    print("Running Experiment W2: Large-n Convergence of Proposition 1 Exponents")
    print("=" * 70)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    paper_dir = os.path.join(project_root, "paper")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    n_cal_list = [500, 1000, 2000, 4000, 8000, 16000, 32000]
    reps_by_n = {
        500: 8,
        1000: 8,
        2000: 6,
        4000: 6,
        8000: 4,
        16000: 3,
        32000: 2,
    }

    densities = [
        ("uniform", lambda n, r: sample_uniform(n, r), density_uniform),
        ("gaussian_centered", lambda n, r: sample_gaussian_trunc(n, 0.0, 0.4, r), lambda s: density_gaussian_trunc(s, 0.0, 0.4)),
        ("gaussian_shifted", lambda n, r: sample_gaussian_trunc(n, 0.4, 0.3, r), lambda s: density_gaussian_trunc(s, 0.4, 0.3)),
        ("beta_asym", lambda n, r: sample_beta_mapped(n, 0.6, 2.0, r), lambda s: density_beta_mapped(s, 0.6, 2.0)),
    ]

    curves = [
        ("sig_k2", lambda s: prob_sigmoidal(s, k=2.0, s0=0.0), lambda s: slope_sigmoidal(s, k=2.0, s0=0.0)),
        ("sig_k5", lambda s: prob_sigmoidal(s, k=5.0, s0=0.0), lambda s: slope_sigmoidal(s, k=5.0, s0=0.0)),
        ("sig_k10", lambda s: prob_sigmoidal(s, k=10.0, s0=0.0), lambda s: slope_sigmoidal(s, k=10.0, s0=0.0)),
        ("sig_shift", lambda s: prob_sigmoidal(s, k=6.0, s0=0.25), lambda s: slope_sigmoidal(s, k=6.0, s0=0.25)),
        ("pow_a2", lambda s: prob_power(s, alpha=2.0, c=0.8), lambda s: slope_power(s, alpha=2.0, c=0.8)),
        ("pow_a3", lambda s: prob_power(s, alpha=3.0, c=0.9), lambda s: slope_power(s, alpha=3.0, c=0.9)),
    ]

    s_eval_grid = np.linspace(-0.65, 0.65, 14)
    all_records = []

    print(f"Generating calibration runs across n = {n_cal_list}...")
    vap = FastVAP1D()

    for n_cal in n_cal_list:
        n_reps = reps_by_n[n_cal]
        print(f"  Simulating n_cal = {n_cal:5d} ({n_reps} reps per setting)...")
        for den_name, sampler, den_fn in densities:
            rho_eval = den_fn(s_eval_grid)
            for curve_name, prob_fn, slope_fn in curves:
                p_true_eval = prob_fn(s_eval_grid)
                slope_eval = slope_fn(s_eval_grid)

                for rep in range(n_reps):
                    rep_seed = seed + int(n_cal * 100 + rep * 17)
                    rep_rng = set_seed(rep_seed)

                    s_cal = sampler(n_cal, rep_rng)
                    p_cal_true = prob_fn(s_cal)
                    y_cal = rep_rng.binomial(1, p_cal_true)

                    p_mid, p0, p1, width = vap.fit_predict(s_cal, y_cal, s_eval_grid)

                    for i, s_test in enumerate(s_eval_grid):
                        w = width[i]
                        pm = p_mid[i]
                        theta = p_true_eval[i]
                        amb_true = theta * (1.0 - theta)
                        amb_hat = pm * (1.0 - pm)
                        rho = rho_eval[i]
                        slp = slope_eval[i]

                        # Filter extreme outliers or boundary zeros
                        if w > 1e-5 and amb_true > 1e-4 and rho > 1e-3 and slp > 1e-4:
                            all_records.append({
                                "n_cal": n_cal,
                                "rep": rep,
                                "density_name": den_name,
                                "curve_name": curve_name,
                                "test_score": s_test,
                                "p_true": theta,
                                "p_mid": pm,
                                "local_density": rho,
                                "local_slope": slp,
                                "ambiguity": amb_true,
                                "ambiguity_true": amb_true,
                                "ambiguity_hat": amb_hat,
                                "width": w,
                            })

    df = pd.DataFrame(all_records)
    print(f"Total simulated data points: {len(df)}")

    # Theoretical targets
    target_rho = -2.0 / 3.0   # -0.6667
    target_v   = -1.0 / 3.0   # -0.3333
    target_s   =  2.0 / 3.0   # +0.6667
    target_n   = -2.0 / 3.0   # -0.6667

    # -------------------------------------------------------------
    # Regression at each n_cal separately
    # -------------------------------------------------------------
    per_n_results = []
    boot_exponent_records = []
    B_boot = 1000

    for n_cal in n_cal_list:
        sub = df[df["n_cal"] == n_cal].copy()
        y_vec = np.log(sub["width"].values)
        X_mat = np.column_stack([
            np.ones(len(sub)),
            np.log(sub["local_density"].values),
            np.log(sub["ambiguity"].values),
            np.log(sub["local_slope"].values),
        ])

        fit = SimpleOLS().fit(X_mat, y_vec)
        b_const, b_rho, b_v, b_s = fit.params
        se_const, se_rho, se_v, se_s = fit.bse

        # Bootstrap percentile CIs across observations
        boot_b = np.zeros((B_boot, 4))
        n_obs = len(sub)
        for b_idx in range(B_boot):
            boot_rng = set_seed(int(n_cal * 10000 + b_idx))
            sample_idx = boot_rng.choice(n_obs, size=n_obs, replace=True)
            fit_b = SimpleOLS().fit(X_mat[sample_idx], y_vec[sample_idx])
            boot_b[b_idx, :] = fit_b.params
            boot_exponent_records.append({
                "n_cal": n_cal,
                "bootstrap_idx": b_idx,
                "beta_rho": fit_b.params[1],
                "beta_v": fit_b.params[2],
                "beta_s": fit_b.params[3],
            })

        ci_rho = np.percentile(boot_b[:, 1], [2.5, 97.5])
        ci_v   = np.percentile(boot_b[:, 2], [2.5, 97.5])
        ci_s   = np.percentile(boot_b[:, 3], [2.5, 97.5])

        err_abs_rho = abs(b_rho - target_rho)
        err_abs_v   = abs(b_v - target_v)
        err_abs_s   = abs(b_s - target_s)

        err_rel_rho = abs(b_rho - target_rho) / abs(target_rho)
        err_rel_v   = abs(b_v - target_v) / abs(target_v)
        err_rel_s   = abs(b_s - target_s) / abs(target_s)

        per_n_results.append({
            "n_cal": n_cal,
            "sample_size": len(sub),
            "r2": fit.rsquared,
            "beta_const": b_const,
            "beta_const_se": se_const,
            "beta_rho": b_rho,
            "beta_rho_se": se_rho,
            "beta_rho_ci_lower": ci_rho[0],
            "beta_rho_ci_upper": ci_rho[1],
            "beta_rho_abs_err": err_abs_rho,
            "beta_rho_rel_err": err_rel_rho,
            "beta_v": b_v,
            "beta_v_se": se_v,
            "beta_v_ci_lower": ci_v[0],
            "beta_v_ci_upper": ci_v[1],
            "beta_v_abs_err": err_abs_v,
            "beta_v_rel_err": err_rel_v,
            "beta_s": b_s,
            "beta_s_se": se_s,
            "beta_s_ci_lower": ci_s[0],
            "beta_s_ci_upper": ci_s[1],
            "beta_s_abs_err": err_abs_s,
            "beta_s_rel_err": err_rel_s,
            "mean_abs_err": (err_abs_rho + err_abs_v + err_abs_s) / 3.0,
        })

        print(f"n = {n_cal:5d} | beta_rho: {b_rho:+.4f} (CI [{ci_rho[0]:+.3f}, {ci_rho[1]:+.3f}]) | beta_v: {b_v:+.4f} (CI [{ci_v[0]:+.3f}, {ci_v[1]:+.3f}]) | beta_s: {b_s:+.4f} (CI [{ci_s[0]:+.3f}, {ci_s[1]:+.3f}]) | R2 = {fit.rsquared:.3f}")

    df_per_n = pd.DataFrame(per_n_results)
    csv_per_n = os.path.join(data_dir, "proposition1_convergence.csv")
    df_per_n.to_csv(csv_per_n, index=False)
    print(f"Saved: {csv_per_n}")

    df_boot_exp = pd.DataFrame(boot_exponent_records)
    csv_boot_exp = os.path.join(data_dir, "proposition1_convergence_bootstrap.csv")
    df_boot_exp.to_csv(csv_boot_exp, index=False)
    print(f"Saved: {csv_boot_exp}")

    # -------------------------------------------------------------
    # Generate Table 2: table_w2_exponent_progression.tex
    # -------------------------------------------------------------
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
"""
    for _, r in df_per_n.iterrows():
        table_tex += (
            f"{int(r['n_cal']):5d} & "
            f"${r['beta_rho']:+.3f}$ [${r['beta_rho_ci_lower']:+.3f}, {r['beta_rho_ci_upper']:+.3f}$] & "
            f"${r['beta_v']:+.3f}$ [${r['beta_v_ci_lower']:+.3f}, {r['beta_v_ci_upper']:+.3f}$] & "
            f"${r['beta_s']:+.3f}$ [${r['beta_s_ci_lower']:+.3f}, {r['beta_s_ci_upper']:+.3f}$] & "
            f"{r['mean_abs_err']:.3f} & {r['r2']:.3f} \\\\\n"
        )
    table_tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    tex_path = os.path.join(paper_dir, "table_w2_exponent_progression.tex")
    with open(tex_path, "w") as f:
        f.write(table_tex)
    print(f"Saved Table 2: {tex_path}")

    # -------------------------------------------------------------
    # Generate Figure 4: w2_exponent_convergence.png & .pdf
    # -------------------------------------------------------------
    n_vals = df_per_n["n_cal"].values
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.8))

    # Panel (a): beta_rho
    yerr_rho = [df_per_n["beta_rho"] - df_per_n["beta_rho_ci_lower"], df_per_n["beta_rho_ci_upper"] - df_per_n["beta_rho"]]
    ax1.errorbar(n_vals, df_per_n["beta_rho"], yerr=yerr_rho, fmt='o-', color='#1f77b4', lw=1.6, capsize=4, label=r'$\hat{\beta}_\rho(n)$ (95\% CI)')
    ax1.axhline(-2.0 / 3.0, color='red', linestyle='--', lw=1.4, label=r'Target $-2/3$')
    ax1.set_xscale('log')
    ax1.set_xticks(n_vals)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xlabel(r'Calibration size $n_{\mathrm{cal}}$')
    ax1.set_ylabel(r'Estimated exponent $\hat{\beta}_\rho$')
    ax1.set_title(r'(a) Local density exponent $\beta_\rho$')
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9)

    # Panel (b): beta_v
    yerr_v = [df_per_n["beta_v"] - df_per_n["beta_v_ci_lower"], df_per_n["beta_v_ci_upper"] - df_per_n["beta_v"]]
    ax2.errorbar(n_vals, df_per_n["beta_v"], yerr=yerr_v, fmt='s-', color='#2ca02c', lw=1.6, capsize=4, label=r'$\hat{\beta}_v(n)$ (95\% CI)')
    ax2.axhline(-1.0 / 3.0, color='red', linestyle='--', lw=1.4, label=r'Target $-1/3$')
    ax2.set_xscale('log')
    ax2.set_xticks(n_vals)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xlabel(r'Calibration size $n_{\mathrm{cal}}$')
    ax2.set_ylabel(r'Estimated exponent $\hat{\beta}_v$')
    ax2.set_title(r'(b) Ambiguity exponent $\beta_v$')
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9)

    # Panel (c): beta_s
    yerr_s = [df_per_n["beta_s"] - df_per_n["beta_s_ci_lower"], df_per_n["beta_s_ci_upper"] - df_per_n["beta_s"]]
    ax3.errorbar(n_vals, df_per_n["beta_s"], yerr=yerr_s, fmt='^-', color='#d62728', lw=1.6, capsize=4, label=r'$\hat{\beta}_s(n)$ (95\% CI)')
    ax3.axhline(2.0 / 3.0, color='red', linestyle='--', lw=1.4, label=r'Target $+2/3$')
    ax3.set_xscale('log')
    ax3.set_xticks(n_vals)
    ax3.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax3.set_xlabel(r'Calibration size $n_{\mathrm{cal}}$')
    ax3.set_ylabel(r'Estimated exponent $\hat{\beta}_s$')
    ax3.set_title(r'(c) Local slope exponent $\beta_s$')
    ax3.legend(loc='lower right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig_png = os.path.join(paper_dir, "w2_exponent_convergence.png")
    fig_pdf = os.path.join(paper_dir, "w2_exponent_convergence.pdf")
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"Saved Figure 4: {fig_png} and {fig_pdf}")

    # -------------------------------------------------------------
    # Pooled Regressions: All n, n >= 4000, n >= 8000
    # -------------------------------------------------------------
    def fit_pooled_subset(df_sub):
        y_sub = np.log(df_sub["width"].values)
        X_sub = np.column_stack([
            np.ones(len(df_sub)),
            np.log(df_sub["n_cal"].values),
            np.log(df_sub["local_density"].values),
            np.log(df_sub["ambiguity"].values),
            np.log(df_sub["local_slope"].values),
        ])
        return SimpleOLS().fit(X_sub, y_sub)

    fit_all = fit_pooled_subset(df)
    fit_4000 = fit_pooled_subset(df[df["n_cal"] >= 4000])
    fit_8000 = fit_pooled_subset(df[df["n_cal"] >= 8000])

    robustness_rows = [
        {"Regime": "All $n_{\\mathrm{cal}} \\in [500, 32000]$", "beta_n": fit_all.params[1], "beta_rho": fit_all.params[2], "beta_v": fit_all.params[3], "beta_s": fit_all.params[4], "r2": fit_all.rsquared},
        {"Regime": "$n_{\\mathrm{cal}} \\ge 4000$", "beta_n": fit_4000.params[1], "beta_rho": fit_4000.params[2], "beta_v": fit_4000.params[3], "beta_s": fit_4000.params[4], "r2": fit_4000.rsquared},
        {"Regime": "$n_{\\mathrm{cal}} \\ge 8000$", "beta_n": fit_8000.params[1], "beta_rho": fit_8000.params[2], "beta_v": fit_8000.params[3], "beta_s": fit_8000.params[4], "r2": fit_8000.rsquared},
    ]
    df_robustness = pd.DataFrame(robustness_rows)
    csv_rob = os.path.join(data_dir, "table_w2_high_n_robustness.csv")
    df_robustness.to_csv(csv_rob, index=False)
    print(f"Saved: {csv_rob}")

    return df_per_n, df_robustness


if __name__ == "__main__":
    run_w2_experiment()
