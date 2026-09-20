"""
src/non_monotonic_experiments.py

Empirically verifying:
1. Invariance of Venn-Abers calibration under strictly monotonic score transformations.
2. Density scaling laws of Venn-Abers interval widths under:
   - Regime A: Monotonic (theta' > 0) -> Expected exponent: -2/3 (-0.667)
   - Regime B: Monotonicity-Violating (theta' < 0) -> Expected exponent: -1.0
   - Regime C: Local Extremum with Violation (middle peak) -> Expected exponent: -1.0
   - Regime D: Local Extremum without Violation (boundary/flat extremum) -> Expected exponent: -4/5 (-0.800)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.isotonic import isotonic_regression

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)


def fast_va_scalar(s_cal: np.ndarray, y_cal: np.ndarray, s_target: float):
    """Exact C-accelerated Venn-Abers at single test score s_target."""
    n = len(s_cal)
    s_all = np.empty(n + 1, dtype=np.float64)
    s_all[:n] = s_cal
    s_all[n] = float(s_target)
    order = np.argsort(s_all, kind="mergesort")
    pos = np.where(order == n)[0][0]
    
    y0_all = np.empty(n + 1, dtype=np.float64)
    y0_all[:n] = y_cal
    y0_all[n] = 0.0
    iso0 = isotonic_regression(y0_all[order], increasing=True)
    p0 = float(iso0[pos])
    
    y1_all = np.empty(n + 1, dtype=np.float64)
    y1_all[:n] = y_cal
    y1_all[n] = 1.0
    iso1 = isotonic_regression(y1_all[order], increasing=True)
    p1 = float(iso1[pos])
    
    p0_val = float(np.clip(p0, 0.0, 1.0))
    p1_val = float(np.clip(p1, 0.0, 1.0))
    width = float(max(0.0, p1_val - p0_val))
    return p0_val, p1_val, width


def test_score_invariance():
    print("=== Running Monotonic Score Invariance Test ===")
    rng = np.random.default_rng(42)
    n_cal = 500
    s_cal = rng.uniform(0.0, 1.0, size=n_cal)
    theta_s = s_cal ** 2
    y_cal = rng.binomial(1, theta_s)
    s_test = 0.5

    p0_orig, p1_orig, w_orig = fast_va_scalar(s_cal, y_cal, s_test)

    # Nonlinear strictly monotonic transformations
    g1_cal, g1_test = np.exp(s_cal), np.exp(s_test)
    _, _, w_g1 = fast_va_scalar(g1_cal, y_cal, g1_test)

    g2_cal, g2_test = s_cal ** 3, s_test ** 3
    _, _, w_g2 = fast_va_scalar(g2_cal, y_cal, g2_test)

    g3_cal, g3_test = np.log(s_cal + 1.0), np.log(s_test + 1.0)
    _, _, w_g3 = fast_va_scalar(g3_cal, y_cal, g3_test)

    print(f"Original width:                   {w_orig:.8f}")
    print(f"Transformed g(s) = exp(s):        {w_g1:.8f} (diff: {abs(w_orig - w_g1):.2e})")
    print(f"Transformed g(s) = s^3:           {w_g2:.8f} (diff: {abs(w_orig - w_g2):.2e})")
    print(f"Transformed g(s) = log(s+1):      {w_g3:.8f} (diff: {abs(w_orig - w_g3):.2e})")
    assert np.isclose(w_orig, w_g1), "Invariance failed for exp(s)"
    assert np.isclose(w_orig, w_g2), "Invariance failed for s^3"
    assert np.isclose(w_orig, w_g3), "Invariance failed for log(s+1)"
    print("Monotonic score transformation invariance holds exactly!\n")


def run_experiment(regime: str, n_values: np.ndarray, n_reps: int = 500, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s_test = 0.0
    mean_widths = []

    for n_cal in n_values:
        widths = []
        for _ in range(n_reps):
            if regime == "monotonic":
                s_cal = rng.uniform(-1, 1, size=n_cal)
                theta_s = np.clip(0.5 + 0.4 * s_cal, 0.0, 1.0)
            elif regime == "violating":
                s_cal = rng.uniform(-1, 1, size=n_cal)
                theta_s = np.clip(0.5 - 0.4 * s_cal, 0.0, 1.0)
            elif regime == "extremum_middle":
                s_cal = rng.uniform(-1, 1, size=n_cal)
                theta_s = np.clip(0.5 - 0.4 * (s_cal ** 2), 0.0, 1.0)
            elif regime == "extremum_boundary":
                s_cal = rng.uniform(-1, 1, size=n_cal)
                theta_s = np.clip(0.5 + 0.4 * (s_cal ** 2) * np.sign(s_cal), 0.0, 1.0)
            else:
                raise ValueError(f"Unknown regime: {regime}")

            y_cal = rng.binomial(1, theta_s)
            _, _, w = fast_va_scalar(s_cal, y_cal, s_test)
            widths.append(w)
        mean_widths.append(np.mean(widths))

    return np.array(mean_widths)


def plot_non_monotonic_scaling(df: pd.DataFrame, paper_dir: str):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    n_values = df["n_cal"].values
    log_x = np.log(n_values)

    regimes_info = [
        ("width_mono", "Regime A: Monotonic", "#1f77b4", "-", "-0.667 (-2/3)"),
        ("width_viol", "Regime B: Violating", "#d62728", "--", "-1.000"),
        ("width_ext_mid", "Regime C: Middle Extremum (Violating)", "#ff7f0e", ":", "-1.000"),
        ("width_ext_bound", "Regime D: Flat Extremum (Non-Violating)", "#2ca02c", "-.", "-0.800 (-4/5)"),
    ]

    slopes = {}
    for col, label, color, style, expected in regimes_info:
        y = df[col].values
        log_y = np.log(y)
        res = linregress(log_x, log_y)
        ax.plot(log_x, log_y, "o", color=color)
        ax.plot(log_x, res.intercept + res.slope * log_x, linestyle=style, color=color,
                label=f"{label} (slope={res.slope:.3f}, Theory={expected})")
        slopes[col] = res.slope

    ax.set_title("Venn--Abers Width Scaling Laws in Different Slope Regimes")
    ax.set_xlabel(r"$\log(N_{\mathrm{cal}})$")
    ax.set_ylabel(r"$\log(\bar{w})$")
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(paper_dir, "non_monotonic_scaling_laws.png")
    pdf_path = os.path.join(paper_dir, "non_monotonic_scaling_laws.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved Figure 5 to:\n  - {png_path}\n  - {pdf_path}")
    return slopes


def main():
    test_score_invariance()

    print("=== Running Monotonicity-Violating Density Scaling Experiments ===")
    n_values = np.array([100, 200, 400, 800, 1600, 3200])
    n_reps = 500

    print("Running Regime A: Standard Monotonic (theta' > 0)...")
    widths_mono = run_experiment("monotonic", n_values, n_reps)

    print("Running Regime B: Monotonicity-Violating (theta' < 0)...")
    widths_viol = run_experiment("violating", n_values, n_reps)

    print("Running Regime C: Local Extremum with Violation (middle peak)...")
    widths_ext_mid = run_experiment("extremum_middle", n_values, n_reps)

    print("Running Regime D: Local Extremum without Violation (flat extremum)...")
    widths_ext_bound = run_experiment("extremum_boundary", n_values, n_reps)

    # Save canonical data
    data_dir = os.path.join(root_dir, "data")
    paper_dir = os.path.join(root_dir, "paper")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)

    df = pd.DataFrame({
        "n_cal": n_values,
        "width_mono": widths_mono,
        "width_viol": widths_viol,
        "width_ext_mid": widths_ext_mid,
        "width_ext_bound": widths_ext_bound,
    })
    csv_path = os.path.join(data_dir, "non_monotonic_scaling_laws.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved canonical non-monotonic data to: {csv_path}")

    # Plot
    slopes = plot_non_monotonic_scaling(df, paper_dir)

    print("\nResults summary:")
    print(f"Regime A (Monotonic) Exponent:       {slopes['width_mono']:.3f} (Theory: -0.667)")
    print(f"Regime B (Violating) Exponent:       {slopes['width_viol']:.3f} (Theory: -1.000)")
    print(f"Regime C (Middle Extremum) Exponent: {slopes['width_ext_mid']:.3f} (Theory: -1.000)")
    print(f"Regime D (Flat Extremum) Exponent:   {slopes['width_ext_bound']:.3f} (Theory: -0.800)")


if __name__ == "__main__":
    main()
