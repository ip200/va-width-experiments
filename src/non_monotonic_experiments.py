"""
non_monotonic_experiments.py

Empirically verifying:
1. Invariance of Venn-Abers calibration under strictly monotonic score transformations.
2. Density scaling laws of Venn-Abers interval widths under:
   - Regime A: Monotonic (theta' > 0) -> Expected exponent: -2/3 (-0.667)
   - Regime B: Monotonicity-Violating (theta' < 0) -> Expected exponent: -1.0
   - Regime C: Local Extremum with Violation (middle peak) -> Expected exponent: -1.0
   - Regime D: Local Extremum without Violation (boundary/flat extremum) -> Expected exponent: -4/5 (-0.800)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
from venn_abers import VennAbers

def run_calibration(s_cal: np.ndarray, y_cal: np.ndarray, s_test: float) -> tuple[float, float, float]:
    """
    Run Venn-Abers calibration on 1D scores and return (p0, p1, width) at s_test.
    """
    p_cal = np.zeros((len(s_cal), 2))
    p_cal[:, 1] = 1.0 / (1.0 + np.exp(-s_cal))
    p_cal[:, 0] = 1.0 - p_cal[:, 1]
    
    p_test = np.zeros((1, 2))
    p_test[0, 1] = 1.0 / (1.0 + np.exp(-s_test))
    p_test[0, 0] = 1.0 - p_test[0, 1]
    
    va = VennAbers()
    va.fit(p_cal, y_cal)
    
    _, pred = va.predict_proba(p_test)
    p0, p1 = pred[0, 0], pred[0, 1]
    
    return p0, p1, max(0.0, float(p1 - p0))

def test_score_invariance():
    print("=== Running Monotonic Score Invariance Test ===")
    rng = np.random.default_rng(42)
    n_cal = 500
    
    # Generate original scores s in [0, 1]
    s_cal = rng.uniform(0.0, 1.0, size=n_cal)
    # True probability is non-linear but monotonic with s
    theta_s = s_cal ** 2
    y_cal = rng.binomial(1, theta_s)
    
    s_test = 0.5
    
    # 1. Original scores
    p0_orig, p1_orig, w_orig = run_calibration(s_cal, y_cal, s_test)
    
    # 2. Monotonic transform 1: s^2
    p0_t1, p1_t1, w_t1 = run_calibration(s_cal ** 2, y_cal, s_test ** 2)
    
    # 3. Monotonic transform 2: exp(s)
    p0_t2, p1_t2, w_t2 = run_calibration(np.exp(s_cal), y_cal, np.exp(s_test))
    
    print(f"Original s:   p0={p0_orig:.6f}, p1={p1_orig:.6f}, width={w_orig:.6f}")
    print(f"Transform s2: p0={p0_t1:.6f}, p1={p1_t1:.6f}, width={w_t1:.6f}")
    print(f"Transform exp:p0={p0_t2:.6f}, p1={p1_t2:.6f}, width={w_t2:.6f}")
    
    # Assert they are exactly equal (within numerical limits)
    assert np.isclose(p0_orig, p0_t1) and np.isclose(p1_orig, p1_t1)
    assert np.isclose(p0_orig, p0_t2) and np.isclose(p1_orig, p1_t2)
    print("Verification SUCCESS: Venn-Abers outputs are invariant under monotonic score transformations!\n")

def run_experiment(regime: str, n_values: np.ndarray, n_reps: int = 500) -> np.ndarray:
    """
    Run experiment for varying calibration sizes (n_values) and return mean widths.
    """
    mean_widths = []
    s_test = 0.0
    
    for n_cal in n_values:
        widths = []
        for seed in range(n_reps):
            rng = np.random.default_rng(seed + n_cal * 100)
            
            if regime == 'monotonic':
                s_cal = rng.uniform(-1, 1, size=n_cal)
                theta_s = np.clip(0.5 + 0.3 * s_cal, 0.0, 1.0)
            elif regime == 'violating':
                s_cal = rng.uniform(-1, 1, size=n_cal)
                theta_s = np.clip(0.5 - 0.3 * s_cal, 0.0, 1.0)
            elif regime == 'extremum_middle':
                s_cal = rng.uniform(-1, 1, size=n_cal)
                theta_s = np.clip(0.5 - 0.4 * (s_cal ** 2), 0.0, 1.0)
            elif regime == 'extremum_boundary':
                # theta(s) = 0.5 + 0.4 * s^2 * sign(s) on s in [-1, 1], evaluated at s_test = 0.0
                # Strictly increasing everywhere, first derivative is 0 at s = 0.0
                s_cal = rng.uniform(-1, 1, size=n_cal)
                theta_s = np.clip(0.5 + 0.4 * (s_cal ** 2) * np.sign(s_cal), 0.0, 1.0)
            else:
                raise ValueError("Unknown regime")
                
            y_cal = rng.binomial(1, theta_s)
            _, _, w = run_calibration(s_cal, y_cal, s_test)
            widths.append(w)
            
        mean_widths.append(np.mean(widths))
        
    return np.array(mean_widths)

def main():
    test_score_invariance()
    
    print("=== Running Monotonicity-Violating Density Scaling Experiments ===")
    n_values = np.array([100, 200, 400, 800, 1600, 3200])
    n_reps = 500
    
    print("Running Regime A: Standard Monotonic (theta' > 0)...")
    widths_mono = run_experiment('monotonic', n_values, n_reps)
    
    print("Running Regime B: Monotonicity-Violating (theta' < 0)...")
    widths_viol = run_experiment('violating', n_values, n_reps)
    
    print("Running Regime C: Local Extremum with Violation (middle peak)...")
    widths_ext_mid = run_experiment('extremum_middle', n_values, n_reps)
    
    print("Running Regime D: Local Extremum without Violation (flat extremum)...")
    widths_ext_bound = run_experiment('extremum_boundary', n_values, n_reps)
    
    # Save directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.join(current_dir, '..', 'paper')
    os.makedirs(paper_dir, exist_ok=True)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(9, 7))
    
    def fit_and_plot(x, y, label, color, style, expected_slope):
        log_x = np.log(x)
        log_y = np.log(y)
        res = linregress(log_x, log_y)
        ax.plot(log_x, log_y, 'o', color=color)
        ax.plot(log_x, res.intercept + res.slope * log_x, linestyle=style, color=color,
                label=f"{label} (slope={res.slope:.3f}, Expected={expected_slope})")
        return res.slope

    slope_mono = fit_and_plot(n_values, widths_mono, "Regime A: Monotonic", "blue", "-", "-0.667 (-2/3)")
    slope_viol = fit_and_plot(n_values, widths_viol, "Regime B: Violating", "red", "--", "-1.000")
    slope_ext_mid = fit_and_plot(n_values, widths_ext_mid, "Regime C: Middle Extremum (Violating)", "orange", ":", "-1.000")
    slope_ext_bound = fit_and_plot(n_values, widths_ext_bound, "Regime D: Flat Extremum (Non-Violating)", "green", "-.", "-0.800 (-4/5)")
    
    ax.set_title("Venn-Abers Width Scaling Laws in Different Slope Regimes")
    ax.set_xlabel("log(N)")
    ax.set_ylabel("log(Interval Width)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(paper_dir, 'non_monotonic_scaling_laws.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("\nResults summary:")
    print(f"Regime A (Monotonic) Exponent:          {slope_mono:.3f} (Theory: -0.667)")
    print(f"Regime B (Violating) Exponent:          {slope_viol:.3f} (Theory: -1.000)")
    print(f"Regime C (Middle Extremum) Exponent:    {slope_ext_mid:.3f} (Theory: -1.000 due to local violation)")
    print(f"Regime D (Flat Extremum) Exponent:      {slope_ext_bound:.3f} (Theory: -0.800)")
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    main()
