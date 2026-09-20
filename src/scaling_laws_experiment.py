"""
scaling_laws_experiment.py

Empirically verifying the scaling laws of Venn-Abers interval width using 
highly controlled 1D synthetic experiments.

Theoretical relationship:
w(s) ~ rho(s)^{-2/3} * (theta(1-theta))^{-1/3} * (theta'(s))^{2/3}
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

# Import VennAbersCalibrator from the venn_abers package
from venn_abers import VennAbersCalibrator
from sklearn.base import BaseEstimator, ClassifierMixin

def run_calibration(s_cal: np.ndarray, y_cal: np.ndarray, s_test: float) -> float:
    """
    Run Venn-Abers calibration on 1D scores and return the interval width at s_test.
    """
    # We bypass the base estimator by creating a dummy that preserves the order of s.
    # PAVA only cares about the sorting order of the scores.
    class IdentityEstimator(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            self.is_fitted_ = True
            return self
        def predict_proba(self, X):
            # X is shape (n, 1), scores in [-1, 1]
            # Map strictly monotonically to [0, 1]
            p1 = (X[:, 0] + 1.0) / 2.0
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
            
    base = IdentityEstimator()
    base.fit(None, None)  # Pre-fit so VennAbersCalibrator knows it's already fitted
    va = VennAbersCalibrator(estimator=base, inductive=True, cal_size=None)
    
    X_cal = s_cal.reshape(-1, 1)
    va.fit(X_cal, y_cal)
    
    X_test = np.array([[s_test]])
    pred = va.predict_proba(X_test, p0_p1_output=True)
    
    # Extract p0 and p1 (handling variations in venn-abers package return formats)
    if isinstance(pred, tuple) and len(pred) == 2:
        a = np.asarray(pred[0])
        b = np.asarray(pred[1])
        if b.ndim == 3: # v1.5.1
            p0, p1 = b[0, 0, 0], b[0, 0, 1]
        elif b.ndim == 2: # older
            p0, p1 = b[0, 0], b[0, 1]
        elif b.ndim == 1:
            p0, p1 = a[0], b[0]
    elif isinstance(pred, tuple) and len(pred) == 3:
        p0, p1 = pred[1][0], pred[2][0]
    else:
        arr = np.asarray(pred)
        if arr.shape[1] == 3:
            p0, p1 = arr[0, 1], arr[0, 2]
        else:
            p0, p1 = arr[0, 0], arr[0, 1]
            
    return max(0.0, float(p1 - p0))

def run_experiment(param_values, param_name, n_reps=500, n_cal_fixed=1000, 
                   theta_0_fixed=0.5, slope_fixed=0.5, s_test=0.0):
    """
    Run the calibration experiment for a list of parameter values.
    Returns the mean widths across n_reps.
    """
    mean_widths = []
    
    for val in param_values:
        widths = []
        for seed in range(n_reps):
            rng = np.random.default_rng(seed + int(val * 10000))
            
            # Determine current parameters
            n_cal = int(val) if param_name == 'density' else n_cal_fixed
            theta_0 = val if param_name == 'variance' else theta_0_fixed
            slope = val if param_name == 'slope' else slope_fixed
            
            # Sample data
            s_cal = rng.uniform(-1, 1, size=n_cal)
            
            # Compute true probabilities
            theta_s = np.clip(theta_0 + slope * (s_cal - s_test), 0.0, 1.0)
            
            # Generate labels
            y_cal = rng.binomial(1, theta_s)
            
            # Compute width
            w = run_calibration(s_cal, y_cal, s_test)
            widths.append(w)
            
        mean_widths.append(np.mean(widths))
        
    return np.array(mean_widths)

def main():
    print("Running Scaling Laws Experiments...")
    os.makedirs('../paper', exist_ok=True)
    
    # 1. Density Experiment
    print("Running Experiment 1: Density...")
    n_values = np.array([100, 200, 400, 800, 1600, 3200])
    widths_density = run_experiment(n_values, 'density', n_reps=500)
    
    # 2. Variance / Ambiguity Experiment
    print("Running Experiment 2: Variance...")
    theta_values = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    variances = theta_values * (1 - theta_values)
    widths_variance = run_experiment(theta_values, 'variance', n_reps=500)
    
    # 3. Slope / Bias Experiment
    print("Running Experiment 3: Slope...")
    slope_values = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    widths_slope = run_experiment(slope_values, 'slope', n_reps=500)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    def plot_loglog(ax, x, y, title, xlabel, expected_slope):
        log_x = np.log(x)
        log_y = np.log(y)
        res = linregress(log_x, log_y)
        
        ax.plot(log_x, log_y, 'o', label='Measured')
        ax.plot(log_x, res.intercept + res.slope * log_x, 'r--', 
                label=f'Fit (slope={res.slope:.3f})')
        
        ax.set_title(f"{title}\n(Expected slope: {expected_slope})")
        ax.set_xlabel(f"log({xlabel})")
        ax.set_ylabel("log(Interval Width)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return res.slope
        
    slope1 = plot_loglog(axes[0], n_values, widths_density, "Exp 1: Density", "N", "-0.667 (-2/3)")
    slope2 = plot_loglog(axes[1], variances, widths_variance, "Exp 2: Variance", "theta(1-theta)", "-0.333 (-1/3)")
    slope3 = plot_loglog(axes[2], slope_values, widths_slope, "Exp 3: Bias-Gradient", "theta'", "0.667 (2/3)")
    
    plt.tight_layout()
    plt.savefig('../paper/scaling_laws.png', dpi=300)
    plt.close()
    
    print("\nResults:")
    print(f"Density Slope:  {slope1:.3f} (Theory: -0.667)")
    print(f"Variance Slope: {slope2:.3f} (Theory: -0.333)")
    print(f"Gradient Slope: {slope3:.3f} (Theory:  0.667)")
    print("\nPlot saved to ../paper/scaling_laws.png")

if __name__ == "__main__":
    main()
