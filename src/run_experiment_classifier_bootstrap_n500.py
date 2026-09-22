"""
src/run_experiment_classifier_bootstrap_n500.py

Runs the classifier-based bootstrap experiment at N_cal = 500.
Generates pointwise interval width w, true midpoint probability p_mid = (p0 + p1) / 2,
standard log-loss merge p_va = p1 / (1 - p0 + p1),
instability index U_cal = sqrt(p_mid * (1 - p_mid) * w),
and bootstrap standard deviation of p_mid across M resamples.

Saves canonical output to data/classifier_bootstrap_n500.csv and prints
the exact Pearson, Spearman, and multiplicative MAE metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier

# Add parent or current directory to path
try:
    from utils import clip01, true_probability_1d, set_seed
    from fast_venn_abers import exact_va_probs
except ImportError:
    from src.utils import clip01, true_probability_1d, set_seed
    from src.fast_venn_abers import exact_va_probs

# Note: computes Venn-Abers directly on the fixed base classifier's scores via
# exact_va_probs rather than routing through VennAbersCalibrator(cal_size=None),
# which internally refits a clone of the base estimator on whatever is passed
# to .fit() and additionally splits it 75/25 (sklearn's train_test_split
# default), so it silently used only ~25% of the N_cal=500 calibration set --
# see Request 1 code review, Priority 1.


def run_classifier_bootstrap_experiment(
    n_train_base: int = 3000,
    n_cal: int = 500,
    n_test: int = 500,
    M: int = 200,
    seed: int = 42,
):
    print(f"Running classifier-based bootstrap experiment (N_cal={n_cal}, M={M}, seed={seed})...")
    rng = set_seed(seed)

    # 1. Generate Training Data
    x_train = rng.uniform(-2.0, 2.0, size=n_train_base)
    p_train = true_probability_1d(x_train)
    y_train = rng.binomial(1, clip01(p_train))

    # 2. Generate Calibration Data (N=500)
    x_cal = rng.uniform(-2.0, 2.0, size=n_cal)
    p_cal = true_probability_1d(x_cal)
    y_cal = rng.binomial(1, clip01(p_cal))

    # 3. Generate Test Data (N=500)
    x_test = np.linspace(-2.0, 2.0, n_test)
    p_test_true = true_probability_1d(x_test)

    # 4. Fit Base Classifier on Train
    X_train = x_train.reshape(-1, 1)
    X_cal = x_cal.reshape(-1, 1)
    X_test = x_test.reshape(-1, 1)

    base_clf = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=250,
        min_samples_leaf=20,
        random_state=seed,
    )
    base_clf.fit(X_train, y_train)

    # 5. Fit Inductive Venn-Abers on Single Calibration Set (scored once with
    # the fixed base model; exact_va_probs consumes the raw scores directly)
    s_cal = base_clf.predict_proba(X_cal)[:, 1]
    s_test = base_clf.predict_proba(X_test)[:, 1]
    p0, p1, p_mid, w = exact_va_probs(s_cal, y_cal, s_test)
    p_va = p1 / (1.0 - p0 + p1)
    u_cal = np.sqrt(p_mid * (1.0 - p_mid) * w)

    # 6. Bootstrap Resampling on Calibration Set (bootstrapping midpoint p_mid;
    # base model and test scores held fixed, only calibration scores/labels resample)
    print(f"Running {M} bootstrap resamples on calibration set...")
    boot_preds = np.zeros((M, n_test))

    for m in range(M):
        boot_rng = set_seed(seed * 1000 + m)
        idx_boot = boot_rng.choice(n_cal, size=n_cal, replace=True)
        s_cal_b = s_cal[idx_boot]
        y_cal_b = y_cal[idx_boot]

        _, _, mid_b, _ = exact_va_probs(s_cal_b, y_cal_b, s_test)
        boot_preds[m, :] = mid_b

    bootstrap_sd = np.std(boot_preds, axis=0)

    # 7. Compute Summary Statistics
    pearson_w, pval_pw = pearsonr(w, bootstrap_sd)
    spearman_w, pval_sw = spearmanr(w, bootstrap_sd)
    pearson_u, pval_pu = pearsonr(u_cal, bootstrap_sd)
    spearman_u, pval_su = spearmanr(u_cal, bootstrap_sd)

    # 8. Held-Out Multiplicative Fit: SD ~ c * metric
    fit_idx = np.arange(0, n_test, 2)
    eval_idx = np.arange(1, n_test, 2)

    c_w = np.sum(bootstrap_sd[fit_idx] * w[fit_idx]) / np.sum(w[fit_idx] ** 2)
    c_u = np.sum(bootstrap_sd[fit_idx] * u_cal[fit_idx]) / np.sum(u_cal[fit_idx] ** 2)

    pred_sd_w = c_w * w[eval_idx]
    pred_sd_u = c_u * u_cal[eval_idx]

    mae_w = float(np.mean(np.abs(bootstrap_sd[eval_idx] - pred_sd_w)))
    mae_u = float(np.mean(np.abs(bootstrap_sd[eval_idx] - pred_sd_u)))

    print("\n" + "=" * 50)
    print("EXPERIMENTAL RESULTS SUMMARY:")
    print("=" * 50)
    print("Raw Width:")
    print(f"  Pearson correlation:  {pearson_w:.4f} (p = {pval_pw:.2e})")
    print(f"  Spearman correlation: {spearman_w:.4f} (p = {pval_sw:.2e})")
    print(f"  Held-out fit MAE:     {mae_w:.4f} (c_w = {c_w:.4f})")
    print("U_cal = sqrt(p_hat*(1-p_hat)*w):")
    print(f"  Pearson correlation:  {pearson_u:.4f} (p = {pval_pu:.2e})")
    print(f"  Spearman correlation: {spearman_u:.4f} (p = {pval_su:.2e})")
    print(f"  Held-out fit MAE:     {mae_u:.4f} (c_u = {c_u:.4f})")
    print("=" * 50)

    # 9. Save Pointwise DataFrame to Canonical Path
    df = pd.DataFrame({
        "x_test": x_test,
        "p_true": p_test_true,
        "p_mid": p_mid,
        "p_va": p_va,
        "p0": p0,
        "p1": p1,
        "width": w,
        "u_cal": u_cal,
        "bootstrap_sd": bootstrap_sd,
    })

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "classifier_bootstrap_n500.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved canonical pointwise results to:\n  - {csv_path}")

    metrics = {
        "pearson_w": float(pearson_w),
        "spearman_w": float(spearman_w),
        "mae_w": float(mae_w),
        "pearson_u": float(pearson_u),
        "spearman_u": float(spearman_u),
        "mae_u": float(mae_u),
        "c_w": float(c_w),
        "c_u": float(c_u),
    }
    return df, metrics


if __name__ == "__main__":
    run_classifier_bootstrap_experiment()
