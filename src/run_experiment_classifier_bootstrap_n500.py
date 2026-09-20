"""
src/run_experiment_classifier_bootstrap_n500.py

Runs the classifier-based bootstrap experiment at N_cal = 500.
Generates pointwise interval width w, midpoint probability p_mid,
instability index U_cal = sqrt(p_mid * (1 - p_mid) * w),
and bootstrap standard deviation across M resamples.

Saves output to data/classifier_bootstrap_n500.csv and prints
the exact Pearson, Spearman, and multiplicative MAE metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from venn_abers import VennAbersCalibrator

# Add current directory to path
try:
    from utils import clip01, true_probability_1d, set_seed
except ImportError:
    from src.utils import clip01, true_probability_1d, set_seed

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
    
    # 5. Fit Inductive Venn-Abers on Single Calibration Set
    va = VennAbersCalibrator(estimator=base_clf, inductive=True, cal_size=None, random_state=seed)
    va.fit(X_cal, y_cal)
    
    pred_test = va.predict_proba(X_test, p0_p1_output=True)
    
    # Parse bounds
    if isinstance(pred_test, tuple) and len(pred_test) == 2:
        a, b = np.asarray(pred_test[0]), np.asarray(pred_test[1])
        if b.ndim == 3 and b.shape[0] == 1:
            p0 = b[0, :, 0].astype(float)
            p1 = b[0, :, 1].astype(float)
            p_mid = a[:, 1].astype(float) if a.ndim == 2 else a.astype(float).reshape(-1)
        elif b.ndim == 2:
            p0 = b[:, 0].astype(float)
            p1 = b[:, 1].astype(float)
            p_mid = a[:, 1].astype(float) if a.ndim == 2 else 0.5 * (p0 + p1)
    else:
        raise RuntimeError(f"Unexpected predict_proba format: {type(pred_test)}")
        
    p0 = clip01(np.minimum(p0, p1))
    p1 = clip01(np.maximum(p0, p1))
    p_mid = clip01(p_mid)
    w = p1 - p0
    u_cal = np.sqrt(p_mid * (1.0 - p_mid) * w)
    
    # 6. Bootstrap Resampling on Calibration Set
    print(f"Running {M} bootstrap resamples on calibration set...")
    boot_preds = np.zeros((M, n_test))
    
    for m in range(M):
        boot_rng = set_seed(seed * 1000 + m)
        idx_boot = boot_rng.choice(n_cal, size=n_cal, replace=True)
        X_cal_b = X_cal[idx_boot]
        y_cal_b = y_cal[idx_boot]
        
        va_b = VennAbersCalibrator(estimator=base_clf, inductive=True, cal_size=None, random_state=seed + m)
        va_b.fit(X_cal_b, y_cal_b)
        pred_b = va_b.predict_proba(X_test, p0_p1_output=True)
        
        if isinstance(pred_b, tuple) and len(pred_b) == 2:
            ab, bb = np.asarray(pred_b[0]), np.asarray(pred_b[1])
            if bb.ndim == 3 and bb.shape[0] == 1:
                mid_b = ab[:, 1].astype(float) if ab.ndim == 2 else ab.astype(float).reshape(-1)
            elif bb.ndim == 2:
                mid_b = ab[:, 1].astype(float) if ab.ndim == 2 else 0.5 * (bb[:, 0] + bb[:, 1])
        boot_preds[m, :] = clip01(mid_b)
        
    bootstrap_sd = np.std(boot_preds, axis=0)
    
    # 7. Compute Summary Statistics
    pearson_w, pval_pw = pearsonr(w, bootstrap_sd)
    spearman_w, pval_sw = spearmanr(w, bootstrap_sd)
    pearson_u, pval_pu = pearsonr(u_cal, bootstrap_sd)
    spearman_u, pval_su = spearmanr(u_cal, bootstrap_sd)
    
    # 8. Held-Out Multiplicative Fit: SD ~ c * metric
    # Split test set into 50% fit and 50% evaluation
    fit_idx = np.arange(0, n_test, 2)
    eval_idx = np.arange(1, n_test, 2)
    
    # Fit scaling factors c on fit_idx via least squares: min sum (sd - c * x)^2 => c = sum(sd * x) / sum(x^2)
    c_w = np.sum(bootstrap_sd[fit_idx] * w[fit_idx]) / np.sum(w[fit_idx] ** 2)
    c_u = np.sum(bootstrap_sd[fit_idx] * u_cal[fit_idx]) / np.sum(u_cal[fit_idx] ** 2)
    
    pred_sd_w = c_w * w[eval_idx]
    pred_sd_u = c_u * u_cal[eval_idx]
    
    mae_w = float(np.mean(np.abs(bootstrap_sd[eval_idx] - pred_sd_w)))
    mae_u = float(np.mean(np.abs(bootstrap_sd[eval_idx] - pred_sd_u)))
    
    print("\n" + "="*50)
    print("EXPERIMENTAL RESULTS SUMMARY:")
    print("="*50)
    print(f"Raw Width:")
    print(f"  Pearson correlation:  {pearson_w:.4f} (p = {pval_pw:.2e})")
    print(f"  Spearman correlation: {spearman_w:.4f} (p = {pval_sw:.2e})")
    print(f"  Held-out fit MAE:     {mae_w:.4f} (c_w = {c_w:.4f})")
    print(f"U_cal = sqrt(p*(1-p)*w):")
    print(f"  Pearson correlation:  {pearson_u:.4f} (p = {pval_pu:.2e})")
    print(f"  Spearman correlation: {spearman_u:.4f} (p = {pval_su:.2e})")
    print(f"  Held-out fit MAE:     {mae_u:.4f} (c_u = {c_u:.4f})")
    print("="*50)
    
    # 9. Save Pointwise DataFrame
    df = pd.DataFrame({
        "x_test": x_test,
        "p_true": p_test_true,
        "p_mid": p_mid,
        "p0": p0,
        "p1": p1,
        "width": w,
        "u_cal": u_cal,
        "bootstrap_sd": bootstrap_sd,
    })
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path1 = os.path.join(data_dir, "classifier_bootstrap_n500.csv")
    csv_path2 = os.path.join(project_root, "classifier_bootstrap_n500.csv")
    df.to_csv(csv_path1, index=False)
    df.to_csv(csv_path2, index=False)
    print(f"\nSaved pointwise results to:\n  - {csv_path1}\n  - {csv_path2}")
    
    metrics = {
        "pearson_w": pearson_w,
        "spearman_w": spearman_w,
        "mae_w": mae_w,
        "pearson_u": pearson_u,
        "spearman_u": spearman_u,
        "mae_u": mae_u,
        "c_w": c_w,
        "c_u": c_u,
    }
    return df, metrics

if __name__ == "__main__":
    run_classifier_bootstrap_experiment()
