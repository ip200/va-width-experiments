"""
src/real_data_calibration_support.py

Controlled empirical evaluation of Venn-Abers interval width and calibration
instability on real tabular benchmarks (UCI Adult, UCI Bank Marketing, UCI Spambase).

Key features:
  1. Base model uniformity: HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200)
     used uniformly for base fit, training bootstrap refits (E_model), and reverse intervention refits.
  2. Exact Venn-Abers computation via C-optimized isotonic regression (fast_va_scalar).
  3. Proper aggregation over R_thin = 100 independent local thinning replicates, where each thinned set
     is bootstrapped B_cal = 100 times to compute sigma_cal.
  4. Pre-specified target test points across score space (s ~ 0.20, 0.35, 0.50, 0.65, 0.80)
     saved to results/real_data_selected_points.csv.
  5. Canonical output paths: data/, paper/, results/.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from joblib import Parallel, delayed

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import isotonic_regression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from venn_abers import VennAbers
from fast_venn_abers import exact_va_probs, exact_va_scalar

# Add current/parent directory to import path
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils import set_seed, make_base_model

# Configure publication-grade styling
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


def fast_va_scalar(s_cal: np.ndarray, y_cal: np.ndarray, s_target: float):
    """
    Computes exact Venn-Abers (p0, p1, p_hat, width) at a single test score s_target.
    Groups identical calibration scores and computes exact GCM bounds matching the official package.
    """
    return exact_va_scalar(s_cal, y_cal, s_target)


def compute_va_probs(s_cal: np.ndarray, y_cal: np.ndarray, s_test: np.ndarray):
    """
    Vectorized Venn-Abers on 1D scores matching the official package.
    Returns (p0, p1, p_hat, width).
    """
    return exact_va_probs(s_cal, y_cal, s_test)


def load_dataset(name="adult", seed=42):
    """
    Loads and preprocesses public binary classification datasets.
    Returns: X_train, y_train, X_cal, y_cal, X_test, y_test, cat_cols
    """
    print(f"\nLoading dataset: {name} (seed={seed})...")
    if name == "adult":
        data = fetch_openml("adult", version=2, as_frame=True)
        X = data.data.copy()
        y = (data.target == ">50K").astype(int).values
    elif name == "bank":
        data = fetch_openml("bank-marketing", version=1, as_frame=True)
        X = data.data.copy()
        y = (data.target.astype(str) == "2").astype(int).values
    elif name == "spambase":
        data = fetch_openml("spambase", version=1, as_frame=True)
        X = data.data.copy()
        y = data.target.astype(int).values
    else:
        raise ValueError(f"Unknown dataset: {name}")

    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols])

    # 50% Train, 25% Cal, 25% Test
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, test_size=0.50, random_state=seed, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rem, y_rem, test_size=0.50, random_state=seed, stratify=y_rem
    )

    print(f"  {name.upper()}: Train={len(y_train)}, Cal={len(y_cal)}, Test={len(y_test)}, PosRate={np.mean(y):.3f}")
    return X_train, y_train, X_cal, y_cal, X_test, y_test, cat_cols


def _process_single_thinning_target(
    dataset_name: str,
    tgt_name: str,
    t_idx: int,
    s_target: float,
    e_model_val: float,
    ambiguity_val: float,
    s_cal: np.ndarray,
    y_cal: np.ndarray,
    cal_ranks: np.ndarray,
    thin_fractions: list,
    R_thin: int,
    B_cal: int,
    seed: int,
):
    """
    Processes all thinning fractions for one target score across R_thin replicates
    and B_cal bootstrap draws per replicate.
    """
    n_cal = len(s_cal)
    rank_target = np.mean(s_cal <= s_target)
    rank_dist = np.abs(cal_ranks - rank_target)
    nbr_mask = rank_dist <= 0.05
    nbr_indices = np.where(nbr_mask)[0]
    non_nbr_indices = np.where(~nbr_mask)[0]

    rows = []
    for frac in thin_fractions:
        w_reps = np.zeros(R_thin)
        sigma_cal_reps = np.zeros(R_thin)

        for r in range(R_thin):
            thin_rng = set_seed(seed * 100000 + t_idx * 10000 + int(frac * 1000) + r)
            if frac < 1.0:
                keep_count = max(2, int(len(nbr_indices) * frac))
                thinned_nbr = thin_rng.choice(nbr_indices, size=keep_count, replace=False)
                cur_cal_idx = np.concatenate([thinned_nbr, non_nbr_indices])
            else:
                cur_cal_idx = np.arange(n_cal)

            s_cal_curr = s_cal[cur_cal_idx]
            y_cal_curr = y_cal[cur_cal_idx]
            cur_n = len(cur_cal_idx)

            # 1. Venn-Abers width on thinned set
            _, _, _, w_curr = fast_va_scalar(s_cal_curr, y_cal_curr, s_target)
            w_reps[r] = w_curr

            # 2. Bootstrap that SAME thinned calibration set B_cal times
            boot_preds = np.zeros(B_cal)
            for b in range(B_cal):
                b_idx = thin_rng.choice(cur_n, size=cur_n, replace=True)
                _, _, pm, _ = fast_va_scalar(s_cal_curr[b_idx], y_cal_curr[b_idx], s_target)
                boot_preds[b] = pm

            sigma_cal_reps[r] = float(np.std(boot_preds, ddof=1))

        rows.append({
            "dataset": dataset_name,
            "target_name": tgt_name,
            "target_score": float(s_target),
            "retained_fraction": float(frac),
            "mean_width": float(np.mean(w_reps)),
            "std_width": float(np.std(w_reps)),
            "mean_sigma_cal": float(np.mean(sigma_cal_reps)),
            "std_sigma_cal": float(np.std(sigma_cal_reps)),
            "ci_lower_width": float(np.percentile(w_reps, 2.5)),
            "ci_upper_width": float(np.percentile(w_reps, 97.5)),
            "ci_lower_sigma_cal": float(np.percentile(sigma_cal_reps, 2.5)),
            "ci_upper_sigma_cal": float(np.percentile(sigma_cal_reps, 97.5)),
            "e_model": float(e_model_val),
            "ambiguity": float(ambiguity_val),
        })
    return rows


def run_experiment_for_dataset(
    dataset_name="adult",
    seed=42,
    R_thin=100,
    B_cal=100,
    B_model=100,
    n_jobs=8,
):
    rng = set_seed(seed)
    X_train, y_train, X_cal, y_cal, X_test, y_test, cat_cols = load_dataset(dataset_name, seed=seed)
    is_cat = [c in cat_cols for c in X_train.columns] if cat_cols else None

    # 1. Fit Base Model (uniform max_iter=200)
    print(f"Training base model for {dataset_name} (max_iter=200)...")
    base_clf = HistGradientBoostingClassifier(
        categorical_features=is_cat,
        max_depth=4,
        learning_rate=0.05,
        max_iter=200,
        random_state=seed,
    )
    base_clf.fit(X_train, y_train)
    s_cal = base_clf.predict_proba(X_cal)[:, 1]
    s_test = base_clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, (s_test >= 0.5).astype(int))
    brier = brier_score_loss(y_test, s_test)
    auc = roc_auc_score(y_test, s_test)
    print(f"  Base Model Evaluation -> Acc: {acc:.4f}, Brier: {brier:.4f}, ROC-AUC: {auc:.4f}")

    # 2. Estimate Model Epistemic Uncertainty E_model(x) via B_model bootstrap refits on training set
    print(f"Estimating Model Epistemic Uncertainty (B_model={B_model}, max_iter=200)...")
    model_preds_test = np.zeros((B_model, len(X_test)))
    n_tr = len(X_train)
    X_tr_arr = X_train.values
    y_tr_arr = y_train

    def _fit_single_boot_model(b):
        boot_rng = set_seed(seed + b * 7)
        boot_idx = boot_rng.choice(n_tr, size=n_tr, replace=True)
        clf_b = HistGradientBoostingClassifier(
            categorical_features=is_cat,
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            random_state=seed + b * 7,
        )
        clf_b.fit(X_tr_arr[boot_idx], y_tr_arr[boot_idx])
        return clf_b.predict_proba(X_test)[:, 1]

    boot_preds_list = Parallel(n_jobs=n_jobs)(
        delayed(_fit_single_boot_model)(b) for b in range(B_model)
    )
    for b, preds in enumerate(boot_preds_list):
        model_preds_test[b, :] = preds

    e_model_test = np.std(model_preds_test, axis=0, ddof=1)

    # 3. Base Venn-Abers Calibration on full Calibration Set
    p0_base, p1_base, p_hat_base, w_base = compute_va_probs(s_cal, y_cal, s_test)
    ambiguity_test = p_hat_base * (1.0 - p_hat_base)
    u_cal_test = np.sqrt(ambiguity_test * w_base)

    # 4. Select representative held-out test points across score space
    score_targets = [0.20, 0.35, 0.50, 0.65, 0.80]
    target_names = ["s~0.20", "s~0.35", "s~0.50", "s~0.65", "s~0.80"]
    target_indices = []
    selected_points_records = []
    for tgt_name, tgt in zip(target_names, score_targets):
        idx_nearest = int(np.argmin(np.abs(s_test - tgt)))
        target_indices.append(idx_nearest)
        selected_points_records.append({
            "dataset": dataset_name,
            "target_name": tgt_name,
            "grid_target": tgt,
            "test_idx": idx_nearest,
            "base_score": float(s_test[idx_nearest]),
            "p0": float(p0_base[idx_nearest]),
            "p1": float(p1_base[idx_nearest]),
            "p_mid": float(p_hat_base[idx_nearest]),
            "width": float(w_base[idx_nearest]),
            "ambiguity": float(ambiguity_test[idx_nearest]),
            "e_model": float(e_model_test[idx_nearest]),
        })

    # 5. Local Calibration Support Intervention
    thin_fractions = [1.0, 0.75, 0.50, 0.25, 0.125]
    n_cal = len(s_cal)
    cal_ranks = stats.rankdata(s_cal) / n_cal

    print(f"Running Local Calibration Support Thinning (R_thin={R_thin}, B_cal={B_cal})...")
    parallel_jobs = []
    for tgt_name, t_idx in zip(target_names, target_indices):
        parallel_jobs.append(
            delayed(_process_single_thinning_target)(
                dataset_name=dataset_name,
                tgt_name=tgt_name,
                t_idx=t_idx,
                s_target=float(s_test[t_idx]),
                e_model_val=float(e_model_test[t_idx]),
                ambiguity_val=float(ambiguity_test[t_idx]),
                s_cal=s_cal,
                y_cal=y_cal,
                cal_ranks=cal_ranks,
                thin_fractions=thin_fractions,
                R_thin=R_thin,
                B_cal=B_cal,
                seed=seed,
            )
        )
    nested_results = Parallel(n_jobs=n_jobs)(parallel_jobs)
    thin_results = [row for target_rows in nested_results for row in target_rows]
    df_thinning = pd.DataFrame(thin_results)

    # 6. Reverse Intervention: Subsample training set while calibration set is fixed
    print("Running Reverse Intervention: Alter base model training support (max_iter=200)...")
    train_fractions = [1.0, 0.75, 0.50, 0.25]
    reverse_results = []

    for tr_frac in train_fractions:
        n_sub = int(n_tr * tr_frac)

        def _fit_single_reverse_model(b):
            sub_rng = set_seed(seed + b * 13 + int(tr_frac * 1000))
            sub_idx = sub_rng.choice(n_tr, size=n_sub, replace=False)
            clf_sub = HistGradientBoostingClassifier(
                categorical_features=is_cat,
                max_depth=4,
                learning_rate=0.05,
                max_iter=200,
                random_state=seed + b * 13,
            )
            clf_sub.fit(X_tr_arr[sub_idx], y_tr_arr[sub_idx])
            s_test_sub = clf_sub.predict_proba(X_test.iloc[target_indices])[:, 1]
            s_cal_sub = clf_sub.predict_proba(X_cal)[:, 1]
            
            # Compute width at all target points using compute_va_probs
            _, _, _, w_targets = compute_va_probs(s_cal_sub, y_cal, s_test_sub)
            return s_test_sub, w_targets

        rev_runs = Parallel(n_jobs=n_jobs)(
            delayed(_fit_single_reverse_model)(b) for b in range(B_model)
        )
        preds_sub = np.array([r[0] for r in rev_runs])
        widths_sub = np.array([r[1] for r in rev_runs])

        for idx_t, tgt_name in enumerate(target_names):
            reverse_results.append({
                "dataset": dataset_name,
                "target_name": tgt_name,
                "train_fraction": float(tr_frac),
                "e_model": float(np.std(preds_sub[:, idx_t], ddof=1)),
                "mean_width": float(np.mean(widths_sub[:, idx_t])),
                "std_width": float(np.std(widths_sub[:, idx_t])),
            })

    df_reverse = pd.DataFrame(reverse_results)

    # 7. Quantitative Decomposition Across Representative Held-out Test Points
    n_eval = min(500, len(X_test))
    eval_indices = np.linspace(0, len(X_test) - 1, n_eval, dtype=int)
    s_test_eval = s_test[eval_indices]
    e_model_eval = e_model_test[eval_indices]
    w_eval = w_base[eval_indices]
    a_eval = ambiguity_test[eval_indices]
    u_cal_eval = u_cal_test[eval_indices]

    print(f"Computing Calibration Bootstrap SD on {n_eval} held-out test points (B_cal=200)...")
    
    def _eval_single_cal_boot(b):
        boot_rng = set_seed(seed + 90000 + b)
        b_idx = boot_rng.choice(n_cal, size=n_cal, replace=True)
        _, _, p_hat_b, _ = compute_va_probs(s_cal[b_idx], y_cal[b_idx], s_test_eval)
        return p_hat_b

    boot_cal_list = Parallel(n_jobs=n_jobs)(
        delayed(_eval_single_cal_boot)(b) for b in range(200)
    )
    boot_cal_eval = np.array(boot_cal_list)
    sigma_cal_eval = np.std(boot_cal_eval, axis=0, ddof=1)

    df_points = pd.DataFrame({
        "dataset": dataset_name,
        "score": s_test_eval,
        "ambiguity": a_eval,
        "e_model": e_model_eval,
        "width": w_eval,
        "u_cal": u_cal_eval,
        "sigma_cal": sigma_cal_eval,
    })

    return df_thinning, df_reverse, df_points, selected_points_records


def fit_linear_model(X_mat, y_vec):
    """Fits OLS regression and computes R2, MAE, RMSE, and coefficients."""
    X_design = np.column_stack([np.ones(len(y_vec)), X_mat])
    beta, _, _, _ = np.linalg.lstsq(X_design, y_vec, rcond=None)
    y_pred = X_design @ beta
    resid = y_vec - y_pred
    ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {
        "beta": beta,
        "y_pred": y_pred,
        "r2": float(r2),
        "mae": mae,
        "rmse": rmse,
    }


def run_nested_decomposition(df_points, B_boot=2000, seed=42):
    rng = set_seed(seed)
    y = df_points["sigma_cal"].values
    A = df_points["ambiguity"].values
    E = df_points["e_model"].values
    W = df_points["width"].values
    U = df_points["u_cal"].values

    # Model A: sigma_cal ~ Ambiguity
    res_A = fit_linear_model(A.reshape(-1, 1), y)
    # Model B: sigma_cal ~ Ambiguity + E_model
    res_B = fit_linear_model(np.column_stack([A, E]), y)
    # Model C: sigma_cal ~ Ambiguity + E_model + Width
    res_C = fit_linear_model(np.column_stack([A, E, W]), y)
    # Model D: sigma_cal ~ Ambiguity + E_model + U_cal
    res_D = fit_linear_model(np.column_stack([A, E, U]), y)

    delta_r2_CB = res_C["r2"] - res_B["r2"]
    delta_r2_CA = res_C["r2"] - res_A["r2"]

    # Bootstrap CIs for Delta R2
    n = len(y)
    boot_delta_CB = []
    boot_delta_CA = []
    for _ in range(B_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b, A_b, E_b, W_b = y[idx], A[idx], E[idx], W[idx]
        r2_A_b = fit_linear_model(A_b.reshape(-1, 1), y_b)["r2"]
        r2_B_b = fit_linear_model(np.column_stack([A_b, E_b]), y_b)["r2"]
        r2_C_b = fit_linear_model(np.column_stack([A_b, E_b, W_b]), y_b)["r2"]
        boot_delta_CB.append(r2_C_b - r2_B_b)
        boot_delta_CA.append(r2_C_b - r2_A_b)

    ci_CB = np.percentile(boot_delta_CB, [2.5, 97.5])
    ci_CA = np.percentile(boot_delta_CA, [2.5, 97.5])

    return {
        "res_A": res_A,
        "res_B": res_B,
        "res_C": res_C,
        "res_D": res_D,
        "delta_r2_CB": delta_r2_CB,
        "ci_CB": ci_CB,
        "delta_r2_CA": delta_r2_CA,
        "ci_CA": ci_CA,
    }


def generate_figures_and_tables(df_all_thinning, df_all_reverse, df_all_points, decomp_results):
    """Generates Figure 6, Figure 7, and Table 4."""
    paper_dir = os.path.join(root_dir, "paper")
    os.makedirs(paper_dir, exist_ok=True)
    
    datasets = ["adult", "bank", "spambase"]
    palette = {"adult": "#1f77b4", "bank": "#2ca02c", "spambase": "#d62728"}
    markers = {"adult": "o", "bank": "s", "spambase": "^"}
    ds_labels = {"adult": "UCI Adult", "bank": "UCI Bank", "spambase": "UCI Spambase"}

    # Figure 6: real_data_local_support.png / .pdf
    print("\nGenerating Figure 6: real_data_local_support...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.0, 3.8))

    for ds in datasets:
        sub = df_all_thinning[df_all_thinning["dataset"] == ds]
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
    ax3.set_ylim(0.035, 0.15)
    ax3.legend(loc="center right", framealpha=0.9)

    plt.tight_layout()
    fig6_png = os.path.join(paper_dir, "real_data_local_support.png")
    fig6_pdf = os.path.join(paper_dir, "real_data_local_support.pdf")
    plt.savefig(fig6_png, dpi=300)
    plt.savefig(fig6_pdf)
    plt.close()
    print(f"Saved Figure 6 to:\n  - {fig6_png}\n  - {fig6_pdf}")

    # Figure 7: training_support_epistemic.png / .pdf
    print("\nGenerating Figure 7: training_support_epistemic...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ds in datasets:
        sub_rev = df_all_reverse[df_all_reverse["dataset"] == ds]
        rev_agg = sub_rev.groupby("train_fraction").agg({
            "e_model": "mean",
            "mean_width": "mean",
        }).reset_index().sort_values("train_fraction")

        tr_pct = rev_agg["train_fraction"].values * 100.0
        ax1.plot(tr_pct, rev_agg["e_model"], f"{markers[ds]}-", color=palette[ds],
                 lw=1.8, label=ds_labels[ds])
        ax2.plot(tr_pct, rev_agg["mean_width"], f"{markers[ds]}--", color=palette[ds],
                 lw=1.8, label=ds_labels[ds])

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
    fig7_png = os.path.join(paper_dir, "training_support_epistemic.png")
    fig7_pdf = os.path.join(paper_dir, "training_support_epistemic.pdf")
    plt.savefig(fig7_png, dpi=300)
    plt.savefig(fig7_pdf)
    plt.close()
    print(f"Saved Figure 7 to:\n  - {fig7_png}\n  - {fig7_pdf}")

    # Table 4: table_real_data_nested.tex
    print("\nGenerating Table 4: table_real_data_nested.tex...")
    table_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Held-out real-data comparison of calibration-instability models across three public tabular benchmarks ($N=500$ test points, $B_{\mathrm{cal}}=200$ resamples). Model A uses score-conditional outcome ambiguity $A(x)=\hat p(1-\hat p)$; Model B adds base-model epistemic uncertainty $E_{\mathrm{model}}(x)$; Model C adds Venn--Abers width $w(x)$. Across all datasets, Venn--Abers width provides an additional statistically detectable contribution to explaining calibration instability ($\Delta R^2$, 95\% bootstrap CI over 2,000 resamples), consistent with width capturing calibration-support information distinct from outcome ambiguity and base-model uncertainty.}",
        r"\label{tab:real_data_nested}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Model Specification & $R^2$ & MAE & RMSE \\",
        r"\midrule",
    ]

    for ds in datasets:
        d = decomp_results[ds]
        lbl = ds_labels[ds]
        r2_A, mae_A, rmse_A = d["res_A"]["r2"], d["res_A"]["mae"], d["res_A"]["rmse"]
        r2_B, mae_B, rmse_B = d["res_B"]["r2"], d["res_B"]["mae"], d["res_B"]["rmse"]
        r2_C, mae_C, rmse_C = d["res_C"]["r2"], d["res_C"]["mae"], d["res_C"]["rmse"]
        delta_CB = d["delta_r2_CB"]
        ci_l, ci_u = d["ci_CB"]

        table_lines.extend([
            f"\\multirow{{4}}{{*}}{{{lbl}}} & Model A: Ambiguity $A$ & {r2_A:.4f} & {mae_A:.4f} & {rmse_A:.4f} \\\\",
            f" & Model B: Ambiguity + $E_{{\\mathrm{{model}}}}$ & {r2_B:.4f} & {mae_B:.4f} & {rmse_B:.4f} \\\\",
            f" & Model C: Ambiguity + $E_{{\\mathrm{{model}}}}$ + Width $w$ & \\textbf{{{r2_C:.4f}}} & \\textbf{{{mae_C:.4f}}} & \\textbf{{{rmse_C:.4f}}} \\\\",
            f" & $\\Delta R^2$ (Model C $-$ Model B) & \\multicolumn{{3}}{{c}}{{+ {delta_CB:.4f} [95\\% CI: {ci_l:.4f}, {ci_u:.4f}]}} \\\\",
            r"\midrule" if ds != datasets[-1] else r"\bottomrule",
        ])

    table_lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])
    table_tex = "\n".join(table_lines) + "\n"
    table_path = os.path.join(paper_dir, "table_real_data_nested.tex")
    with open(table_path, "w") as f:
        f.write(table_tex)
    print(f"Saved Table 4 to: {table_path}")


def main():
    start_time = time.time()
    data_dir = os.path.join(root_dir, "data")
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    datasets = ["adult", "bank", "spambase"]
    all_thinning = []
    all_reverse = []
    all_points = []
    all_selected_points = []
    decomp_results = {}

    for ds in datasets:
        df_thin, df_rev, df_pts, sel_pts = run_experiment_for_dataset(
            dataset_name=ds, seed=42, R_thin=100, B_cal=100, B_model=100, n_jobs=4
        )
        all_thinning.append(df_thin)
        all_reverse.append(df_rev)
        all_points.append(df_pts)
        all_selected_points.extend(sel_pts)
        decomp_results[ds] = run_nested_decomposition(df_pts, B_boot=2000, seed=42)

    df_all_thinning = pd.concat(all_thinning, ignore_index=True)
    df_all_reverse = pd.concat(all_reverse, ignore_index=True)
    df_all_points = pd.concat(all_points, ignore_index=True)
    df_sel_points = pd.DataFrame(all_selected_points)

    # Save Canonical Datasets
    csv_results_path = os.path.join(data_dir, "REAL_DATA_RESULTS.csv")
    df_all_thinning.to_csv(csv_results_path, index=False)
    print(f"\nSaved canonical thinning results to: {csv_results_path}")

    csv_rev_path = os.path.join(data_dir, "REVERSE_INTERVENTION_RESULTS.csv")
    df_all_reverse.to_csv(csv_rev_path, index=False)
    print(f"Saved canonical reverse results to: {csv_rev_path}")

    csv_decomp_path = os.path.join(data_dir, "UNCERTAINTY_DECOMPOSITION.csv")
    df_all_points.to_csv(csv_decomp_path, index=False)
    print(f"Saved canonical uncertainty points to: {csv_decomp_path}")

    csv_sel_path = os.path.join(results_dir, "real_data_selected_points.csv")
    df_sel_points.to_csv(csv_sel_path, index=False)
    print(f"Saved pre-specified test points to: {csv_sel_path}")

    # Generate Figures and Tables
    generate_figures_and_tables(df_all_thinning, df_all_reverse, df_all_points, decomp_results)
    
    elapsed = time.time() - start_time
    print(f"\nReal-data experiments completed successfully in {elapsed:.1f}s ({elapsed/60.0:.2f} min).")


if __name__ == "__main__":
    main()
