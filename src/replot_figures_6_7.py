"""
src/replot_figures_6_7.py

Script to replot Figures 6 and 7 with properly formatted North-East legends
and sufficient y-axis headroom to prevent any overlap between curve data and legend boxes.
Saves updated plots and data to both paper/ and target repositories.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from venn_abers import VennAbers
import warnings
warnings.filterwarnings('ignore')

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

def compute_va_probs(s_cal, y_cal, s_test):
    p_cal = np.zeros((len(s_cal), 2))
    p_cal[:, 1] = s_cal
    p_cal[:, 0] = 1.0 - s_cal

    p_test = np.zeros((len(s_test), 2))
    p_test[:, 1] = s_test
    p_test[:, 0] = 1.0 - s_test

    va = VennAbers()
    va.fit(p_cal, y_cal)
    _, pred = va.predict_proba(p_test)
    p0 = pred[:, 0]
    p1 = pred[:, 1]
    p_hat = (p0 + p1) / 2.0
    width = np.maximum(0.0, p1 - p0)
    return p0, p1, p_hat, width

def plot_figure_6(df_thinning, output_paths):
    print("Plotting Figure 6 (real_data_local_support)...")
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
            "e_model": "mean"
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
    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300 if p.endswith('.png') else None)
        print(f"Saved Figure 6 to {p}")
    plt.close()

def run_or_load_reverse(datasets, seed=42, B_model=100):
    rev_csv_candidates = [
        "data/REVERSE_INTERVENTION_RESULTS.csv",
        "/Users/ivanpetej/Projects/va-width-experiments/data/REVERSE_INTERVENTION_RESULTS.csv"
    ]
    for c in rev_csv_candidates:
        if os.path.exists(c):
            print(f"Loading existing reverse intervention results from {c}...")
            return pd.read_csv(c)

    print("Running reverse training intervention (subsampling training set, B_model=100)...")
    from real_data_calibration_support import load_dataset
    
    all_reverse = []
    rng = np.random.default_rng(seed)

    for ds in datasets:
        print(f"  Dataset: {ds}...")
        X_tr, y_tr, X_cal, y_cal, X_te, y_te, cat_cols = load_dataset(ds, seed)
        is_cat = [c in cat_cols for c in X_tr.columns]

        # Initial fit on full train to identify target instances
        clf_base = HistGradientBoostingClassifier(
            categorical_features=is_cat,
            max_depth=4,
            learning_rate=0.05,
            max_iter=100,
            random_state=seed
        )
        clf_base.fit(X_tr, y_tr)
        s_test = clf_base.predict_proba(X_te)[:, 1]

        target_scores = [0.20, 0.35, 0.50, 0.65, 0.80]
        target_indices = []
        for t_s in target_scores:
            idx = int(np.argmin(np.abs(s_test - t_s)))
            target_indices.append(idx)

        X_tr_arr = X_tr.to_numpy() if hasattr(X_tr, 'to_numpy') else np.array(X_tr)
        y_tr_arr = np.array(y_tr)
        n_tr = len(X_tr)

        train_fractions = [1.0, 0.75, 0.50, 0.25]
        reverse_results = []

        for tr_frac in train_fractions:
            n_sub = int(n_tr * tr_frac)
            preds_sub = np.zeros((B_model, len(target_indices)))
            widths_sub = np.zeros((B_model, len(target_indices)))

            for b in range(B_model):
                sub_idx = rng.choice(n_tr, size=n_sub, replace=False)
                clf_sub = HistGradientBoostingClassifier(
                    categorical_features=is_cat,
                    max_depth=4,
                    learning_rate=0.05,
                    max_iter=100,
                    random_state=seed + b * 13
                )
                clf_sub.fit(X_tr_arr[sub_idx], y_tr_arr[sub_idx])
                s_test_sub = clf_sub.predict_proba(X_te.iloc[target_indices])[:, 1]
                s_cal_sub = clf_sub.predict_proba(X_cal)[:, 1]
                preds_sub[b, :] = s_test_sub

                _, _, _, w_pts = compute_va_probs(s_cal_sub, y_cal, s_test_sub)
                widths_sub[b, :] = w_pts

            for idx_t, tgt_name in enumerate(["s~0.20", "s~0.35", "s~0.50", "s~0.65", "s~0.80"]):
                reverse_results.append({
                    "dataset": ds,
                    "target_name": tgt_name,
                    "train_fraction": tr_frac,
                    "e_model": float(np.std(preds_sub[:, idx_t])),
                    "mean_width": float(np.mean(widths_sub[:, idx_t])),
                    "std_width": float(np.std(widths_sub[:, idx_t]))
                })

        all_reverse.extend(reverse_results)

    df_rev = pd.DataFrame(all_reverse)
    os.makedirs("data", exist_ok=True)
    df_rev.to_csv("data/REVERSE_INTERVENTION_RESULTS.csv", index=False)
    os.makedirs("/Users/ivanpetej/Projects/va-width-experiments/data", exist_ok=True)
    df_rev.to_csv("/Users/ivanpetej/Projects/va-width-experiments/data/REVERSE_INTERVENTION_RESULTS.csv", index=False)
    print("Saved data/REVERSE_INTERVENTION_RESULTS.csv")
    return df_rev

def plot_figure_7(df_reverse, output_paths):
    print("Plotting Figure 7 (training_support_epistemic)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    
    palette = {"adult": "#1f77b4", "bank": "#2ca02c", "spambase": "#d62728"}
    markers = {"adult": "o", "bank": "s", "spambase": "^"}
    ds_labels = {"adult": "UCI Adult", "bank": "UCI Bank", "spambase": "UCI Spambase"}
    datasets = ["adult", "bank", "spambase"]

    for ds in datasets:
        sub_rev = df_reverse[df_reverse["dataset"] == ds]
        rev_agg = sub_rev.groupby("train_fraction").agg({
            "e_model": "mean",
            "mean_width": "mean"
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
    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300 if p.endswith('.png') else None)
        print(f"Saved Figure 7 to {p}")
    plt.close()

if __name__ == "__main__":
    thin_csv = "data/REAL_DATA_RESULTS.csv"
    if not os.path.exists(thin_csv):
        thin_csv = "/Users/ivanpetej/Projects/va-width-experiments/data/REAL_DATA_RESULTS.csv"
    df_thinning = pd.read_csv(thin_csv)

    fig6_paths = [
        "paper/real_data_local_support.png",
        "paper/real_data_local_support.pdf",
        "/Users/ivanpetej/Projects/va-width-experiments/paper/real_data_local_support.png",
        "/Users/ivanpetej/Projects/va-width-experiments/paper/real_data_local_support.pdf"
    ]
    plot_figure_6(df_thinning, fig6_paths)

    datasets = ["adult", "bank", "spambase"]
    df_reverse = run_or_load_reverse(datasets, seed=42, B_model=100)

    fig7_paths = [
        "paper/training_support_epistemic.png",
        "paper/training_support_epistemic.pdf",
        "/Users/ivanpetej/Projects/va-width-experiments/paper/training_support_epistemic.png",
        "/Users/ivanpetej/Projects/va-width-experiments/paper/training_support_epistemic.pdf"
    ]
    plot_figure_7(df_reverse, fig7_paths)
    print("Replotting complete!")
