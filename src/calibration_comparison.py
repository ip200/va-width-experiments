import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from venn_abers import VennAbers

# 1. Define Histogram Binning Class
class HistogramBinning:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_values = None

    def fit(self, s, y):
        # Equal width bins on [-1, 1]
        self.bin_edges = np.linspace(-1.0, 1.0, self.n_bins + 1)
        self.bin_values = np.zeros(self.n_bins)
        # Compute bin values (empirical means)
        for i in range(self.n_bins):
            mask = (s >= self.bin_edges[i]) & (s < self.bin_edges[i+1])
            if i == self.n_bins - 1:
                mask |= (s == self.bin_edges[i+1])
            if np.sum(mask) > 0:
                self.bin_values[i] = np.mean(y[mask])
            else:
                self.bin_values[i] = np.mean(y)

    def predict(self, s):
        s = np.asarray(s)
        preds = np.zeros_like(s, dtype=float)
        for i in range(self.n_bins):
            mask = (s >= self.bin_edges[i]) & (s < self.bin_edges[i+1])
            if i == self.n_bins - 1:
                mask |= (s == self.bin_edges[i+1])
            preds[mask] = self.bin_values[i]
        return preds

# 2. Venn-Abers Point and Interval Wrapper
class VennAbersWrapper:
    def __init__(self):
        self.va = VennAbers()

    def fit(self, s, y):
        # Sigmoid transform to construct base probabilities
        p_cal = np.zeros((len(s), 2))
        p_cal[:, 1] = 1.0 / (1.0 + np.exp(-s))
        p_cal[:, 0] = 1.0 - p_cal[:, 1]
        self.va.fit(p_cal, y)

    def predict(self, s):
        p_test = np.zeros((len(s), 2))
        p_test[:, 1] = 1.0 / (1.0 + np.exp(-s))
        p_test[:, 0] = 1.0 - p_test[:, 1]
        
        _, pred = self.va.predict_proba(p_test)
        p0, p1 = pred[:, 0], pred[:, 1]
        p_point = p1 / (1.0 - p0 + p1)
        return p_point, p0, p1

# 3. Setup settings
def get_theta(s, setting):
    if setting == 'sigmoidal':
        return 1.0 / (1.0 + np.exp(-5.0 * s))
    elif setting == 'step':
        return 0.2 + 0.6 * (s > 0.0).astype(float)
    elif setting == 'peak':
        return 0.5 - 0.4 * (s ** 2)
    elif setting == 'wave':
        return 0.5 + 0.3 * np.sin(np.pi * s)
    else:
        raise ValueError("Unknown setting")

# Brier Score Decomposition (Method A)
def brier_decomposition(y_true, y_pred, n_bins=20):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_pred, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    
    n_total = len(y_true)
    y_bar = np.mean(y_true)
    
    reliability = 0.0
    resolution = 0.0
    
    for k in range(n_bins):
        mask = (bin_ids == k)
        n_k = np.sum(mask)
        if n_k > 0:
            p_bar_k = np.mean(y_pred[mask])
            y_bar_k = np.mean(y_true[mask])
            reliability += (n_k / n_total) * ((p_bar_k - y_bar_k) ** 2)
            resolution += (n_k / n_total) * ((y_bar_k - y_bar) ** 2)
            
    uncertainty = y_bar * (1.0 - y_bar)
    brier_score = np.mean((y_true - y_pred) ** 2)
    return brier_score, reliability, resolution, uncertainty

def main():
    rng = np.random.default_rng(42)
    
    # Target paper directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.join(current_dir, '..', 'paper')
    os.makedirs(paper_dir, exist_ok=True)
    
    settings = ['sigmoidal', 'step', 'peak', 'wave']
    setting_labels = {
        'sigmoidal': 'Monotonic Sigmoidal',
        'step': 'Monotonic Step-like',
        'peak': 'Non-Monotonic Peak',
        'wave': 'Non-Monotonic Wave'
    }
    
    # Prepare tables / plots
    # Method A and B
    N_cal = 1000
    N_test = 10000
    
    # Figures for Method B
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    
    brier_results = []
    
    for idx, setting in enumerate(settings):
        # Generate Calibration data
        s_cal = rng.uniform(-1, 1, size=N_cal)
        theta_cal = get_theta(s_cal, setting)
        y_cal = rng.binomial(1, theta_cal)
        
        # Generate Test data
        s_test = rng.uniform(-1, 1, size=N_test)
        theta_test = get_theta(s_test, setting)
        y_test = rng.binomial(1, theta_test)
        
        # Fit models
        # Platt Scaling
        ps = LogisticRegression()
        ps.fit(s_cal.reshape(-1, 1), y_cal)
        pred_ps = ps.predict_proba(s_test.reshape(-1, 1))[:, 1]
        
        # Isotonic Regression
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(s_cal, y_cal)
        pred_ir = ir.predict(s_test)
        
        # Histogram Binning (10 bins)
        hb = HistogramBinning(n_bins=10)
        hb.fit(s_cal, y_cal)
        pred_hb = hb.predict(s_test)
        
        # Venn-Abers
        va = VennAbersWrapper()
        va.fit(s_cal, y_cal)
        pred_va, p0_test, p1_test = va.predict(s_test)
        
        # Compute decompositions
        models = {
            'VAP': pred_va,
            'IR': pred_ir,
            'PS': pred_ps,
            'HB': pred_hb
        }
        
        for name, pred in models.items():
            bs, rel, res, unc = brier_decomposition(y_test, pred, n_bins=20)
            brier_results.append({
                'setting': setting_labels[setting],
                'calibrator': name,
                'brier': bs,
                'reliability': rel,
                'resolution': res,
                'uncertainty': unc
            })
            
        # Plotting (Method B)
        # Sort for clean lines
        sort_idx = np.argsort(s_test)
        s_plot = s_test[sort_idx]
        
        ax = axes[idx]
        ax.plot(s_plot, theta_test[sort_idx], 'k-', lw=2.5, label='True Probability')
        ax.plot(s_plot, pred_ps[sort_idx], 'r--', lw=1.5, label='Platt Scaling')
        ax.plot(s_plot, pred_ir[sort_idx], 'g-.', lw=1.5, label='Isotonic Reg.')
        ax.plot(s_plot, pred_hb[sort_idx], 'm:', lw=1.5, label='Histogram Binning')
        ax.plot(s_plot, pred_va[sort_idx], 'b-', lw=1.5, label='Venn-Abers')
        
        # Add VAP intervals as shaded region
        ax.fill_between(s_plot, p0_test[sort_idx], p1_test[sort_idx], color='blue', alpha=0.1, label='VAP $[p_0, p_1]$')
        
        ax.set_title(setting_labels[setting], fontsize=13)
        ax.set_xlabel('Score ($s$)', fontsize=11)
        ax.set_ylabel('Probability', fontsize=11)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)
            
    plt.tight_layout()
    fits_plot_path = os.path.join(paper_dir, 'calibration_comparison_fits.png')
    plt.savefig(fits_plot_path, dpi=300)
    plt.close()
    print(f"Calibration fits plot saved to {fits_plot_path}")
    
    # Write LaTeX Table with bolded best values
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in brier_results:
        grouped[row['setting']].append(row)
        
    for setting_name, rows in grouped.items():
        min_brier = min(r['brier'] for r in rows)
        min_reliability = min(r['reliability'] for r in rows)
        max_resolution = max(r['resolution'] for r in rows)
        for r in rows:
            r['is_best_brier'] = np.isclose(r['brier'], min_brier, atol=1e-4)
            r['is_best_reliability'] = np.isclose(r['reliability'], min_reliability, atol=1e-4)
            r['is_best_resolution'] = np.isclose(r['resolution'], max_resolution, atol=1e-4)




            
    table_path = os.path.join(paper_dir, 'table_brier_decomp.tex')
    with open(table_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Brier score decomposition (Reliability vs. Resolution) across different monotonicity settings. Lower Brier score and Reliability are better, while higher Resolution is better. Uncertainty represents the intrinsic label variance.}\n")
        f.write("\\label{tab:brier_decomposition}\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\toprule\n")
        f.write("Setting & Calibrator & Brier Score $\\downarrow$ & Reliability $\\downarrow$ & Resolution $\\uparrow$ & Uncertainty \\\\\n")
        f.write("\\midrule\n")
        
        current_setting = ""
        for row in brier_results:
            brier_str = f"\\textbf{{{row['brier']:.4f}}}" if row.get('is_best_brier') else f"{row['brier']:.4f}"
            rel_str = f"\\textbf{{{row['reliability']:.4f}}}" if row.get('is_best_reliability') else f"{row['reliability']:.4f}"
            res_str = f"\\textbf{{{row['resolution']:.4f}}}" if row.get('is_best_resolution') else f"{row['resolution']:.4f}"
            unc_str = f"{row['uncertainty']:.4f}"
            
            if row['setting'] != current_setting:
                if current_setting != "":
                    f.write("\\midrule\n")
                current_setting = row['setting']
                f.write(f"\\multirow{{4}}{{*}}{{{current_setting}}} & {row['calibrator']} & {brier_str} & {rel_str} & {res_str} & {unc_str} \\\\\n")
            else:
                f.write(f" & {row['calibrator']} & {brier_str} & {rel_str} & {res_str} & {unc_str} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX Table saved to {table_path}")

    
    # Method C: Epistemic Uncertainty vs. Resampling Instability
    print("Running Method C (Bootstrap Resampling Uncertainty)...")
    N_c = 500
    M = 100
    
    # Grid of test points
    s_grid = np.linspace(-1.0, 1.0, 200)
    theta_grid = get_theta(s_grid, 'sigmoidal')
    
    # Containers for bootstrap predictions
    preds_va_boot = []
    preds_ir_boot = []
    preds_ps_boot = []
    preds_hb_boot = []
    
    # We will also compute VAP widths on a single representative calibration set
    # to compare its local interval width against the bootstrap variance
    s_cal_single = rng.uniform(-1, 1, size=N_c)
    y_cal_single = rng.binomial(1, get_theta(s_cal_single, 'sigmoidal'))
    
    va_single = VennAbersWrapper()
    va_single.fit(s_cal_single, y_cal_single)
    _, p0_single, p1_single = va_single.predict(s_grid)
    vap_width_single = p1_single - p0_single
    
    # Run bootstrap resamples
    for m in range(M):
        # Sample with replacement
        idx_boot = rng.choice(N_c, size=N_c, replace=True)
        s_boot = s_cal_single[idx_boot]
        y_boot = y_cal_single[idx_boot]
        
        # Fit and predict for all methods
        ps = LogisticRegression()
        ps.fit(s_boot.reshape(-1, 1), y_boot)
        preds_ps_boot.append(ps.predict_proba(s_grid.reshape(-1, 1))[:, 1])
        
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(s_boot, y_boot)
        preds_ir_boot.append(ir.predict(s_grid))
        
        hb = HistogramBinning(n_bins=10)
        hb.fit(s_boot, y_boot)
        preds_hb_boot.append(hb.predict(s_grid))
        
        va = VennAbersWrapper()
        va.fit(s_boot, y_boot)
        p_pt, _, _ = va.predict(s_grid)
        preds_va_boot.append(p_pt)
        
    # Compute standard deviations across bootstrap runs
    std_ps = np.std(preds_ps_boot, axis=0)
    std_ir = np.std(preds_ir_boot, axis=0)
    std_hb = np.std(preds_hb_boot, axis=0)
    std_va = np.std(preds_va_boot, axis=0)
    
    # Plot Method C
    plt.figure(figsize=(9, 6))
    plt.plot(s_grid, vap_width_single, 'b-', lw=2.5, label='VAP Interval Width ($p_1 - p_0$)')
    plt.plot(s_grid, std_va, 'b--', lw=1.5, label='VAP Bootstrap Std Dev')
    plt.plot(s_grid, std_ir, 'g-.', lw=1.5, label='Isotonic Reg. Bootstrap Std Dev')
    plt.plot(s_grid, std_ps, 'r--', lw=1.5, label='Platt Scaling Bootstrap Std Dev')
    plt.plot(s_grid, std_hb, 'm:', lw=1.5, label='Histogram Binning Bootstrap Std Dev')
    
    plt.title('Comparison of Finite-Sample Instability (Bootstrap Std. Dev. vs. VAP Width)', fontsize=13)
    plt.xlabel('Score ($s$)', fontsize=11)
    plt.ylabel('Uncertainty / Instability Measure', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    bootstrap_plot_path = os.path.join(paper_dir, 'calibration_uncertainty_bootstrap.png')
    plt.savefig(bootstrap_plot_path, dpi=300)
    plt.close()
    print(f"Bootstrap instability plot saved to {bootstrap_plot_path}")

    # Method C Scatter Plot: VAP Width vs Bootstrap SD
    from scipy.stats import pearsonr, spearmanr
    p_corr, _ = pearsonr(std_va, vap_width_single)
    s_corr, _ = spearmanr(std_va, vap_width_single)
    plt.figure(figsize=(7, 5))
    plt.scatter(std_va, vap_width_single, color='blue', alpha=0.6, edgecolors='none')
    plt.title(f'VAP Width vs. Bootstrap Instability (Pearson={p_corr:.3f}, Spearman={s_corr:.3f})')
    plt.xlabel('Bootstrap Standard Deviation of VAP')
    plt.ylabel('VAP Interval Width ($p_1 - p_0$)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_plot_path = os.path.join(paper_dir, 'alternative_calibrator_instability.png')
    plt.savefig(scatter_plot_path, dpi=300)
    plt.close()
    print(f"Scatter plot saved to {scatter_plot_path}")

    # CIFAR-10H Macro-Averaged Correlation Experiment (Priority 4)
    print("Running CIFAR-10H Macro-Averaged Correlation Experiment...")
    n_items_c, d_c, n_ann_c = 2000, 64, 50
    rng_c = np.random.default_rng(0)
    X_c = rng_c.normal(size=(n_items_c, d_c))
    logits_c = rng_c.normal(size=(n_items_c, 10))
    exp_logits_c = np.exp(logits_c - np.max(logits_c, axis=1, keepdims=True))
    soft_labels_c = exp_logits_c / np.sum(exp_logits_c, axis=1, keepdims=True)

    train_size_c = 1000
    cal_size_c = 500
    test_size_c = 500

    X_train_c = X_c[:train_size_c]
    X_cal_c = X_c[train_size_c:train_size_c+cal_size_c]
    X_test_c = X_c[train_size_c+cal_size_c:]

    soft_train_c = soft_labels_c[:train_size_c]
    soft_cal_c = soft_labels_c[train_size_c:train_size_c+cal_size_c]
    soft_test_c = soft_labels_c[train_size_c+cal_size_c:]

    pearson_sd, spearman_sd = [], []
    pearson_dis, spearman_dis = [], []
    pearson_ent, spearman_ent = [], []

    from sklearn.ensemble import HistGradientBoostingClassifier
    import scipy.stats as stats

    for class_idx in range(10):
        y_mean_train = soft_train_c[:, class_idx]
        y_mean_cal = soft_cal_c[:, class_idx]
        y_mean_test = soft_test_c[:, class_idx]
        
        y_hard_train = (np.argmax(soft_train_c, axis=1) == class_idx).astype(int)
        
        base_clf = HistGradientBoostingClassifier(max_depth=5, random_state=42 + class_idx)
        base_clf.fit(X_train_c, y_hard_train)
        scores_cal = base_clf.predict_proba(X_cal_c)[:, 1]
        scores_test = base_clf.predict_proba(X_test_c)[:, 1]
        
        p_cal = np.zeros((len(scores_cal), 2))
        p_cal[:, 1] = scores_cal
        p_cal[:, 0] = 1.0 - scores_cal
        
        p_test = np.zeros((len(scores_test), 2))
        p_test[:, 1] = scores_test
        p_test[:, 0] = 1.0 - scores_test
        
        y_cal_real = rng_c.binomial(1, y_mean_cal)
        
        va = VennAbers()
        va.fit(p_cal, y_cal_real)
        _, pred = va.predict_proba(p_test)
        widths = pred[:, 1] - pred[:, 0]
        
        # Bootstrap VAP
        M_b = 100
        boot_preds = []
        for m in range(M_b):
            idx = rng_c.choice(cal_size_c, size=cal_size_c, replace=True)
            va_b = VennAbers()
            va_b.fit(p_cal[idx], y_cal_real[idx])
            _, pred_b = va_b.predict_proba(p_test)
            pb0, pb1 = pred_b[:, 0], pred_b[:, 1]
            p_hat = 0.5 * (pb0 + pb1)
            boot_preds.append(p_hat)
        boot_sd = np.std(boot_preds, axis=0)
        
        disagreement = 2.0 * y_mean_test * (1.0 - y_mean_test)
        eps = 1e-12
        entropy = -y_mean_test * np.log2(y_mean_test + eps) - (1.0 - y_mean_test) * np.log2(1.0 - y_mean_test + eps)
        
        pearson_sd.append(stats.pearsonr(widths, boot_sd)[0])
        spearman_sd.append(stats.spearmanr(widths, boot_sd)[0])
        
        pearson_dis.append(stats.pearsonr(widths, disagreement)[0])
        spearman_dis.append(stats.spearmanr(widths, disagreement)[0])
        
        pearson_ent.append(stats.pearsonr(widths, entropy)[0])
        spearman_ent.append(stats.spearmanr(widths, entropy)[0])

    # Save canonical CSV for CIFAR correlations
    data_dir = os.path.join(os.path.dirname(paper_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    cifar_df = pd.DataFrame([
        {"Quantity": "Bootstrap Instability", "Pearson": float(np.mean(pearson_sd)), "Spearman": float(np.mean(spearman_sd))},
        {"Quantity": "Annotator Disagreement", "Pearson": float(np.mean(pearson_dis)), "Spearman": float(np.mean(spearman_dis))},
        {"Quantity": "Annotator Entropy", "Pearson": float(np.mean(pearson_ent)), "Spearman": float(np.mean(spearman_ent))},
    ])
    cifar_csv_path = os.path.join(data_dir, "table_cifar_correlations.csv")
    cifar_df.to_csv(cifar_csv_path, index=False)
    print(f"CIFAR correlations CSV saved to {cifar_csv_path}")

    # Output LaTeX table for CIFAR correlations
    cifar_table_path = os.path.join(paper_dir, 'table_cifar_correlations.tex')
    with open(cifar_table_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Macro-averaged Pearson and Spearman correlations of Venn--Abers interval width with bootstrap instability, annotator disagreement, and annotator entropy in the synthetic CIFAR-10H-inspired crowd-annotation experiment.}\n")
        f.write("\\label{tab:cifar_correlations}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Quantity & Pearson Correlation & Spearman Correlation \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Bootstrap Instability & {np.mean(pearson_sd):.4f} & {np.mean(spearman_sd):.4f} \\\\\n")
        f.write(f"Annotator Disagreement & {np.mean(pearson_dis):.4f} & {np.mean(spearman_dis):.4f} \\\\\n")
        f.write(f"Annotator Entropy & {np.mean(pearson_ent):.4f} & {np.mean(spearman_ent):.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"CIFAR correlations table saved to {cifar_table_path}")

if __name__ == '__main__':
    main()

