#!/usr/bin/env python3
"""
src/generate_manifest.py

Reads canonical data files from data/ and generates:
1. results/paper_results_manifest.json (machine-readable numerical results, full precision)
2. paper/generated_results.tex (LaTeX macros for manuscript)

Run:
    python src/generate_manifest.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress, t

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(root_dir, "data")
    paper_dir = os.path.join(root_dir, "paper")
    results_dir = os.path.join(root_dir, "results")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)
    
    manifest = {
        "metadata": {
            "description": "Canonical numerical results for 'The Meaning and Scaling of Venn-Abers Probability Intervals'",
            "tolerance": 5e-5,
        }
    }
    
    # -------------------------------------------------------------
    # 1. Table 1: Calibration Resampling Baseline
    # -------------------------------------------------------------
    df_cr = pd.read_csv(os.path.join(data_dir, "calibration_resampling_baseline.csv"))
    cr_records = []
    for _, r in df_cr.iterrows():
        cr_records.append({
            "n_cal": int(r["n_cal"]),
            "mean_width": float(r["mean_width"]),
            "mean_sd": float(r["mean_sd"]),
            "corr": float(r["corr"]),
            "fresh_brier": float(r["fresh_brier"]),
        })
    manifest["table_1_calibration_resampling"] = cr_records

    # -------------------------------------------------------------
    # 2. Figure 2: Classifier Bootstrap (N=500)
    # -------------------------------------------------------------
    df_boot = pd.read_csv(os.path.join(data_dir, "classifier_bootstrap_n500.csv"))
    w = df_boot["width"].values
    u = df_boot["u_cal"].values
    sd = df_boot["bootstrap_sd"].values
    n_test = len(df_boot)
    
    pw, _ = pearsonr(w, sd)
    sw, _ = spearmanr(w, sd)
    pu, _ = pearsonr(u, sd)
    su, _ = spearmanr(u, sd)
    
    fit_idx = np.arange(0, n_test, 2)
    eval_idx = np.arange(1, n_test, 2)
    c_w = float(np.sum(sd[fit_idx] * w[fit_idx]) / np.sum(w[fit_idx] ** 2))
    c_u = float(np.sum(sd[fit_idx] * u[fit_idx]) / np.sum(u[fit_idx] ** 2))
    mae_w = float(np.mean(np.abs(sd[eval_idx] - c_w * w[eval_idx])))
    mae_u = float(np.mean(np.abs(sd[eval_idx] - c_u * u[eval_idx])))
    
    manifest["figure_2_classifier_bootstrap"] = {
        "n_test": n_test,
        "width_pearson": float(pw),
        "width_spearman": float(sw),
        "width_mae": float(mae_w),
        "c_w": float(c_w),
        "ucal_pearson": float(pu),
        "ucal_spearman": float(su),
        "ucal_mae": float(mae_u),
        "c_u": float(c_u),
    }

    # -------------------------------------------------------------
    # 3. Figure 3: Idealised Scaling Laws
    # -------------------------------------------------------------
    df_id = pd.read_csv(os.path.join(data_dir, "SCALING_IDEALISED.csv"))
    log_n = np.log(df_id["n_cal"].values)
    dof = len(df_id) - 2
    t_crit = t.ppf(0.975, dof)
    
    scaling_res = {}
    for col, key in [("mean_width", "width"), ("sd_p_mid", "sd"), ("mean_ucal", "ucal")]:
        log_y = np.log(df_id[col].values)
        res = linregress(log_n, log_y)
        ci_lower = res.slope - t_crit * res.stderr
        ci_upper = res.slope + t_crit * res.stderr
        scaling_res[f"{key}_slope"] = float(res.slope)
        scaling_res[f"{key}_ci_lower"] = float(ci_lower)
        scaling_res[f"{key}_ci_upper"] = float(ci_upper)
        scaling_res[f"{key}_r2"] = float(res.rvalue ** 2)
    manifest["figure_3_idealised_scaling"] = scaling_res

    # -------------------------------------------------------------
    # 4. Table 2: W2 Exponent Progression
    # -------------------------------------------------------------
    df_w2 = pd.read_csv(os.path.join(data_dir, "proposition1_convergence.csv"))
    w2_dict = {}
    for _, r in df_w2.iterrows():
        n = int(r["n_cal"])
        w2_dict[str(n)] = {
            "beta_rho": float(r["beta_rho"]),
            "beta_rho_ci_lower": float(r["beta_rho_ci_lower"]),
            "beta_rho_ci_upper": float(r["beta_rho_ci_upper"]),
            "beta_v": float(r["beta_v"]),
            "beta_v_ci_lower": float(r["beta_v_ci_lower"]),
            "beta_v_ci_upper": float(r["beta_v_ci_upper"]),
            "beta_s": float(r["beta_s"]),
            "beta_s_ci_lower": float(r["beta_s_ci_lower"]),
            "beta_s_ci_upper": float(r["beta_s_ci_upper"]),
            "mae": float(r["mean_abs_err"]),
            "r2": float(r["r2"]),
        }
    manifest["table_2_w2_exponent_progression"] = w2_dict

    # -------------------------------------------------------------
    # 5. Figure 5: Non-Monotonic Scaling Regimes
    # -------------------------------------------------------------
    df_nm = pd.read_csv(os.path.join(data_dir, "non_monotonic_scaling_laws.csv"))
    log_n_nm = np.log(df_nm["n_cal"].values)
    manifest["figure_5_non_monotonic"] = {
        "regime_a_slope": float(np.polyfit(log_n_nm, np.log(df_nm["width_mono"].values), 1)[0]),
        "regime_b_slope": float(np.polyfit(log_n_nm, np.log(df_nm["width_viol"].values), 1)[0]),
        "regime_c_slope": float(np.polyfit(log_n_nm, np.log(df_nm["width_ext_mid"].values), 1)[0]),
        "regime_d_slope": float(np.polyfit(log_n_nm, np.log(df_nm["width_ext_bound"].values), 1)[0]),
    }

    # -------------------------------------------------------------
    # 6. Table 3: CIFAR-10H Crowd Correlations
    # -------------------------------------------------------------
    df_cifar = pd.read_csv(os.path.join(data_dir, "table_cifar_correlations.csv"))
    cifar_dict = {}
    for _, r in df_cifar.iterrows():
        name = r["Quantity"].lower().replace(" ", "_")
        cifar_dict[name] = {
            "pearson": float(r["Pearson"]),
            "spearman": float(r["Spearman"]),
        }
    manifest["table_3_cifar_correlations"] = cifar_dict

    # -------------------------------------------------------------
    # 7. Figure 6: Real-Data Thinning Intervention
    # -------------------------------------------------------------
    df_rd = pd.read_csv(os.path.join(data_dir, "REAL_DATA_RESULTS.csv"))
    rd_dict = {}
    for ds in ["adult", "bank", "spambase"]:
        sub1 = df_rd[(df_rd["dataset"] == ds) & (df_rd["retained_fraction"] == 1.0)]
        sub12 = df_rd[(df_rd["dataset"] == ds) & (df_rd["retained_fraction"] == 0.125)]
        rd_dict[ds] = {
            "width_100": float(sub1["mean_width"].mean()),
            "width_12_5": float(sub12["mean_width"].mean()),
            "sigma_cal_100": float(sub1["mean_sigma_cal"].mean()),
            "sigma_cal_12_5": float(sub12["mean_sigma_cal"].mean()),
            "e_model": float(sub1["e_model"].mean()),
        }
    manifest["figure_6_real_data_thinning"] = rd_dict

    # -------------------------------------------------------------
    # 7b. Base-model evaluation metrics (accuracy, Brier, ROC-AUC)
    # -------------------------------------------------------------
    # Previously these were computed and printed by
    # real_data_calibration_support.run_experiment_for_dataset() but never
    # exported, so the accuracy figures quoted in the paper's prose had no
    # generated macro tying them to the code and could silently go stale.
    # Read from the exported metrics CSV when available; fields not yet
    # populated by a from-scratch rerun (e.g. brier/auc, which require
    # OpenML access to regenerate) are left out of the macro set rather
    # than emitted as bogus zeros.
    metrics_path = os.path.join(data_dir, "REAL_DATA_MODEL_METRICS.csv")
    metrics_dict = {}
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        for _, r in df_metrics.iterrows():
            entry = {}
            for col in ("accuracy", "brier", "auc"):
                if col in df_metrics.columns and pd.notna(r[col]):
                    entry[col] = float(r[col])
            metrics_dict[r["dataset"]] = entry
    manifest["real_data_model_metrics"] = metrics_dict

    # -------------------------------------------------------------
    # 8. Figure 7: Reverse Training Intervention
    # -------------------------------------------------------------
    df_rev = pd.read_csv(os.path.join(data_dir, "REVERSE_INTERVENTION_RESULTS.csv"))
    rev_dict = {}
    for ds in ["adult", "bank", "spambase"]:
        sub1 = df_rev[(df_rev["dataset"] == ds) & (df_rev["train_fraction"] == 1.0)]
        sub25 = df_rev[(df_rev["dataset"] == ds) & (df_rev["train_fraction"] == 0.25)]
        rev_dict[ds] = {
            "e_model_100": float(sub1["e_model"].mean()),
            "e_model_25": float(sub25["e_model"].mean()),
            "width_100": float(sub1["mean_width"].mean()),
            "width_25": float(sub25["mean_width"].mean()),
        }
    manifest["figure_7_reverse_intervention"] = rev_dict

    # -------------------------------------------------------------
    # 9. Table 4: Real-Data Nested Regressions
    # -------------------------------------------------------------
    # Delegates to real_data_calibration_support.run_nested_decomposition so
    # this manifest can never drift out of sync with the canonical fitting
    # code (as an inline duplicate previously did). That function also reports
    # a permutation test and k-fold cross-validated Delta R^2 alongside the
    # in-sample Delta R^2: the in-sample bootstrap CI can never contain zero
    # by construction (Model C's regressors are a superset of Model B's, so
    # OLS in-sample R^2 cannot decrease), so it is retained only as a
    # descriptive fit statistic, not as evidence that width helps.
    sys.path.insert(0, os.path.join(root_dir, "src"))
    from real_data_calibration_support import run_nested_decomposition

    df_pts = pd.read_csv(os.path.join(data_dir, "UNCERTAINTY_DECOMPOSITION.csv"))
    nested_dict = {}
    for ds in ["adult", "bank", "spambase"]:
        sub = df_pts[df_pts["dataset"] == ds].reset_index(drop=True)
        res = run_nested_decomposition(sub, B_boot=2000, n_perm=5000, k_folds=10, seed=42)
        nested_dict[ds] = {
            "model_a_r2": res["res_A"]["r2"],
            "model_a_mae": res["res_A"]["mae"],
            "model_a_rmse": res["res_A"]["rmse"],
            "model_b_r2": res["res_B"]["r2"],
            "model_b_mae": res["res_B"]["mae"],
            "model_b_rmse": res["res_B"]["rmse"],
            "model_c_r2": res["res_C"]["r2"],
            "model_c_mae": res["res_C"]["mae"],
            "model_c_rmse": res["res_C"]["rmse"],
            "model_d_r2": res["res_D"]["r2"],
            "model_d_mae": res["res_D"]["mae"],
            "model_d_rmse": res["res_D"]["rmse"],
            "delta_r2": res["delta_r2_CB"],
            "delta_r2_ci_lower": float(res["ci_CB"][0]),
            "delta_r2_ci_upper": float(res["ci_CB"][1]),
            "perm_p_value": res["perm_p_value"],
            "delta_r2_cv": res["delta_r2_cv"],
            "r2_b_cv": res["r2_B_cv"],
            "r2_c_cv": res["r2_C_cv"],
        }
    manifest["table_4_real_data_nested"] = nested_dict

    # -------------------------------------------------------------
    # Write JSON manifest
    # -------------------------------------------------------------
    manifest_path = os.path.join(results_dir, "paper_results_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved machine-readable manifest: {manifest_path}")

    # -------------------------------------------------------------
    # Write paper/generated_results.tex (LaTeX macros)
    # -------------------------------------------------------------
    _macro_name = {"adult": "Adult", "bank": "Bank", "spambase": "Spambase"}
    accuracy_macro_lines = []
    for ds, entry in manifest["real_data_model_metrics"].items():
        if "accuracy" in entry:
            accuracy_macro_lines.append(
                f"\\newcommand{{\\{_macro_name.get(ds, ds.capitalize())}Accuracy}}{{{entry['accuracy'] * 100:.1f}}}"
            )
    accuracy_macros_block = "\n".join(accuracy_macro_lines)

    tex_content = f"""% generated_results.tex
% Auto-generated by src/generate_manifest.py -- DO NOT EDIT DIRECTLY

% Base-model evaluation metrics (held-out test accuracy, percent)
{accuracy_macros_block}

% Figure 2: Classifier Bootstrap (N=500)
\\newcommand{{\\WidthPearson}}{{{manifest['figure_2_classifier_bootstrap']['width_pearson']:.4f}}}
\\newcommand{{\\WidthSpearman}}{{{manifest['figure_2_classifier_bootstrap']['width_spearman']:.4f}}}
\\newcommand{{\\WidthMAE}}{{{manifest['figure_2_classifier_bootstrap']['width_mae']:.4f}}}
\\newcommand{{\\UCalPearson}}{{{manifest['figure_2_classifier_bootstrap']['ucal_pearson']:.4f}}}
\\newcommand{{\\UCalSpearman}}{{{manifest['figure_2_classifier_bootstrap']['ucal_spearman']:.4f}}}
\\newcommand{{\\UCalMAE}}{{{manifest['figure_2_classifier_bootstrap']['ucal_mae']:.4f}}}

% Figure 3: Idealised Scaling Laws
\\newcommand{{\\IdealWidthSlope}}{{{manifest['figure_3_idealised_scaling']['width_slope']:.4f}}}
\\newcommand{{\\IdealWidthCILow}}{{{manifest['figure_3_idealised_scaling']['width_ci_lower']:.4f}}}
\\newcommand{{\\IdealWidthCIHigh}}{{{manifest['figure_3_idealised_scaling']['width_ci_upper']:.4f}}}
\\newcommand{{\\IdealSDSlope}}{{{manifest['figure_3_idealised_scaling']['sd_slope']:.4f}}}
\\newcommand{{\\IdealSDCILow}}{{{manifest['figure_3_idealised_scaling']['sd_ci_lower']:.4f}}}
\\newcommand{{\\IdealSDCIHigh}}{{{manifest['figure_3_idealised_scaling']['sd_ci_upper']:.4f}}}
\\newcommand{{\\IdealUCalSlope}}{{{manifest['figure_3_idealised_scaling']['ucal_slope']:.4f}}}
\\newcommand{{\\IdealUCalCILow}}{{{manifest['figure_3_idealised_scaling']['ucal_ci_lower']:.4f}}}
\\newcommand{{\\IdealUCalCIHigh}}{{{manifest['figure_3_idealised_scaling']['ucal_ci_upper']:.4f}}}

% Figure 5: Non-Monotonic Regimes
\\newcommand{{\\NonMonoRegimeASlope}}{{{manifest['figure_5_non_monotonic']['regime_a_slope']:.3f}}}
\\newcommand{{\\NonMonoRegimeBSlope}}{{{manifest['figure_5_non_monotonic']['regime_b_slope']:.3f}}}
\\newcommand{{\\NonMonoRegimeCSlope}}{{{manifest['figure_5_non_monotonic']['regime_c_slope']:.3f}}}
\\newcommand{{\\NonMonoRegimeDSlope}}{{{manifest['figure_5_non_monotonic']['regime_d_slope']:.3f}}}

% Table 3: CIFAR-10H Crowd Correlations
\\newcommand{{\\CifarBootPearson}}{{{manifest['table_3_cifar_correlations']['bootstrap_instability']['pearson']:.4f}}}
\\newcommand{{\\CifarBootSpearman}}{{{manifest['table_3_cifar_correlations']['bootstrap_instability']['spearman']:.4f}}}
\\newcommand{{\\CifarDisPearson}}{{{manifest['table_3_cifar_correlations']['annotator_disagreement']['pearson']:.4f}}}
\\newcommand{{\\CifarDisSpearman}}{{{manifest['table_3_cifar_correlations']['annotator_disagreement']['spearman']:.4f}}}
\\newcommand{{\\CifarEntPearson}}{{{manifest['table_3_cifar_correlations']['annotator_entropy']['pearson']:.4f}}}
\\newcommand{{\\CifarEntSpearman}}{{{manifest['table_3_cifar_correlations']['annotator_entropy']['spearman']:.4f}}}

% Figure 6 & Table 4: Real Tabular Data - Adult
\\newcommand{{\\AdultWidthStart}}{{{manifest['figure_6_real_data_thinning']['adult']['width_100']:.4f}}}
\\newcommand{{\\AdultWidthEnd}}{{{manifest['figure_6_real_data_thinning']['adult']['width_12_5']:.4f}}}
\\newcommand{{\\AdultSigmaCalStart}}{{{manifest['figure_6_real_data_thinning']['adult']['sigma_cal_100']:.4f}}}
\\newcommand{{\\AdultSigmaCalEnd}}{{{manifest['figure_6_real_data_thinning']['adult']['sigma_cal_12_5']:.4f}}}
\\newcommand{{\\AdultEModel}}{{{manifest['figure_6_real_data_thinning']['adult']['e_model']:.4f}}}
\\newcommand{{\\AdultModelARsq}}{{{manifest['table_4_real_data_nested']['adult']['model_a_r2']:.4f}}}
\\newcommand{{\\AdultDeltaRsq}}{{{manifest['table_4_real_data_nested']['adult']['delta_r2']:.4f}}}
\\newcommand{{\\AdultDeltaRsqCILow}}{{{manifest['table_4_real_data_nested']['adult']['delta_r2_ci_lower']:.4f}}}
\\newcommand{{\\AdultDeltaRsqCIHigh}}{{{manifest['table_4_real_data_nested']['adult']['delta_r2_ci_upper']:.4f}}}
\\newcommand{{\\AdultModelDRsq}}{{{manifest['table_4_real_data_nested']['adult']['model_d_r2']:.4f}}}
\\newcommand{{\\AdultPermPValue}}{{{manifest['table_4_real_data_nested']['adult']['perm_p_value']:.4f}}}
\\newcommand{{\\AdultDeltaRsqCV}}{{{manifest['table_4_real_data_nested']['adult']['delta_r2_cv']:.4f}}}

% Figure 6 & Table 4: Real Tabular Data - Bank
\\newcommand{{\\BankWidthStart}}{{{manifest['figure_6_real_data_thinning']['bank']['width_100']:.4f}}}
\\newcommand{{\\BankWidthEnd}}{{{manifest['figure_6_real_data_thinning']['bank']['width_12_5']:.4f}}}
\\newcommand{{\\BankSigmaCalStart}}{{{manifest['figure_6_real_data_thinning']['bank']['sigma_cal_100']:.4f}}}
\\newcommand{{\\BankSigmaCalEnd}}{{{manifest['figure_6_real_data_thinning']['bank']['sigma_cal_12_5']:.4f}}}
\\newcommand{{\\BankEModel}}{{{manifest['figure_6_real_data_thinning']['bank']['e_model']:.4f}}}
\\newcommand{{\\BankModelARsq}}{{{manifest['table_4_real_data_nested']['bank']['model_a_r2']:.4f}}}
\\newcommand{{\\BankDeltaRsq}}{{{manifest['table_4_real_data_nested']['bank']['delta_r2']:.4f}}}
\\newcommand{{\\BankDeltaRsqCILow}}{{{manifest['table_4_real_data_nested']['bank']['delta_r2_ci_lower']:.4f}}}
\\newcommand{{\\BankDeltaRsqCIHigh}}{{{manifest['table_4_real_data_nested']['bank']['delta_r2_ci_upper']:.4f}}}
\\newcommand{{\\BankModelDRsq}}{{{manifest['table_4_real_data_nested']['bank']['model_d_r2']:.4f}}}
\\newcommand{{\\BankPermPValue}}{{{manifest['table_4_real_data_nested']['bank']['perm_p_value']:.4f}}}
\\newcommand{{\\BankDeltaRsqCV}}{{{manifest['table_4_real_data_nested']['bank']['delta_r2_cv']:.4f}}}

% Figure 6 & Table 4: Real Tabular Data - Spambase
\\newcommand{{\\SpambaseWidthStart}}{{{manifest['figure_6_real_data_thinning']['spambase']['width_100']:.4f}}}
\\newcommand{{\\SpambaseWidthEnd}}{{{manifest['figure_6_real_data_thinning']['spambase']['width_12_5']:.4f}}}
\\newcommand{{\\SpambaseSigmaCalStart}}{{{manifest['figure_6_real_data_thinning']['spambase']['sigma_cal_100']:.4f}}}
\\newcommand{{\\SpambaseSigmaCalEnd}}{{{manifest['figure_6_real_data_thinning']['spambase']['sigma_cal_12_5']:.4f}}}
\\newcommand{{\\SpambaseEModel}}{{{manifest['figure_6_real_data_thinning']['spambase']['e_model']:.4f}}}
\\newcommand{{\\SpambaseModelARsq}}{{{manifest['table_4_real_data_nested']['spambase']['model_a_r2']:.4f}}}
\\newcommand{{\\SpambaseDeltaRsq}}{{{manifest['table_4_real_data_nested']['spambase']['delta_r2']:.4f}}}
\\newcommand{{\\SpambaseDeltaRsqCILow}}{{{manifest['table_4_real_data_nested']['spambase']['delta_r2_ci_lower']:.4f}}}
\\newcommand{{\\SpambaseDeltaRsqCIHigh}}{{{manifest['table_4_real_data_nested']['spambase']['delta_r2_ci_upper']:.4f}}}
\\newcommand{{\\SpambaseModelDRsq}}{{{manifest['table_4_real_data_nested']['spambase']['model_d_r2']:.4f}}}
\\newcommand{{\\SpambasePermPValue}}{{{manifest['table_4_real_data_nested']['spambase']['perm_p_value']:.4f}}}
\\newcommand{{\\SpambaseDeltaRsqCV}}{{{manifest['table_4_real_data_nested']['spambase']['delta_r2_cv']:.4f}}}

% Figure 7: Reverse Intervention Refits
\\newcommand{{\\AdultRevEModelStart}}{{{manifest['figure_7_reverse_intervention']['adult']['e_model_100']:.4f}}}
\\newcommand{{\\AdultRevEModelEnd}}{{{manifest['figure_7_reverse_intervention']['adult']['e_model_25']:.4f}}}
\\newcommand{{\\AdultRevWidthStart}}{{{manifest['figure_7_reverse_intervention']['adult']['width_100']:.4f}}}
\\newcommand{{\\AdultRevWidthEnd}}{{{manifest['figure_7_reverse_intervention']['adult']['width_25']:.4f}}}

\\newcommand{{\\BankRevEModelStart}}{{{manifest['figure_7_reverse_intervention']['bank']['e_model_100']:.4f}}}
\\newcommand{{\\BankRevEModelEnd}}{{{manifest['figure_7_reverse_intervention']['bank']['e_model_25']:.4f}}}
\\newcommand{{\\BankRevWidthStart}}{{{manifest['figure_7_reverse_intervention']['bank']['width_100']:.4f}}}
\\newcommand{{\\BankRevWidthEnd}}{{{manifest['figure_7_reverse_intervention']['bank']['width_25']:.4f}}}

\\newcommand{{\\SpambaseRevEModelStart}}{{{manifest['figure_7_reverse_intervention']['spambase']['e_model_100']:.4f}}}
\\newcommand{{\\SpambaseRevEModelEnd}}{{{manifest['figure_7_reverse_intervention']['spambase']['e_model_25']:.4f}}}
\\newcommand{{\\SpambaseRevWidthStart}}{{{manifest['figure_7_reverse_intervention']['spambase']['width_100']:.4f}}}
\\newcommand{{\\SpambaseRevWidthEnd}}{{{manifest['figure_7_reverse_intervention']['spambase']['width_25']:.4f}}}
"""
    tex_path = os.path.join(paper_dir, "generated_results.tex")
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"Saved LaTeX macros: {tex_path}")

if __name__ == "__main__":
    main()
