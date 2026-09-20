#!/usr/bin/env python3
"""
src/verify_paper_consistency.py

Comprehensive consistency verifier for 'The Meaning and Scaling of Venn-Abers Probability Intervals'.

Performs:
1. Recomputes all manifest quantities from committed data/ CSVs and verifies
   numerical agreement within strict tolerance (5e-5).
2. Verifies all required paper assets (PDF/PNG figures and LaTeX tables) exist.
3. Verifies no duplicate publication output names across producer scripts.
4. Verifies no obsolete agent/LLM artifacts exist in reproduction scripts.
5. Exits 0 on full consistency, non-zero on any mismatch or failure.

Usage:
    python src/verify_paper_consistency.py
"""

import os
import sys
import json
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress, t

TOLERANCE = 5e-5

def log_pass(msg):
    print(f"[\033[92mPASS\033[0m] {msg}")

def log_fail(msg):
    print(f"[\033[91mFAIL\033[0m] {msg}")

def check_close(actual, expected, label, tol=TOLERANCE):
    diff = abs(actual - expected)
    if diff > tol:
        log_fail(f"{label}: actual {actual} != expected {expected} (diff={diff:.2e} > tol={tol})")
        return False
    return True

def verify_manifest_consistency(root_dir):
    manifest_path = os.path.join(root_dir, "results", "paper_results_manifest.json")
    if not os.path.exists(manifest_path):
        log_fail(f"Manifest missing: {manifest_path}")
        return False
        
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    data_dir = os.path.join(root_dir, "data")
    all_ok = True
    
    # 1. Calibration Resampling Baseline
    df_cr = pd.read_csv(os.path.join(data_dir, "calibration_resampling_baseline.csv"))
    cr_manifest = manifest["table_1_calibration_resampling"]
    for i, r in df_cr.iterrows():
        m_item = cr_manifest[i]
        all_ok &= check_close(float(r["mean_width"]), m_item["mean_width"], f"Table 1 n={r['n_cal']} mean_width")
        all_ok &= check_close(float(r["mean_sd"]), m_item["mean_sd"], f"Table 1 n={r['n_cal']} mean_sd")
        all_ok &= check_close(float(r["corr"]), m_item["corr"], f"Table 1 n={r['n_cal']} corr")
        all_ok &= check_close(float(r["fresh_brier"]), m_item["fresh_brier"], f"Table 1 n={r['n_cal']} fresh_brier")

    # 2. Classifier Bootstrap
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
    
    cb_m = manifest["figure_2_classifier_bootstrap"]
    all_ok &= check_close(pw, cb_m["width_pearson"], "Fig 2 width_pearson")
    all_ok &= check_close(sw, cb_m["width_spearman"], "Fig 2 width_spearman")
    all_ok &= check_close(mae_w, cb_m["width_mae"], "Fig 2 width_mae")
    all_ok &= check_close(pu, cb_m["ucal_pearson"], "Fig 2 ucal_pearson")
    all_ok &= check_close(su, cb_m["ucal_spearman"], "Fig 2 ucal_spearman")
    all_ok &= check_close(mae_u, cb_m["ucal_mae"], "Fig 2 ucal_mae")

    # 3. Idealised Scaling
    df_id = pd.read_csv(os.path.join(data_dir, "SCALING_IDEALISED.csv"))
    log_n = np.log(df_id["n_cal"].values)
    dof = len(df_id) - 2
    t_crit = t.ppf(0.975, dof)
    id_m = manifest["figure_3_idealised_scaling"]
    for col, key in [("mean_width", "width"), ("sd_p_mid", "sd"), ("mean_ucal", "ucal")]:
        log_y = np.log(df_id[col].values)
        res = linregress(log_n, log_y)
        ci_lower = res.slope - t_crit * res.stderr
        ci_upper = res.slope + t_crit * res.stderr
        all_ok &= check_close(res.slope, id_m[f"{key}_slope"], f"Fig 3 {key}_slope")
        all_ok &= check_close(ci_lower, id_m[f"{key}_ci_lower"], f"Fig 3 {key}_ci_lower")
        all_ok &= check_close(ci_upper, id_m[f"{key}_ci_upper"], f"Fig 3 {key}_ci_upper")

    # 4. W2 Exponent Progression
    df_w2 = pd.read_csv(os.path.join(data_dir, "proposition1_convergence.csv"))
    w2_m = manifest["table_2_w2_exponent_progression"]
    for _, r in df_w2.iterrows():
        n_str = str(int(r["n_cal"]))
        item = w2_m[n_str]
        all_ok &= check_close(r["beta_rho"], item["beta_rho"], f"Table 2 n={n_str} beta_rho")
        all_ok &= check_close(r["beta_v"], item["beta_v"], f"Table 2 n={n_str} beta_v")
        all_ok &= check_close(r["beta_s"], item["beta_s"], f"Table 2 n={n_str} beta_s")
        all_ok &= check_close(r["mean_abs_err"], item["mae"], f"Table 2 n={n_str} mae")
        all_ok &= check_close(r["r2"], item["r2"], f"Table 2 n={n_str} r2")

    # 5. Non-monotonic Scaling
    df_nm = pd.read_csv(os.path.join(data_dir, "non_monotonic_scaling_laws.csv"))
    log_n_nm = np.log(df_nm["n_cal"].values)
    nm_m = manifest["figure_5_non_monotonic"]
    slope_a = np.polyfit(log_n_nm, np.log(df_nm["width_mono"].values), 1)[0]
    slope_b = np.polyfit(log_n_nm, np.log(df_nm["width_viol"].values), 1)[0]
    slope_c = np.polyfit(log_n_nm, np.log(df_nm["width_ext_mid"].values), 1)[0]
    slope_d = np.polyfit(log_n_nm, np.log(df_nm["width_ext_bound"].values), 1)[0]
    all_ok &= check_close(slope_a, nm_m["regime_a_slope"], "Fig 5 Regime A slope")
    all_ok &= check_close(slope_b, nm_m["regime_b_slope"], "Fig 5 Regime B slope")
    all_ok &= check_close(slope_c, nm_m["regime_c_slope"], "Fig 5 Regime C slope")
    all_ok &= check_close(slope_d, nm_m["regime_d_slope"], "Fig 5 Regime D slope")

    # 6. Real-Data Thinning
    df_rd = pd.read_csv(os.path.join(data_dir, "REAL_DATA_RESULTS.csv"))
    rd_m = manifest["figure_6_real_data_thinning"]
    for ds in ["adult", "bank", "spambase"]:
        sub1 = df_rd[(df_rd["dataset"] == ds) & (df_rd["retained_fraction"] == 1.0)]
        sub12 = df_rd[(df_rd["dataset"] == ds) & (df_rd["retained_fraction"] == 0.125)]
        all_ok &= check_close(sub1["mean_width"].mean(), rd_m[ds]["width_100"], f"Fig 6 {ds} width_100")
        all_ok &= check_close(sub12["mean_width"].mean(), rd_m[ds]["width_12_5"], f"Fig 6 {ds} width_12_5")
        all_ok &= check_close(sub1["mean_sigma_cal"].mean(), rd_m[ds]["sigma_cal_100"], f"Fig 6 {ds} sigma_cal_100")
        all_ok &= check_close(sub12["mean_sigma_cal"].mean(), rd_m[ds]["sigma_cal_12_5"], f"Fig 6 {ds} sigma_cal_12_5")
        all_ok &= check_close(sub1["e_model"].mean(), rd_m[ds]["e_model"], f"Fig 6 {ds} e_model")

    # 7. Reverse Intervention
    df_rev = pd.read_csv(os.path.join(data_dir, "REVERSE_INTERVENTION_RESULTS.csv"))
    rev_m = manifest["figure_7_reverse_intervention"]
    for ds in ["adult", "bank", "spambase"]:
        sub1 = df_rev[(df_rev["dataset"] == ds) & (df_rev["train_fraction"] == 1.0)]
        sub25 = df_rev[(df_rev["dataset"] == ds) & (df_rev["train_fraction"] == 0.25)]
        all_ok &= check_close(sub1["e_model"].mean(), rev_m[ds]["e_model_100"], f"Fig 7 {ds} e_model_100")
        all_ok &= check_close(sub25["e_model"].mean(), rev_m[ds]["e_model_25"], f"Fig 7 {ds} e_model_25")
        all_ok &= check_close(sub1["mean_width"].mean(), rev_m[ds]["width_100"], f"Fig 7 {ds} width_100")
        all_ok &= check_close(sub25["mean_width"].mean(), rev_m[ds]["width_25"], f"Fig 7 {ds} width_25")

    if all_ok:
        log_pass("Manifest numbers match data to < 5e-5 tolerance")
    return all_ok

def verify_assets_exist(root_dir):
    required_figures = [
        "paper/width_vs_bootstrap_instability.png",
        "paper/width_vs_bootstrap_instability.pdf",
        "paper/idealised_scaling_laws.png",
        "paper/idealised_scaling_laws.pdf",
        "paper/w2_exponent_convergence.png",
        "paper/w2_exponent_convergence.pdf",
        "paper/non_monotonic_scaling_laws.png",
        "paper/non_monotonic_scaling_laws.pdf",
        "paper/real_data_local_support.png",
        "paper/real_data_local_support.pdf",
        "paper/training_support_epistemic.png",
        "paper/training_support_epistemic.pdf",
        "paper/calibration_uncertainty_bootstrap.png",
        "paper/alternative_calibrator_instability.png",
    ]
    required_tables = [
        "paper/table_calibration_resampling.tex",
        "paper/table_w2_exponent_progression.tex",
        "paper/table_cifar_correlations.tex",
        "paper/table_real_data_nested.tex",
        "paper/generated_results.tex",
    ]
    required_data = [
        "data/calibration_resampling_baseline.csv",
        "data/classifier_bootstrap_n500.csv",
        "data/SCALING_IDEALISED.csv",
        "data/proposition1_convergence.csv",
        "data/proposition1_convergence_bootstrap.csv",
        "data/non_monotonic_scaling_laws.csv",
        "data/table_cifar_correlations.csv",
        "data/REAL_DATA_RESULTS.csv",
        "data/REVERSE_INTERVENTION_RESULTS.csv",
        "data/UNCERTAINTY_DECOMPOSITION.csv",
    ]
    
    all_ok = True
    for item in required_figures + required_tables + required_data:
        path = os.path.join(root_dir, item)
        if not os.path.exists(path):
            log_fail(f"Required asset missing: {item}")
            all_ok = False
        else:
            log_pass(f"Asset found: {item}")
    return all_ok

def verify_no_duplicate_producers(root_dir):
    """Verifies that no two scripts claim to produce the same publication figure or table."""
    script_files = glob.glob(os.path.join(root_dir, "src", "*.py"))
    outputs = {}
    
    for sf in script_files:
        if os.path.basename(sf) in ["generate_paper_artifacts.py", "run_all_experiments.py", "verify_paper_consistency.py", "generate_manifest.py"]:
            continue
        with open(sf) as f:
            content = f.read()
        matches = re.findall(r'["\'](?:paper|data)/([^"\']+\.(?:png|pdf|tex|csv))["\']', content)
        for m in set(matches):
            if m not in outputs:
                outputs[m] = []
            outputs[m].append(os.path.basename(sf))
            
    all_ok = True
    for out, scripts in outputs.items():
        if len(scripts) > 1:
            log_fail(f"Duplicate producer collision for {out}: {scripts}")
            all_ok = False
        else:
            log_pass(f"Unique producer for {out}: {scripts[0]}")
    return all_ok

def verify_no_obsolete_artifacts(root_dir):
    """Ensures no obsolete LLM/agent code exists in current scripts or active generation paths."""
    forbidden_terms = ["gsm8k", "hotpotqa", "triviaqa", "bfcl", "0.1119"]
    script_files = glob.glob(os.path.join(root_dir, "src", "*.py"))
    all_ok = True
    for sf in script_files:
        if os.path.basename(sf) in ["verify_paper_consistency.py"]:
            continue
        with open(sf) as f:
            content = f.read().lower()
        for term in forbidden_terms:
            if term in content:
                log_fail(f"Obsolete agent artifact term '{term}' found in {sf}")
                all_ok = False
    if all_ok:
        log_pass("No obsolete agent/LLM artifacts found in src/")
    return all_ok

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    print("=" * 70)
    print("Running Repository-Paper Consistency Verification")
    print("=" * 70)
    
    ok_manifest = verify_manifest_consistency(root_dir)
    ok_assets = verify_assets_exist(root_dir)
    ok_dups = verify_no_duplicate_producers(root_dir)
    ok_obsolete = verify_no_obsolete_artifacts(root_dir)
    
    print("=" * 70)
    if ok_manifest and ok_assets and ok_dups and ok_obsolete:
        print("\033[92mALL CONSISTENCY CHECKS PASSED SUCCESSFULLY!\033[0m")
        sys.exit(0)
    else:
        print("\033[91mONE OR MORE CONSISTENCY CHECKS FAILED!\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()
