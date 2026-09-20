# Repo–Paper Consistency Audit: Before Report

**Repository**: `https://github.com/ip200/va-width-experiments`  
**Branch**: `repo-paper-consistency-audit`  
**Manuscript**: *The Meaning and Scaling of Venn–Abers Probability Intervals*  
**Date**: September 20, 2026  

---

## 1. Inventory of Empirical Paper Elements

The following table maps every empirical table and figure in the manuscript against the repository scripts, data sources, assets, and initial audit status.

| Paper Item | Scientific Claim | Script | Raw / Generated Data | Figure / Table Asset | Initial Audit Status |
|---|---|---|---|---|---|
| **Figure 1** | GCM schematic of isotonic regression; test instance shifts local cumulative sum by 1, yielding $w_n(s) \approx 1/N_{\mathrm{eff}}(s)$. | Pure LaTeX/TikZ | None (geometric diagram) | Embedded in `arxiv_va_interval_width.tex` (L145–180) | **PASS** (LaTeX native) |
| **Table 1** | Baseline calibration resampling analysis across $n_{\mathrm{cal}} \in [100, 250, 500, 1000, 2000]$ reporting mean width, bootstrap SD, correlation, Brier score. | Hardcoded in `.tex` | None committed | Inline in `arxiv_va_interval_width.tex` (L397–414) | **MISSING REPRODUCTION PATH** (Needs canonical generator & raw CSV) |
| **Figure 2** | Pointwise comparison of interval width $w$ vs bootstrap SD $\sigma_{\mathrm{boot}}$ ($r=0.4287, \rho=0.4072$) and $U_{\mathrm{cal}}$ vs $\sigma_{\mathrm{boot}}$ ($r=0.5300, \rho=0.4247$). | `src/run_experiment_classifier_bootstrap_n500.py`, `src/generate_figure_width_vs_instability.py` | `data/classifier_bootstrap_n500.csv` | `paper/width_vs_bootstrap_instability.png`, `.pdf` | **FAIL: Critical Fix A** (Uses $p_{\mathrm{VA}}$ from `predict_proba` instead of true midpoint $\hat p$) |
| **Figure 3** | Idealised scaling of interval width (slope $\approx -0.6390$) and probability instability $\sigma_{\mathrm{boot}}$ (slope $\approx -0.3172$), $U_{\mathrm{cal}}$ (slope $\approx -0.3285$). | `src/scaling_laws_experiment.py` (partial/different) | `data/SCALING_IDEALISED.csv` | `paper/idealised_scaling_laws.png`, `.pdf` | **FAIL: Missing Canonical Generator** (`scaling_laws_experiment.py` does not generate `SCALING_IDEALISED.csv`) |
| **Table 2** | Progression of multivariable scaling exponents over $n_{\mathrm{cal}} \in [500, \dots, 32000]$: $\beta_\rho \to -2/3, \beta_v \to -1/3, \beta_s \to 2/3$. | `src/run_experiment_w2_convergence.py` | `data/proposition1_convergence.csv` | `paper/table_w2_exponent_progression.tex` | **FAIL: Critical Fix B** (Computes ambiguity using fitted midpoint proxy `pm*(1-pm)` instead of true DGP variance $\theta(1-\theta)$) |
| **Figure 4** | Multivariable exponent convergence with 95% bootstrap confidence intervals over calibration sample sizes. | `src/run_experiment_w2_convergence.py` | `data/proposition1_convergence_bootstrap.csv` | `paper/w2_exponent_convergence.png`, `.pdf` | **FAIL: Critical Fix B** (Dependent on true ambiguity fix in W2) |
| **Figure 5** | Scaling under monotonic and non-monotonic probability curves (Regimes A, B, C, D) verifying $-2/3, -1, -1, -4/5$ rates. | `src/non_monotonic_experiments.py` | Generated dynamically (not saved to canonical CSV) | `paper/non_monotonic_scaling_laws.png` | **PASS / REFACTOR** (Rates match; needs canonical data export to `data/`) |
| **Table 3** | Synthetic CIFAR-10H-inspired crowd experiment: width correlates with calibration bootstrap instability ($r\approx0.69$), but not annotator disagreement ($r\approx0.01$). | `src/generate_all_figures_tables.py` (inline) | Generated dynamically | `paper/table_cifar_correlations.tex` | **REFACTOR** (Needs standalone generator, clear synthetic labeling, and verification of midpoint $\hat p$) |
| **Figure 6** | Controlled local calibration-support thinning on UCI Adult, Bank, Spambase: width and $\sigma_{\mathrm{cal}}$ rise as support decreases, while $E_{\mathrm{model}}$ is fixed. | `src/real_data_calibration_support.py`, `src/plot_figures_6_7.py` | `data/REAL_DATA_RESULTS.csv` | `paper/real_data_local_support.png`, `.pdf` | **FAIL: Critical Fix C & D** (Bootstrap $\sigma_{\mathrm{cal}}$ computed only on last replicate; `std_sigma_cal` misassigned; base-model refits use `max_iter=100`) |
| **Figure 7** | Reverse intervention subsampling base training partition (100% down to 25%): $E_{\mathrm{model}}$ inflates strongly while width is comparatively unresponsive. | `src/real_data_calibration_support.py`, `src/plot_figures_6_7.py` | `data/REVERSE_INTERVENTION_RESULTS.csv` | `paper/training_support_epistemic.png`, `.pdf` | **FAIL: Critical Fix D** (Reverse refits use `max_iter=100` instead of canonical `max_iter=200`) |
| **Table 4** | Real-data nested uncertainty decomposition regressions: Model A ($A(x)$), Model B ($+ E_{\mathrm{model}}$), Model C ($+ w$). $\Delta R^2 > 0$ strictly positive. | `src/real_data_calibration_support.py` | `data/UNCERTAINTY_DECOMPOSITION.csv` | `paper/table_real_data_nested.tex` | **FAIL: Critical Fix D** (Affected by $E_{\mathrm{model}}$ `max_iter=100` refits) |
| **Figure 8** | Alternative-calibrator comparison (Histogram, Isotonic, Platt, Venn-Abers) across calibration set sizes. | `src/calibration_comparison.py` | Generated dynamically | `paper/calibration_uncertainty_bootstrap.png` | **FAIL: Filename Collision Risk** (Script had conflicting default filename) |

---

## 2. Detailed Findings & Discrepancies

### Critical Fix A: Midpoint vs Standard Log-Loss Merge
- **Location**: `src/run_experiment_classifier_bootstrap_n500.py`
- **Issue**: `VennAbersCalibrator.predict_proba(...)` returns $(p_{\mathrm{VA}}, 1 - p_{\mathrm{VA}})$, where $p_{\mathrm{VA}} = p_1 / (1 - p_0 + p_1)$. The script assigned this merged value to `p_mid`, while the manuscript defines the midpoint $\hat p = (p_0 + p_1) / 2$.
- **Impact**: The calibration instability index $U_{\mathrm{cal}}$ and bootstrap standard deviation $\sigma_{\mathrm{boot}}$ were computed on the log-loss merge rather than the arithmetic midpoint.

### Critical Fix B: Ambiguity Regressor in W2 Convergence
- **Location**: `src/run_experiment_w2_convergence.py`
- **Issue**: The script calculated ambiguity via `pm * (1.0 - pm)` where `pm` is an estimated midpoint from the fitted calibrator. The manuscript specifies evaluating Proposition 1 against the true data-generating Bernoulli variance $v(s) = \theta(s)(1 - \theta(s))$.
- **Impact**: Introduced estimation noise and attenuation into the ambiguity scaling exponent $\beta_v$.

### Critical Fix C: Real-Data Calibration-Thinning Bootstrap Aggregation
- **Location**: `src/real_data_calibration_support.py`
- **Issue**: In `run_local_support_experiment()`, the script thinned the calibration set $R_{\mathrm{thin}} = 100$ times to average width, but saved only the last thinning replicate to compute $\sigma_{\mathrm{cal}}$ via bootstrap. Furthermore, `"std_sigma_cal": float(np.std(w_reps))` mistakenly stored width standard deviation under the name of calibration instability standard deviation.
- **Impact**: `mean_sigma_cal` came from an arbitrary single thinning replicate rather than the expectation over thinning draws, causing non-monotonic noise in Spambase ($0.0873 \to 0.0768$).

### Critical Fix D: Base Model Hyperparameter Inconsistency
- **Location**: `src/real_data_calibration_support.py`
- **Issue**: The full base model fit used `HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200)`, whereas `run_bootstrap_model_epistemic()` and `run_training_support_experiment()` used `max_iter=100`.
- **Impact**: The model uncertainty $E_{\mathrm{model}}$ and reverse intervention curves evaluated a different, less-trained model than the predictor being evaluated.

### Critical Fix E: Missing Generator for Idealised Scaling
- **Location**: `data/SCALING_IDEALISED.csv`
- **Issue**: The CSV exists, but `src/scaling_laws_experiment.py` uses a different setup and does not reproduce it end-to-end.
- **Impact**: Figure 3 lacked a reproducible generator script.

### Critical Fix F: Missing Generator for Table 1
- **Location**: `arxiv_va_interval_width.tex` (L397–414)
- **Issue**: The baseline calibration-resampling table numbers were hardcoded directly in LaTeX without an executable generator or committed CSV.
- **Impact**: Unverifiable reproduction path.

### Obsolete Artifacts & Collisions
- `src/generate_final_figures_tables.py` contains obsolete LLM / agent benchmark tables (GSM8K, MATH, HotpotQA, TriviaQA, BFCL, $\Delta R^2 = 0.1119$).
- `src/calibration_comparison.py` had conflicting output paths that could overwrite Figure 2.

---

## 3. Corrective Action Plan
1. Implement unified helpers in `src/utils.py` (`extract_va_outputs`, `make_base_model`).
2. Fix `src/run_experiment_classifier_bootstrap_n500.py` and rerun.
3. Fix `src/run_experiment_w2_convergence.py` with true $\theta(1-\theta)$ and rerun.
4. Fix `src/real_data_calibration_support.py` bootstrap aggregation across thinning replicates and unify `max_iter=200`; rerun Adult, Bank, Spambase.
5. Create `src/run_idealised_scaling.py` and `src/run_calibration_resampling_baseline.py`.
6. Fix `src/calibration_comparison.py` output paths.
7. Remove `src/generate_final_figures_tables.py`.
8. Create `src/generate_paper_artifacts.py`, `src/run_all_experiments.py`, and `src/verify_paper_consistency.py`.
9. Document all numerical changes in `results/RESULT_CHANGES.md`.
