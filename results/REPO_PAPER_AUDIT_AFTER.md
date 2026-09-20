# Post-Audit Experiment Inventory and Verification Report

**Repository**: `va-width-experiments`  
**Branch**: `repo-paper-consistency-audit`  
**Target Manuscript**: *The Meaning and Scaling of Venn–Abers Probability Intervals*  
**Date**: September 20, 2026  
**Status**: **ALL AUDIT ITEMS RESOLVED & VERIFIED**

---

## 1. Post-Audit Inventory Table

| # | Paper Item | Scientific Claim | Canonical Script | Raw / Generated Data | Publication Asset | Status |
|---|---|---|---|---|---|---|
| 1 | **Table 1** (Sec 5.1) | Calibration Resampling Baseline ($N \in [100, 2000]$) | `src/run_calibration_resampling_baseline.py` | `data/calibration_resampling_baseline.csv` | `paper/table_calibration_resampling.tex` | **VERIFIED** |
| 2 | **Figure 2** (Sec 5.1) | Pointwise width vs calibration bootstrap instability ($N=500$) | `src/run_experiment_classifier_bootstrap_n500.py` | `data/classifier_bootstrap_n500.csv` | `paper/width_vs_bootstrap_instability.png` / `.pdf` | **VERIFIED** |
| 3 | **Figure 3** (Sec 5.2) | Idealised sample-size scaling laws ($s_0=0.5, N \in [500, 32000]$) | `src/run_idealised_scaling.py` | `data/SCALING_IDEALISED.csv` | `paper/idealised_scaling_laws.png` / `.pdf` | **VERIFIED** |
| 4 | **Table 2 & Figure 4** (Sec 5.3) | Large-$n$ multivariate exponent convergence (Prop 1) | `src/run_experiment_w2_convergence.py` | `data/proposition1_convergence.csv`, `proposition1_convergence_bootstrap.csv` | `paper/table_w2_exponent_progression.tex`, `paper/w2_exponent_convergence.png` / `.pdf` | **VERIFIED** |
| 5 | **Figure 5** (Sec 6) | Non-monotonic scaling laws & flat points (Regimes A, B, C, D) | `src/non_monotonic_experiments.py` | `data/non_monotonic_scaling_laws.csv` | `paper/non_monotonic_scaling_laws.png` / `.pdf` | **VERIFIED** |
| 6 | **Sec 6.1 / Appendix** | Score-transformation invariance ($s, s^2, \exp(s)$) | `src/non_monotonic_experiments.py` | Runtime assertion (0.0 numerical diff) | Verified invariant | **VERIFIED** |
| 7 | **Table 3** (Sec 5.4) | Synthetic CIFAR-10H-inspired crowd-annotation experiment | `src/calibration_comparison.py` | `data/table_cifar_correlations.csv` | `paper/table_cifar_correlations.tex` | **VERIFIED** |
| 8 | **Figure 8 / Alt** (Sec 6.2) | Alternative calibrators instability comparison | `src/calibration_comparison.py` | Synthetic run data | `paper/alternative_calibrator_instability.png`, `calibration_uncertainty_bootstrap.png` | **VERIFIED** |
| 9 | **Figure 6** (Sec 7.5) | Real tabular data local calibration-support thinning | `src/real_data_calibration_support.py` | `data/REAL_DATA_RESULTS.csv` | `paper/real_data_local_support.png` / `.pdf` | **VERIFIED** |
| 10 | **Figure 7** (Sec 7.5) | Reverse training-support intervention | `src/real_data_calibration_support.py` | `data/REVERSE_INTERVENTION_RESULTS.csv` | `paper/training_support_epistemic.png` / `.pdf` | **VERIFIED** |
| 11 | **Table 4** (Sec 7.5) | Quantitative uncertainty decomposition (Models A, B, C) | `src/real_data_calibration_support.py` | `data/UNCERTAINTY_DECOMPOSITION.csv` | `paper/table_real_data_nested.tex` | **VERIFIED** |
| 12 | **Pre-specified Points** | Pre-specified evaluation locations ($s^* \approx 0.20, 0.35, 0.50, 0.65, 0.80$) | `src/real_data_calibration_support.py` | `results/real_data_selected_points.csv` | Pre-intervention selection audit log | **VERIFIED** |

---

## 2. Completed Non-Negotiable Audit Checklist

- [x] **Every empirical paper result has a generating script**: All 12 items map 1-to-1 to committed scripts in `src/`.
- [x] **Every script writes to canonical `data/`, `paper/`, `results/` directories**: No temporary files or CSVs written to root.
- [x] **Every paper figure/table has exactly one producer**: Verified by `verify_paper_consistency.py` AST/path scanner with 0 collisions.
- [x] **No publication asset filename collisions remain**: `src/calibration_comparison.py` writes to `paper/alternative_calibrator_instability.png`, preventing collision with Figure 2.
- [x] **$\hat p$ always means arithmetic midpoint**: Centralized helper `extract_va_outputs(pred)` defines $\hat p = 0.5(p_0 + p_1)$, and $p_{\mathrm{VA}} = p_1 / (1 - p_0 + p_1)$ is labeled separately.
- [x] **$p_{\mathrm{VA}}$ is named separately wherever used**: Explicitly labeled and distinguished from midpoint $\hat p$.
- [x] **$W_2$ uses true $\theta(1-\theta)$**: Proposition 1 exponent experiment calculates true Bernoulli variance from known DGP $\theta(s)$.
- [x] **Figure 6 $\sigma_{\mathrm{cal}}$ is averaged consistently across thinning replicates**: Aggregates all $R_{\mathrm{thin}}=100$ thinned sets with $B_{\mathrm{cal}}=100$ bootstrap resamples per replicate.
- [x] **`std_sigma_cal` is actually $\sigma_{\mathrm{cal}}$ variability**: Fixed field mapping to represent standard deviation of calibration bootstrap SD across thinning draws.
- [x] **All `HistGradientBoosting` refits use one shared specification**: Unified helper `make_base_classifier` with `max_depth=4, learning_rate=0.05, max_iter=200` across all baseline, bootstrap, and reverse-intervention fits.
- [x] **Idealised scaling CSV has a committed generator**: Canonical script `src/run_idealised_scaling.py` reproduces `data/SCALING_IDEALISED.csv` and Figure 3.
- [x] **Baseline calibration-resampling table has a committed generator**: Canonical script `src/run_calibration_resampling_baseline.py` reproduces `data/calibration_resampling_baseline.csv` and Table 1.
- [x] **Crowd experiment is labelled synthetic**: Labeled as "synthetic CIFAR-10H-inspired crowd-annotation experiment" in paper, code, table, and manifest.
- [x] **No active LLM/agent experiment remains**: Deleted obsolete `src/generate_final_figures_tables.py`, repository confirmed free of obsolete LLM artifact names.
- [x] **README matches actual git tree**: README provides complete instructions for fast artifact regeneration and full experiment reproduction.
- [x] **No absolute local paths remain**: All path references use relative project root paths (`os.path.dirname(...)`).
- [x] **Exact environment is recorded**: Pinned in `results/environment.txt` and `requirements-lock.txt`.
- [x] **Manuscript numerical claims equal regenerated outputs**: Machine-readable manifest `results/paper_results_manifest.json` and generated macros `paper/generated_results.tex` match data to $< 5\times 10^{-5}$.
- [x] **Clean-room verification passes**: `python src/verify_paper_consistency.py` exits 0 with all checks passing.
