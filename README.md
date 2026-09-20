# The Meaning and Scaling of Venn--Abers Probability Intervals

This repository contains the complete, self-contained codebase, empirical benchmarks, precomputed experimental data, and LaTeX source for the paper:

> **The Meaning and Scaling of Venn--Abers Probability Intervals**  
> Ivan Petej  
> *Department of Computer Science, Royal Holloway, University of London*  
> arXiv: [arXiv preprint] | Code: [GitHub](https://github.com/ip200/va-width-experiments)

---

## 1. Abstract & Key Claims

Venn--Abers predictors (VAPs) output a multiprobabilistic pair $[p_0, p_1]$ with finite-sample validity guarantees under exchangeability. However, the exact semantic interpretation of the interval width $w = p_1 - p_0$ has remained an open question. In applications, width is often informally treated as a measure of outcome ambiguity (aleatoric uncertainty) or model parameter uncertainty (epistemic uncertainty).

This paper establishes that **Venn--Abers width primarily reflects local calibration leverage, or equivalently inverse effective local calibration support**:

1. **Inverse Block Size Mechanism**:
   A test instance falls into a local Pool Adjacent Violators (PAV) active block containing $N_{\text{eff}}(s)$ calibration observations. Perturbing the test point's hypothetical label shifts the local cumulative sum by $1$, yielding:
   $$w(s) \approx \frac{1}{N_{\text{eff}}(s)}$$

2. **Asymptotic Scaling Law**:
   Balancing systematic probability variation against local Bernoulli noise under the Greatest Convex Minorant (GCM) representation gives the scaling relationship:
   $$w_n(s) \propto \underbrace{n^{-2/3} f(s)^{-2/3}}_{\text{calibration leverage}} \;\times\; \underbrace{[\theta(s)(1 - \theta(s))]^{-1/3}}_{\text{outcome variability}} \;\times\; \underbrace{[\theta'(s)]^{2/3}}_{\text{local geometry}}$$
   where $n$ is calibration size, $f(s)$ is score density, $\theta(s)$ is true conditional probability, and $\theta'(s)$ is local slope.

3. **Probability-Scale Instability Index**:
   The index:
   $$U_{\text{cal}}(s) = \sqrt{\hat p(s)(1 - \hat p(s)) w(s)} \propto n^{-1/3}$$
   where $\hat p = (p_0 + p_1)/2$ is the arithmetic midpoint, exhibits the classical isotonic cube-root rate and provides a calibrated probability-scale instability summary.

4. **Tripartite Uncertainty Separation**:
   $$\text{outcome ambiguity} \;\neq\; \text{base-model epistemic uncertainty} \;\neq\; \text{calibration leverage / support}$$
   - **Outcome Ambiguity** (label noise proxy): $A(x) = \hat p(x)(1 - \hat p(x))$
   - **Base-Model Epistemic Uncertainty**: $E_{\text{model}}(x) = \text{SD}_b(g_b(x))$ across bootstrap training refits
   - **Calibration Leverage / Support**: $w(x) = p_1(x) - p_0(x)$

Controlled interventions on real tabular benchmarks (**UCI Adult**, **UCI Bank Marketing**, **UCI Spambase**) directly confirm this separation: thinning local calibration support systematically inflates width and calibration instability while base-model uncertainty is held fixed by construction.

---

## 2. Repository Structure

```text
va-width-experiments/
├── README.md                           # Repository documentation and reproduction guide
├── requirements.txt                    # Minimal package requirements
├── requirements-lock.txt               # Exact pinned dependencies for bit-for-bit reproduction
├── LICENSE                             # MIT License
├── data/                               # Canonical precomputed experimental results
│   ├── calibration_resampling_baseline.csv   # Table 1: Calibration resampling baseline
│   ├── classifier_bootstrap_n500.csv         # Figure 2: Pointwise classifier bootstrap (N=500)
│   ├── SCALING_IDEALISED.csv                 # Figure 3: Idealised sample-size scaling laws
│   ├── proposition1_convergence.csv          # Table 2 & Figure 4: W2 exponent convergence
│   ├── proposition1_convergence_bootstrap.csv# Table 2: 1000 bootstrap draws per calibration size
│   ├── non_monotonic_scaling_laws.csv        # Figure 5: Regimes A, B, C, D scaling laws
│   ├── table_cifar_correlations.csv          # Table 3: Synthetic CIFAR-10H crowd correlations
│   ├── REAL_DATA_RESULTS.csv                 # Figure 6: Real-data calibration thinning
│   ├── REVERSE_INTERVENTION_RESULTS.csv      # Figure 7: Reverse training-support intervention
│   ├── UNCERTAINTY_DECOMPOSITION.csv         # Table 4: Held-out nested regression points
│   └── table_w2_high_n_robustness.csv        # Pooled W2 regressions
├── paper/                              # Publication LaTeX source, figures, tables, and PDF
│   ├── arxiv_va_interval_width.tex     # Master LaTeX manuscript (22 pages)
│   ├── arxiv_va_interval_width.pdf     # Compiled publication PDF
│   ├── generated_results.tex           # Auto-generated LaTeX macros from data manifest
│   ├── width_vs_bootstrap_instability.png # Figure 2 (PNG & PDF)
│   ├── idealised_scaling_laws.png      # Figure 3 (PNG & PDF)
│   ├── w2_exponent_convergence.png     # Figure 4 (PNG & PDF)
│   ├── non_monotonic_scaling_laws.png  # Figure 5 (PNG & PDF)
│   ├── real_data_local_support.png     # Figure 6 (PNG & PDF)
│   ├── training_support_epistemic.png  # Figure 7 (PNG & PDF)
│   ├── calibration_uncertainty_bootstrap.png # Figure 8: Calibrator comparison
│   ├── alternative_calibrator_instability.png# Alternative calibrators scatter plot
│   ├── table_calibration_resampling.tex# Table 1 LaTeX source
│   ├── table_w2_exponent_progression.tex# Table 2 LaTeX source
│   ├── table_cifar_correlations.tex    # Table 3 LaTeX source
│   └── table_real_data_nested.tex      # Table 4 LaTeX source
├── results/                            # Audit reports, selected points, and manifests
│   ├── REPO_PAPER_AUDIT_BEFORE.md      # Pre-audit consistency inventory
│   ├── REPO_PAPER_AUDIT_AFTER.md       # Post-audit verification report and checklist
│   ├── RESULT_CHANGES.md               # Systematic table of old vs corrected numbers
│   ├── paper_results_manifest.json     # Machine-readable numerical results manifest
│   ├── environment.txt                 # Exact environment, platform, and git commit
│   └── real_data_selected_points.csv   # Pre-specified test evaluation locations
└── src/                                # Reproducible experiment scripts
    ├── generate_paper_artifacts.py     # Fast generator: produces Figures 2-8 & Tables 1-4 (< 10s)
    ├── verify_paper_consistency.py     # Automated audit verifier (checks data, assets, manifest)
    ├── generate_manifest.py            # Generates manifest.json and generated_results.tex
    ├── run_all_experiments.py          # Master runner executing all experiments sequentially
    ├── utils.py                        # Common Venn-Abers extraction helpers and metrics
    ├── run_calibration_resampling_baseline.py # Generator for Table 1
    ├── run_experiment_classifier_bootstrap_n500.py # Generator for Figure 2
    ├── run_idealised_scaling.py        # Generator for Figure 3
    ├── run_experiment_w2_convergence.py# Generator for Figure 4 & Table 2
    ├── non_monotonic_experiments.py    # Generator for Figure 5 & invariance test
    ├── calibration_comparison.py       # Generator for Table 3 & Figure 8
    └── real_data_calibration_support.py# Generator for Figures 6, 7 & Table 4
```

---

## 3. Quickstart & Fast Reproduction

### Installation
Clone the repository and install dependencies in Python 3.10+:

```bash
git clone https://github.com/ip200/va-width-experiments.git
cd va-width-experiments
pip install -r requirements.txt
```

For bit-for-bit exact environment reproduction:
```bash
pip install -r requirements-lock.txt
```

### Fast Artifact Regeneration (< 10 seconds)
Regenerate all publication-quality figures (PDF & PNG) and LaTeX tables directly from canonical precomputed experimental data:

```bash
python src/generate_paper_artifacts.py
```

### Automated Consistency Verification (< 2 seconds)
Verify that all committed data files, manifests, figures, and tables agree to $< 5\times 10^{-5}$ numerical tolerance with zero obsolete artifacts or collisions:

```bash
python src/verify_paper_consistency.py
```

### Recompile the LaTeX Manuscript
Compile the camera-ready 22-page paper with `pdflatex` + `bibtex`:

```bash
cd paper
pdflatex -interaction=nonstopmode arxiv_va_interval_width.tex
bibtex arxiv_va_interval_width
pdflatex -interaction=nonstopmode arxiv_va_interval_width.tex
pdflatex -interaction=nonstopmode arxiv_va_interval_width.tex
```

---

## 4. Full End-to-End Experiment Execution

To rerun the entire experimental suite from scratch and regenerate all raw data:

```bash
python src/run_all_experiments.py
```

Individual experiments can also be executed independently:

| Experiment | Target Assets | Command | Expected Runtime |
|---|---|---|---|
| **1. Resampling Baseline** | Table 1 | `python src/run_calibration_resampling_baseline.py` | ~15 sec |
| **2. Classifier Bootstrap** | Figure 2 | `python src/run_experiment_classifier_bootstrap_n500.py` | ~25 sec |
| **3. Idealised Scaling** | Figure 3 | `python src/run_idealised_scaling.py` | ~20 sec |
| **4. Exponent Convergence** | Table 2, Figure 4 | `python src/run_experiment_w2_convergence.py` | ~2 min |
| **5. Non-Monotonic Scaling** | Figure 5 | `python src/non_monotonic_experiments.py` | ~10 sec |
| **6. Synthetic Crowd Experiment** | Table 3, Figure 8 | `python src/calibration_comparison.py` | ~45 sec |
| **7. Real-Data Support Thinning** | Figures 6, 7, Table 4 | `python src/real_data_calibration_support.py` | ~6 min |

---

## 5. Dataset Provenance & Benchmark Details

All real-world benchmarks are automatically fetched via OpenML with pinned version IDs:

| Dataset | OpenML Name / ID | Version | Total $N$ | Train / Cal / Test | Positive Class Definition |
|---|---|---|---|---|---|
| **Adult** | `adult` / 1590 | 2 | 48,842 | 24,421 / 12,210 / 12,211 | Income `>50K` |
| **Bank Marketing** | `bank-marketing` / 1461 | 1 | 45,211 | 22,605 / 11,303 / 11,303 | Subscription `"2"` (`"yes"`) |
| **Spambase** | `spambase` / 44 | 1 | 4,601 | 2,300 / 1,150 / 1,151 | Spam label `1` |

Base models utilize a uniform specification across all fits:
`HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=seed)`.

---

## 6. Citation

If you find this work or codebase useful, please cite:

```bibtex
@article{petej2026meaning,
  title={The Meaning and Scaling of Venn--Abers Probability Intervals},
  author={Petej, Ivan},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 7. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
