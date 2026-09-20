# The Meaning and Scaling of Venn--Abers Probability Intervals

This repository contains the complete, self-contained codebase, empirical benchmarks, precomputed experimental data, and LaTeX source for the paper:

> **The Meaning and Scaling of Venn--Abers Probability Intervals**  
> Ivan Petej  
> *Department of Computer Science, Royal Holloway, University of London*  
> arXiv: [arXiv preprint] | Code: [GitHub](https://github.com/ip200/va-width-experiments)

---

## 1. Abstract & Key Claims

Venn--Abers predictors (VAPs) output a multiprobabilistic pair $[p_0, p_1]$ with finite-sample validity guarantees under exchangeability. However, the exact semantic interpretation of the interval width $w = p_1 - p_0$ has remained an open question. In applications, width is often informally treated as a measure of outcome ambiguity (aleatoric uncertainty) or model parameter uncertainty (epistemic uncertainty).

This paper establishes that **Venn--Abers width is a measure of local calibration leverage**, reflecting the finite calibration support available in score space:

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
   exhibits the classical isotonic cube-root rate and provides a calibrated probability-scale instability summary.

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
├── README.md               # Repository documentation and reproduction guide
├── requirements.txt        # Python package dependencies
├── LICENSE                 # MIT License
├── paper/                  # Publication LaTeX source, figures, tables, and PDF
│   ├── arxiv_va_interval_width.tex     # Master LaTeX manuscript
│   ├── arxiv_va_interval_width.pdf     # Compiled publication PDF (22 pages)
│   ├── real_data_local_support.png     # Figure 6: Local calibration-support thinning
│   ├── training_support_epistemic.png  # Figure 7: Reverse training-support intervention
│   ├── table_real_data_nested.tex      # Table 3: Real-data nested regression decomposition
│   ├── idealised_scaling_laws.png      # Figure 3: Synthetic scaling laws (w ~ n^-2/3)
│   ├── table_w2_exponent_progression.tex # Table 1: Exponent convergence progression
│   ├── w2_exponent_convergence.png     # Figure 4: Exponent convergence plots
│   ├── non_monotonic_scaling_laws.png  # Figure 5: Non-monotonic & cusp scaling regimes
│   ├── width_vs_bootstrap_instability.png # Figure 2: Pointwise classifier bootstrap
│   ├── table_cifar_correlations.tex    # Table 2: CIFAR crowd-annotation simulation
│   └── calibration_uncertainty_bootstrap.png # Figure 8: Alternative calibrator comparison
├── src/                    # Modular Python experiment and plotting scripts
│   ├── utils.py                        # Common statistical utilities and data generators
│   ├── real_data_calibration_support.py # Primary real-data experiments (Adult, Bank, Spambase)
│   ├── replot_figures_6_7.py           # Publication rendering for Figures 6 and 7
│   ├── scaling_laws_experiment.py       # 1D controlled synthetic scaling experiments
│   ├── run_experiment_w2_convergence.py # Multivariate exponent convergence (n up to 32,000)
│   ├── non_monotonic_experiments.py    # Score invariance & non-monotonic slope regimes
│   ├── run_experiment_classifier_bootstrap_n500.py # Pointwise bootstrap evaluation
│   ├── generate_figure_width_vs_instability.py # Generates Figure 2
│   ├── calibration_comparison.py       # IR, PS, HB comparison & CIFAR crowd simulation
│   └── generate_all_figures_tables.py  # Fast reproduction of all figures and tables
├── data/                   # Experimental CSV data and precomputed resamples
│   ├── REAL_DATA_RESULTS.csv           # Real-data thinning metrics
│   ├── REVERSE_INTERVENTION_RESULTS.csv# Real-data reverse training intervention metrics
│   ├── UNCERTAINTY_DECOMPOSITION.csv   # N=500 pointwise decomposition evaluations
│   ├── SCALING_IDEALISED.csv           # 1D scaling law simulation data
│   ├── proposition1_convergence.csv    # Large-n multivariate exponent progression data
│   └── classifier_bootstrap_n500.csv   # Pointwise bootstrap evaluations
└── results/                # Detailed experiment execution reports and markdown logs
    └── REAL_DATA_EXPERIMENT_REPORT.md
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, and 3.13)
- TeX Live / MacTeX (with `pdflatex`) for compiling the paper

### Environment Installation
```bash
# 1. Clone repository
git clone https://github.com/ip200/va-width-experiments.git
cd va-width-experiments

# 2. Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and install package dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Reproducing Experiments & Figures

You can reproduce all paper artifacts in two ways:
1. **Fast Reproduction**: Regenerate all publication figures and LaTeX tables from precomputed experimental data in ~5 seconds.
2. **End-to-End Reproduction**: Re-run the full Monte Carlo simulations, bootstrap refits, and real-data interventions from scratch.

### 4.1 Fast Reproduction (From Precomputed Data)

To regenerate all figures and LaTeX tables directly from the experimental datasets:

```bash
# 1. Regenerate Figures 2, 3, 4, Table 1 from precomputed data
python src/generate_all_figures_tables.py

# 2. Regenerate Figures 6 and 7 (Real-Data Thinning & Reverse Intervention)
python src/replot_figures_6_7.py
```

---

### 4.2 End-to-End Reproduction (From Scratch)

#### Experiment 1: Real-Data Local Support Thinning & Reverse Intervention (Figures 6 & 7, Table 3)
Evaluates UCI Adult, UCI Bank Marketing, and UCI Spambase under controlled local calibration thinning (100% down to 12.5%, $R=100$) and reverse training support subsampling (100% down to 25%, $B=100$). Also estimates nested uncertainty regressions across $N=500$ held-out test observations per dataset with $B_{\text{cal}}=200$ bootstrap resamples and 2,000 bootstrap CI draws.

```bash
python src/real_data_calibration_support.py
```
*Expected Runtime*: ~3–5 minutes  
*Generated Artifacts*:
- `paper/real_data_local_support.png` (Figure 6)
- `paper/training_support_epistemic.png` (Figure 7)
- `paper/table_real_data_nested.tex` (Table 3)
- `data/REAL_DATA_RESULTS.csv`
- `data/REVERSE_INTERVENTION_RESULTS.csv`
- `data/UNCERTAINTY_DECOMPOSITION.csv`

#### Experiment 2: Synthetic Scaling Laws (Figure 3)
Simulates calibration sets across $n \in [100, 3200]$ under controlled density, ambiguity, and slope settings to verify the $w \propto n^{-2/3}$ and $U_{\text{cal}} \propto n^{-1/3}$ scaling rates.

```bash
python src/scaling_laws_experiment.py
```
*Expected Runtime*: ~20 seconds  
*Generated Artifacts*:
- `data/SCALING_IDEALISED.csv`
- `paper/idealised_scaling_laws.png` (Figure 3)

#### Experiment 3: Multivariate Exponent Convergence (Figure 4, Table 1)
Estimates local density ($f$), ambiguity ($v$), and slope ($\theta'$) exponents via multivariate log-linear regressions across calibration sizes up to $n = 32{,}000$, validating finite-sample convergence toward the asymptotic targets $(-2/3, -1/3, +2/3)$.

```bash
python src/run_experiment_w2_convergence.py
```
*Expected Runtime*: ~1–2 minutes  
*Generated Artifacts*:
- `data/proposition1_convergence.csv`
- `paper/w2_exponent_convergence.png` (Figure 4)
- `paper/table_w2_exponent_progression.tex` (Table 1)

#### Experiment 4: Score Invariance & Non-Monotonic Scaling Regimes (Figure 5)
Verifies empirical invariance under strictly increasing non-linear score transformations and benchmarks sample-size scaling across four distinct slope regimes:
- **Regime A (Standard Monotonic)**: $\theta' > 0$ (rate $-2/3$)
- **Regime B (Decreasing / Violating)**: $\theta' < 0$ (rate $-1$)
- **Regime C (Parabolic Extremum)**: local peak with violation (rate $-1$)
- **Regime D (Cusp / Flat Point)**: order-two flat point (rate $-4/5$)

```bash
python src/non_monotonic_experiments.py
```
*Expected Runtime*: ~30 seconds  
*Generated Artifacts*:
- `paper/non_monotonic_scaling_laws.png` (Figure 5)

#### Experiment 5: Pointwise Classifier Bootstrap & Instability Index (Figure 2)
Evaluates pointwise Venn--Abers width and $U_{\text{cal}}$ against empirical standard deviations from $M=200$ calibration bootstrap resamples on held-out test points ($N=500$).

```bash
# Step 1: Compute pointwise bootstrap data
python src/run_experiment_classifier_bootstrap_n500.py

# Step 2: Plot Figure 2
python src/generate_figure_width_vs_instability.py
```
*Expected Runtime*: ~45 seconds  
*Generated Artifacts*:
- `data/classifier_bootstrap_n500.csv`
- `paper/width_vs_bootstrap_instability.png` (Figure 2)

#### Experiment 6: Alternative Calibration Methods & CIFAR-10H Crowd Simulation (Figure 8, Table 2)
Compares Venn--Abers interval width against the sampling variability of Isotonic Regression (IR), Platt Scaling (PS), and Histogram Binning (HB). Runs a multiclass crowd-annotation simulation comparing width against bootstrap instability, annotator disagreement ($2p(1-p)$), and annotator entropy.

```bash
python src/calibration_comparison.py
```
*Expected Runtime*: ~45 seconds  
*Generated Artifacts*:
- `paper/calibration_uncertainty_bootstrap.png` (Figure 8)
- `paper/table_cifar_correlations.tex` (Table 2)

---

## 5. Mapping of Paper Elements to Code and Data

| Paper Element | Description | Script | Input Data | Output File |
| :--- | :--- | :--- | :--- | :--- |
| **Figure 1** | GCM Cumulative Sum Diagram | *(TikZ in LaTeX)* | N/A | `arxiv_va_interval_width.tex` |
| **Figure 2** | Width vs Bootstrap Instability ($U_{\text{cal}}$) | `generate_figure_width_vs_instability.py` | `classifier_bootstrap_n500.csv` | `paper/width_vs_bootstrap_instability.png` |
| **Figure 3** | Sample-Size Scaling Laws ($n^{-2/3}, n^{-1/3}$) | `scaling_laws_experiment.py` | `SCALING_IDEALISED.csv` | `paper/idealised_scaling_laws.png` |
| **Figure 4** | Exponent Convergence across $n$ | `run_experiment_w2_convergence.py` | `proposition1_convergence.csv` | `paper/w2_exponent_convergence.png` |
| **Figure 5** | Non-Monotonic & Cusp Scaling Regimes | `non_monotonic_experiments.py` | Synthetic generation | `paper/non_monotonic_scaling_laws.png` |
| **Figure 6** | Real-Data Calibration Support Thinning | `replot_figures_6_7.py` | `REAL_DATA_RESULTS.csv` | `paper/real_data_local_support.png` |
| **Figure 7** | Reverse Training Support Intervention | `replot_figures_6_7.py` | `REVERSE_INTERVENTION_RESULTS.csv` | `paper/training_support_epistemic.png` |
| **Figure 8** | Alternative Calibration Comparison (IR/PS/HB) | `calibration_comparison.py` | Synthetic generation | `paper/calibration_uncertainty_bootstrap.png` |
| **Table 1** | Exponent Progression across $n$ | `run_experiment_w2_convergence.py` | `proposition1_convergence.csv` | `paper/table_w2_exponent_progression.tex` |
| **Table 2** | CIFAR Crowd Simulation Correlations | `calibration_comparison.py` | Synthetic embeddings | `paper/table_cifar_correlations.tex` |
| **Table 3** | Real-Data Nested Regressions ($N=500$) | `real_data_calibration_support.py` | `UNCERTAINTY_DECOMPOSITION.csv` | `paper/table_real_data_nested.tex` |

---

## 6. Compiling the LaTeX Paper

The complete camera-ready paper can be compiled from the `paper/` directory using standard `pdflatex`:

```bash
cd paper
pdflatex -interaction=nonstopmode arxiv_va_interval_width.tex
pdflatex -interaction=nonstopmode arxiv_va_interval_width.tex
```

The resulting document is:
- **Output PDF**: `paper/arxiv_va_interval_width.pdf` (22 pages, 0 errors, 0 undefined references).

---

## 7. Citation

If you find this work or codebase useful in your research, please cite:

```bibtex
@article{petej2026meaning,
  title={The Meaning and Scaling of Venn--Abers Probability Intervals},
  author={Petej, Ivan},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
