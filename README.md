# The Meaning and Scaling of Venn--Abers Probability Intervals

This repository contains the complete, self-contained codebase, empirical benchmarks, and LaTeX source for the paper:

> **The Meaning and Scaling of Venn--Abers Probability Intervals**  
> Ivan Petej  
> *Department of Computer Science, Royal Holloway, University of London*

---

## Abstract & Overview

Venn--Abers predictors (VAPs) return two calibrated probability values, $p_0$ and $p_1$, defining an interval $[p_0, p_1]$ with proven multi-probabilistic validity under exchangeability. However, the exact semantic interpretation of the interval width $w = p_1 - p_0$ has remained unclear.

In this work, we investigate the interval width through the local active block structure of isotonic regression and the Greatest Convex Minorant (GCM):
1. **Inverse Block Size**: A test point falls into a local PAV block of effective size $N_{\text{eff}}(s)$. Perturbing the test point's hypothetical label shifts the local cumulative sum by $1$, yielding:
   $$w(s) \approx \frac{1}{N_{\text{eff}}(s)}$$
2. **Local Scaling Law**: Balancing systematic probability variation against local Bernoulli noise yields the asymptotic scaling relationship:
   $$w_n(s) \propto n^{-2/3} f(s)^{-2/3} [\theta(s)(1 - \theta(s))]^{-1/3} [\theta'(s)]^{2/3}$$
   where $n$ is calibration set size, $f(s)$ is calibration score density, $\theta(s)$ is the true conditional probability, and $\theta'(s)$ is the local slope.
3. **Probability-Scale Instability**: The probability-scale fluctuation index:
   $$U_{\text{cal}}(s) = \sqrt{\hat p(s)(1 - \hat p(s)) w(s)} \propto n^{-1/3}$$
   exhibits the classical isotonic cube-root rate and closely tracks empirical calibration bootstrap standard deviations.
4. **Three Distinct Uncertainties**:
   - **Outcome Ambiguity** (Aleatoric proxy): $A(x) = \hat p(x)(1 - \hat p(x))$
   - **Base Model Epistemic Uncertainty**: $E_{\text{model}}(x) = \text{SD}_b(g_b(x))$
   - **Calibration Epistemic Uncertainty / Support**: $w(x) = p_1(x) - p_0(x)$
   Controlled local-support thinning experiments on real tabular benchmarks (**UCI Adult**, **UCI Bank Marketing**, **UCI Spambase**) demonstrate that reducing local calibration support increases Venn--Abers width and calibration instability without affecting base-model epistemic uncertainty.

---

## Repository Structure

```text
va-width-experiments/
├── README.md               # Repository documentation and instructions
├── requirements.txt        # Python package dependencies
├── LICENSE                 # License
├── paper/                  # LaTeX paper source, figures, tables, and compiled PDF
│   ├── arxiv_va_interval_width.tex   # Main paper LaTeX source
│   ├── arxiv_va_interval_width.pdf   # Compiled camera-ready PDF
│   ├── idealised_scaling_laws.png    # Figure 1: Sample-size scaling
│   ├── table_w2_exponent_progression.tex # Table 1: Exponent progression across n
│   ├── w2_exponent_convergence.png   # Figure 2: Convergence to theoretical exponents
│   ├── non_monotonic_scaling_laws.png# Figure 3: Non-monotonic scaling regimes
│   ├── width_vs_bootstrap_instability.png # Figure 4: Classifier bootstrap comparison
│   ├── table_cifar_correlations.tex  # Table 2: CIFAR simulation correlations
│   ├── real_data_local_support.png   # Figure 5: Real-data local support thinning
│   ├── table_real_data_nested.tex    # Table 3: Real-data nested regression decomposition
│   └── calibration_uncertainty_bootstrap.png # Figure 6: Alternative calibration comparison
├── src/                    # Modular Python experiment and plotting scripts
│   ├── utils.py                      # Shared utilities and synthetic generators
│   ├── real_data_calibration_support.py # Primary real-data experiments (Adult, Bank, Spambase)
│   ├── scaling_laws_experiment.py     # 1D controlled synthetic scaling experiments
│   ├── run_experiment_w2_convergence.py # Large-n multivariate exponent convergence
│   ├── non_monotonic_experiments.py  # Monotonic invariance & non-monotonic slope regimes
│   ├── run_experiment_classifier_bootstrap_n500.py # Pointwise classifier bootstrap
│   ├── generate_figure_width_vs_instability.py # Generates Figure 4
│   ├── calibration_comparison.py     # IR, PS, HB comparison & CIFAR crowd simulation
│   └── generate_final_figures_tables.py # Master figure/table reproduction script
├── data/                   # Experimental data and precomputed summary CSVs
│   ├── SCALING_IDEALISED.csv
│   ├── proposition1_convergence.csv
│   ├── classifier_bootstrap_n500.csv
│   ├── REAL_DATA_RESULTS.csv
│   └── UNCERTAINTY_DECOMPOSITION.csv
└── results/                # Experiment logs and execution reports
    └── REAL_DATA_EXPERIMENT_REPORT.md
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- TeX Live / MacTeX (with `pdflatex`) for compiling the paper

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ip200/va-width-experiments.git
cd va-width-experiments

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Reproducing Experiments and Figures

### 1. Real-Data Calibration Support Thinning (Primary Experiment)
Evaluates UCI Adult, UCI Bank Marketing, and UCI Spambase under local calibration support thinning (100% down to 12.5%) and base model training subsampling:
```bash
python src/real_data_calibration_support.py
```
*Outputs:*
- `paper/real_data_local_support.png` (and `.pdf`)
- `paper/training_support_epistemic.png` (and `.pdf`)
- `paper/table_real_data_nested.tex`
- `data/REAL_DATA_RESULTS.csv`
- `data/UNCERTAINTY_DECOMPOSITION.csv`
- `results/REAL_DATA_EXPERIMENT_REPORT.md`

### 2. Theoretical Scaling Laws & Convergence
```bash
# 1D controlled scaling laws
python src/scaling_laws_experiment.py

# Multivariate exponent convergence across n_cal in [500, 32000]
python src/run_experiment_w2_convergence.py

# Non-monotonic regimes (Regimes A, B, C, D) & monotonic score invariance
python src/non_monotonic_experiments.py
```

### 3. Classifier Bootstrap & Instability Index
```bash
# Run classifier bootstrap experiment at N_cal = 500
python src/run_experiment_classifier_bootstrap_n500.py

# Plot Figure 4: Raw width vs U_cal vs bootstrap SD
python src/generate_figure_width_vs_instability.py
```

### 4. Alternative Calibration Methods & CIFAR-10H Crowd Simulation
```bash
# Compares Isotonic Regression, Platt Scaling, Histogram Binning, and Venn-Abers
# and runs crowd annotation simulation
python src/calibration_comparison.py
```

### 5. Generate All Publication Figures and Tables
To regenerate all figures and tables from computed data:
```bash
python src/generate_final_figures_tables.py
```

---

## Compiling the Paper

To compile the LaTeX source into the final publication PDF:

```bash
cd paper
pdflatex -interaction=nonstopmode arxiv_va_interval_width.tex
pdflatex -interaction=nonstopmode arxiv_va_interval_width.tex
```

The compiled PDF will be output at `paper/arxiv_va_interval_width.pdf`.

---

## Citation

```bibtex
@article{petej2026meaning,
  title={The Meaning and Scaling of Venn--Abers Probability Intervals},
  author={Petej, Ivan},
  journal={arXiv preprint},
  year={2026}
}
```

## License
MIT License. See [LICENSE](LICENSE) for details.
