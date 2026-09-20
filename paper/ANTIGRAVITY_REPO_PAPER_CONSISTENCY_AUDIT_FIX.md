# Antigravity — Repo–Paper Consistency Audit and Fix

## Repository and manuscript

Repository:

```text
https://github.com/ip200/va-width-experiments
```

Audit the current `main` branch of this repository against the **latest manuscript**:

```text
The Meaning and Scaling of Venn–Abers Probability Intervals
```

The purpose of this task is **not** to improve the empirical results or preserve previously reported numbers. The purpose is to make the repository, code, committed data, figures, tables, README, and manuscript describe **one identical set of experiments**.

The manuscript is the conceptual source of truth for definitions and experimental intent. The corrected code and newly generated experimental outputs are the numerical source of truth.

If corrected code changes a paper result, **update the paper**. Never alter code, random seeds, filters, samples, or analysis choices merely to recover an old number.

---

# 0. Non-negotiable rules

1. **Do not target-match existing paper numbers.**
2. **Do not silently retain old CSVs after changing code.**
3. **Do not silently change the scientific protocol.**
4. If a methodological correction changes a result, record the old and new values in the audit report.
5. All paper-critical results must be reproducible from committed source code.
6. Every table and figure in the paper must have a unique generating script and a clearly identified input data file.
7. No two scripts may write different experiments to the same output filename.
8. All variables named `p_mid` or `p_hat` must mean the arithmetic midpoint
   \[
   \hat p = (p_0+p_1)/2.
   \]
9. If the standard Venn–Abers log-loss merge is used,
   \[
   p_{\rm VA}=\frac{p_1}{1-p_0+p_1},
   \]
   it must be named `p_va`, `p_logloss`, or equivalent and must not be described as the midpoint.
10. Width is always
    \[
    w=p_1-p_0.
    \]
11. In synthetic experiments where the true conditional probability \(\theta(s)\) is known, a theoretical quantity such as
    \[
    v(s)=\theta(s)(1-\theta(s))
    \]
    must be calculated from the known \(\theta\), not from a fitted probability, unless the manuscript explicitly states that an estimated proxy is being used.
12. Create a new branch for this work. Do not make unreviewed destructive edits directly to `main`.

Suggested branch:

```text
repo-paper-consistency-audit
```

---

# 1. First produce an experiment inventory

Before changing code, inspect:

```text
README.md
requirements.txt
.gitignore
src/
data/
results/
```

and the latest manuscript.

Create:

```text
results/REPO_PAPER_AUDIT_BEFORE.md
```

with a complete mapping:

| Paper item | Scientific claim | Script | Raw/generated data | Figure/table asset | Status |
|---|---|---|---|---|---|

The inventory must cover **every empirical paper element**, including:

- calibration-resampling baseline table;
- pointwise width/bootstrap-instability experiment;
- idealised sample-size scaling experiment;
- multivariate exponent-convergence experiment;
- score-transformation invariance experiment;
- non-monotonic / flat-point experiments;
- synthetic crowd-annotation experiment;
- real-data local-calibration-support intervention;
- reverse training-support intervention;
- real-data uncertainty-decomposition regression;
- alternative-calibrator comparison, if retained in the manuscript.

Use the actual current manuscript numbering.

Do not proceed until every empirical figure/table has been mapped or explicitly marked **MISSING REPRODUCTION PATH**.

---

# 2. Critical fix A — midpoint versus standard Venn–Abers merge

## Current problem

In:

```text
src/run_experiment_classifier_bootstrap_n500.py
```

the object currently called `p_mid` comes from the first output of `VennAbersCalibrator.predict_proba(...)`.

That first output is the standard Venn–Abers log-loss merge:

\[
p_{\rm VA}=\frac{p_1}{1-p_0+p_1},
\]

not

\[
\hat p=\frac{p_0+p_1}{2}.
\]

The manuscript now defines

\[
\hat p=\frac{p_0+p_1}{2}
\]

as the scalar summary used in the empirical instability index

\[
U_{\rm cal}=\sqrt{\hat p(1-\hat p)w}.
\]

The committed classifier-bootstrap data confirm that the current `p_mid` is effectively \(p_{\rm VA}\), not the midpoint.

## Required fix

Refactor all Venn–Abers probability extraction into one common helper, for example:

```python
def extract_va_outputs(...):
    # return p0, p1, p_mid, p_va, width
```

with:

```python
p_mid = 0.5 * (p0 + p1)
p_va = p1 / (1.0 - p0 + p1)
width = p1 - p0
```

Audit the entire repository for:

```text
p_mid
p_hat
p_prime
p_va
predict_proba
```

and ensure each use is semantically correct.

### Paper rule

If the manuscript says the bootstrap instability of \(\hat p\) is being measured, bootstrap the **midpoint**.

If a separate experiment intentionally studies the standard VA merged point probability, state this explicitly in the paper and code.

Do not silently mix them.

## Mandatory reruns after this fix

At minimum rerun:

```text
src/run_experiment_classifier_bootstrap_n500.py
```

and any other experiment whose calibration-bootstrap SD is based on a VA scalar point prediction, including the crowd-annotation experiment if it currently bootstraps \(p_{\rm VA}\) while the manuscript describes \(\hat p\).

Regenerate all affected:

```text
CSV
figure
table
manuscript numbers
captions
README results
```

The old U_cal correlations are **checkpoints only**, not targets:

```text
Pearson(U_cal, bootstrap SD) ≈ 0.5300
Spearman(U_cal, bootstrap SD) ≈ 0.4247
held-out multiplicative MAE ≈ 0.0389
```

They are allowed to change.

Width-only statistics should normally be unchanged by this particular correction, but verify rather than assume.

---

# 3. Critical fix B — use true ambiguity in the W2 synthetic exponent experiment

## Current problem

In:

```text
src/run_experiment_w2_convergence.py
```

the code currently computes:

```python
amb = pm * (1.0 - pm)
```

where `pm` is an estimated VA midpoint.

However, the manuscript's Proposition-1 experiment is described as testing the scaling with the **known synthetic**

\[
v(s)=\theta(s)(1-\theta(s)).
\]

Because \(\theta\) is known exactly in this simulation, using the fitted midpoint creates an unnecessary estimation layer and can contaminate the estimated ambiguity exponent.

## Required fix

Replace the empirical ambiguity regressor with the known DGP value:

```python
theta = p_true_eval[i]
ambiguity_true = theta * (1.0 - theta)
```

Store both quantities if useful:

```text
ambiguity_true
ambiguity_hat
```

but the Proposition-1 scaling regression must use:

```text
ambiguity_true
```

unless the manuscript is explicitly rewritten to test the proxy instead.

## Mandatory rerun

Rerun the full experiment:

```text
src/run_experiment_w2_convergence.py
```

for:

```text
n_cal = 500, 1000, 2000, 4000, 8000, 16000, 32000
```

using the manuscript's existing density families, probability curves, score grid, replication schedule, and bootstrap count.

Regenerate:

```text
data/proposition1_convergence.csv
data/proposition1_convergence_bootstrap.csv
paper/table_w2_exponent_progression.tex
paper/w2_exponent_convergence.png
paper/w2_exponent_convergence.pdf
```

Also regenerate any pooled-regression result quoted in the manuscript.

Do **not** preserve the old exponent values if the corrected experiment changes them.

The previous values are audit checkpoints only:

```text
n=500:   beta_rho=-0.4250, beta_v=-0.1286, beta_s=+0.3903
n=1000:  beta_rho=-0.5235, beta_v=-0.2427, beta_s=+0.5358
n=2000:  beta_rho=-0.5386, beta_v=-0.2372, beta_s=+0.5543
n=4000:  beta_rho=-0.6022, beta_v=-0.3134, beta_s=+0.6069
n=8000:  beta_rho=-0.6570, beta_v=-0.2955, beta_s=+0.5895
n=16000: beta_rho=-0.6601, beta_v=-0.3878, beta_s=+0.6899
n=32000: beta_rho=-0.6147, beta_v=-0.3025, beta_s=+0.6143
```

If they change, update the manuscript and explain the correction in the audit report.

---

# 4. Critical fix C — real-data calibration-thinning bootstrap bug

## Current problem

In:

```text
src/real_data_calibration_support.py
```

the local-support intervention currently:

1. generates `R_thin` thinning replicates and averages width;
2. retains only the **last** `rep_cal_idx`;
3. computes `sigma_cal` using bootstrap resampling from that final thinning replicate.

Therefore:

```text
mean_width
```

is averaged over the thinning distribution, while:

```text
mean_sigma_cal
```

comes from one arbitrary last thinning draw.

There is also an incorrect field:

```python
"std_sigma_cal": float(np.std(w_reps))
```

which stores width variability under the name `std_sigma_cal`.

This must be fixed.

## Correct protocol

For each:

```text
dataset
target score
retained_fraction
```

run:

```text
R_thin = 100
```

independent local-thinning replicates.

For each thinning replicate \(r\):

1. create the thinned calibration set;
2. compute Venn–Abers width \(w_r\);
3. bootstrap that **same** thinned calibration set `B_cal` times;
4. compute
   \[
   \sigma_{{\rm cal},r}
   =
   \operatorname{SD}_b(\hat p_{r,b});
   \]
5. store both \(w_r\) and \(\sigma_{{\rm cal},r}\).

Then aggregate across thinning replicates:

```python
mean_width = mean(w_r)
std_width = std(w_r)

mean_sigma_cal = mean(sigma_cal_r)
std_sigma_cal = std(sigma_cal_r)
```

Prefer also storing 2.5% and 97.5% quantiles across thinning replicates.

### Computational cost

Do not replace this with the old "last replicate" shortcut.

If the full:

```text
R_thin=100
B_cal=100
```

combination is computationally expensive, optimize the implementation, parallelize safely, or cache intermediate results.

Do not reduce either count without explicitly documenting the change and updating the manuscript.

## Mandatory rerun

Rerun all three datasets:

```text
Adult
Bank Marketing
Spambase
```

for retained fractions:

```text
1.00
0.75
0.50
0.25
0.125
```

and the five pre-specified score targets:

```text
0.20
0.35
0.50
0.65
0.80
```

Regenerate the real-data support CSV and Figure 6.

### Important current mismatch

The currently committed Spambase calibration-bootstrap SD aggregated over the five target points is approximately:

```text
100%  : 0.0671
75%   : 0.0688
50%   : 0.0751
25%   : 0.0873
12.5% : 0.0768
```

This is **not monotonic** and does not match the manuscript's currently quoted terminal value.

After the bug fix, report the corrected pattern exactly as observed.

Do not state "monotonic" unless the corrected results are actually monotonic.

---

# 5. Critical fix D — use one base-model specification everywhere

## Current problem

The main real-data classifier uses:

```python
HistGradientBoostingClassifier(
    max_depth=4,
    learning_rate=0.05,
    max_iter=200,
    ...
)
```

but model-epistemic bootstrap refits and reverse-intervention refits currently use:

```text
max_iter=100
```

This means the "same model under resampling" experiment is not actually using the same model specification.

## Required fix

Create a single constructor, e.g.:

```python
def make_base_model(seed):
    return HistGradientBoostingClassifier(
        categorical_features=...,
        max_depth=4,
        learning_rate=0.05,
        max_iter=200,
        random_state=seed,
    )
```

and use it for:

- full-data base fit;
- training-bootstrap refits used for \(E_{\rm model}\);
- reverse training-support intervention;
- any related real-data robustness calculation.

Do not have duplicated hyperparameter definitions.

Preserve the existing early-stopping setting unless the manuscript explicitly changes it. The goal here is consistency, not a new model-selection exercise.

## Mandatory rerun

Because this change can alter:

```text
E_model
reverse-intervention curves
nested regression results
```

rerun:

- Figure 6 panel (c) quantities;
- Figure 7;
- Table 4 / real-data nested decomposition;
- all associated manuscript numbers.

The previously reported Table 4 values are audit checkpoints only:

```text
Adult:    ΔR² ≈ 0.0208
Bank:     ΔR² ≈ 0.0171
Spambase: ΔR² ≈ 0.0045
```

They may change.

---

# 6. Fix the idealised scaling experiment reproduction path

## Current problem

The committed:

```text
data/SCALING_IDEALISED.csv
```

reproduces the manuscript's current slopes:

```text
width slope       ≈ -0.6390
bootstrap SD      ≈ -0.3172
U_cal slope       ≈ -0.3285
```

but no committed script clearly generates this CSV.

The existing:

```text
src/scaling_laws_experiment.py
```

is a different experiment and is not a valid end-to-end generator for the paper's current Figure 3.

## Required fix

Create one canonical generator:

```text
src/run_idealised_scaling.py
```

that implements **exactly the DGP and protocol described in the manuscript**.

The intended paper DGP is:

\[
S\sim {\rm Uniform}(0,1),
\qquad
Y\mid S=s\sim{\rm Bernoulli}(0.2+0.6s),
\]

evaluated at:

\[
s_0=0.5.
\]

Audit the manuscript for:

- calibration sizes;
- number of Monte Carlo repetitions;
- number of bootstrap resamples;
- confidence-interval procedure;
- midpoint definition.

Make all of these explicit in both code and paper.

The script must generate:

```text
data/SCALING_IDEALISED.csv
paper/idealised_scaling_laws.png
paper/idealised_scaling_laws.pdf
```

from scratch.

Do not merely preserve the current CSV.

---

# 7. Restore a reproduction path for the calibration-resampling baseline table

The manuscript contains a calibration-resampling baseline table with calibration sizes:

```text
100
250
500
1000
2000
```

and quantities such as:

```text
mean width
bootstrap SD
correlation
Brier score
```

but the current repository does not contain a clear script/data path reproducing this table.

Create a canonical script, for example:

```text
src/run_calibration_resampling_baseline.py
```

and a committed raw/summary CSV:

```text
data/calibration_resampling_baseline.csv
```

The script must implement the methodology stated in the manuscript.

If the original experiment code exists in git history or elsewhere in the working tree, recover and clean it rather than reverse-engineering numbers.

If it cannot be recovered, reconstruct the experiment **from the manuscript protocol only**.

Do not tune the reconstruction to obtain the old values.

The old values are checkpoints, not targets:

```text
n=100:  width≈0.1564, bootstrap SD≈0.1131, corr≈0.9127, Brier≈0.1236
n=250:  width≈0.1045, bootstrap SD≈0.1118, corr≈0.8603, Brier≈0.1234
n=500:  width≈0.0658, bootstrap SD≈0.0990, corr≈0.7802, Brier≈0.1197
n=1000: width≈0.0423, bootstrap SD≈0.0774, corr≈0.8520, Brier≈0.1197
n=2000: width≈0.0261, bootstrap SD≈0.0586, corr≈0.8779, Brier≈0.1161
```

If the corrected reproducible implementation yields different values, update the table and manuscript discussion.

---

# 8. Audit the crowd-annotation experiment terminology and point probability

The repository currently implements a **synthetic CIFAR-10H-inspired crowd experiment**, not an experiment on the actual CIFAR-10H annotation dataset.

Ensure the repository and manuscript consistently call it something like:

```text
synthetic CIFAR-10H-inspired crowd-annotation experiment
```

or:

```text
controlled synthetic crowd-annotation experiment
```

Do not call it simply "CIFAR-10H" in a way that implies real CIFAR-10H data were used.

Retain:

\[
\text{annotator disagreement}=2p(1-p)
\]

only with the correct interpretation:

> the probability that two independently sampled annotators give different binary labels.

Do not call \(2p(1-p)\) the Bernoulli variance.

Bernoulli variance is:

\[
p(1-p).
\]

Also audit whether crowd bootstrap instability is computed from \(p_{\rm VA}\) or the manuscript-defined midpoint \(\hat p\). Make code and manuscript agree.

Rerun the table if this changes.

Old correlation values are checkpoints only:

```text
Width vs bootstrap instability:
Pearson ≈ 0.6899
Spearman ≈ 0.5298

Width vs annotator disagreement:
Pearson ≈ 0.0134
Spearman ≈ -0.0081

Width vs annotator entropy:
Pearson ≈ 0.0163
Spearman ≈ -0.0081
```

---

# 9. Keep the non-monotonic and invariance experiments, but verify exact correspondence

Audit:

```text
src/non_monotonic_experiments.py
```

against the manuscript.

Verify that the four regimes are exactly:

```text
A: theta(s) = 0.5 + 0.3s
B: theta(s) = 0.5 - 0.3s
C: theta(s) = 0.5 - 0.4s^2
D: theta(s) = 0.5 + 0.4 s^2 sign(s)
```

and that the paper describes Regime D correctly as an order-two cusp/flat-point regime, not a smooth quadratic extremum.

Verify the reported rates by rerunning the experiment.

Current checkpoints:

```text
A ≈ -0.660
B ≈ -1.004
C ≈ -1.097
D ≈ -0.802
```

Also verify invariance under strictly **increasing** transformations only:

```text
s
s^2   on the restricted score domain used in the experiment
exp(s)
```

The manuscript and README must say **strictly increasing**, not merely "monotonic" if decreasing transformations are not included in the invariance statement.

---

# 10. Remove all obsolete LLM/agent artifacts

The paper no longer contains the LLM/agent experiment.

However:

```text
src/generate_final_figures_tables.py
```

still contains hard-coded obsolete real-agent tables/figures involving:

```text
GSM8K
MATH
HotpotQA
TriviaQA
BFCL
ΔR² = 0.1119
```

This file must not remain as a "final" paper-generation script.

Choose one of:

```text
A. delete it; or
B. rewrite it so it generates only the current paper.
```

There must be **zero current-paper generation paths** that recreate removed LLM content.

Search the entire repository case-insensitively for:

```text
agent
LLM
GPT
GSM8K
MATH
HotpotQA
TriviaQA
BFCL
0.1119
```

Any retained occurrence must be clearly historical and outside the reproducibility path.

Prefer removing obsolete artifacts entirely.

---

# 11. Prevent filename collisions

Currently:

```text
src/calibration_comparison.py
```

can write a different plot to:

```text
paper/width_vs_bootstrap_instability.png
```

which is also the filename of the actual paper pointwise bootstrap figure.

This is unacceptable.

Give every experiment a unique output filename.

For example:

```text
paper/width_vs_bootstrap_instability.png
paper/alternative_calibrator_instability.png
```

No script may overwrite another paper figure.

Add an automated check that all declared publication assets have unique producer scripts.

---

# 12. Fix output paths and stale-data risk

All experiment scripts must write to canonical repository directories.

Use only:

```text
data/
paper/
results/
```

for publication outputs.

Do not write duplicate experiment CSVs into the repository root.

For example, correct scripts that currently create:

```text
REAL_DATA_RESULTS.csv
UNCERTAINTY_DECOMPOSITION.csv
proposition1_convergence.csv
```

in the root while plotting scripts read from:

```text
data/
```

This creates a serious stale-data risk.

Canonicalize to:

```text
data/REAL_DATA_RESULTS.csv
data/REVERSE_INTERVENTION_RESULTS.csv
data/UNCERTAINTY_DECOMPOSITION.csv
data/proposition1_convergence.csv
data/proposition1_convergence_bootstrap.csv
...
```

Every plotting/table script must read those exact canonical files.

No hidden absolute local paths such as:

```text
/Users/ivanpetej/...
```

may remain in the final reproducibility code.

---

# 13. Fix README versus actual repository structure

The current README claims a committed:

```text
paper/
```

directory containing the LaTeX manuscript, figures, tables and PDF.

But `.gitignore` currently ignores:

```text
paper/
```

and the GitHub repository does not contain that directory.

Resolve this contradiction.

Preferred solution:

1. remove `paper/` from `.gitignore`;
2. commit:
   - manuscript `.tex`;
   - bibliography;
   - generated `.tex` tables;
   - publication figures;
3. PDF is optional if repository size is a concern.

If the manuscript source is intentionally not to be committed, rewrite the README so it does not claim it is present.

The final README structure listing must match the actual git tree exactly.

---

# 14. Create one canonical paper-generation entry point

Replace the current overlapping generation scripts with one canonical command:

```bash
python src/generate_paper_artifacts.py
```

This command must regenerate all publication figures and tables from the canonical committed data.

It must not rerun expensive Monte Carlo experiments.

It should generate exactly the current paper assets, for example:

```text
Figure 1  GCM schematic — generated in LaTeX/TikZ if applicable
Figure 2  pointwise width vs calibration instability
Figure 3  idealised sample-size scaling
Figure 4  exponent convergence
Figure 5  non-monotonic regimes
Figure 6  real-data calibration-support thinning
Figure 7  reverse training-support intervention
Figure 8  alternative-calibrator comparison, if retained

Table 1  calibration-resampling baseline
Table 2  exponent progression
Table 3  synthetic crowd-annotation correlations
Table 4  real-data nested regressions
```

Use the **actual latest manuscript numbering**, not this example if numbering has changed.

Remove or archive redundant generation scripts after the canonical script has been verified.

---

# 15. Create one canonical full-experiment entry point

Add:

```bash
python src/run_all_experiments.py
```

This should execute the paper-critical experiments in a documented order and regenerate the canonical data files.

It may call subprocesses or imported functions, but it must fail loudly if any experiment fails.

Suggested order:

```text
1. calibration-resampling baseline
2. classifier bootstrap
3. idealised scaling
4. W2 exponent convergence
5. non-monotonic/invariance
6. synthetic crowd experiment
7. real-data support intervention
8. reverse training-support intervention
9. real-data nested decomposition
10. alternative calibrator comparison
```

At the end:

```text
python src/generate_paper_artifacts.py
python src/verify_paper_consistency.py
```

---

# 16. Add a machine-readable results manifest

Create:

```text
results/paper_results_manifest.json
```

containing every numerical result quoted in the manuscript.

Example structure:

```json
{
  "classifier_bootstrap": {
    "width_pearson": ...,
    "width_spearman": ...,
    "width_mae": ...,
    "ucal_pearson": ...,
    "ucal_spearman": ...,
    "ucal_mae": ...
  },
  "idealised_scaling": {
    "width_slope": ...,
    "bootstrap_sd_slope": ...,
    "ucal_slope": ...
  },
  "w2": {
    "500": {...},
    "1000": {...}
  },
  "real_data_nested": {
    "adult": {...},
    "bank": {...},
    "spambase": {...}
  }
}
```

This file must be **generated from result data**, not manually typed to match the manuscript.

Then create:

```text
src/verify_paper_consistency.py
```

which:

1. recomputes all manifest quantities from committed CSVs;
2. verifies they agree with the manifest within a strict tolerance;
3. verifies every expected figure/table source file exists;
4. verifies there are no duplicate publication output names;
5. verifies no obsolete agent artifact is in the current generation path;
6. exits non-zero on any inconsistency.

Suggested numerical tolerance:

```text
absolute tolerance = 5e-5
```

for values printed to four decimal places.

---

# 17. Prefer generated LaTeX macros for quoted numerical results

To reduce future drift, generate:

```text
paper/generated_results.tex
```

from the verified manifest.

For example:

```latex
\newcommand{\WidthPearson}{0.4287}
\newcommand{\UCalPearson}{0.xxxx}
\newcommand{\IdealWidthSlope}{-0.6390}
\newcommand{\AdultDeltaRsq}{0.xxxx}
```

Where practical, use these macros in the manuscript rather than manually duplicating numbers.

Do not convert every number into a macro if it harms readability; prioritize headline numerical results repeated in prose and tables.

---

# 18. Environment reproducibility

The current dependency constraints are too broad for exact numerical reproduction.

After obtaining one verified clean run, create:

```text
requirements-lock.txt
```

or an equivalent reproducible environment file containing exact package versions used for the final results.

At minimum record exact versions of:

```text
python
numpy
scipy
pandas
scikit-learn
venn-abers
matplotlib
```

If `seaborn` is no longer needed, remove it.

Every full run should write:

```text
results/environment.txt
```

including:

```text
Python version
OS/platform
package versions
git commit SHA
random seeds
timestamp
```

Keep `requirements.txt` as a human-readable minimal dependency file if desired, but the README must point to the exact lock file for reproduction.

---

# 19. Dataset provenance and target coding

For each real dataset, record in the README/report:

```text
OpenML dataset name
OpenML version
raw N
target labels/classes
positive-class mapping
post-preprocessing N
train/calibration/test sizes
random seed
```

Explicitly verify the Bank Marketing positive-class mapping.

Do not assume that:

```python
target == "2"
```

is correct without printing and documenting the actual OpenML class labels for the pinned dataset version.

For Adult, verify:

```text
N = 48,842
train = 24,421
cal ≈ 12,210
test ≈ 12,211
```

subject to exact sklearn split rounding.

For Bank:

```text
N = 45,211
```

For Spambase:

```text
N = 4,601
```

If the actual loaded dataset differs, use the actual values and update the manuscript.

---

# 20. Preserve the five real-data intervention points mechanically

The manuscript states that the representative test locations are pre-specified at scores near:

```text
0.20
0.35
0.50
0.65
0.80
```

The code must select them mechanically using nearest-score matching before any intervention outcome is inspected.

Store selected test indices and actual scores in:

```text
results/real_data_selected_points.csv
```

This provides an audit trail against post-hoc point selection.

---

# 21. Statistical-reporting audit

Verify that the manuscript exactly matches the implemented uncertainty calculations.

For the real-data nested regressions, report:

```text
N_eval = 500 test points per dataset
B_cal = 200 calibration bootstraps
B_boot = 2000 point-level bootstrap resamples for ΔR² CIs
```

or update these values if the corrected code uses something else.

Clarify that the 500 test points are selected mechanically.

The current implementation uses:

```python
np.linspace(0, len(X_test)-1, 500, dtype=int)
```

which samples by dataset row order rather than score.

Decide whether this is the intended manuscript protocol.

If the manuscript says "500 held-out test observations", this is acceptable if the held-out set order is random from the split, but document it.

If the manuscript implies a random sample, replace with a seeded random sample and rerun.

Do not leave an ambiguity between "500 points" and "the five representative intervention points".

---

# 22. Audit all claims for overstatement after corrected reruns

After the corrected results are available, search the manuscript and repository reports for:

```text
confirm
proves
decisive
decisively
independent
monotonic
substantial
```

Retain these words only if the corrected results and experimental design justify them.

Preferred language:

```text
supports
is consistent with
provides additional explanatory power
is less responsive to
is held fixed by construction
```

Do not describe base-model uncertainty as empirically invariant under Figure 6; it is held fixed by design.

Do not claim complete independence between model uncertainty and width from the reverse intervention, because retraining changes score geometry.

---

# 23. README rewrite

Rewrite the README only after all corrected reruns are complete.

The README must:

1. use the same definition of \(\hat p\), \(w\), \(U_{\rm cal}\), \(v\), and \(E_{\rm model}\) as the manuscript;
2. use the corrected numerical results;
3. call the crowd experiment synthetic/CIFAR-10H-inspired;
4. contain no LLM/agent material;
5. list the actual repository tree;
6. distinguish:
   - fast artifact regeneration;
   - full expensive experiment reproduction;
7. provide exact commands;
8. specify expected runtimes conservatively;
9. state exact package/environment instructions;
10. link each paper figure/table to its producer script and data source.

---

# 24. Required final clean-room test

After all changes:

1. create a completely fresh environment;
2. clone the audit branch into a clean directory;
3. install only from the declared environment file;
4. run:

```bash
python src/generate_paper_artifacts.py
python src/verify_paper_consistency.py
```

using committed data;

5. then, if computational resources allow, run:

```bash
python src/run_all_experiments.py
```

from scratch;

6. regenerate all paper artifacts;
7. run the verifier again.

Record both runs.

There must be no dependence on:

```text
/Users/ivanpetej/...
local untracked files
manually copied CSVs
files outside the repository
```

---

# 25. Final deliverables

Produce all of the following:

```text
results/REPO_PAPER_AUDIT_BEFORE.md
results/REPO_PAPER_AUDIT_AFTER.md
results/RESULT_CHANGES.md
results/paper_results_manifest.json
results/environment.txt
requirements-lock.txt

src/run_all_experiments.py
src/generate_paper_artifacts.py
src/verify_paper_consistency.py
```

and the corrected experiment scripts/data/assets.

## `RESULT_CHANGES.md`

For every numerical change, include:

| Result | Old manuscript value | Old repo value | Corrected value | Reason |
|---|---:|---:|---:|---|

Reasons should be explicit, e.g.:

```text
midpoint/log-loss merge mismatch
true-v versus estimated-v mismatch
last-thinning-replicate bootstrap bug
base-model max_iter inconsistency
stale manually copied CSV
output filename collision
```

## `REPO_PAPER_AUDIT_AFTER.md`

End with a checklist:

```text
[ ] Every empirical paper result has a generating script
[ ] Every script writes to canonical data/paper/results directories
[ ] Every paper figure/table has exactly one producer
[ ] No publication asset filename collisions remain
[ ] p_hat always means arithmetic midpoint
[ ] p_va is named separately wherever used
[ ] W2 uses true theta(1-theta)
[ ] Figure 6 sigma_cal is averaged consistently across thinning replicates
[ ] std_sigma_cal is actually sigma_cal variability
[ ] all HistGradientBoosting refits use one shared specification
[ ] idealised scaling CSV has a committed generator
[ ] baseline calibration-resampling table has a committed generator
[ ] crowd experiment is labelled synthetic
[ ] no active LLM/agent experiment remains
[ ] README matches actual git tree
[ ] no absolute local paths remain
[ ] exact environment is recorded
[ ] manuscript numerical claims equal regenerated outputs
[ ] clean-room verification passes
```

---

# 26. Stop condition

Do not add new scientific experiments beyond those needed to reproduce or correct the existing paper.

Do not change the theory.

Do not add a new benchmark.

Do not optimize results.

Do not selectively preserve favourable old values.

The task is complete only when:

\[
\boxed{
\text{paper}
=
\text{code}
=
\text{committed data}
=
\text{figures/tables}
=
\text{README}
}
\]

for one fully reproducible experimental version.

If a corrected result weakens a claim, update the claim rather than changing the experiment.
