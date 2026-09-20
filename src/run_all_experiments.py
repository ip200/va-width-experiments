"""
src/run_all_experiments.py

Master script to execute all paper-critical experiments end-to-end in order
and regenerate all canonical data files, tables, and figures.

Order of execution:
  1. Calibration-resampling baseline (Table 1)
  2. Pointwise classifier bootstrap instability (Figure 2)
  3. Idealised sample-size scaling laws (Figure 3)
  4. Multivariate W2 exponent convergence (Figure 4, Table 2)
  5. Monotonic score invariance and non-monotonic scaling regimes (Figure 5)
  6. Alternative calibrator comparison & synthetic CIFAR-10H crowd experiment (Figure 8, Table 3)
  7. Real-data calibration support thinning, reverse intervention & nested decomposition (Figure 6, Figure 7, Table 4)
"""

import os
import sys
import time
import subprocess

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
py_exec = sys.executable

EXPERIMENTS = [
    ("1. Calibration Resampling Baseline (Table 1)", "run_calibration_resampling_baseline.py"),
    ("2. Classifier Pointwise Bootstrap Instability (Figure 2)", "run_experiment_classifier_bootstrap_n500.py"),
    ("3. Idealised Scaling Laws (Figure 3)", "run_idealised_scaling.py"),
    ("4. W2 Exponent Convergence (Figure 4, Table 2)", "run_experiment_w2_convergence.py"),
    ("5. Score Invariance & Non-Monotonic Regimes (Figure 5)", "non_monotonic_experiments.py"),
    ("6. Alternative Calibrators & Crowd Experiment (Figure 8, Table 3)", "calibration_comparison.py"),
    ("7. Real-Data Support Thinning & Nested Decomposition (Figures 6 & 7, Table 4)", "real_data_calibration_support.py"),
]


def run_all():
    total_start = time.time()
    print("=" * 70)
    print("RUNNING ALL PAPER EXPERIMENTS END-TO-END")
    print("=" * 70)
    print(f"Python interpreter: {py_exec}")
    print(f"Working directory:  {root_dir}\n")

    for idx, (desc, script_name) in enumerate(EXPERIMENTS, start=1):
        script_path = os.path.join(curr_dir, script_name)
        print("-" * 70)
        print(f"[{idx}/{len(EXPERIMENTS)}] Starting: {desc}")
        print(f"Script: {script_path}")
        print("-" * 70)
        t0 = time.time()
        res = subprocess.run([py_exec, script_path], cwd=root_dir)
        elapsed = time.time() - t0
        if res.returncode != 0:
            print(f"\n[ERROR] Experiment failed with exit code {res.returncode}: {script_name}")
            sys.exit(res.returncode)
        print(f"Completed in {elapsed:.1f}s ({elapsed/60.0:.2f} min)\n")

    # Finally run artifact generator to ensure all assets are synchronized
    print("=" * 70)
    print("Synchronizing all paper artifacts...")
    gen_script = os.path.join(curr_dir, "generate_paper_artifacts.py")
    res = subprocess.run([py_exec, gen_script], cwd=root_dir)
    if res.returncode != 0:
        print(f"\n[ERROR] generate_paper_artifacts failed with exit code {res.returncode}")
        sys.exit(res.returncode)

    total_time = time.time() - total_start
    print("=" * 70)
    print(f"ALL EXPERIMENTS AND ARTIFACTS COMPLETED SUCCESSFULLY in {total_time:.1f}s ({total_time/60.0:.2f} min)!")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
