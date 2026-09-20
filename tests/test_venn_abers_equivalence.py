"""
Unit tests establishing exact equivalence between exact_va_probs / exact_va_scalar
and the official open-source venn_abers package across continuous, tied,
discrete, and bootstrap scenarios.
"""
import os
import sys
import pytest
import numpy as np
from venn_abers import VennAbers

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from fast_venn_abers import exact_va_probs, exact_va_scalar


def _eval_official_va(s_cal, y_cal, s_test):
    s_test_arr = np.atleast_1d(np.asarray(s_test, dtype=np.float64))
    p_cal = np.column_stack([1.0 - s_cal, s_cal])
    p_test = np.column_stack([1.0 - s_test_arr, s_test_arr])
    va = VennAbers()
    va.fit(p_cal, y_cal)
    _, pred = va.predict_proba(p_test)
    p0 = np.clip(pred[:, 0], 0.0, 1.0)
    p1 = np.clip(pred[:, 1], 0.0, 1.0)
    p_hat = 0.5 * (p0 + p1)
    width = np.maximum(0.0, p1 - p0)
    return p0, p1, p_hat, width


def test_continuous_random_scores():
    """Tests agreement across random continuous calibration scores."""
    rng = np.random.RandomState(101)
    for rep in range(25):
        n = rng.randint(50, 400)
        s_cal = rng.uniform(0.0, 1.0, size=n)
        y_cal = rng.binomial(1, s_cal)
        s_test = rng.uniform(0.0, 1.0, size=15)
        
        p0_exp, p1_exp, pm_exp, w_exp = exact_va_probs(s_cal, y_cal, s_test)
        p0_act, p1_act, pm_act, w_act = _eval_official_va(s_cal, y_cal, s_test)
        
        np.testing.assert_allclose(p0_exp, p0_act, atol=1e-12, err_msg=f"p0 mismatch at rep {rep}")
        np.testing.assert_allclose(p1_exp, p1_act, atol=1e-12, err_msg=f"p1 mismatch at rep {rep}")
        np.testing.assert_allclose(w_exp, w_act, atol=1e-12, err_msg=f"width mismatch at rep {rep}")


def test_discrete_scores_with_frequent_ties():
    """Tests agreement when scores have many exact ties."""
    rng = np.random.RandomState(202)
    grid = np.linspace(0.05, 0.95, 12)
    for rep in range(25):
        n = 500
        s_cal = rng.choice(grid, size=n)
        y_cal = rng.binomial(1, s_cal)
        s_test = np.linspace(0.0, 1.0, 21)
        
        p0_exp, p1_exp, pm_exp, w_exp = exact_va_probs(s_cal, y_cal, s_test)
        p0_act, p1_act, pm_act, w_act = _eval_official_va(s_cal, y_cal, s_test)
        
        np.testing.assert_allclose(p0_exp, p0_act, atol=1e-12, err_msg=f"ties p0 mismatch at rep {rep}")
        np.testing.assert_allclose(p1_exp, p1_act, atol=1e-12, err_msg=f"ties p1 mismatch at rep {rep}")
        np.testing.assert_allclose(w_exp, w_act, atol=1e-12, err_msg=f"ties width mismatch at rep {rep}")


def test_bootstrap_resamples_with_duplicates():
    """Tests agreement on bootstrap resamples (which duplicate scores by construction)."""
    rng = np.random.RandomState(303)
    s_orig = rng.uniform(0.1, 0.9, size=300)
    y_orig = rng.binomial(1, s_orig)
    n = len(s_orig)
    
    for b in range(50):
        b_idx = rng.choice(n, size=n, replace=True)
        s_boot = s_orig[b_idx]
        y_boot = y_orig[b_idx]
        s_targets = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
        
        p0_exp, p1_exp, pm_exp, w_exp = exact_va_probs(s_boot, y_boot, s_targets)
        p0_act, p1_act, pm_act, w_act = _eval_official_va(s_boot, y_boot, s_targets)
        
        np.testing.assert_allclose(p0_exp, p0_act, atol=1e-12, err_msg=f"boot p0 mismatch at b={b}")
        np.testing.assert_allclose(p1_exp, p1_act, atol=1e-12, err_msg=f"boot p1 mismatch at b={b}")
        np.testing.assert_allclose(w_exp, w_act, atol=1e-12, err_msg=f"boot width mismatch at b={b}")


def test_boundary_and_scalar_equivalence():
    """Tests scalar helper and boundary values (0.0 and 1.0)."""
    rng = np.random.RandomState(404)
    s_cal = np.clip(rng.normal(0.5, 0.2, size=200), 0.0, 1.0)
    y_cal = rng.binomial(1, s_cal)
    
    for s_pt in [0.0, 0.05, 0.5, 0.95, 1.0]:
        p0_sc, p1_sc, pm_sc, w_sc = exact_va_scalar(s_cal, y_cal, s_pt)
        p0_exp, p1_exp, pm_exp, w_exp = _eval_official_va(s_cal, y_cal, [s_pt])
        
        assert abs(p0_sc - p0_exp[0]) < 1e-12
        assert abs(p1_sc - p1_exp[0]) < 1e-12
        assert abs(w_sc - w_exp[0]) < 1e-12
