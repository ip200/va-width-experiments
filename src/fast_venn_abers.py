"""
Fast, exact Venn-Abers calculation matching the official venn_abers package.
Implements Algorithms 1-6 from Vovk, Petej & Fedorova (2015).
Groups identical calibration scores and computes exact isotonic GCM bounds.
"""
import os
import sys
import ctypes
import subprocess
import numpy as np
from venn_abers import VennAbers

_c_lib = None
_lib_dir = os.path.dirname(os.path.abspath(__file__))
_c_src = os.path.join(_lib_dir, "c_venn_abers.c")
_c_so = os.path.join(_lib_dir, "c_venn_abers.so")


def _load_c_lib():
    global _c_lib
    if _c_lib is not None:
        return _c_lib

    # Compile if not present or older than source
    if os.path.exists(_c_src):
        compile_needed = not os.path.exists(_c_so)
        if not compile_needed:
            try:
                compile_needed = os.path.getmtime(_c_src) > os.path.getmtime(_c_so)
            except OSError:
                compile_needed = True

        if compile_needed:
            try:
                cmd = ["clang", "-O3", "-shared", "-fPIC", _c_src, "-o", _c_so]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if res.returncode != 0:
                    # Try gcc
                    cmd = ["gcc", "-O3", "-shared", "-fPIC", _c_src, "-o", _c_so]
                    subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except Exception:
                pass

    if os.path.exists(_c_so):
        try:
            lib = ctypes.CDLL(_c_so)
            lib.c_calc_p0p1.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)
            ]
            lib.c_calc_p0p1.restype = None
            _c_lib = lib
            return _c_lib
        except Exception:
            _c_lib = None
    return None


def exact_va_probs(s_cal: np.ndarray, y_cal: np.ndarray, s_test: np.ndarray):
    """
    Computes exact Venn-Abers (p0, p1, p_hat, width) at test scores s_test.
    Respects tie grouping and GCM geometry exactly as in the official package.
    """
    s_test_arr = np.atleast_1d(np.asarray(s_test, dtype=np.float64))
    lib = _load_c_lib()
    
    if lib is not None:
        s_cal_arr = np.asarray(s_cal, dtype=np.float64)
        y_cal_arr = np.asarray(y_cal, dtype=np.float64)
        ix = np.argsort(s_cal_arr)
        k_sort = s_cal_arr[ix]
        k_label_sort = y_cal_arr[ix]
        
        c = np.unique(k_sort)
        ia = np.searchsorted(k_sort, c)
        w = np.zeros(len(c), dtype=np.float64)
        w[:-1] = np.diff(ia)
        w[-1] = len(k_sort) - ia[-1]
        
        k_dash = len(c)
        w_cumsum = np.cumsum(w).astype(np.float64)
        y_cumsum = np.empty(k_dash, dtype=np.float64)
        y_cumsum[:-1] = np.cumsum(k_label_sort)[(ia - 1)[1:]]
        y_cumsum[-1] = np.cumsum(k_label_sort)[-1]
        
        out_p0 = np.empty(k_dash + 1, dtype=np.float64)
        out_p1 = np.empty(k_dash + 1, dtype=np.float64)
        
        lib.c_calc_p0p1(
            k_dash,
            w_cumsum.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_cumsum.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_p0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_p1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        
        idx0 = np.searchsorted(c, s_test_arr, 'right')
        idx1 = np.searchsorted(c, s_test_arr, 'left')
        p0 = np.clip(out_p0[idx0], 0.0, 1.0)
        p1 = np.clip(out_p1[idx1], 0.0, 1.0)
        p_hat = 0.5 * (p0 + p1)
        width = np.maximum(0.0, p1 - p0)
        return p0, p1, p_hat, width

    # Pure Python / official package fallback
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


def exact_va_scalar(s_cal: np.ndarray, y_cal: np.ndarray, s_target: float):
    """Exact scalar evaluation at single score s_target."""
    p0, p1, p_hat, width = exact_va_probs(s_cal, y_cal, np.array([s_target]))
    return float(p0[0]), float(p1[0]), float(p_hat[0]), float(width[0])
