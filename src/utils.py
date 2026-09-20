"""
src/utils.py

Shared utility and helper functions for Venn--Abers width scaling experiments.
"""

from typing import Tuple, Any, Dict
import numpy as np


def set_seed(seed: int = 42) -> np.random.Generator:
    """Initializes and returns a NumPy random generator with a fixed seed."""
    return np.random.default_rng(seed)


def clip01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Clips array to (eps, 1 - eps) to prevent numerical degeneration."""
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Standard logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def true_probability_1d(x: np.ndarray) -> np.ndarray:
    """Sigmoidal true probability mapping for 1D synthetic experiments."""
    return 0.1 + 0.8 * sigmoid(6.0 * x)


def extract_va_outputs(pred_output: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses output from VennAbersCalibrator.predict_proba(..., p0_p1_output=True).

    Returns:
        p0: np.ndarray, lower Venn--Abers bound
        p1: np.ndarray, upper Venn--Abers bound
        p_mid: np.ndarray, arithmetic midpoint (p0 + p1) / 2
        p_va: np.ndarray, standard log-loss minimax merge p1 / (1 - p0 + p1)
        width: np.ndarray, p1 - p0
    """
    if isinstance(pred_output, tuple) and len(pred_output) == 2:
        a, b = np.asarray(pred_output[0]), np.asarray(pred_output[1])
        if b.ndim == 3 and b.shape[0] == 1:
            p0 = b[0, :, 0].astype(float)
            p1 = b[0, :, 1].astype(float)
        elif b.ndim == 2:
            p0 = b[:, 0].astype(float)
            p1 = b[:, 1].astype(float)
        else:
            raise ValueError(f"Unexpected b shape: {b.shape}")

        if a.ndim == 2 and a.shape[1] >= 2:
            p_va = a[:, 1].astype(float)
        else:
            p_va = a.astype(float).reshape(-1)
    else:
        raise ValueError(f"Unexpected pred_output format: {type(pred_output)}")

    p0_clean = clip01(np.minimum(p0, p1))
    p1_clean = clip01(np.maximum(p0, p1))
    p_mid = clip01(0.5 * (p0_clean + p1_clean))

    # Analytical fallback for p_va if needed
    denom = 1.0 - p0_clean + p1_clean
    denom = np.where(denom <= 0, 1e-12, denom)
    p_va_calc = clip01(p1_clean / denom)
    p_va_clean = clip01(p_va if p_va is not None else p_va_calc)

    width = p1_clean - p0_clean
    return p0_clean, p1_clean, p_mid, p_va_clean, width


def make_base_model(seed: int = 42, **kwargs) -> Any:
    """
    Unified constructor for HistGradientBoostingClassifier across the repository.
    Ensures identical hyperparameters (max_depth=4, learning_rate=0.05, max_iter=200)
    for full-fit base model, bootstrap refits (E_model), and reverse interventions.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    params: Dict[str, Any] = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "max_iter": 200,
        "random_state": seed,
    }
    params.update(kwargs)
    return HistGradientBoostingClassifier(**params)
