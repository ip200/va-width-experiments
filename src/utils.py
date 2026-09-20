"""
src/utils.py

Shared utility and helper functions for Venn--Abers width scaling experiments.
"""

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
