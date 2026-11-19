"""Recall metrics and uncertainty estimates."""

import numpy as np
from numpy.random import Generator


def recall_overlap(recalled: np.ndarray, target: np.ndarray, K: int) -> float:
    """Return the fraction of target cells active in the recalled state."""
    if K < 1:
        raise ValueError("K must be positive")
    if recalled.shape != target.shape:
        raise ValueError("recalled and target must have the same shape")
    return float(np.dot(recalled, target) / K)


def interference(
    recalled: np.ndarray, patterns: np.ndarray, target_index: int, K: int
) -> float:
    """Return the greatest recalled overlap with any non-target pattern."""
    if patterns.ndim != 2 or patterns.shape[1] != recalled.size:
        raise ValueError("patterns must have shape (M, len(recalled))")
    if not 0 <= target_index < patterns.shape[0]:
        raise ValueError("target_index is outside patterns")
    if patterns.shape[0] < 2:
        return 0.0
    overlaps = patterns @ recalled
    overlaps[target_index] = -1
    return float(np.max(overlaps) / K)


def bootstrap_ci(
    values: np.ndarray,
    rng: Generator,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Estimate a percentile confidence interval for the sample mean."""
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap must be positive and confidence in (0, 1)")
    indices = rng.integers(0, samples.size, size=(n_bootstrap, samples.size))
    means = samples[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(low), float(high)
