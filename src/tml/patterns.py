"""Memory pattern and chemical signature generation."""

import numpy as np
from numpy.random import Generator


def make_patterns(N: int, K: int, M: int, rng: Generator) -> np.ndarray:
    """Create M binary patterns of N neurons, with exactly K active neurons each."""
    if N < 1 or M < 1:
        raise ValueError("N and M must be positive")
    if not 1 <= K <= N:
        raise ValueError("K must be between 1 and N")

    patterns = np.zeros((M, N), dtype=np.int8)
    for memory in range(M):
        active = rng.choice(N, size=K, replace=False)
        patterns[memory, active] = 1
    return patterns


def make_signatures(M: int, D: int, rng: Generator) -> np.ndarray:
    """Create M random chemical signatures with D bipolar components."""
    if M < 1 or D < 1:
        raise ValueError("M and D must be positive")
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(M, D))
