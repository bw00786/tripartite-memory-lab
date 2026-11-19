"""Memory model implementations."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator


class Model(ABC):
    """Interface shared by models that store memories and complete cues."""

    def __init__(self, N: int = 200, K: int = 20, iters: int = 3) -> None:
        if N < 1 or not 1 <= K <= N or iters < 1:
            raise ValueError("Require N >= K >= 1 and iters >= 1")
        self.N = N
        self.K = K
        self.iters = iters
        self.patterns: np.ndarray | None = None

    @abstractmethod
    def store(self, P: np.ndarray, S: np.ndarray | None = None) -> None:
        """Store memory patterns and, when applicable, chemical signatures."""

    @abstractmethod
    def recall(self, cue: np.ndarray, rng: Generator) -> np.ndarray:
        """Complete a cue and return a binary state."""

    def _validate_patterns(self, P: np.ndarray) -> None:
        if P.ndim != 2 or P.shape[1] != self.N:
            raise ValueError(f"P must have shape (M, {self.N})")
        if not np.all((P == 0) | (P == 1)):
            raise ValueError("P must be binary")
        if not np.all(P.sum(axis=1) == self.K):
            raise ValueError(f"Every pattern must contain exactly K={self.K} ones")

    def _validate_cue(self, cue: np.ndarray) -> np.ndarray:
        state = np.asarray(cue, dtype=np.int8)
        if state.shape != (self.N,) or not np.all((state == 0) | (state == 1)):
            raise ValueError(f"cue must be a binary vector of length {self.N}")
        if state.sum() > self.K:
            raise ValueError("cue cannot contain more than K active neurons")
        return state


def _normalized(scores: np.ndarray) -> np.ndarray:
    standard_deviation = float(np.std(scores))
    if standard_deviation <= np.finfo(float).eps:
        return np.zeros_like(scores, dtype=float)
    return scores / standard_deviation


def _top_k(scores: np.ndarray, K: int) -> np.ndarray:
    """Select exactly K cells, resolving equal scores by lower neuron index."""
    order = np.lexsort((np.arange(scores.size), -scores))
    state = np.zeros(scores.size, dtype=np.int8)
    state[order[:K]] = 1
    return state


class NeuronOnlyModel(Model):
    """Associative-memory baseline using only pairwise synaptic weights."""

    def __init__(self, N: int = 200, K: int = 20, iters: int = 3) -> None:
        super().__init__(N, K, iters)
        self.weights: np.ndarray | None = None

    def store(self, P: np.ndarray, S: np.ndarray | None = None) -> None:
        self._validate_patterns(P)
        self.patterns = np.asarray(P, dtype=float)
        self.weights = self.patterns.T @ self.patterns
        np.fill_diagonal(self.weights, 0.0)

    def recall(self, cue: np.ndarray, rng: Generator) -> np.ndarray:
        del rng
        state = self._validate_cue(cue)
        if self.weights is None:
            raise RuntimeError("store must be called before recall")
        for _ in range(self.iters):
            state = _top_k(_normalized(self.weights @ state), self.K)
        return state
