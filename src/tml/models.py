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


class CueMatchModel(Model):
    """Retrieve the stored pattern with the greatest overlap with the cue."""

    def store(self, P: np.ndarray, S: np.ndarray | None = None) -> None:
        self._validate_patterns(P)
        self.patterns = np.asarray(P, dtype=np.int8).copy()

    def recall(self, cue: np.ndarray, rng: Generator) -> np.ndarray:
        del rng
        state = self._validate_cue(cue)
        if self.patterns is None:
            raise RuntimeError("store must be called before recall")
        cue_matches = self.patterns @ state
        memory_index = int(np.argmax(cue_matches))
        return self.patterns[memory_index].copy()


class MatrixModel(NeuronOnlyModel):
    """Associative-memory baseline augmented with a blurred chemical imprint."""

    def __init__(
        self,
        N: int = 200,
        K: int = 20,
        D: int = 16,
        noise: float = 0.0,
        alpha: float = 1.0,
        iters: int = 3,
        seed: int = 1,
    ) -> None:
        super().__init__(N, K, iters)
        if D < 1 or noise < 0:
            raise ValueError("D must be positive and noise must be non-negative")
        self.D = D
        self.noise = noise
        self.alpha = alpha
        self.seed = seed
        self.imprint: np.ndarray | None = None
        self._noise_rng = np.random.default_rng(seed)

    def _chemical_scores(self, state: np.ndarray) -> np.ndarray:
        if self.imprint is None:
            raise RuntimeError("store must be called before recall")
        chemical_state = self.imprint[state.astype(bool)].sum(axis=0)
        return _normalized(self.imprint @ chemical_state)

    def store(self, P: np.ndarray, S: np.ndarray | None = None) -> None:
        super().store(P)
        if S is None:
            raise ValueError("MatrixModel requires chemical signatures S")
        signatures = np.asarray(S)
        if signatures.shape != (P.shape[0], self.D):
            raise ValueError(f"S must have shape ({P.shape[0]}, {self.D})")
        if not self._valid_signatures(signatures):
            raise ValueError("S contains values unsupported by this model")

        self.imprint = np.asarray(P, dtype=float).T @ signatures
        blur_scale = self.noise * np.sqrt(P.shape[0] * self.K / self.N)
        if blur_scale:
            self.imprint += self._noise_rng.normal(
                0.0, blur_scale, size=self.imprint.shape
            )

    def _valid_signatures(self, signatures: np.ndarray) -> bool:
        return bool(np.all((signatures == -1) | (signatures == 1)))

    def recall(self, cue: np.ndarray, rng: Generator) -> np.ndarray:
        del rng
        state = self._validate_cue(cue)
        if self.weights is None or self.imprint is None:
            raise RuntimeError("store must be called before recall")
        for _ in range(self.iters):
            h_synaptic = _normalized(self.weights @ state)
            h_chemical = self._chemical_scores(state)
            state = _top_k(h_synaptic + self.alpha * h_chemical, self.K)
        return state


class LocalDomainMatrix(MatrixModel):
    """Matrix model with thresholded local chemical binding in neuron domains."""

    def __init__(
        self,
        N: int = 200,
        K: int = 20,
        D: int = 16,
        noise: float = 0.0,
        alpha: float = 1.0,
        iters: int = 3,
        seed: int = 1,
        A: int = 5,
        theta: int = 2,
        no_astrocyte: bool = False,
    ) -> None:
        super().__init__(N, K, D, noise, alpha, iters, seed)
        if not 1 <= A <= self.N or theta < 1:
            raise ValueError("A must be in [1, N] and theta must be positive")
        self.A = A
        self.theta = theta
        self.no_astrocyte = no_astrocyte
        self.domains = np.array_split(np.arange(self.N), A)

    def _chemical_scores(self, state: np.ndarray) -> np.ndarray:
        if self.imprint is None:
            raise RuntimeError("store must be called before recall")
        if self.no_astrocyte:
            return super()._chemical_scores(state)
        scores = np.zeros(self.N, dtype=float)
        for domain in self.domains:
            domain_state = state[domain]
            if np.count_nonzero(domain_state) < self.theta:
                continue
            domain_imprint = self.imprint[domain]
            chemical_state = domain_imprint.T @ domain_state
            scores[domain] = domain_imprint @ chemical_state
        return _normalized(scores)


class NoAstrocyteMatrix(LocalDomainMatrix):
    """Local-domain ablation without astrocyte threshold amplification."""

    def __init__(
        self,
        N: int = 200,
        K: int = 20,
        D: int = 16,
        noise: float = 0.0,
        alpha: float = 1.0,
        iters: int = 3,
        seed: int = 1,
        A: int = 5,
        theta: int = 2,
    ) -> None:
        super().__init__(N, K, D, noise, alpha, iters, seed, A, theta, True)


class DiffusionMatrix(MatrixModel):
    """Matrix model with signatures spread over a Gaussian spatial kernel."""

    def __init__(
        self,
        N: int = 200,
        K: int = 20,
        D: int = 16,
        noise: float = 0.0,
        alpha: float = 1.0,
        iters: int = 3,
        seed: int = 1,
        sigma: float = 1.0,
    ) -> None:
        super().__init__(N, K, D, noise, alpha, iters, seed)
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.sigma = sigma

    def store(self, P: np.ndarray, S: np.ndarray | None = None) -> None:
        super().store(P, S)
        if self.sigma == 0:
            return
        positions = np.column_stack(
            (np.arange(self.N) // int(np.ceil(np.sqrt(self.N))),
             np.arange(self.N) % int(np.ceil(np.sqrt(self.N))))
        )
        differences = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        squared_distances = np.sum(differences**2, axis=2)
        kernel = np.exp(-squared_distances / (2 * self.sigma**2))
        kernel /= kernel.sum(axis=1, keepdims=True)
        if self.imprint is None:
            raise RuntimeError("store must initialize the chemical imprint")
        self.imprint = kernel @ self.imprint


class SparseSignatureMatrix(MatrixModel):
    """Matrix model supporting sparse ternary chemical signatures."""

    def __init__(
        self,
        N: int = 200,
        K: int = 20,
        D: int = 16,
        noise: float = 0.0,
        alpha: float = 1.0,
        iters: int = 3,
        seed: int = 1,
        density: float = 0.25,
    ) -> None:
        super().__init__(N, K, D, noise, alpha, iters, seed)
        if not 0 < density <= 1:
            raise ValueError("density must be in (0, 1]")
        self.density = density

    def _valid_signatures(self, signatures: np.ndarray) -> bool:
        return bool(np.all(np.isin(signatures, (-1, 0, 1))))


class GatedMatrix(MatrixModel):
    """Use chemical recall only when the synaptic top-K margin is weak."""

    def __init__(
        self,
        N: int = 200,
        K: int = 20,
        D: int = 16,
        noise: float = 0.0,
        alpha: float = 1.0,
        iters: int = 3,
        seed: int = 1,
        margin_threshold: float = 0.1,
    ) -> None:
        super().__init__(N, K, D, noise, alpha, iters, seed)
        if margin_threshold < 0:
            raise ValueError("margin_threshold must be non-negative")
        self.margin_threshold = margin_threshold

    def recall(self, cue: np.ndarray, rng: Generator) -> np.ndarray:
        del rng
        state = self._validate_cue(cue)
        if self.weights is None or self.imprint is None:
            raise RuntimeError("store must be called before recall")
        for _ in range(self.iters):
            h_synaptic = _normalized(self.weights @ state)
            ranked = np.sort(h_synaptic)
            margin = (
                float(ranked[-self.K] - ranked[-self.K - 1])
                if self.K < self.N
                else 0
            )
            scores = h_synaptic
            if margin < self.margin_threshold:
                scores = scores + self.alpha * self._chemical_scores(state)
            state = _top_k(scores, self.K)
        return state
