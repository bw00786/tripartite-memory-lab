import numpy as np
import pytest

from tml.models import (
    DiffusionMatrix,
    GatedMatrix,
    LocalDomainMatrix,
    MatrixModel,
    NeuronOnlyModel,
    NoAstrocyteMatrix,
    SparseSignatureMatrix,
)
from tml.patterns import make_patterns, make_signatures, make_sparse_signatures


def test_neuron_only_recall_meets_low_load_acceptance() -> None:
    rng = np.random.default_rng(1)
    patterns = make_patterns(200, 20, 5, rng)
    model = NeuronOnlyModel(N=200, K=20, iters=3)
    model.store(patterns)

    scores = []
    for target in range(5):
        for _ in range(30):
            cue = np.zeros(200, dtype=np.int8)
            cue_cells = rng.choice(
                np.flatnonzero(patterns[target]), size=6, replace=False
            )
            cue[cue_cells] = 1
            recalled = model.recall(cue, rng)
            scores.append(np.dot(recalled, patterns[target]) / 20)

    assert float(np.mean(scores)) >= 0.95


def test_neuron_only_checks_input_and_storage() -> None:
    model = NeuronOnlyModel(N=10, K=3)
    with pytest.raises(RuntimeError, match="store must be called"):
        model.recall(np.zeros(10), np.random.default_rng(0))
    with pytest.raises(ValueError, match="exactly K"):
        model.store(np.zeros((2, 10)))


def test_matrix_model_is_deterministic_and_uses_expected_shapes() -> None:
    data_rng = np.random.default_rng(42)
    patterns = make_patterns(40, 5, 6, data_rng)
    signatures = make_signatures(6, 8, data_rng)
    first = MatrixModel(N=40, K=5, D=8, noise=0.5, seed=13)
    second = MatrixModel(N=40, K=5, D=8, noise=0.5, seed=13)
    first.store(patterns, signatures)
    second.store(patterns, signatures)

    assert first.imprint is not None
    assert first.imprint.shape == (40, 8)
    np.testing.assert_array_equal(first.imprint, second.imprint)

    cue = np.zeros(40, dtype=np.int8)
    cue[np.flatnonzero(patterns[0])[:2]] = 1
    np.testing.assert_array_equal(
        first.recall(cue, np.random.default_rng(5)),
        second.recall(cue, np.random.default_rng(6)),
    )


def test_matrix_model_requires_matching_signatures() -> None:
    model = MatrixModel(N=10, K=3, D=4)
    patterns = make_patterns(10, 3, 2, np.random.default_rng(0))
    with pytest.raises(ValueError, match="shape"):
        model.store(patterns, np.ones((2, 3)))


@pytest.mark.parametrize(
    ("model_type", "variant_options"),
    [
        (LocalDomainMatrix, {"A": 4, "theta": 2}),
        (DiffusionMatrix, {"sigma": 1.5}),
        (SparseSignatureMatrix, {"density": 0.25}),
        (GatedMatrix, {"margin_threshold": 0.1}),
        (NoAstrocyteMatrix, {"A": 4, "theta": 2}),
    ],
)
def test_variant_models_return_deterministic_k_active_states(
    model_type, variant_options
) -> None:
    data_rng = np.random.default_rng(42)
    patterns = make_patterns(40, 5, 6, data_rng)
    signatures = make_signatures(6, 8, data_rng)
    if model_type is SparseSignatureMatrix:
        signatures = make_sparse_signatures(6, 8, data_rng, density=0.25)
    options = {
        "N": 40,
        "K": 5,
        "D": 8,
        "seed": 12,
        **variant_options,
    }
    first = model_type(**options)
    second = model_type(**options)
    first.store(patterns, signatures)
    second.store(patterns, signatures)
    cue = np.zeros(40, dtype=np.int8)
    cue[np.flatnonzero(patterns[0])[:2]] = 1

    recalled = first.recall(cue, np.random.default_rng(1))

    assert recalled.shape == (40,)
    assert recalled.sum() == 5
    np.testing.assert_array_equal(
        recalled, second.recall(cue, np.random.default_rng(2))
    )
