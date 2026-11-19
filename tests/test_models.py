import numpy as np
import pytest

from tml.models import NeuronOnlyModel
from tml.patterns import make_patterns


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
