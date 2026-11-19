import numpy as np

from tml.experiment import RESULT_COLUMNS, run_experiment, run_setting
from tml.metrics import bootstrap_ci, interference, recall_overlap


def _small_config() -> dict[str, dict[str, int | float | str]]:
    return {
        "model": {
            "N": 30,
            "K": 5,
            "D": 4,
            "M": 3,
            "cue_frac": 0.4,
            "noise": 0.2,
            "alpha": 1.0,
            "A": 3,
            "theta": 2,
            "sigma": 1.0,
            "density": 0.25,
            "margin_threshold": 0.1,
            "iters": 2,
            "trials": 12,
            "seed": 17,
            "bootstrap_samples": 100,
        },
        "experiment": {"type": "baseline"},
    }


def test_experiment_returns_required_columns_and_is_reproducible() -> None:
    first = run_experiment(_small_config())
    second = run_experiment(_small_config())

    assert set(RESULT_COLUMNS).issubset(first.columns)
    assert first["model"].tolist() == ["NeuronOnly", "CueMatch", "Matrix"]
    np.testing.assert_array_equal(
        first.select_dtypes("number"), second.select_dtypes("number")
    )
    assert (first["ci_low"] <= first["mean"]).all()
    assert (first["ci_high"] >= first["mean"]).all()


def test_capacity_sweep_records_each_setting() -> None:
    config = _small_config()
    config["experiment"] = {"type": "capacity", "values": [2, 4]}

    results = run_experiment(config)

    assert results["M"].tolist() == [2, 2, 2, 4, 4, 4]
    assert results["sweep_value"].tolist() == [2, 2, 2, 4, 4, 4]


def test_noise_sweep_reuses_neuron_patterns_and_trials() -> None:
    config = _small_config()
    config["experiment"] = {"type": "noise", "values": [0.0, 3.0]}

    results = run_experiment(config)
    neuron_scores = results[results["model"] == "NeuronOnly"]["mean"]

    assert neuron_scores.nunique() == 1


def test_requested_variants_appear_in_sweep_results() -> None:
    config = _small_config()
    config["experiment"] = {
        "type": "noise",
        "values": [0.0],
        "models": [
            "NeuronOnly",
            "CueMatch",
            "Matrix",
            "LocalDomain",
            "Diffusion",
            "SparseSignature",
            "Gated",
            "NoAstrocyte",
        ],
    }

    results = run_experiment(config)

    assert set(config["experiment"]["models"]) == set(results["model"])


def test_capacity_sweep_reproduces_prototype_within_tolerance() -> None:
    config = {
        "model": {
            "N": 200,
            "K": 20,
            "D": 16,
            "M": 20,
            "cue_frac": 0.3,
            "noise": 0,
            "alpha": 1.0,
            "A": 5,
            "theta": 2,
            "sigma": 1.0,
            "density": 0.25,
            "margin_threshold": 0.1,
            "iters": 3,
            "trials": 150,
            "bootstrap_samples": 500,
            "seed": 1,
        },
        "experiment": {"type": "capacity", "values": [5, 20, 40]},
    }

    results = run_experiment(config)

    expected = {
        "NeuronOnly": {5: 1.00, 20: 0.99, 40: 0.71},
        "CueMatch": {5: 1.00, 20: 1.00, 40: 1.00},
        "Matrix": {5: 1.00, 20: 0.86, 40: 0.50},
    }
    for row in results.itertuples():
        assert abs(row.mean - expected[row.model][row.M]) <= 0.05


def test_setting_validates_required_parameters() -> None:
    try:
        run_setting({"seed": 1})
    except ValueError as error:
        assert "Missing model parameters" in str(error)
    else:
        raise AssertionError("Missing parameters should fail")


def test_recall_metrics() -> None:
    target = np.array([1, 1, 0, 0])
    recalled = np.array([1, 0, 1, 0])
    patterns = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])
    assert recall_overlap(recalled, target, 2) == 0.5
    assert interference(recalled, patterns, 0, 2) == 1.0
    assert bootstrap_ci(np.array([0.0, 1.0]), np.random.default_rng(1))[0] <= 0.5
