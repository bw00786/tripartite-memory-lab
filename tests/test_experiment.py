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
    assert first["model"].tolist() == ["NeuronOnly", "Matrix"]
    np.testing.assert_array_equal(
        first.select_dtypes("number"), second.select_dtypes("number")
    )
    assert (first["ci_low"] <= first["mean"]).all()
    assert (first["ci_high"] >= first["mean"]).all()


def test_capacity_sweep_records_each_setting() -> None:
    config = _small_config()
    config["experiment"] = {"type": "capacity", "values": [2, 4]}

    results = run_experiment(config)

    assert results["M"].tolist() == [2, 2, 4, 4]
    assert results["sweep_value"].tolist() == [2, 2, 4, 4]


def test_default_load_reproduces_matrix_prototype_within_tolerance() -> None:
    config = {
        "model": {
            "N": 200,
            "K": 20,
            "D": 16,
            "M": 20,
            "cue_frac": 0.3,
            "noise": 0,
            "alpha": 1.0,
            "iters": 3,
            "trials": 150,
            "bootstrap_samples": 500,
            "seed": 1,
        },
        "experiment": {"type": "baseline"},
    }

    results = run_experiment(config).set_index("model")

    assert abs(results.loc["NeuronOnly", "mean"] - 0.99) <= 0.05
    assert abs(results.loc["Matrix", "mean"] - 0.86) <= 0.05


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
