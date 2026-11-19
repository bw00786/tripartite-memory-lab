"""Experiment runners and parameter sweeps."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tml.metrics import bootstrap_ci, interference, recall_overlap
from tml.models import MatrixModel, Model, NeuronOnlyModel
from tml.patterns import make_patterns, make_signatures

RESULT_COLUMNS = [
    "model",
    "M",
    "cue_frac",
    "noise",
    "alpha",
    "mean",
    "ci_low",
    "ci_high",
    "trials",
    "seed",
]


def _model_registry(parameters: Mapping[str, Any]) -> dict[str, Model]:
    common = {
        "N": int(parameters["N"]),
        "K": int(parameters["K"]),
        "iters": int(parameters["iters"]),
    }
    return {
        "NeuronOnly": NeuronOnlyModel(**common),
        "Matrix": MatrixModel(
            **common,
            D=int(parameters["D"]),
            noise=float(parameters["noise"]),
            alpha=float(parameters["alpha"]),
            seed=int(parameters["seed"]),
        ),
    }


def run_setting(
    parameters: Mapping[str, Any],
    models: Mapping[str, Model] | None = None,
    setting_index: int = 0,
) -> pd.DataFrame:
    """Run one setting and return per-model recall and interference summaries."""
    p = dict(parameters)
    required = (
        "N",
        "K",
        "D",
        "M",
        "cue_frac",
        "noise",
        "alpha",
        "iters",
        "trials",
        "seed",
    )
    missing = [key for key in required if key not in p]
    if missing:
        raise ValueError(f"Missing model parameters: {', '.join(missing)}")
    if int(p["trials"]) < 1:
        raise ValueError("trials must be positive")
    if not 0 < float(p["cue_frac"]) <= 1:
        raise ValueError("cue_frac must be in (0, 1]")
    if int(p["M"]) < 1:
        raise ValueError("M must be positive")

    seed = int(p["seed"])
    data_seed, trials_seed = np.random.SeedSequence([seed, setting_index]).spawn(2)
    data_rng = np.random.default_rng(data_seed)
    trial_rng = np.random.default_rng(trials_seed)
    patterns = make_patterns(int(p["N"]), int(p["K"]), int(p["M"]), data_rng)
    signatures = make_signatures(int(p["M"]), int(p["D"]), data_rng)
    active_cue_count = max(
        1, min(int(p["K"]), round(int(p["K"]) * float(p["cue_frac"])))
    )
    model_set = dict(models) if models is not None else _model_registry(p)
    if not model_set:
        raise ValueError("At least one model is required")
    for model in model_set.values():
        model.store(patterns, signatures)

    overlaps = {name: [] for name in model_set}
    interference_scores = {name: [] for name in model_set}
    for trial in range(int(p["trials"])):
        target_index = int(trial_rng.integers(int(p["M"])))
        cue = np.zeros(int(p["N"]), dtype=np.int8)
        cue_cells = trial_rng.choice(
            np.flatnonzero(patterns[target_index]),
            size=active_cue_count,
            replace=False,
        )
        cue[cue_cells] = 1
        for model_index, (name, model) in enumerate(model_set.items()):
            recall_rng = np.random.default_rng(
                np.random.SeedSequence([seed, setting_index, trial, model_index + 1])
            )
            recalled = model.recall(cue, recall_rng)
            overlaps[name].append(
                recall_overlap(recalled, patterns[target_index], int(p["K"]))
            )
            interference_scores[name].append(
                interference(recalled, patterns, target_index, int(p["K"]))
            )

    bootstrap_samples = int(p.get("bootstrap_samples", 2000))
    rows = []
    for model_index, name in enumerate(model_set):
        metric_rng = np.random.default_rng(
            np.random.SeedSequence([seed, setting_index, model_index, 2])
        )
        overlap_values = np.asarray(overlaps[name])
        low, high = bootstrap_ci(
            overlap_values, metric_rng, n_bootstrap=bootstrap_samples
        )
        interference_values = np.asarray(interference_scores[name])
        interference_low, interference_high = bootstrap_ci(
            interference_values, metric_rng, n_bootstrap=bootstrap_samples
        )
        rows.append(
            {
                "model": name,
                "M": int(p["M"]),
                "cue_frac": float(p["cue_frac"]),
                "noise": float(p["noise"]),
                "alpha": float(p["alpha"]),
                "mean": float(overlap_values.mean()),
                "ci_low": low,
                "ci_high": high,
                "mean_interference": float(interference_values.mean()),
                "interference_ci_low": interference_low,
                "interference_ci_high": interference_high,
                "trials": int(p["trials"]),
                "seed": seed,
                "setting_index": setting_index,
            }
        )
    return pd.DataFrame(rows)


def run_experiment(config: Mapping[str, Any]) -> pd.DataFrame:
    """Run a baseline experiment or one of the configured one-dimensional sweeps."""
    if "model" not in config or "experiment" not in config:
        raise ValueError("Config must contain model and experiment sections")
    base = dict(config["model"])
    experiment = dict(config["experiment"])
    sweep_type = experiment.get("type", "baseline")
    if sweep_type == "baseline":
        return run_setting(base)

    parameter_name = {
        "capacity": "M",
        "noise": "noise",
        "cue": "cue_frac",
        "sparse_dimension": "D",
        "diffusion": "sigma",
    }.get(sweep_type)
    if parameter_name is None:
        raise ValueError(f"Unknown experiment type: {sweep_type}")
    values = experiment.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("Sweep experiment must define a non-empty values list")

    results = []
    for setting_index, value in enumerate(values):
        setting = {**base, parameter_name: value}
        if setting_index:
            setting["seed"] = int(base["seed"])
        frame = run_setting(setting, setting_index=setting_index)
        frame["sweep"] = sweep_type
        frame["sweep_value"] = value
        results.append(frame)
    return pd.concat(results, ignore_index=True)
