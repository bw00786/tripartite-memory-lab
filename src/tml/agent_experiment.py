"""Seeded, model-free benchmark for agent memory and coordination strategies."""

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from tml.metrics import bootstrap_ci

TASKS = (
    "delayed_retrieval",
    "corrected_fact",
    "distributed_clues",
    "uncertainty_and_staleness",
)
SETUPS = (
    "single_agent",
    "private_memory",
    "shared_memory",
    "shared_memory_noisy",
    "shared_memory_limited",
)
RECORD_COLUMNS = (
    "key",
    "value",
    "version",
    "timestamp",
    "confidence",
)


def _record(
    key: str,
    value: str,
    timestamp: int,
    version: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    return {
        "key": key,
        "value": value,
        "version": version,
        "timestamp": timestamp,
        "confidence": confidence,
    }


def _make_scenario(
    task: str, trial: int, rng: np.random.Generator, distractors: int, delay: int
) -> tuple[list[dict[str, Any]], list[str], dict[str, str | None], int, set[str]]:
    records: list[dict[str, Any]] = []
    now = delay + 2
    stale_values: set[str] = set()

    if task == "delayed_retrieval":
        now = delay
        answer = f"delayed-answer-{trial}"
        records.append(_record("delayed_target", answer, 0))
        query_keys = ["delayed_target"]
        expected = {"delayed_target": answer}
    elif task == "corrected_fact":
        now = 2
        old_value = f"original-{trial}"
        new_value = f"corrected-{trial}"
        records.extend(
            [
                _record("account_status", old_value, 0, version=0),
                _record("account_status", new_value, 1, version=1),
            ]
        )
        stale_values.add(old_value)
        query_keys = ["account_status"]
        expected = {"account_status": new_value}
    elif task == "distributed_clues":
        now = 2
        expected = {
            f"clue_{index}": f"piece-{trial}-{index}" for index in range(3)
        }
        records.extend(
            _record(key, value, index)
            for index, (key, value) in enumerate(expected.items())
        )
        query_keys = list(expected)
    elif task == "uncertainty_and_staleness":
        records.extend(
            [
                _record("expired_fact", f"expired-{trial}", 0),
                _record("uncertain_fact", f"uncertain-{trial}", now, confidence=0.45),
            ]
        )
        query_keys = ["expired_fact", "uncertain_fact"]
        expected = {key: None for key in query_keys}
    else:
        raise ValueError(f"Unknown agent task: {task}")

    for index in range(distractors):
        key = f"distractor_{index}"
        records.append(
            _record(
                key,
                f"distractor-value-{trial}-{index}",
                int(rng.integers(1, max(now, 2))),
                confidence=float(rng.uniform(0.7, 1.0)),
            )
        )
    rng.shuffle(records)
    return records, query_keys, expected, now, stale_values


def _store_records(
    setup: str,
    team_size: int,
    records: list[dict[str, Any]],
    noise_rate: float,
    rng: np.random.Generator,
) -> list[list[dict[str, Any]]]:
    agent_count = team_size if setup == "private_memory" else 1
    memories = [[] for _ in range(agent_count)]
    for index, record in enumerate(records):
        stored = dict(record)
        if setup == "shared_memory_noisy" and rng.random() < noise_rate:
            stored["value"] = f"corrupted-{record['value']}"
        memories[index % agent_count].append(stored)
    return memories


def _answer(
    setup: str,
    memories: list[list[dict[str, Any]]],
    query_keys: list[str],
    now: int,
    freshness_limit: int,
    confidence_threshold: float,
    access_limit: int,
    communication_budget: int,
) -> tuple[dict[str, str | None], dict[str, float], int, int]:
    if setup == "shared_memory_limited":
        shared_records = memories[0]
        readable = sorted(
            enumerate(shared_records),
            key=lambda item: (item[1]["timestamp"], item[0]),
            reverse=True,
        )[:access_limit]
        visible = [record for _, record in readable]
        scanned = len(readable)
        candidate_groups = [visible]
    elif setup == "private_memory":
        candidate_groups = memories
        scanned = sum(len(memory) for memory in memories)
    else:
        candidate_groups = [record for memory in memories for record in memory]
        scanned = len(candidate_groups)
        candidate_groups = [candidate_groups]

    candidates: dict[str, list[dict[str, Any]]] = {key: [] for key in query_keys}
    messages = 0
    for group in candidate_groups:
        matched = [
            record for record in group if record["key"] in candidates
        ]
        if setup == "private_memory":
            if not matched:
                continue
            if messages >= communication_budget:
                continue
            messages += 1
        for record in matched:
            candidates[record["key"]].append(record)

    answers: dict[str, str | None] = {}
    confidences: dict[str, float] = {}
    for key in query_keys:
        matching = candidates[key]
        if not matching:
            answers[key] = None
            confidences[key] = 0.5
            continue
        newest = max(
            matching,
            key=lambda item: (
                item["version"],
                item["timestamp"],
                item["confidence"],
            ),
        )
        is_expired = now - newest["timestamp"] > freshness_limit
        is_uncertain = newest["confidence"] < confidence_threshold
        if is_expired or is_uncertain:
            answers[key] = None
            confidences[key] = 0.95
        else:
            answers[key] = str(newest["value"])
            confidences[key] = float(newest["confidence"])
    return answers, confidences, messages, scanned


def _trial_metrics(
    task: str,
    answers: Mapping[str, str | None],
    confidences: Mapping[str, float],
    expected: Mapping[str, str | None],
    stale_values: set[str],
) -> dict[str, float]:
    keys = list(expected)
    matches = [
        answers.get(key) == expected[key]
        for key in keys
    ]
    exact = float(all(matches))
    stale_errors = sum(answers.get(key) in stale_values for key in keys)
    conflict_errors = (
        sum(answers.get(key) != expected[key] for key in keys)
        if task == "corrected_fact"
        else 0
    )
    stale_task_errors = float(
        task == "uncertainty_and_staleness"
        and answers.get("expired_fact") is not None
    )
    uncertainty_errors = float(
        task == "uncertainty_and_staleness"
        and answers.get("uncertain_fact") is not None
    )
    brier_scores = [
        (float(confidences.get(key, 0.5)) - float(correct)) ** 2
        for key, correct in zip(keys, matches, strict=True)
    ]
    return {
        "accuracy": float(np.mean(matches)),
        "exact_retrieval": exact,
        "stale_error_rate": float((stale_errors + stale_task_errors) / len(keys)),
        "conflict_error_rate": float(conflict_errors / len(keys)),
        "uncertainty_error_rate": float(uncertainty_errors / len(keys)),
        "uncertainty_brier": float(np.mean(brier_scores)),
    }


def run_agent_experiment(config: Mapping[str, Any]) -> pd.DataFrame:
    """Run paired memory-task trials across agent team and memory configurations."""
    model = dict(config.get("model", {}))
    experiment = dict(config.get("experiment", {}))
    intervention = experiment.get("intervention")
    if intervention is not None:
        parameter_names = {
            "communication_budget": "communication_budget",
            "shared_access_limit": "shared_access_limit",
            "shared_noise_rate": "shared_noise_rate",
            "freshness_limit": "freshness_limit",
            "confidence_threshold": "confidence_threshold",
            "team_size": "team_sizes",
        }
        if intervention not in parameter_names:
            raise ValueError(f"Unknown agent intervention: {intervention}")
        values = experiment.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError("Agent intervention must define a non-empty values list")
        results = []
        for value in values:
            setting_model = dict(model)
            setting_experiment = dict(experiment)
            setting_experiment.pop("intervention")
            setting_experiment.pop("values")
            if intervention == "team_size":
                setting_experiment["team_sizes"] = [int(value)]
            else:
                setting_model[parameter_names[intervention]] = value
            frame = run_agent_experiment(
                {"model": setting_model, "experiment": setting_experiment}
            )
            frame["sweep"] = intervention
            frame["sweep_value"] = value
            results.append(frame)
        return pd.concat(results, ignore_index=True)

    if model.get("backend") != "deterministic_simulator":
        raise ValueError("agent memory experiments require the deterministic_simulator")
    if model.get("agent_policy") != "highest_version_then_latest_timestamp":
        raise ValueError("Unsupported agent_policy")
    seed = int(model["seed"])
    trials = int(model["trials"])
    bootstrap_samples = int(model["bootstrap_samples"])
    team_sizes = [int(value) for value in experiment["team_sizes"]]
    tasks = list(experiment.get("tasks", TASKS))
    setups = list(experiment.get("setups", SETUPS))
    noise_rate = float(model["shared_noise_rate"])
    freshness_limit = int(model["freshness_limit"])
    confidence_threshold = float(model["confidence_threshold"])
    access_limit = int(model["shared_access_limit"])
    communication_budget = int(model["communication_budget"])
    distractors = int(model["distractors"])
    delay = int(model["delay_turns"])
    api_cost_per_call = float(model["api_cost_per_call_usd"])

    if trials < 1 or bootstrap_samples < 1:
        raise ValueError("trials and bootstrap_samples must be positive")
    if not team_sizes or any(size < 1 for size in team_sizes):
        raise ValueError("team_sizes must contain positive integers")
    if set(tasks) - set(TASKS):
        raise ValueError(f"Unknown tasks: {', '.join(sorted(set(tasks) - set(TASKS)))}")
    if set(setups) - set(SETUPS):
        raise ValueError(
            f"Unknown setups: {', '.join(sorted(set(setups) - set(SETUPS)))}"
        )
    if not 0 <= noise_rate <= 1:
        raise ValueError("shared_noise_rate must be in [0, 1]")
    if freshness_limit < 0 or access_limit < 1 or communication_budget < 0:
        raise ValueError("freshness/access limits and communication budget are invalid")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if api_cost_per_call < 0:
        raise ValueError("api_cost_per_call_usd must be non-negative")
    if distractors < 0 or delay < 1:
        raise ValueError("distractors must be non-negative and delay_turns positive")

    raw: dict[tuple[str, str, int], dict[str, list[float]]] = {}
    for task in tasks:
        for team_size in team_sizes:
            for setup_index, setup in enumerate(setups):
                group_key = (task, setup, team_size)
                raw[group_key] = {
                    metric: []
                    for metric in (
                        "accuracy",
                        "exact_retrieval",
                        "stale_error_rate",
                        "conflict_error_rate",
                        "uncertainty_error_rate",
                        "uncertainty_brier",
                        "latency_ms",
                        "compute_units",
                        "agent_messages",
                    )
                }
                for trial in range(trials):
                    scenario_seed = np.random.SeedSequence([seed, trial])
                    records, query_keys, expected, now, stale_values = _make_scenario(
                        task,
                        trial,
                        np.random.default_rng(scenario_seed),
                        distractors,
                        delay,
                    )
                    timer = perf_counter()
                    memories = _store_records(
                        setup,
                        team_size,
                        records,
                        noise_rate,
                        np.random.default_rng(
                            np.random.SeedSequence(
                                [seed, trial, setup_index, team_size]
                            )
                        ),
                    )
                    answers, confidences, messages, scanned = _answer(
                        setup,
                        memories,
                        query_keys,
                        now,
                        freshness_limit,
                        confidence_threshold,
                        access_limit,
                        communication_budget,
                    )
                    elapsed_ms = (perf_counter() - timer) * 1000
                    metrics = _trial_metrics(
                        task, answers, confidences, expected, stale_values
                    )
                    for name, value in metrics.items():
                        raw[group_key][name].append(value)
                    raw[group_key]["latency_ms"].append(elapsed_ms)
                    raw[group_key]["compute_units"].append(
                        float(len(records) + scanned + messages)
                    )
                    raw[group_key]["agent_messages"].append(float(messages))
    rows = []
    for task, setup, team_size in raw:
        metrics = raw[(task, setup, team_size)]
        row: dict[str, Any] = {
            "task": task,
            "setup": setup,
            "team_size": team_size,
            "backend": model["backend"],
            "agent_policy": model["agent_policy"],
            "taskset_version": int(model["taskset_version"]),
            "trials": trials,
            "seed": seed,
            "shared_noise_rate": noise_rate,
            "communication_budget": communication_budget,
            "shared_access_limit": access_limit,
            "freshness_limit": freshness_limit,
            "external_model_calls": 0,
            "api_cost_per_call_usd": api_cost_per_call,
            "estimated_api_cost_usd": 0.0 * api_cost_per_call,
        }
        for metric, values in metrics.items():
            samples = np.asarray(values, dtype=float)
            row[f"mean_{metric}"] = float(samples.mean())
            rng = np.random.default_rng(
                np.random.SeedSequence(
                [
                    seed,
                    team_size,
                    setups.index(setup),
                    tasks.index(task),
                    len(metric),
                ]
                )
            )
            low, high = bootstrap_ci(
                samples, rng, n_bootstrap=bootstrap_samples
            )
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)
