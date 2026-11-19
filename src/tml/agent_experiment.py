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
    "correction_chain",
    "distributed_clues",
    "heldout_cue_retrieval",
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
    task: str,
    trial: int,
    rng: np.random.Generator,
    distractors: int,
    delay: int,
    taskset_version: int,
    min_correction_updates: int,
    max_correction_updates: int,
    min_distributed_clues: int,
    max_distributed_clues: int,
    heldout_lexicon: list[str],
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
    elif task == "correction_chain":
        if taskset_version < 2:
            raise ValueError("correction_chain requires taskset_version 2 or newer")
        update_count = int(
            rng.integers(min_correction_updates, max_correction_updates + 1)
        )
        now = update_count
        key = "account_status"
        values = [f"revision-{trial}-{index}" for index in range(update_count)]
        records.extend(
            _record(key, value, index, version=index)
            for index, value in enumerate(values)
        )
        stale_values.update(values[:-1])
        query_keys = [key]
        expected = {key: values[-1]}
    elif task == "distributed_clues":
        now = 2
        clue_count = (
            int(rng.integers(min_distributed_clues, max_distributed_clues + 1))
            if taskset_version >= 2
            else 3
        )
        expected = {
            f"clue_{index}": f"piece-{trial}-{index}"
            for index in range(clue_count)
        }
        records.extend(
            _record(key, value, index)
            for index, (key, value) in enumerate(expected.items())
        )
        query_keys = list(expected)
    elif task == "heldout_cue_retrieval":
        if taskset_version < 3:
            raise ValueError(
                "heldout_cue_retrieval requires taskset_version 3 or newer"
            )
        if len(set(heldout_lexicon)) < 8:
            raise ValueError("heldout_lexicon must contain at least 8 unique terms")
        selected = rng.choice(heldout_lexicon, size=4, replace=False).tolist()
        query_terms = selected[:2]
        query = " ".join(query_terms)
        expected_value = f"heldout-answer-{trial}"
        records.append(
            {
                **_record(f"opaque-target-{trial}", expected_value, 0),
                "content": " ".join(selected),
            }
        )
        for index in range(distractors):
            overlap_term = query_terms[index % len(query_terms)]
            non_query_terms = [
                term for term in heldout_lexicon if term not in query_terms
            ]
            distractor_terms = rng.choice(
                non_query_terms, size=3, replace=False
            ).tolist()
            records.append(
                {
                    **_record(
                        f"opaque-distractor-{trial}-{index}",
                        f"distractor-answer-{trial}-{index}",
                        0,
                    ),
                    "content": " ".join([overlap_term, *distractor_terms]),
                }
            )
        now = 1
        query_keys = [query]
        expected = {query: expected_value}
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

    if task != "heldout_cue_retrieval":
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
    task: str,
    setup: str,
    memories: list[list[dict[str, Any]]],
    query_keys: list[str],
    now: int,
    freshness_limit: int,
    confidence_threshold: float,
    access_limit: int,
    communication_budget: int,
    retrieval_policy: str,
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
        if task == "heldout_cue_retrieval" and retrieval_policy == "cue_token_overlap":
            scored: dict[str, list[tuple[int, dict[str, Any]]]] = {}
            for query in query_keys:
                query_terms = set(query.casefold().split())
                scored[query] = []
                for record in group:
                    content_terms = set(
                        str(record.get("content", "")).casefold().split()
                    )
                    score = len(query_terms & content_terms)
                    if score:
                        scored[query].append((score, record))
            matched_by_query: dict[str, list[dict[str, Any]]] = {}
            for query, scored_records in scored.items():
                if not scored_records:
                    matched_by_query[query] = []
                    continue
                best_score = max(score for score, _ in scored_records)
                matched_by_query[query] = [
                    record
                    for score, record in scored_records
                    if score == best_score
                ]
        else:
            matched_by_query = {
                query: [record for record in group if record["key"] == query]
                for query in query_keys
            }
        matched = [
            record
            for query in query_keys
            for record in matched_by_query[query]
        ]
        if setup == "private_memory":
            if not matched:
                continue
            if messages >= communication_budget:
                continue
            messages += 1
        for query in query_keys:
            candidates[query].extend(matched_by_query[query])

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
        if task in {"corrected_fact", "correction_chain"}
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


def run_agent_experiment(
    config: Mapping[str, Any], *, _include_trial_samples: bool = False
) -> pd.DataFrame:
    """Run paired memory-task trials across agent team and memory configurations."""
    model = dict(config.get("model", {}))
    experiment = dict(config.get("experiment", {}))
    intervention = experiment.get("intervention")
    if intervention is not None:
        seed = int(model["seed"])
        bootstrap_samples = int(model["bootstrap_samples"])
        parameter_names = {
            "communication_budget": "communication_budget",
            "shared_access_limit": "shared_access_limit",
            "shared_noise_rate": "shared_noise_rate",
            "freshness_limit": "freshness_limit",
            "confidence_threshold": "confidence_threshold",
            "team_size": "team_sizes",
            "retrieval_policy": "retrieval_policy",
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
            elif intervention == "retrieval_policy":
                setting_model[parameter_names[intervention]] = str(value)
            else:
                setting_model[parameter_names[intervention]] = value
            frame = run_agent_experiment(
                {"model": setting_model, "experiment": setting_experiment},
                _include_trial_samples=intervention == "retrieval_policy",
            )
            frame["sweep"] = intervention
            frame["sweep_value"] = value
            results.append(frame)
        combined = pd.concat(results, ignore_index=True)
        if intervention == "retrieval_policy":
            combined["accuracy_delta_vs_exact_key"] = np.nan
            combined["accuracy_delta_vs_exact_key_ci_low"] = np.nan
            combined["accuracy_delta_vs_exact_key_ci_high"] = np.nan
            combined["exact_retrieval_delta_vs_exact_key"] = np.nan
            combined["exact_retrieval_delta_vs_exact_key_ci_low"] = np.nan
            combined["exact_retrieval_delta_vs_exact_key_ci_high"] = np.nan
            reference = combined[combined["sweep_value"] == "exact_key"]
            for index, row in combined.iterrows():
                matching_reference = reference[
                    (reference["task"] == row["task"])
                    & (reference["setup"] == row["setup"])
                    & (reference["team_size"] == row["team_size"])
                ]
                if matching_reference.empty:
                    continue
                reference_row = matching_reference.iloc[0]
                for metric in ("accuracy", "exact_retrieval"):
                    differences = np.asarray(row[f"__{metric}_samples"]) - np.asarray(
                        reference_row[f"__{metric}_samples"]
                    )
                    combined.at[index, f"{metric}_delta_vs_exact_key"] = float(
                        differences.mean()
                    )
                    delta_rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                seed,
                                int(row["team_size"]),
                                SETUPS.index(row["setup"]),
                                TASKS.index(row["task"]),
                                881 if metric == "accuracy" else 882,
                            ]
                        )
                    )
                    low, high = bootstrap_ci(
                        differences, delta_rng, n_bootstrap=bootstrap_samples
                    )
                    combined.at[
                        index, f"{metric}_delta_vs_exact_key_ci_low"
                    ] = low
                    combined.at[
                        index, f"{metric}_delta_vs_exact_key_ci_high"
                    ] = high
            combined = combined.drop(
                columns=["__accuracy_samples", "__exact_retrieval_samples"]
            )
        return combined

    if model.get("backend") != "deterministic_simulator":
        raise ValueError("agent memory experiments require the deterministic_simulator")
    if model.get("agent_policy") != "highest_version_then_latest_timestamp":
        raise ValueError("Unsupported agent_policy")
    seed = int(model["seed"])
    trials = int(model["trials"])
    bootstrap_samples = int(model["bootstrap_samples"])
    taskset_version = int(model["taskset_version"])
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
    min_correction_updates = int(model["min_correction_updates"])
    max_correction_updates = int(model["max_correction_updates"])
    min_distributed_clues = int(model["min_distributed_clues"])
    max_distributed_clues = int(model["max_distributed_clues"])
    retrieval_policy = str(model.get("retrieval_policy", "exact_key"))
    heldout_lexicon_value = model.get("heldout_lexicon", [])
    if not isinstance(heldout_lexicon_value, list):
        raise ValueError("heldout_lexicon must be a list of single-token strings")
    heldout_lexicon = heldout_lexicon_value
    api_cost_per_call = float(model["api_cost_per_call_usd"])

    if taskset_version not in (1, 2, 3):
        raise ValueError("taskset_version must be 1, 2, or 3")
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
    if (
        min_correction_updates < 2
        or max_correction_updates < min_correction_updates
        or min_distributed_clues < 2
        or max_distributed_clues < min_distributed_clues
    ):
        raise ValueError("task-set difficulty bounds are invalid")
    if "correction_chain" in tasks and taskset_version < 2:
        raise ValueError("correction_chain requires taskset_version 2 or newer")
    if "heldout_cue_retrieval" in tasks and taskset_version < 3:
        raise ValueError("heldout_cue_retrieval requires taskset_version 3 or newer")
    if retrieval_policy not in {"exact_key", "cue_token_overlap"}:
        raise ValueError(f"Unknown retrieval_policy: {retrieval_policy}")
    if "heldout_cue_retrieval" in tasks and len(set(heldout_lexicon)) < 8:
        raise ValueError("heldout_lexicon must contain at least 8 unique terms")
    if any(
        not isinstance(term, str) or not term or len(term.split()) != 1
        for term in heldout_lexicon
    ):
        raise ValueError("heldout_lexicon terms must be non-empty single-token strings")

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
                        taskset_version,
                        min_correction_updates,
                        max_correction_updates,
                        min_distributed_clues,
                        max_distributed_clues,
                        heldout_lexicon,
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
                        task,
                        setup,
                        memories,
                        query_keys,
                        now,
                        freshness_limit,
                        confidence_threshold,
                        access_limit,
                        communication_budget,
                        retrieval_policy,
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
        baseline = raw.get((task, "single_agent", team_size))
        row: dict[str, Any] = {
            "task": task,
            "setup": setup,
            "team_size": team_size,
            "backend": model["backend"],
            "agent_policy": model["agent_policy"],
            "retrieval_policy": retrieval_policy,
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
            if _include_trial_samples and metric in {
                "accuracy",
                "exact_retrieval",
            }:
                row[f"__{metric}_samples"] = samples.tolist()
        for metric in ("accuracy", "exact_retrieval"):
            delta_name = f"{metric}_delta_vs_single_agent"
            if baseline is None:
                row[delta_name] = np.nan
                row[f"{delta_name}_ci_low"] = np.nan
                row[f"{delta_name}_ci_high"] = np.nan
                continue
            differences = np.asarray(metrics[metric]) - np.asarray(baseline[metric])
            row[delta_name] = float(differences.mean())
            delta_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [seed, team_size, setups.index(setup), tasks.index(task), 991]
                )
            )
            low, high = bootstrap_ci(
                differences, delta_rng, n_bootstrap=bootstrap_samples
            )
            row[f"{delta_name}_ci_low"] = low
            row[f"{delta_name}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)
