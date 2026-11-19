"""Local Ollama benchmark for agent memory and coordination strategies."""

import json
import time
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from tml.agent_experiment import (
    SETUPS,
    TASKS,
    _make_scenario,
    _store_records,
    _trial_metrics,
)
from tml.metrics import bootstrap_ci


class ChatClient(Protocol):
    model_name: str
    model_digest: str

    def chat(self, system: str, user: str) -> dict[str, Any]: ...


class OllamaChatClient:
    """Small standard-library client for Ollama's local chat API."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        seed: int,
        num_ctx: int,
        num_predict: int,
        timeout_seconds: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.seed = seed
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.timeout_seconds = timeout_seconds
        self.model_digest = self._find_model_digest()

    def _find_model_digest(self) -> str:
        request = Request(f"{self.base_url}/api/tags")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Unable to query local Ollama models: {error}"
            ) from error
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ValueError("Ollama tags response must contain a models list")
        for model in payload["models"]:
            if model.get("name") == self.model_name:
                return str(model.get("digest", "unknown"))
        available = ", ".join(
            str(model.get("name", "")) for model in payload.get("models", [])
        )
        raise ValueError(
            f"Ollama model {self.model_name!r} is not installed; available: {available}"
        )

    def chat(self, system: str, user: str) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "seed": self.seed,
                "temperature": 0,
                "top_p": 1,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Local Ollama chat request failed: {error}") from error
        if not isinstance(result, dict):
            raise ValueError("Ollama chat response must be a JSON object")
        if result.get("error"):
            raise RuntimeError(f"Local Ollama returned an error: {result['error']}")
        message = result.get("message")
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), str
        ):
            raise ValueError("Ollama chat response is missing message.content")
        return result


def _system_prompt(freshness_limit: int, confidence_threshold: float) -> str:
    return (
        "You are a memory-retrieval agent. Use only the supplied memory records. "
        "For each query, return the value supported by the records, or null if "
        "there is no supported answer. Prefer the greatest version, then the "
        "latest timestamp. Abstain when record age exceeds the freshness limit "
        "or record confidence is below the confidence threshold. For lexical "
        "cues, use the record content that best matches the query. Return only "
        'JSON shaped as {"answers":{"query":"value or null"},'
        '"confidence":{"query":0.0}}. Confidence must be numeric from 0 to 1. '
        "Use each exact query key as its corresponding JSON object key. "
        f"Current freshness limit: {freshness_limit}; confidence threshold: "
        f"{confidence_threshold}."
    )


def _query_text(task: str, key: str) -> str:
    if task == "heldout_cue_retrieval":
        return (
            "Which stored value is associated with content matching these terms: "
            f"{key}"
        )
    if task in {"corrected_fact", "correction_chain"}:
        return f"What is the latest value for {key}?"
    if task == "uncertainty_and_staleness":
        return f"Should I retrieve {key}, or abstain if it is stale or uncertain?"
    return f"What value is stored for {key}?"


def _parse_answer(
    content: str, query_keys: list[str]
) -> tuple[dict[str, str | None], dict[str, float], int]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return (
            {key: None for key in query_keys},
            {key: 0.5 for key in query_keys},
            1,
        )
    if not isinstance(payload, dict):
        return (
            {key: None for key in query_keys},
            {key: 0.5 for key in query_keys},
            1,
        )
    answer_payload = payload.get("answers", {})
    confidence_payload = payload.get("confidence", {})
    if not isinstance(answer_payload, dict) or not isinstance(confidence_payload, dict):
        return (
            {key: None for key in query_keys},
            {key: 0.5 for key in query_keys},
            1,
        )
    answers: dict[str, str | None] = {}
    confidences: dict[str, float] = {}
    failures = 0
    for key in query_keys:
        if key not in answer_payload:
            failures += 1
        if key not in confidence_payload:
            failures += 1
        answer = answer_payload.get(key)
        if answer is not None and not isinstance(answer, str):
            answer = None
            failures += 1
        confidence = confidence_payload.get(key, 0.5)
        if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
            confidence = 0.5
            failures += 1
        answers[key] = answer
        confidences[key] = float(confidence)
    return answers, confidences, failures


def _chat_for_memory(
    client: OllamaChatClient,
    task: str,
    query_keys: list[str],
    records: list[dict[str, Any]],
    now: int,
    freshness_limit: int,
    confidence_threshold: float,
) -> tuple[dict[str, str | None], dict[str, float], dict[str, int], int]:
    user_payload = {
        "queries": {key: _query_text(task, key) for key in query_keys},
        "current_time": now,
        "records": records,
    }
    started = time.perf_counter()
    result = client.chat(
        _system_prompt(freshness_limit, confidence_threshold),
        json.dumps(user_payload, ensure_ascii=True),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    message = result["message"]
    answers, confidences, failures = _parse_answer(message["content"], query_keys)
    usage = {
        "prompt_tokens": int(result.get("prompt_eval_count", 0)),
        "completion_tokens": int(result.get("eval_count", 0)),
        "latency_ms": int(result.get("total_duration", elapsed_ms * 1_000_000))
        / 1_000_000,
        "model_calls": 1,
    }
    return answers, confidences, usage, failures


def _latest_records(
    records: list[dict[str, Any]], access_limit: int
) -> list[dict[str, Any]]:
    indexed = sorted(
        enumerate(records),
        key=lambda item: (item[1]["timestamp"], item[0]),
        reverse=True,
    )[:access_limit]
    return [record for _, record in indexed]


def _run_trial(
    client: ChatClient,
    task: str,
    setup: str,
    team_size: int,
    memory_groups: list[list[dict[str, Any]]],
    query_keys: list[str],
    now: int,
    freshness_limit: int,
    confidence_threshold: float,
    access_limit: int,
    communication_budget: int,
) -> tuple[dict[str, str | None], dict[str, float], dict[str, float], int]:
    if setup == "private_memory":
        allowed_agents = min(team_size, communication_budget)
        agent_memories = memory_groups[:allowed_agents]
    else:
        records = memory_groups[0]
        visible = (
            _latest_records(records, access_limit)
            if setup == "shared_memory_limited"
            else records
        )
        agent_count = 1 if setup == "single_agent" else team_size
        agent_memories = [visible for _ in range(agent_count)]

    if not agent_memories:
        return (
            {key: None for key in query_keys},
            {key: 0.5 for key in query_keys},
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0.0,
                "model_calls": 0,
                "agent_messages": 0,
                "records_scanned": 0,
            },
            0,
        )

    system = _system_prompt(freshness_limit, confidence_threshold)
    worker_reports: list[dict[str, Any]] = []
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0.0,
        "model_calls": 0,
        "agent_messages": 0,
        "records_scanned": 0,
    }
    parse_failures = 0
    for agent_index, memory in enumerate(agent_memories):
        answers, confidences, usage, failures = _chat_for_memory(
            client,
            task,
            query_keys,
            memory,
            now,
            freshness_limit,
            confidence_threshold,
        )
        parse_failures += failures
        for name, value in usage.items():
            total_usage[name] += value
        total_usage["records_scanned"] += len(memory)
        worker_reports.append(
            {
                "agent": agent_index,
                "answers": answers,
                "confidence": confidences,
            }
        )
    if len(worker_reports) == 1:
        report = worker_reports[0]
        return (
            report["answers"],
            report["confidence"],
            total_usage,
            parse_failures,
        )

    coordinator_payload = {
        "queries": {key: _query_text(task, key) for key in query_keys},
        "agent_reports": worker_reports,
        "instruction": (
            "Resolve the reports using evidence in the reports. For corrections, "
            "prefer the highest version represented. If reports disagree or "
            "evidence is insufficient, abstain."
        ),
    }
    started = time.perf_counter()
    result = client.chat(system, json.dumps(coordinator_payload, ensure_ascii=True))
    elapsed_ms = (time.perf_counter() - started) * 1000
    message = result["message"]
    answers, confidences, failures = _parse_answer(
        message["content"], query_keys
    )
    parse_failures += failures
    total_usage["prompt_tokens"] += int(result.get("prompt_eval_count", 0))
    total_usage["completion_tokens"] += int(result.get("eval_count", 0))
    total_usage["latency_ms"] += int(
        result.get("total_duration", elapsed_ms * 1_000_000)
    ) / 1_000_000
    total_usage["model_calls"] += 1
    total_usage["agent_messages"] = len(worker_reports)
    return answers, confidences, total_usage, parse_failures


def run_local_agent_experiment(
    config: Mapping[str, Any], *, client: ChatClient | None = None
) -> pd.DataFrame:
    """Run a seeded experiment using a local Ollama chat model."""
    model = dict(config["model"])
    experiment = dict(config["experiment"])
    if model.get("backend") != "ollama":
        raise ValueError("local model experiments require backend: ollama")
    seed = int(model["seed"])
    trials = int(model["trials"])
    bootstrap_samples = int(model["bootstrap_samples"])
    if trials < 1 or bootstrap_samples < 1:
        raise ValueError("trials and bootstrap_samples must be positive")
    tasks = list(experiment["tasks"])
    setups = list(experiment["setups"])
    team_sizes = [int(value) for value in experiment["team_sizes"]]
    if set(tasks) - set(TASKS) or not tasks:
        raise ValueError("tasks must be a non-empty list of supported tasks")
    if set(setups) - set(SETUPS) or not setups:
        raise ValueError("setups must be a non-empty list of supported setups")
    if not team_sizes or any(size < 1 for size in team_sizes):
        raise ValueError("team_sizes must contain positive integers")

    if client is None:
        client = OllamaChatClient(
            str(model["base_url"]),
            str(model["model_name"]),
            seed,
            int(model["num_ctx"]),
            int(model["num_predict"]),
            float(model["timeout_seconds"]),
        )
    freshness_limit = int(model["freshness_limit"])
    confidence_threshold = float(model["confidence_threshold"])
    access_limit = int(model["shared_access_limit"])
    communication_budget = int(model["communication_budget"])
    noise_rate = float(model["shared_noise_rate"])
    distractors = int(model["distractors"])
    delay = int(model["delay_turns"])
    taskset_version = int(model["taskset_version"])
    min_correction_updates = int(model["min_correction_updates"])
    max_correction_updates = int(model["max_correction_updates"])
    min_distributed_clues = int(model["min_distributed_clues"])
    max_distributed_clues = int(model["max_distributed_clues"])
    heldout_lexicon = list(model.get("heldout_lexicon", []))

    trial_metrics: dict[tuple[str, str, int], dict[str, list[float]]] = {}
    for task in tasks:
        for team_size in team_sizes:
            for setup_index, setup in enumerate(setups):
                group = (task, setup, team_size)
                trial_metrics[group] = {
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
                        "model_calls",
                        "records_scanned",
                        "prompt_tokens",
                        "completion_tokens",
                        "parse_failures",
                    )
                }
                for trial in range(trials):
                    records, query_keys, expected, now, stale_values = _make_scenario(
                        task,
                        trial,
                        np.random.default_rng(
                            np.random.SeedSequence([seed, trial])
                        ),
                        distractors,
                        delay,
                        taskset_version,
                        min_correction_updates,
                        max_correction_updates,
                        min_distributed_clues,
                        max_distributed_clues,
                        heldout_lexicon,
                    )
                    stored = _store_records(
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
                    answers, confidences, usage, parse_failures = _run_trial(
                        client,
                        task,
                        setup,
                        team_size,
                        stored,
                        query_keys,
                        now,
                        freshness_limit,
                        confidence_threshold,
                        access_limit,
                        communication_budget,
                    )
                    metrics = _trial_metrics(
                        task, answers, confidences, expected, stale_values
                    )
                    values = trial_metrics[group]
                    for name, value in metrics.items():
                        values[name].append(value)
                    values["latency_ms"].append(usage["latency_ms"])
                    values["compute_units"].append(
                        float(
                            usage["records_scanned"]
                            + usage["agent_messages"]
                        )
                    )
                    values["records_scanned"].append(
                        float(usage["records_scanned"])
                    )
                    values["agent_messages"].append(
                        float(usage["agent_messages"])
                    )
                    values["model_calls"].append(float(usage["model_calls"]))
                    values["prompt_tokens"].append(float(usage["prompt_tokens"]))
                    values["completion_tokens"].append(
                        float(usage["completion_tokens"])
                    )
                    values["parse_failures"].append(float(parse_failures))

    rows = []
    for task, setup, team_size in trial_metrics:
        values = trial_metrics[(task, setup, team_size)]
        row: dict[str, Any] = {
            "task": task,
            "setup": setup,
            "team_size": team_size,
            "backend": "ollama",
            "model_name": client.model_name,
            "model_digest": client.model_digest,
            "agent_policy": str(model["agent_policy"]),
            "retrieval_policy": str(model.get("retrieval_policy", "model_generated")),
            "taskset_version": taskset_version,
            "trials": trials,
            "seed": seed,
            "shared_noise_rate": noise_rate,
            "communication_budget": communication_budget,
            "shared_access_limit": access_limit,
            "freshness_limit": freshness_limit,
            "external_model_calls": 0,
            "api_cost_per_call_usd": 0.0,
            "estimated_api_cost_usd": 0.0,
        }
        for metric, samples_list in values.items():
            samples = np.asarray(samples_list, dtype=float)
            row[f"mean_{metric}"] = float(samples.mean())
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        seed,
                        team_size,
                        SETUPS.index(setup),
                        TASKS.index(task),
                        len(metric),
                    ]
                )
            )
            low, high = bootstrap_ci(samples, rng, n_bootstrap=bootstrap_samples)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        baseline = trial_metrics.get((task, "single_agent", team_size))
        differences = (
            np.asarray(values["accuracy"]) - np.asarray(baseline["accuracy"])
            if baseline
            else np.full(trials, np.nan)
        )
        row["accuracy_delta_vs_single_agent"] = float(differences.mean())
        if baseline:
            delta_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [seed, team_size, SETUPS.index(setup), TASKS.index(task), 991]
                )
            )
            low, high = bootstrap_ci(
                differences, delta_rng, n_bootstrap=bootstrap_samples
            )
            row["accuracy_delta_vs_single_agent_ci_low"] = low
            row["accuracy_delta_vs_single_agent_ci_high"] = high
        else:
            row["accuracy_delta_vs_single_agent_ci_low"] = np.nan
            row["accuracy_delta_vs_single_agent_ci_high"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
