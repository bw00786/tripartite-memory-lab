import json

import numpy as np

from tml.agent_experiment import run_agent_experiment
from tml.local_model_experiment import run_local_agent_experiment


def _config(
    tasks: list[str],
    setups: list[str],
    *,
    team_sizes: list[int] | None = None,
    trials: int = 20,
    communication_budget: int = 3,
    noise_rate: float = 0.0,
    access_limit: int = 8,
    taskset_version: int = 1,
    retrieval_policy: str = "exact_key",
    heldout_lexicon: list[str] | None = None,
) -> dict:
    return {
        "model": {
            "backend": "deterministic_simulator",
            "agent_policy": "highest_version_then_latest_timestamp",
            "taskset_version": taskset_version,
            "retrieval_policy": retrieval_policy,
            "heldout_lexicon": heldout_lexicon or [],
            "seed": 12,
            "trials": trials,
            "bootstrap_samples": 200,
            "distractors": 12,
            "delay_turns": 10,
            "min_correction_updates": 2,
            "max_correction_updates": 5,
            "min_distributed_clues": 2,
            "max_distributed_clues": 5,
            "freshness_limit": 10,
            "confidence_threshold": 0.6,
            "shared_noise_rate": noise_rate,
            "shared_access_limit": access_limit,
            "communication_budget": communication_budget,
            "api_cost_per_call_usd": 0.0,
        },
        "experiment": {
            "type": "agent_memory",
            "tasks": tasks,
            "setups": setups,
            "team_sizes": team_sizes or [3],
        },
    }


def test_agent_memory_results_are_seeded_and_include_cost_metrics() -> None:
    config = _config(["delayed_retrieval"], ["single_agent", "shared_memory"])

    first = run_agent_experiment(config)
    second = run_agent_experiment(config)

    numeric = first.select_dtypes("number").columns
    numeric = numeric[
        ~numeric.str.contains("latency_ms")
    ]
    np.testing.assert_array_equal(first[numeric], second[numeric])
    assert (first["mean_accuracy"] == 1.0).all()
    assert (first["external_model_calls"] == 0).all()
    assert (first["estimated_api_cost_usd"] == 0.0).all()
    assert (first["mean_latency_ms"] > 0).all()
    assert (first["accuracy_delta_vs_single_agent"] == 0.0).all()
    assert (first["accuracy_delta_vs_single_agent_ci_low"] == 0.0).all()
    assert (first["accuracy_delta_vs_single_agent_ci_high"] == 0.0).all()


def test_paired_accuracy_delta_uses_the_same_trials_as_baseline() -> None:
    config = _config(
        ["delayed_retrieval"],
        ["single_agent", "shared_memory_noisy"],
        noise_rate=1.0,
        trials=30,
    )

    results = run_agent_experiment(config).set_index("setup")
    noisy = results.loc["shared_memory_noisy"]

    assert noisy["accuracy_delta_vs_single_agent"] == -1.0
    assert noisy["accuracy_delta_vs_single_agent_ci_low"] == -1.0
    assert noisy["accuracy_delta_vs_single_agent_ci_high"] == -1.0


def test_private_memory_team_requires_communication_to_combine_clues() -> None:
    config = _config(
        ["distributed_clues"],
        ["private_memory"],
        communication_budget=1,
        trials=50,
    )

    results = run_agent_experiment(config).iloc[0]

    assert results["mean_accuracy"] < 1.0
    assert results["mean_agent_messages"] <= 1.0


def test_shared_limited_memory_can_lose_delayed_facts() -> None:
    config = _config(
        ["delayed_retrieval"],
        ["shared_memory", "shared_memory_limited"],
        team_sizes=[3],
        access_limit=1,
        trials=40,
    )

    results = run_agent_experiment(config).set_index("setup")

    assert results.loc["shared_memory", "mean_accuracy"] == 1.0
    assert results.loc["shared_memory_limited", "mean_accuracy"] < 1.0


def test_corrected_facts_and_uncertainty_are_handled_explicitly() -> None:
    config = _config(
        ["corrected_fact", "uncertainty_and_staleness"],
        ["single_agent", "shared_memory"],
        trials=10,
    )

    results = run_agent_experiment(config)

    assert (results["mean_accuracy"] == 1.0).all()
    assert (results["mean_stale_error_rate"] == 0.0).all()
    assert (results["mean_conflict_error_rate"] == 0.0).all()
    assert (results["mean_uncertainty_brier"] < 0.01).all()


def test_noisy_shared_memory_records_a_failure_when_all_writes_are_corrupted() -> None:
    config = _config(
        ["delayed_retrieval"],
        ["shared_memory", "shared_memory_noisy"],
        noise_rate=1.0,
        trials=8,
    )

    results = run_agent_experiment(config).set_index("setup")

    assert results.loc["shared_memory", "mean_accuracy"] == 1.0
    assert results.loc["shared_memory_noisy", "mean_accuracy"] == 0.0


def test_communication_intervention_changes_private_clue_retrieval() -> None:
    config = _config(
        ["distributed_clues"],
        ["private_memory"],
        communication_budget=3,
        trials=30,
    )
    config["experiment"]["intervention"] = "communication_budget"
    config["experiment"]["values"] = [0, 3]

    results = run_agent_experiment(config)

    assert results["sweep_value"].tolist() == [0, 3]
    assert results["mean_accuracy"].iloc[0] == 0.0
    assert results["mean_accuracy"].iloc[1] == 1.0


def test_noise_and_access_sweeps_preserve_condition_labels() -> None:
    noise_config = _config(
        ["delayed_retrieval"],
        ["shared_memory_noisy"],
        noise_rate=0.0,
        trials=12,
    )
    noise_config["experiment"].update(
        {"intervention": "shared_noise_rate", "values": [0.0, 1.0]}
    )
    noise_results = run_agent_experiment(noise_config)
    assert noise_results["mean_accuracy"].tolist() == [1.0, 0.0]

    access_config = _config(
        ["delayed_retrieval"],
        ["shared_memory_limited"],
        access_limit=1,
        trials=20,
    )
    access_config["experiment"].update(
        {"intervention": "shared_access_limit", "values": [1, 32]}
    )
    access_results = run_agent_experiment(access_config)
    assert access_results["mean_accuracy"].iloc[0] < 1.0
    assert access_results["mean_accuracy"].iloc[1] == 1.0


def test_freshness_and_confidence_interventions_expose_ablation_errors() -> None:
    config = _config(
        ["uncertainty_and_staleness"],
        ["single_agent"],
        team_sizes=[1],
        trials=12,
    )
    config["experiment"].update(
        {"intervention": "freshness_limit", "values": [3, 20]}
    )

    freshness_results = run_agent_experiment(config)

    assert freshness_results["mean_stale_error_rate"].tolist() == [0.0, 0.5]
    assert freshness_results["mean_accuracy"].tolist() == [1.0, 0.5]

    config["experiment"].update(
        {"intervention": "confidence_threshold", "values": [0.3, 0.6]}
    )
    confidence_results = run_agent_experiment(config)

    assert confidence_results["mean_uncertainty_error_rate"].tolist() == [0.5, 0.0]
    assert confidence_results["mean_accuracy"].tolist() == [0.5, 1.0]


def test_taskset_v2_varies_correction_depth_and_clue_count_reproducibly() -> None:
    config = _config(
        ["correction_chain", "distributed_clues"],
        ["single_agent", "shared_memory_noisy"],
        team_sizes=[3],
        trials=40,
        noise_rate=1.0,
        taskset_version=2,
    )

    first = run_agent_experiment(config)
    second = run_agent_experiment(config)

    metrics = [
        "mean_accuracy",
        "mean_exact_retrieval",
        "mean_conflict_error_rate",
        "accuracy_delta_vs_single_agent",
        "accuracy_delta_vs_single_agent_ci_low",
        "accuracy_delta_vs_single_agent_ci_high",
    ]
    np.testing.assert_array_equal(
        first[metrics].to_numpy(),
        second[metrics].to_numpy(),
    )
    noisy = first[first["setup"] == "shared_memory_noisy"]
    noisy_chain = noisy[noisy["task"] == "correction_chain"].iloc[0]
    assert noisy_chain["mean_accuracy"] == 0.0
    assert noisy_chain["mean_conflict_error_rate"] == 1.0

    communication_config = _config(
        ["distributed_clues"],
        ["private_memory"],
        team_sizes=[3],
        trials=40,
        communication_budget=3,
        taskset_version=2,
    )
    communication_config["experiment"].update(
        {"intervention": "communication_budget", "values": [0, 3]}
    )
    communication_results = run_agent_experiment(communication_config)
    assert communication_results["mean_accuracy"].iloc[0] < 1.0
    assert communication_results["mean_accuracy"].iloc[1] == 1.0


def test_heldout_cue_retrieval_uses_text_not_an_explicit_record_key() -> None:
    lexicon = [
        "amber",
        "anchor",
        "orchard",
        "vessel",
        "cobalt",
        "meadow",
        "ribbon",
        "summit",
        "lantern",
        "marble",
        "harbor",
        "thimble",
    ]
    config = _config(
        ["heldout_cue_retrieval"],
        ["single_agent", "shared_memory"],
        team_sizes=[3],
        trials=40,
        taskset_version=3,
        retrieval_policy="cue_token_overlap",
        heldout_lexicon=lexicon,
    )

    first = run_agent_experiment(config)
    second = run_agent_experiment(config)

    np.testing.assert_array_equal(
        first[["mean_accuracy", "accuracy_delta_vs_single_agent"]].to_numpy(),
        second[["mean_accuracy", "accuracy_delta_vs_single_agent"]].to_numpy(),
    )
    assert (first["mean_accuracy"] == 1.0).all()

    config["model"]["retrieval_policy"] = "exact_key"
    exact_key_results = run_agent_experiment(config)
    assert (exact_key_results["mean_accuracy"] == 0.0).all()


def test_heldout_cue_policy_ablation_runs_as_a_retrieval_intervention() -> None:
    config = _config(
        ["heldout_cue_retrieval"],
        ["single_agent"],
        team_sizes=[1],
        trials=20,
        taskset_version=3,
        retrieval_policy="cue_token_overlap",
        heldout_lexicon=[
            "amber",
            "anchor",
            "orchard",
            "vessel",
            "cobalt",
            "meadow",
            "ribbon",
            "summit",
        ],
    )
    config["experiment"].update(
        {
            "intervention": "retrieval_policy",
            "values": ["exact_key", "cue_token_overlap"],
        }
    )

    results = run_agent_experiment(config)

    assert results["sweep_value"].tolist() == ["exact_key", "cue_token_overlap"]
    assert results["mean_accuracy"].tolist() == [0.0, 1.0]
    assert results["accuracy_delta_vs_exact_key"].tolist() == [0.0, 1.0]
    assert results["accuracy_delta_vs_exact_key_ci_low"].tolist() == [0.0, 1.0]
    assert results["accuracy_delta_vs_exact_key_ci_high"].tolist() == [0.0, 1.0]


def test_cue_token_policy_keeps_structured_tasks_on_exact_keys() -> None:
    config = _config(
        ["delayed_retrieval", "heldout_cue_retrieval"],
        ["single_agent"],
        team_sizes=[1],
        trials=12,
        taskset_version=3,
        retrieval_policy="cue_token_overlap",
        heldout_lexicon=[
            "amber",
            "anchor",
            "orchard",
            "vessel",
            "cobalt",
            "meadow",
            "ribbon",
            "summit",
        ],
    )

    results = run_agent_experiment(config)

    assert (results["mean_accuracy"] == 1.0).all()


class _FakeChatClient:
    model_name = "fake-local"
    model_digest = "fake-digest"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, system: str, user: str) -> dict:
        del system
        self.calls += 1
        payload = json.loads(user)
        if "records" in payload:
            records = payload["records"]
            query_keys = payload["queries"]
            answers = {}
            for key, query in query_keys.items():
                record = next(
                    (item for item in records if item["key"] in query), None
                )
                answers[key] = record["value"] if record else None
            confidences = {key: 0.95 for key in query_keys}
        else:
            report = payload["agent_reports"][0]
            answers = report["answers"]
            confidences = report["confidence"]
        return {
            "message": {
                "content": json.dumps(
                    {"answers": answers, "confidence": confidences}
                )
            },
            "prompt_eval_count": 12,
            "eval_count": 8,
            "total_duration": 1_000_000,
        }


def test_local_model_runner_uses_client_and_records_inference_metadata() -> None:
    config = _config(
        ["delayed_retrieval"],
        ["single_agent"],
        team_sizes=[1],
        trials=2,
    )
    config["model"].update(
        {
            "backend": "ollama",
            "model_name": "fake-local",
            "base_url": "http://localhost:11434",
            "num_ctx": 1024,
            "num_predict": 32,
            "timeout_seconds": 2,
        }
    )
    client = _FakeChatClient()

    results = run_local_agent_experiment(config, client=client)

    assert client.calls == 2
    assert results.loc[0, "mean_accuracy"] == 1.0
    assert results.loc[0, "model_name"] == "fake-local"
    assert results.loc[0, "model_digest"] == "fake-digest"
    assert results.loc[0, "mean_model_calls"] == 1.0
    assert results.loc[0, "mean_prompt_tokens"] == 12.0
    assert results.loc[0, "mean_completion_tokens"] == 8.0
    assert results.loc[0, "mean_parse_failures"] == 0.0
