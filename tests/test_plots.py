import pandas as pd

from tml.plots import (
    create_summary,
    plot_agent_results,
    plot_results,
    write_agent_summary,
)


def test_sweep_plot_is_written_with_reference_and_ci_data(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {
                "model": model,
                "sweep": "noise",
                "sweep_value": noise,
                "mean": mean,
                "ci_low": mean - 0.05,
                "ci_high": mean + 0.05,
            }
            for noise, baseline, variant in [(0, 0.8, 0.7), (1, 0.6, 0.5)]
            for model, mean in [
                ("NeuronOnly", baseline),
                ("Matrix", variant),
            ]
        ]
    )
    output = tmp_path / "noise.png"

    plot_results(results, output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_summary_plainly_reports_negative_and_noise_disappearance() -> None:
    rows = []
    for noise, baseline, matrix, local in [
        (0.0, 0.8, 0.7, 0.9),
        (1.0, 0.6, 0.5, 0.55),
    ]:
        for model, mean in [
            ("NeuronOnly", baseline),
            ("Matrix", matrix),
            ("LocalDomain", local),
        ]:
            rows.append(
                {
                    "model": model,
                    "sweep": "noise",
                    "sweep_value": noise,
                    "setting_index": int(noise),
                    "M": 20,
                    "cue_frac": 0.3,
                    "noise": noise,
                    "alpha": 1.0,
                    "seed": 1,
                    "mean": mean,
                    "ci_low": mean - 0.02,
                    "ci_high": mean + 0.02,
                }
            )

    summary = create_summary({"sweep_noise": pd.DataFrame(rows)})

    assert "Matrix | No | None observed" in summary
    assert "LocalDomain | Yes, in tested settings" in summary
    assert "noise=0" in summary
    assert "First non-positive tested at noise=1" in summary


def test_agent_plot_and_summary_include_task_metrics(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {
                "task": "delayed_retrieval",
                "setup": setup,
                "team_size": team_size,
                "mean_accuracy": accuracy,
                "accuracy_ci_low": accuracy - 0.05,
                "accuracy_ci_high": accuracy + 0.05,
                "accuracy_delta_vs_single_agent": accuracy - 1.0,
                "accuracy_delta_vs_single_agent_ci_low": accuracy - 1.0,
                "accuracy_delta_vs_single_agent_ci_high": accuracy - 1.0,
                "mean_exact_retrieval": accuracy,
                "exact_retrieval_ci_low": accuracy - 0.05,
                "exact_retrieval_ci_high": accuracy + 0.05,
                "mean_stale_error_rate": 0.0,
                "mean_conflict_error_rate": 0.0,
                "mean_uncertainty_error_rate": 0.0,
                "mean_uncertainty_brier": 0.01,
                "mean_latency_ms": 1.5,
                "mean_compute_units": 10.0,
                "estimated_api_cost_usd": 0.0,
            }
            for setup, team_size, accuracy in [
                ("single_agent", 1, 1.0),
                ("private_memory", 1, 0.8),
            ]
        ]
    )
    image_path = tmp_path / "agent.png"
    summary_path = tmp_path / "agent.md"

    plot_agent_results(results, image_path)
    write_agent_summary({"agent_test": results}, summary_path)

    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "Brier score" in summary_path.read_text(encoding="utf-8")
    assert "private_memory" in summary_path.read_text(encoding="utf-8")
    assert "below zero in 1/1" in summary_path.read_text(encoding="utf-8")


def test_agent_summary_compares_matching_sweep_values(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {
                "task": "uncertainty",
                "setup": setup,
                "team_size": 1,
                "mean_accuracy": accuracy,
                "accuracy_ci_low": accuracy,
                "accuracy_ci_high": accuracy,
                "accuracy_delta_vs_single_agent": 0.0,
                "accuracy_delta_vs_single_agent_ci_low": 0.0,
                "accuracy_delta_vs_single_agent_ci_high": 0.0,
                "mean_exact_retrieval": accuracy,
                "exact_retrieval_ci_low": accuracy,
                "exact_retrieval_ci_high": accuracy,
                "mean_stale_error_rate": 0.0,
                "mean_conflict_error_rate": 0.0,
                "mean_uncertainty_error_rate": 0.0,
                "mean_uncertainty_brier": 0.0,
                "mean_latency_ms": 1.0,
                "mean_compute_units": 1.0,
                "estimated_api_cost_usd": 0.0,
                "sweep": "confidence",
                "sweep_value": value,
            }
            for value, accuracy in [
                (0.3, 0.5),
                (0.6, 1.0),
            ]
            for setup in ("single_agent", "shared_memory")
        ]
    )
    summary_path = tmp_path / "sweep.md"

    write_agent_summary({"sweep": results}, summary_path)

    assert (
        "shared_memory: paired CI above zero in 0/2, below zero in 0/2, "
        "and crossing zero in 2/2"
        in summary_path.read_text(encoding="utf-8")
    )
