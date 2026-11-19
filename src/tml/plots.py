"""Experiment result plotting and plain-language comparison summaries."""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

SWEEP_LABELS = {
    "capacity": ("Stored memories (M)", "Memory capacity sweep"),
    "noise": ("Chemical noise", "Chemical noise sweep"),
    "cue": ("Cue fraction", "Cue-size sweep"),
    "cue_noise": ("Cue corruption (fraction replaced)", "Corrupted-cue sweep"),
    "diffusion": ("Diffusion width (sigma)", "Diffusion-width sweep"),
    "sparse_dimension": (
        "Signature dimensions (D)",
        "Sparse-signature dimension sweep",
    ),
}


def plot_results(results: pd.DataFrame, output_path: Path) -> None:
    """Plot all model means and percentile confidence bands for one sweep."""
    if "sweep" not in results or "sweep_value" not in results:
        raise ValueError("Plot data must include sweep and sweep_value columns")
    sweep_types = results["sweep"].unique()
    if len(sweep_types) != 1 or sweep_types[0] not in SWEEP_LABELS:
        raise ValueError("Plot data must contain exactly one supported sweep")
    sweep = str(sweep_types[0])
    x_label, title = SWEEP_LABELS[sweep]

    fig, axis = plt.subplots(figsize=(8, 5))
    for model_name, model_rows in results.groupby("model", sort=False):
        model_rows = model_rows.sort_values("sweep_value")
        x_values = model_rows["sweep_value"].to_numpy(dtype=float)
        means = model_rows["mean"].to_numpy(dtype=float)
        lows = model_rows["ci_low"].to_numpy(dtype=float)
        highs = model_rows["ci_high"].to_numpy(dtype=float)
        is_reference = model_name == "NeuronOnly"
        line = axis.plot(
            x_values,
            means,
            linestyle="--" if is_reference else "-",
            marker="o",
            label=f"{model_name} (reference)" if is_reference else str(model_name),
        )[0]
        axis.fill_between(
            x_values,
            lows,
            highs,
            color=line.get_color(),
            alpha=0.10 if is_reference else 0.16,
        )

    axis.set(
        xlabel=x_label,
        ylabel="Mean recall overlap (95% bootstrap CI)",
        title=title,
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_agent_results(results: pd.DataFrame, output_path: Path) -> None:
    """Plot task accuracy by memory setup and team size."""
    required = {
        "task",
        "setup",
        "team_size",
        "mean_accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
    }
    if not required.issubset(results.columns):
        raise ValueError(f"Agent results must include: {', '.join(sorted(required))}")
    tasks = list(results["task"].drop_duplicates())
    if "sweep" in results and "sweep_value" in results:
        sweep_names = results["sweep"].unique()
        if len(sweep_names) != 1:
            raise ValueError("Agent plot data must contain one intervention at a time")
        fig, axes = plt.subplots(
            1, len(tasks), figsize=(5 * len(tasks), 5), squeeze=False, sharey=True
        )
        for axis, task in zip(axes[0], tasks, strict=True):
            task_rows = results[results["task"] == task]
            for (setup, team_size), group in task_rows.groupby(
                ["setup", "team_size"], sort=False
            ):
                group = group.sort_values("sweep_value")
                means = group["mean_accuracy"].to_numpy(dtype=float)
                axis.errorbar(
                    group["sweep_value"],
                    means,
                    yerr=[
                        means - group["accuracy_ci_low"].to_numpy(dtype=float),
                        group["accuracy_ci_high"].to_numpy(dtype=float) - means,
                    ],
                    marker="o",
                    capsize=2,
                    label=f"{setup}, {team_size} agents",
                )
            axis.set_title(task.replace("_", " "))
            axis.set_ylim(0, 1.05)
            axis.grid(alpha=0.25)
        axes[0, 0].set_ylabel("Task accuracy (95% bootstrap CI)")
        axes[0, len(tasks) // 2].set_xlabel(str(sweep_names[0]).replace("_", " "))
        axes[0, -1].legend(fontsize="small")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    fig, axes = plt.subplots(
        1, len(tasks), figsize=(5 * len(tasks), 5), squeeze=False, sharey=True
    )
    setups = list(results["setup"].drop_duplicates())
    team_sizes = sorted(results["team_size"].unique())
    positions = np.arange(len(setups), dtype=float)
    width = 0.8 / len(team_sizes)
    for axis, task in zip(axes[0], tasks, strict=True):
        task_rows = results[results["task"] == task]
        for size_index, team_size in enumerate(team_sizes):
            means = []
            lower = []
            upper = []
            for setup in setups:
                row = task_rows[
                    (task_rows["setup"] == setup)
                    & (task_rows["team_size"] == team_size)
                ]
                if row.empty:
                    means.append(np.nan)
                    lower.append(0.0)
                    upper.append(0.0)
                    continue
                result = row.iloc[0]
                means.append(float(result["mean_accuracy"]))
                lower.append(
                    float(result["mean_accuracy"] - result["accuracy_ci_low"])
                )
                upper.append(
                    float(result["accuracy_ci_high"] - result["mean_accuracy"])
                )
            offset = (size_index - (len(team_sizes) - 1) / 2) * width
            axis.bar(
                positions + offset,
                means,
                width,
                yerr=np.asarray([lower, upper]),
                capsize=2,
                label=f"{team_size} agents",
            )
        axis.set_xticks(positions, setups, rotation=45, ha="right")
        axis.set_title(task.replace("_", " "))
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Task accuracy (95% bootstrap CI)")
    axes[0, -1].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _condition_label(row: pd.Series) -> str:
    sweep = row.get("sweep")
    value = row.get("sweep_value")
    if sweep == "capacity":
        return f"M={value:g}"
    if sweep == "noise":
        return f"noise={value:g}"
    if sweep == "cue":
        return f"cue_frac={value:g}"
    if sweep == "cue_noise":
        return f"cue_noise={value:g}"
    if sweep == "diffusion":
        return f"sigma={value:g}"
    if sweep == "sparse_dimension":
        return f"D={value:g}"
    return f"M={row['M']:g}, cue_frac={row['cue_frac']:g}, noise={row['noise']:g}"


def create_summary(results_by_config: dict[str, pd.DataFrame]) -> str:
    """Compare every modeled variant with its neuron-only baseline."""
    comparisons: list[dict[str, Any]] = []
    model_names: set[str] = set()
    noise_differences: dict[str, list[tuple[float, float]]] = {}
    for config_name, frame in results_by_config.items():
        if "model" not in frame or "mean" not in frame:
            continue
        model_names.update(frame["model"].unique())
        neuron_rows = frame[frame["model"] == "NeuronOnly"]
        if neuron_rows.empty:
            continue
        join_keys = [
            key
            for key in (
                "sweep",
                "sweep_value",
                "setting_index",
                "M",
                "cue_frac",
                "noise",
                "alpha",
                "seed",
            )
            if key in frame.columns
        ]
        variant_rows = frame[frame["model"] != "NeuronOnly"]
        paired = variant_rows.merge(
            neuron_rows,
            on=join_keys,
            suffixes=("_variant", "_baseline"),
            validate="many_to_one",
        )
        for _, row in paired.iterrows():
            difference = float(row["mean_variant"] - row["mean_baseline"])
            variant = str(row["model_variant"])
            if str(row.get("sweep", "")) == "noise":
                noise_differences.setdefault(variant, []).append(
                    (float(row["noise"]), difference)
                )
            comparisons.append(
                {
                    "variant": variant,
                    "condition": _condition_label(
                        pd.Series(
                            {
                                "sweep": row.get("sweep"),
                                "sweep_value": row.get("sweep_value"),
                                "M": row.get("M"),
                                "cue_frac": row.get("cue_frac"),
                                "noise": row.get("noise"),
                            }
                        )
                    ),
                    "difference": difference,
                    "config": config_name,
                    "variant_low": float(row["ci_low_variant"]),
                    "baseline_high": float(row["ci_high_baseline"]),
                }
            )

    lines = [
        "# Experiment summary",
        "",
        "Positive mean differences are descriptive; they are not by themselves a "
        "significance test. CSV files report each model's 95% bootstrap interval.",
        "CueMatch is a direct lookup baseline that scans the stored patterns; it "
        "is an algorithmic comparator, not a chemical-memory mechanism.",
        "Positive overlap differences do not necessarily mean exact retrieval; "
        "use exact_match_rate and its interval in the CSV for that outcome.",
        "",
        (
            "| Model | Mean beats neuron-only? | Conditions with higher mean "
            "recall | Noise advantage disappears |"
        ),
        "|---|---|---|---|",
    ]
    for variant in sorted(model_names - {"NeuronOnly"}):
        variant_comparisons = [
            item for item in comparisons if item["variant"] == variant
        ]
        better = [item for item in variant_comparisons if item["difference"] > 0]
        if better:
            condition_text = "; ".join(
                f"{item['condition']} ({item['config']}, delta "
                f"{item['difference']:+.3f})"
                for item in better
            )
            status = "Yes, in tested settings"
        elif variant_comparisons:
            condition_text = "None observed"
            status = "No"
        else:
            condition_text = "No matched neuron-only results available"
            status = "Not evaluated"

        noise_points = sorted(noise_differences.get(variant, []))
        positive_noises = [
            noise for noise, difference in noise_points if difference > 0
        ]
        if variant == "CueMatch":
            noise_status = "Not applicable; algorithm does not use chemical noise"
        elif not positive_noises:
            noise_status = "No advantage observed at tested noise levels"
        else:
            first_advantage_noise = min(positive_noises)
            first_nonpositive = next(
                (
                    noise
                    for noise, difference in noise_points
                    if noise > first_advantage_noise and difference <= 0
                ),
                None,
            )
            noise_status = (
                f"First non-positive tested at noise={first_nonpositive:g}"
                if first_nonpositive is not None
                else f"Advantage persisted through noise={noise_points[-1][0]:g}"
            )
        lines.append(
            f"| {variant} | {status} | {condition_text} | {noise_status} |"
        )
    if len(lines) == 6:
        lines.append(
            "| No variants | Not evaluated | Run a sweep including variants and "
            "NeuronOnly | Not evaluated |"
        )
    lines.extend(
        [
            "",
            "A condition is listed when the variant's observed mean exceeds the "
            "neuron-only observed mean. No positive difference means no tested "
            "condition showed a mean advantage.",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary(
    results_by_config: dict[str, pd.DataFrame], output_path: Path
) -> None:
    """Write a Markdown comparison summary from completed experiment CSVs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(create_summary(results_by_config), encoding="utf-8")


def write_agent_summary(
    results_by_config: dict[str, pd.DataFrame], output_path: Path
) -> None:
    """Write functional agent-memory outcomes without consciousness claims."""
    lines = [
        "# Agent memory experiment summary",
        "",
        "Backend and model are recorded for each row. deterministic_simulator "
        "rows are policy simulations; Ollama rows use the named local model. "
        "Latency is local wall-clock time. API cost is zero because no paid "
        "external inference API is used; local compute and energy costs are not "
        "estimated. Team-size setting is configured team size; single_agent "
        "always uses one agent.",
        "",
        (
            "| Config | Task | Setup | Backend/model | Retrieval policy | "
            "Intervention | "
            "Team-size setting | Accuracy (95% CI) | Exact "
            "retrieval (95% CI) | Paired accuracy delta vs single-agent (95% CI) | "
            "Paired accuracy delta vs exact-key (95% CI) | "
            "Stale errors | Conflict errors | "
            "Uncertainty errors | Brier score | Model calls | Prompt tokens | "
            "Completion tokens | Parse failures | "
            "Latency ms | Compute units | API cost USD |"
        ),
        "|---|---|---|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config_name, frame in results_by_config.items():
        if not {"task", "setup", "team_size", "mean_accuracy"}.issubset(
            frame.columns
        ):
            continue
        for row in frame.itertuples():
            retrieval_policy = getattr(row, "retrieval_policy", "exact_key")
            model_name = getattr(row, "model_name", "deterministic_simulator")
            model_digest = getattr(row, "model_digest", "")
            model_label = (
                f"{model_name} ({model_digest[:12]})"
                if model_digest
                else model_name
            )
            sweep_name = getattr(row, "sweep", "")
            sweep_value = getattr(row, "sweep_value", "")
            intervention = (
                f"{sweep_name}={sweep_value}" if sweep_name else "none"
            )
            has_paired_delta = {
                "accuracy_delta_vs_single_agent",
                "accuracy_delta_vs_single_agent_ci_low",
                "accuracy_delta_vs_single_agent_ci_high",
            }.issubset(frame.columns)
            paired_delta = "N/A"
            if has_paired_delta and np.isfinite(row.accuracy_delta_vs_single_agent):
                paired_delta = (
                    f"{row.accuracy_delta_vs_single_agent:.3f} "
                    f"[{row.accuracy_delta_vs_single_agent_ci_low:.3f}, "
                    f"{row.accuracy_delta_vs_single_agent_ci_high:.3f}]"
                )
            policy_delta = "N/A"
            if hasattr(row, "accuracy_delta_vs_exact_key") and np.isfinite(
                row.accuracy_delta_vs_exact_key
            ):
                policy_delta = (
                    f"{row.accuracy_delta_vs_exact_key:.3f} "
                    f"[{row.accuracy_delta_vs_exact_key_ci_low:.3f}, "
                    f"{row.accuracy_delta_vs_exact_key_ci_high:.3f}]"
                )
            lines.append(
                f"| {config_name} | {row.task} | {row.setup} | "
                f"{model_label} | {retrieval_policy} | {intervention} | "
                f"{row.team_size} | "
                f"{row.mean_accuracy:.3f} "
                f"[{row.accuracy_ci_low:.3f}, {row.accuracy_ci_high:.3f}] | "
                f"{row.mean_exact_retrieval:.3f} "
                f"[{row.exact_retrieval_ci_low:.3f}, "
                f"{row.exact_retrieval_ci_high:.3f}] | "
                f"{paired_delta} | "
                f"{policy_delta} | "
                f"{row.mean_stale_error_rate:.3f} | "
                f"{row.mean_conflict_error_rate:.3f} | "
                f"{row.mean_uncertainty_error_rate:.3f} | "
                f"{row.mean_uncertainty_brier:.3f} | "
                f"{getattr(row, 'mean_model_calls', 0.0):.1f} | "
                f"{getattr(row, 'mean_prompt_tokens', 0.0):.1f} | "
                f"{getattr(row, 'mean_completion_tokens', 0.0):.1f} | "
                f"{getattr(row, 'mean_parse_failures', 0.0):.3f} | "
                f"{row.mean_latency_ms:.3f} | "
                f"{row.mean_compute_units:.1f} | "
                f"{row.estimated_api_cost_usd:.4f} |"
            )
    lines.extend(
        [
            "",
            "Paired accuracy contrasts use trial-level differences from the "
            "single-agent setup and a bootstrap interval over those differences. "
            "Intervals crossing zero are inconclusive; these intervals are not "
            "adjusted for multiple comparisons.",
        ]
    )
    for config_name, frame in results_by_config.items():
        if not {"task", "setup", "team_size", "mean_accuracy"}.issubset(
            frame.columns
        ):
            continue
        if "accuracy_delta_vs_single_agent_ci_low" not in frame.columns:
            continue
        comparisons: dict[str, list[str]] = {}
        group_columns = ["task", "team_size"]
        if {"sweep", "sweep_value"}.issubset(frame.columns):
            group_columns.extend(["sweep", "sweep_value"])
        for _, group in frame.groupby(group_columns, dropna=False):
            for row in group.itertuples():
                if row.setup == "single_agent":
                    continue
                if not np.isfinite(row.accuracy_delta_vs_single_agent):
                    continue
                outcome = "inconclusive"
                if row.accuracy_delta_vs_single_agent_ci_low > 0:
                    outcome = "positive"
                elif row.accuracy_delta_vs_single_agent_ci_high < 0:
                    outcome = "negative"
                comparisons.setdefault(row.setup, []).append(outcome)
        if not comparisons:
            continue
        lines.append(f"- **{config_name}**:")
        for setup, outcomes in sorted(comparisons.items()):
            counts = {
                outcome: outcomes.count(outcome)
                for outcome in ("positive", "negative", "inconclusive")
            }
            lines.append(
                f"  - {setup}: paired CI above zero in {counts['positive']}/"
                f"{len(outcomes)}, below zero in {counts['negative']}/"
                f"{len(outcomes)}, and crossing zero in "
                f"{counts['inconclusive']}/{len(outcomes)} comparisons."
            )
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
