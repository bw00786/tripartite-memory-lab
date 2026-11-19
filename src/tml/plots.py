"""Experiment result plotting and plain-language comparison summaries."""

from pathlib import Path
from typing import Any

import matplotlib

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
