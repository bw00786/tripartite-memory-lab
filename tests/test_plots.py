import pandas as pd

from tml.plots import create_summary, plot_results


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
