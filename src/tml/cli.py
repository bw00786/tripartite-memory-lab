"""Command-line entry point for experiment runs."""

import argparse
from pathlib import Path
from typing import Any

import yaml

from tml.experiment import RESULT_COLUMNS, run_experiment
from tml.plots import plot_results, write_summary


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible memory experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one YAML experiment config")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--output", type=Path)
    all_parser = subparsers.add_parser("all", help="run every YAML config in configs")
    all_parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    all_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.command == "run":
        config = _read_config(args.config)
        results = run_experiment(config)
        output_path = args.output or Path("results") / f"{args.config.stem}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        columns = RESULT_COLUMNS + [
            column for column in results if column not in RESULT_COLUMNS
        ]
        results.to_csv(output_path, index=False, columns=columns)
        print(f"Wrote {len(results)} setting/model rows to {output_path}")
        if config["experiment"].get("type", "baseline") != "baseline":
            plot_path = output_path.with_suffix(".png")
            plot_results(results, plot_path)
            print(f"Wrote plot to {plot_path}")
        write_summary({args.config.stem: results}, output_path.parent / "summary.md")
    elif args.command == "all":
        config_paths = sorted(args.config_dir.glob("*.yaml"))
        if not config_paths:
            parser.error(f"No YAML configs found in {args.config_dir}")
        results_by_config = {}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for config_path in config_paths:
            config = _read_config(config_path)
            results = run_experiment(config)
            csv_path = args.output_dir / f"{config_path.stem}.csv"
            columns = RESULT_COLUMNS + [
                column for column in results if column not in RESULT_COLUMNS
            ]
            results.to_csv(csv_path, index=False, columns=columns)
            results_by_config[config_path.stem] = results
            print(f"Wrote {len(results)} setting/model rows to {csv_path}")
            if config["experiment"].get("type", "baseline") != "baseline":
                plot_path = args.output_dir / f"{config_path.stem}.png"
                plot_results(results, plot_path)
                print(f"Wrote plot to {plot_path}")
        summary_path = args.output_dir / "summary.md"
        write_summary(results_by_config, summary_path)
        print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
