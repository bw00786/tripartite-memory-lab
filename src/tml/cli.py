"""Command-line entry point for experiment runs."""

import argparse
from pathlib import Path
from typing import Any

import yaml

from tml.experiment import RESULT_COLUMNS, run_experiment


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
    args = parser.parse_args()

    if args.command == "run":
        config = _read_config(args.config)
        results = run_experiment(config)
        output_path = args.output or Path("results") / f"{args.config.stem}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        columns = RESULT_COLUMNS + [
            column
            for column in results.columns
            if column not in RESULT_COLUMNS
        ]
        results.to_csv(output_path, index=False, columns=columns)
        print(f"Wrote {len(results)} setting/model rows to {output_path}")


if __name__ == "__main__":
    main()
