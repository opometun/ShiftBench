"""Command-line runner for reproducible ShiftBench experiments."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any
import json
import platform
import random
import subprocess
import sys

from shiftbench.config import ExperimentConfig, load_experiment_config
from shiftbench.datasets.schema import validate_csv_dataset


def run_experiment(config_path: Path, output_root: Path | None = None) -> Path:
    """Run a deterministic ShiftBench smoke experiment.

    The current scaffold validates the configured dataset and writes structured
    run artifacts. Later model training and distribution-shift metrics should
    attach to this same run directory contract.

    Args:
        config_path: Path to a TOML experiment config.
        output_root: Optional output root override, useful for tests.

    Returns:
        Path to the created run directory.

    Raises:
        ValueError: If dataset validation fails.
    """

    config = load_experiment_config(config_path)
    if output_root is not None:
        config = ExperimentConfig(
            name=config.name,
            seed=config.seed,
            output_root=output_root.resolve(),
            dataset=config.dataset,
        )

    random.seed(config.seed)
    run_dir = _create_run_dir(config)
    copy2(config_path, run_dir / "config.toml")

    validation = validate_csv_dataset(config.dataset)
    _write_json(run_dir / "validation.json", validation.to_jsonable())

    if not validation.is_valid:
        _write_log(
            run_dir,
            config,
            status="failed",
            details={"errors": list(validation.errors)},
        )
        formatted_errors = "; ".join(validation.errors)
        msg = f"Dataset validation failed: {formatted_errors}"
        raise ValueError(msg)

    metrics = {
        "experiment": config.name,
        "seed": config.seed,
        "dataset_rows": validation.rows,
        "split_counts": dict(validation.split_counts),
        "source_counts": dict(validation.source_counts),
        "label_counts": dict(validation.label_counts),
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "config.normalized.json", config.to_jsonable())
    _write_log(run_dir, config, status="completed", details=metrics)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    """Run the ShiftBench experiment CLI.

    Args:
        argv: Optional command-line arguments. Defaults to ``sys.argv``.

    Returns:
        Process exit code.
    """

    args = _parse_args(argv)
    try:
        run_dir = run_experiment(
            config_path=args.config,
            output_root=args.output_root,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(f"Run artifacts written to: {run_dir}")
    return 0


def _parse_args(argv: list[str] | None) -> Namespace:
    parser = ArgumentParser(description="Run a reproducible ShiftBench experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a TOML experiment config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the configured output root.",
    )
    return parser.parse_args(argv)


def _create_run_dir(config: ExperimentConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in config.name.lower()
    )
    run_dir = config.output_root / f"{timestamp}_{safe_name}_seed-{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _write_log(
    run_dir: Path,
    config: ExperimentConfig,
    status: str,
    details: dict[str, Any],
) -> None:
    log = {
        "status": status,
        "experiment": config.name,
        "seed": config.seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "details": details,
    }
    _write_json(run_dir / "logs.json", log)
    (run_dir / "logs.txt").write_text(
        "\n".join(
            [
                f"status={status}",
                f"experiment={config.name}",
                f"seed={config.seed}",
                f"python={platform.python_version()}",
                f"git_commit={log['git_commit']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
