"""Command-line runner for reproducible ShiftBench experiments."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any
import json
import platform
import numpy as np
import random
import subprocess
import sys
import torch

from shiftbench.config import ExperimentConfig, ShiftConfig, load_experiment_config
from shiftbench.datasets.schema import validate_csv_dataset
from shiftbench.metrics import SUMMARIES, Metric, get_metric
from shiftbench.provenance import (
    UNKNOWN,
    describe,
    describe_summary,
    require_comparable,
)


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

    # Load config and safe a copy of it in this run's folder
    config = load_experiment_config(config_path)
    if output_root is not None:
        config = replace(config, output_root=output_root.resolve())

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    run_dir = _create_run_dir(config)
    copy2(config_path, run_dir / "config.toml")

    # Validate CSV manifest and safe validation in this run's folder
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
        "train_selection": config.train_selection,
        "dataset_rows": validation.rows,
        "split_counts": dict(validation.split_counts),
        "source_counts": dict(validation.source_counts),
        "label_counts": dict(validation.label_counts),
    }

    # Compute the shift scores
    if config.shift is not None:
        try:
            metrics["shift"] = _compute_shift_metrics(config.shift, run_dir, split_column=config.dataset.split_column)
        except ValueError as error:
            _write_log(
                run_dir,
                config,
                status="failed",
                details={"errors": [str(error)]},
            )
            raise

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


def _compute_shift_metrics(shift: ShiftConfig, run_dir:Path, split_column:str) -> dict[str, Any]:
    """Compare two datasets with the configured metrics.

    Metrics are grouped by the input they declare, each input is materialized
    once, and every distance is recorded together with the provenance and
    hyperparameters it was computed under.

    Args:
        shift: Input sources and the distances to compute.
        run_dir: Path to folder to store run results.
        split_column: Name of the manifest column holding split information.

    Returns:
        The distances plus what produced them.

    Raises:
        ValueError: If an input is unreadable, the stats artifacts are not
            comparable, or a requested metric cannot run from these inputs.
    """
    summary_dir = run_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Get the distance measure for each metric requested
    grouped: dict[str, list[Metric]] = {}
    for name in shift.metrics:
        metric = get_metric(name)
        grouped.setdefault(metric.input, []).append(metric)

    report: dict[str, Any] = {"distances": {}, "params": {}}

    # Measure shift for selected embedding-based metrics
    if "embeddings" in grouped:
        try:
            stats_a = np.load(shift.stats_a)
            stats_b = np.load(shift.stats_b)
        except OSError as error:
            msg = f"Could not read shift stats: {error}"
            raise ValueError(msg) from error

        require_comparable(str(shift.stats_a), stats_a, str(shift.stats_b), stats_b)
        kind, _ = describe_summary(stats_a)
        report["encoder"], report["model"] = describe(stats_a)
        report["stats_a"] = str(shift.stats_a)
        report["stats_b"] = str(shift.stats_b)

        for metric in grouped["embeddings"]:
            if kind != UNKNOWN and kind != metric.summary:
                msg = (
                    f"Metric '{metric.name}' expects '{metric.summary}' "
                    f"summaries, these stats hold '{kind}'"
                )
                raise ValueError(msg)
            report["distances"][metric.name] = metric.compare(stats_a, stats_b)
            report["params"][metric.name] = dict(
                SUMMARIES[metric.summary].default_params
            )

    if "images" in grouped or "masks" in grouped or "image_dirs" in grouped:
        report["manifest_a"] = str(shift.manifest_a)
        report["manifest_b"] = str(shift.manifest_b)
        report["split_a"] = shift.split_a
        report["split_b"] = shift.split_b

    if "images" in grouped:
        images_a, images_b = _load_shift_images(shift, split_column)
        for metric in grouped["images"]:
            summary = SUMMARIES[metric.summary]

            # compute dataset histograms
            stats_a = summary.make(images_a)
            stats_b = summary.make(images_b)

            # save histograms
            np.savez(summary_dir / f"{metric.name}_a_summary.npz", **stats_a)
            np.savez(summary_dir / f"{metric.name}_b_summary.npz", **stats_b)

            # compute distance between histograms
            report["distances"][metric.name] = metric.compare(stats_a, stats_b)
            report["params"][metric.name] = {
                **summary.default_params,
                "image_column": shift.image_column,
            }

    if "masks" in grouped:
        masks_a, masks_b = _load_shift_masks(shift, split_column)
        for metric in grouped["masks"]:
            summary = SUMMARIES[metric.summary]

            # compute dataset histograms
            stats_a = summary.make(masks_a, num_classes=shift.num_classes)
            stats_b = summary.make(masks_b, num_classes=shift.num_classes)

            # save histograms
            np.savez(summary_dir / f"{metric.name}_a_summary.npz", **stats_a)
            np.savez(summary_dir / f"{metric.name}_b_summary.npz", **stats_b)

            # compute distance between histograms
            report["distances"][metric.name] = metric.compare(stats_a, stats_b)
            report["params"][metric.name] = {
                **summary.default_params,
                "mask_column": shift.mask_column,
                "num_classes": shift.num_classes,
            }

    if "image_dirs" in grouped:
        image_paths_a, image_paths_b = _load_sadge_image_paths(shift)
        for metric in grouped["image_dirs"]:
            try:
                value = metric.pairwise(image_paths_a, image_paths_b)
            except ModuleNotFoundError as error:
                msg = (
                    f"Metric '{metric.name}' needs the '[sadge]' extras "
                    f"(missing: {error.name})"
                )
                raise ValueError(msg) from error
            except RuntimeError as error:
                # e.g. the MASt3R submodule gating message, verbatim.
                raise ValueError(str(error)) from error
            report["distances"][metric.name] = value
            report["params"][metric.name] = dict(metric.params)

    return report


def _load_shift_images(shift: ShiftConfig, split_column: str) -> tuple[list, list]:
    from shiftbench.datasets.manifest import load_image_paths

    loaders = _import_loaders()
    try:
        return (
            loaders.load_rgb_images(
                load_image_paths(str(shift.manifest_a), shift.image_column, shift.split_a, split_column)
            ),
            loaders.load_rgb_images(
                load_image_paths(str(shift.manifest_b), shift.image_column, shift.split_b, split_column)
            ),
        )
    except OSError as error:
        msg = f"Could not read shift manifest data: {error}"
        raise ValueError(msg) from error


def _load_shift_masks(shift: ShiftConfig, split_column:str) -> tuple[list, list]:
    from shiftbench.datasets.manifest import load_mask_paths

    loaders = _import_loaders()
    try:
        return (
            loaders.load_masks(
                load_mask_paths(str(shift.manifest_a), shift.mask_column, shift.split_a, split_column)
            ),
            loaders.load_masks(
                load_mask_paths(str(shift.manifest_b), shift.mask_column, shift.split_b, split_column)
            ),
        )
    except OSError as error:
        msg = f"Could not read shift manifest data: {error}"
        raise ValueError(msg) from error


def _load_sadge_image_paths(shift: ShiftConfig, split_column:str) -> tuple[list[str], list[str]]:
    from shiftbench.datasets.manifest import load_image_paths
    image_paths_a = load_image_paths(
        str(shift.manifest_a),
        shift.image_column,
        shift.split_a,
        split_column,
    )
    image_paths_b = load_image_paths(
        str(shift.manifest_b),
        shift.image_column,
        shift.split_b,
        split_column,
    )
    return image_paths_a, image_paths_b


def _import_loaders():
    try:
        from shiftbench.datasets import loaders
    except ModuleNotFoundError as error:  # pragma: no cover - env specific
        msg = "Image and mask shift metrics need pillow; install '.[image]'"
        raise ValueError(msg) from error
    return loaders


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
