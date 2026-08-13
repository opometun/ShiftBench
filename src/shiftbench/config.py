"""Configuration loading for ShiftBench experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import tomllib
import warnings

from shiftbench.metrics import METRICS


@dataclass(frozen=True)
class DatasetSchemaConfig:
    """Dataset schema expected by a ShiftBench experiment.

    Attributes:
        path: CSV dataset path.
        required_columns: List of column names that must exist in the CSV file.
        id_column: Name of column containing unique row identifiers.
        split_column: Name of column containing dataset split information.
        source_column: Name of column containing data source information.
        input_column: Name of column containing the input (or path to input).
        label_column: Name of column containing the label (or path to label).
        allowed_splits: List of accepted split names.
        allowed_sources: List of accepted data source names.
        num_classes (optional): Class count. Please provide it for classification applications.
        description_column (optional): Name of column to store additional descriptive information.
    """

    path: Path
    required_columns: tuple[str, ...]
    id_column: str
    split_column: str
    source_column: str
    input_column: str
    label_column: str
    allowed_splits: tuple[str, ...]
    allowed_sources: tuple[str, ...]
    num_classes: int | None = None
    description_column: str | None = None


@dataclass(frozen=True)
class ShiftConfig:
    """The two datasets to compare during a run, per metric input kind.

    Attributes:
        metrics: Names of the distances to compute, from shiftbench.metrics.
        stats_a (optional): Path to .npz summary file that stores precomputed statistics 
            of dataset A. Required for embedding-based metrics.
        stats_b (optional): Path to .npz summary file that stores precomputed statistics 
            of dataset B. Required for embedding-based metrics.
        manifest_a (optional): Path to CSV manifest of dataset A.
            Required for image- or mask-based metrics.
        manifest_b (optional): Path to CSV manifest of dataset B.
            Required for image- or mask-based metrics.
        split_a (optional): The split of selected data samples from dataset A.
            If not provided, then all data samples will be considered.
        split_b (optional): The split of selected data samples from dataset B.
            If not provided, then all data samples will be considered.
        image_column (optional): Name of manifest column holding image paths.
            Required for image-based metrics.
        mask_column (optional): Name of manifest column holding mask paths.
            Required for mask-based metrics.
        num_classes (optional): Class count of the masks.
            Required for mask-based metrics.

    Which fields must be set follows from the requested metrics and is
    validated at load time, so a run fails at config parse rather than after
    work has started.
    """

    metrics: tuple[str, ...]
    stats_a: Path | None = None
    stats_b: Path | None = None
    manifest_a: Path | None = None
    manifest_b: Path | None = None
    split_a: str | None = None
    split_b: str | None = None
    image_column: str | None = None
    mask_column: str | None = None
    num_classes: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration for a reproducible ShiftBench run.

    Attributes:
        name: Human-readable experiment name used in run folders.
        seed: Random seed for deterministic experiment behavior.
        output_root: Root directory where run artifacts are written.
        train_selection: Training data composition of data sources 
            and their number of samples.
        dataset: Dataset schema and location.
        shift (optional): Distribution-shift comparison to run, if any.
    """

    name: str
    seed: int
    output_root: Path
    train_selection: dict
    dataset: DatasetSchemaConfig
    shift: ShiftConfig | None = None

    def to_jsonable(self) -> dict[str, Any]:
        """Return the config as JSON-serializable data."""

        data = asdict(self)
        data["output_root"] = str(self.output_root)
        data["dataset"]["path"] = str(self.dataset.path)
        if self.shift is not None:
            path_keys = (
                "stats_a", "stats_b", "manifest_a", "manifest_b",
            )
            for key in path_keys:
                value = getattr(self.shift, key)
                data["shift"][key] = None if value is None else str(value)
        return data


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration file.

    Args:
        path: TOML configuration file path.

    Returns:
        Parsed experiment configuration.

    Raises:
        ValueError: If required config fields are missing or invalid.
    """
    # Load .toml file
    config_path = path.resolve()
    with config_path.open("rb") as file:
        raw_config = tomllib.load(file)

    # Check whether all arguments are valid
    experiment = _required_table(raw_config, "experiment")
    dataset = _required_table(raw_config, "dataset")

    name = _required_str(experiment, "name")
    seed = _required_int(experiment, "seed")
    output_root = _path_from_config(config_path, _required_str(experiment, "output_root"))
    train_selection = _required_dict(experiment, "train_selection")

    required_columns = tuple(_required_list(dataset, "required_columns"))
    id_column = _required_str(dataset, "id_column")
    split_column = _required_str(dataset, "split_column")
    source_column = _required_str(dataset, "source_column")
    input_column = _required_str(dataset, "input_column")
    label_column = _required_str(dataset, "label_column")
    num_classes = _optional_int(dataset, "num_classes")
    description_column = _optional_str(dataset, "description_column")
    allowed_splits = tuple(_required_list(dataset, "allowed_splits"))
    allowed_sources = tuple(_required_list(dataset, "allowed_sources"))

    # Check whether train_selection lists invalid sources
    invalid_sources = set(train_selection.keys()) - set(allowed_sources)
    if invalid_sources:
        raise ValueError(
            f"train_selection contains unknown sources: {', '.join(invalid_sources)}"
        )

    # Check whether required_columns contains all schema columns
    schema_columns = {
        id_column,
        split_column,
        source_column,
        label_column,
        input_column,
    }
    missing_schema_columns = schema_columns.difference(required_columns)
    if missing_schema_columns:
        formatted = ", ".join(sorted(missing_schema_columns))
        msg = f"required_columns must include schema columns: {formatted}"
        raise ValueError(msg)

    # Get DatasetSchemaConfig and final ExperimentConfig
    dataset_config = DatasetSchemaConfig(
        path=_path_from_config(config_path, _required_str(dataset, "path")),
        required_columns=required_columns,
        id_column=id_column,
        split_column=split_column,
        source_column=source_column,
        label_column=label_column,
        input_column=input_column,
        allowed_splits=allowed_splits,
        allowed_sources=allowed_sources, 
        num_classes=num_classes,
        description_column=description_column,
    )
    return ExperimentConfig(
        name=name,
        seed=seed,
        output_root=output_root,
        train_selection=train_selection,
        dataset=dataset_config,
        shift=_shift_from_config(config_path, raw_config.get("shift"), split_column),
    )


def _shift_from_config(
    config_path: Path,
    raw_shift: Any,
    dataset_split_column: str,
) -> ShiftConfig | None:
    """Parse the optional [shift] table.

    Validates metric names and, per requested metric, that the inputs it needs
    are present — so a bad shift setup fails at load, not mid-run.
    """
    if raw_shift is None:
        return None
    if not isinstance(raw_shift, dict):
        msg = "[shift] must be a table"
        raise ValueError(msg)

    metrics = tuple(_required_list(raw_shift, "metrics"))
    unknown = sorted(set(metrics).difference(METRICS))
    if unknown:
        available = ", ".join(sorted(METRICS))
        formatted = ", ".join(unknown)
        msg = f"Unknown shift metrics: {formatted}. Available: {available}"
        raise ValueError(msg)

    def optional_path(key: str) -> Path | None:
        value = _optional_str(raw_shift, key)
        return None if value is None else _path_from_config(config_path, value)

    shift = ShiftConfig(
        metrics=metrics,
        stats_a=optional_path("stats_a"),
        stats_b=optional_path("stats_b"),
        manifest_a=optional_path("manifest_a"),
        manifest_b=optional_path("manifest_b"),
        split_a=_optional_str(raw_shift, "split_a"),
        split_b=_optional_str(raw_shift, "split_b"),
        image_column=_optional_str(raw_shift, "image_column"),
        mask_column=_optional_str(raw_shift, "mask_column"),
        num_classes=_optional_int(raw_shift, "num_classes"),
    )

    if (shift.stats_a is None) != (shift.stats_b is None):
        msg = "stats_a and stats_b must be set together"
        raise ValueError(msg)
    if (shift.manifest_a is None) != (shift.manifest_b is None):
        msg = "manifest_a and manifest_b must be set together"
        raise ValueError(msg)

    for name in metrics:
        metric = METRICS[name]
        if metric.input == "embeddings":
            if (shift.stats_a is None) or (shift.stats_b is None):
                msg = f"Metric '{name}' needs stats_a and stats_b"
                raise ValueError(msg)
        if metric.input in ("images", "masks", "image_dirs"): 
            if (shift.manifest_a is None) or (shift.manifest_b is None):
                msg = f"Metric '{name}' needs manifest_a and manifest_b"
                raise ValueError(msg)
        if metric.input == "masks" and (
            shift.mask_column is None or shift.num_classes is None
        ):
            msg = f"Metric '{name}' needs mask_column and num_classes"
            raise ValueError(msg)

    if (shift.split_a is not None) and (shift.manifest_a is None):
        warnings.warn(
            "split_a is ignored because metrics require no manifest_a",
            category=UserWarning,
        )
    if (shift.split_b is not None) and (shift.manifest_b is None):
        warnings.warn(
            "split_b is ignored because metrics require no manifest_b",
            category=UserWarning,
        )

    return shift


def _path_from_config(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _required_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        msg = f"Missing [{key}] table"
        raise ValueError(msg)
    return value


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"Missing non-empty string field: {key}"
        raise ValueError(msg)
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"Field must be a non-empty string when set: {key}"
        raise ValueError(msg)
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        msg = f"Field must be a positive integer when set: {key}"
        raise ValueError(msg)
    return value


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        msg = f"Missing integer field: {key}"
        raise ValueError(msg)
    return value


def _required_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        msg = f"Missing non-empty list field: {key}"
        raise ValueError(msg)
    if not all(isinstance(item, str) and item.strip() for item in value):
        msg = f"Field must contain only non-empty strings: {key}"
        raise ValueError(msg)
    return value


def _required_dict(data: dict[str, Any], key: str) -> dict[str, int]:
    value = data.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Missing non-empty dict field: {key}")
    for k, v in value.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"train_selection keys must be non-empty strings: {k}")
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"train_selection values must be positive integers: {k} -> {v}")
    return value