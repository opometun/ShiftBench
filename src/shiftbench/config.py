"""Configuration loading for ShiftBench experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class DatasetSchemaConfig:
    """Dataset schema expected by a ShiftBench experiment.

    Attributes:
        path: CSV dataset path.
        required_columns: Columns that must exist in the CSV file.
        id_column: Unique row identifier column.
        split_column: Dataset split column.
        source_column: Data source column, for example real or synthetic.
        label_column: Target label column.
        text_column: Input text column.
        allowed_splits: Accepted split names.
        allowed_sources: Accepted data source names.
    """

    path: Path
    required_columns: tuple[str, ...]
    id_column: str
    split_column: str
    source_column: str
    label_column: str
    text_column: str
    allowed_splits: tuple[str, ...]
    allowed_sources: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration for a reproducible ShiftBench run.

    Attributes:
        name: Human-readable experiment name used in run folders.
        seed: Random seed for deterministic experiment behavior.
        output_root: Root directory where run artifacts are written.
        dataset: Dataset schema and location.
    """

    name: str
    seed: int
    output_root: Path
    dataset: DatasetSchemaConfig

    def to_jsonable(self) -> dict[str, Any]:
        """Return the config as JSON-serializable data."""

        data = asdict(self)
        data["output_root"] = str(self.output_root)
        data["dataset"]["path"] = str(self.dataset.path)
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

    config_path = path.resolve()
    with config_path.open("rb") as file:
        raw_config = tomllib.load(file)

    experiment = _required_table(raw_config, "experiment")
    dataset = _required_table(raw_config, "dataset")

    name = _required_str(experiment, "name")
    seed = _required_int(experiment, "seed")
    output_root = _path_from_config(config_path, _required_str(experiment, "output_root"))

    required_columns = tuple(_required_list(dataset, "required_columns"))
    id_column = _required_str(dataset, "id_column")
    split_column = _required_str(dataset, "split_column")
    source_column = _required_str(dataset, "source_column")
    label_column = _required_str(dataset, "label_column")
    text_column = _required_str(dataset, "text_column")
    allowed_splits = tuple(_required_list(dataset, "allowed_splits"))
    allowed_sources = tuple(_required_list(dataset, "allowed_sources"))

    schema_columns = {
        id_column,
        split_column,
        source_column,
        label_column,
        text_column,
    }
    missing_schema_columns = schema_columns.difference(required_columns)
    if missing_schema_columns:
        formatted = ", ".join(sorted(missing_schema_columns))
        msg = f"required_columns must include schema columns: {formatted}"
        raise ValueError(msg)

    dataset_config = DatasetSchemaConfig(
        path=_path_from_config(config_path, _required_str(dataset, "path")),
        required_columns=required_columns,
        id_column=id_column,
        split_column=split_column,
        source_column=source_column,
        label_column=label_column,
        text_column=text_column,
        allowed_splits=allowed_splits,
        allowed_sources=allowed_sources,
    )
    return ExperimentConfig(
        name=name,
        seed=seed,
        output_root=output_root,
        dataset=dataset_config,
    )


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
