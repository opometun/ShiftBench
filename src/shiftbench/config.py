"""Configuration loading for ShiftBench experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import tomllib

from shiftbench.metrics import METRICS


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
        allowed_splits: Accepted split names.
        allowed_sources: Accepted data source names.
        text_column: Input text column, for text datasets.
        image_column: Image path column, for image datasets.
        mask_column: Semantic-mask path column, for label-based shift metrics.
        num_classes: Class count the masks are labeled with. Required whenever
            mask_column is set, so every label metric records the class count
            it assumed instead of falling back to a silent default.

    Exactly which of text_column and image_column is set depends on the
    dataset's modality. At least one must be, but neither is required on its
    own, so an image-only dataset does not have to invent a text column.
    mask_column is supplementary, not a modality on its own.
    """

    path: Path
    required_columns: tuple[str, ...]
    id_column: str
    split_column: str
    source_column: str
    label_column: str
    allowed_splits: tuple[str, ...]
    allowed_sources: tuple[str, ...]
    text_column: str | None = None
    image_column: str | None = None
    mask_column: str | None = None
    num_classes: int | None = None


@dataclass(frozen=True)
class ShiftConfig:
    """The two datasets to compare during a run, per metric input kind.

    Attributes:
        metrics: Names of the distances to compute, from shiftbench.metrics.
        stats_a: First .npz summary file, for embeddings metrics. Statistics
            are precomputed rather than extracted here so that running an
            experiment does not require torch.
        stats_b: Second .npz summary file.
        manifest_a: First CSV manifest, for image/mask metrics that read the
            raw data directly.
        manifest_b: Second CSV manifest.
        image_column: Manifest column holding image paths, for image metrics.
        mask_column: Manifest column holding mask paths, for mask metrics.
        num_classes: Class count of the masks, for mask metrics.
        image_dir_a: First image directory, for pairwise metrics (SADGE).
        image_dir_b: Second image directory.

    Which fields must be set follows from the requested metrics and is
    validated at load time, so a run fails at config parse rather than after
    work has started.
    """

    metrics: tuple[str, ...]
    stats_a: Path | None = None
    stats_b: Path | None = None
    manifest_a: Path | None = None
    manifest_b: Path | None = None
    image_column: str | None = None
    mask_column: str | None = None
    num_classes: int | None = None
    image_dir_a: Path | None = None
    image_dir_b: Path | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration for a reproducible ShiftBench run.

    Attributes:
        name: Human-readable experiment name used in run folders.
        seed: Random seed for deterministic experiment behavior.
        output_root: Root directory where run artifacts are written.
        dataset: Dataset schema and location.
        shift: Distribution-shift comparison to run, if any.
    """

    name: str
    seed: int
    output_root: Path
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
                "image_dir_a", "image_dir_b",
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
    text_column = _optional_str(dataset, "text_column")
    image_column = _optional_str(dataset, "image_column")
    mask_column = _optional_str(dataset, "mask_column")
    num_classes = _optional_int(dataset, "num_classes")
    allowed_splits = tuple(_required_list(dataset, "allowed_splits"))
    allowed_sources = tuple(_required_list(dataset, "allowed_sources"))

    if text_column is None and image_column is None:
        msg = "dataset must set text_column, image_column, or both"
        raise ValueError(msg)
    if mask_column is not None and num_classes is None:
        msg = "Setting mask_column requires num_classes"
        raise ValueError(msg)
    if num_classes is not None and mask_column is None:
        msg = "num_classes is set but mask_column is not"
        raise ValueError(msg)

    schema_columns = {
        id_column,
        split_column,
        source_column,
        label_column,
    }
    schema_columns.update(
        column
        for column in (text_column, image_column, mask_column)
        if column is not None
    )
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
        image_column=image_column,
        mask_column=mask_column,
        num_classes=num_classes,
    )
    return ExperimentConfig(
        name=name,
        seed=seed,
        output_root=output_root,
        dataset=dataset_config,
        shift=_shift_from_config(config_path, raw_config.get("shift")),
    )


def _shift_from_config(
    config_path: Path,
    raw_shift: Any,
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
        image_column=_optional_str(raw_shift, "image_column"),
        mask_column=_optional_str(raw_shift, "mask_column"),
        num_classes=_optional_int(raw_shift, "num_classes"),
        image_dir_a=optional_path("image_dir_a"),
        image_dir_b=optional_path("image_dir_b"),
    )

    if (shift.stats_a is None) != (shift.stats_b is None):
        msg = "stats_a and stats_b must be set together"
        raise ValueError(msg)
    if (shift.manifest_a is None) != (shift.manifest_b is None):
        msg = "manifest_a and manifest_b must be set together"
        raise ValueError(msg)
    if (shift.image_dir_a is None) != (shift.image_dir_b is None):
        msg = "image_dir_a and image_dir_b must be set together"
        raise ValueError(msg)

    for name in metrics:
        metric = METRICS[name]
        if metric.summary is None and shift.image_dir_a is None:
            msg = f"Metric '{name}' needs image_dir_a and image_dir_b"
            raise ValueError(msg)
        if metric.input == "embeddings" and shift.stats_a is None:
            msg = f"Metric '{name}' needs stats_a and stats_b"
            raise ValueError(msg)
        if metric.input in ("images", "masks") and shift.manifest_a is None:
            msg = f"Metric '{name}' needs manifest_a and manifest_b"
            raise ValueError(msg)
        if metric.input == "masks" and (
            shift.mask_column is None or shift.num_classes is None
        ):
            msg = f"Metric '{name}' needs mask_column and num_classes"
            raise ValueError(msg)

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
