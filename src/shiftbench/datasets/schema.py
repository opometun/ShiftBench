"""Dataset schema validation for ShiftBench CSV inputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import csv

from shiftbench.config import DatasetSchemaConfig
from shiftbench.datasets.manifest import resolve_image_path


@dataclass(frozen=True)
class DatasetValidationResult:
    """Summary of a dataset schema validation run.

    Attributes:
        rows: Number of data rows checked.
        split_counts: Row counts by split.
        source_counts: Row counts by data source.
        label_counts: Row counts by label.
        errors: Validation errors. Empty when the dataset is valid.
    """

    rows: int
    split_counts: Mapping[str, int]
    source_counts: Mapping[str, int]
    label_counts: Mapping[str, int]
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether the dataset passed all validation checks."""

        return not self.errors

    def to_jsonable(self) -> dict[str, object]:
        """Return the validation result as JSON-serializable data."""

        return {
            "rows": self.rows,
            "split_counts": dict(self.split_counts),
            "source_counts": dict(self.source_counts),
            "label_counts": dict(self.label_counts),
            "errors": list(self.errors),
            "is_valid": self.is_valid,
        }


def validate_csv_dataset(schema: DatasetSchemaConfig) -> DatasetValidationResult:
    """Validate a CSV dataset against the configured ShiftBench schema.

    Args:
        schema: Dataset schema and path configuration.

    Returns:
        Validation result containing errors and basic dataset counts.
    """

    errors: list[str] = []
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    rows = 0

    if not schema.path.exists():
        return DatasetValidationResult(
            rows=0,
            split_counts={},
            source_counts={},
            label_counts={},
            errors=(f"Dataset file does not exist: {schema.path}",),
        )

    with schema.path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = sorted(set(schema.required_columns).difference(fieldnames))
        if missing_columns:
            formatted = ", ".join(missing_columns)
            errors.append(f"Missing required columns: {formatted}")

        for row_index, row in enumerate(reader, start=2):
            rows += 1
            _validate_required_values(row, row_index, schema, errors)
            _validate_unique_id(row, row_index, schema, seen_ids, errors)
            _validate_allowed_value(
                row=row,
                row_index=row_index,
                column=schema.split_column,
                allowed_values=schema.allowed_splits,
                errors=errors,
            )
            _validate_allowed_value(
                row=row,
                row_index=row_index,
                column=schema.source_column,
                allowed_values=schema.allowed_sources,
                errors=errors,
            )
            _validate_image_exists(row, row_index, schema, errors)

            split_counts[row.get(schema.split_column, "")] += 1
            source_counts[row.get(schema.source_column, "")] += 1
            label_counts[row.get(schema.label_column, "")] += 1

    if rows == 0:
        errors.append("Dataset contains no data rows")

    return DatasetValidationResult(
        rows=rows,
        split_counts=dict(sorted(split_counts.items())),
        source_counts=dict(sorted(source_counts.items())),
        label_counts=dict(sorted(label_counts.items())),
        errors=tuple(errors),
    )


def _validate_required_values(
    row: dict[str, str],
    row_index: int,
    schema: DatasetSchemaConfig,
    errors: list[str],
) -> None:
    for column in schema.required_columns:
        if not row.get(column, "").strip():
            errors.append(f"Row {row_index}: missing value for column '{column}'")


def _validate_unique_id(
    row: dict[str, str],
    row_index: int,
    schema: DatasetSchemaConfig,
    seen_ids: set[str],
    errors: list[str],
) -> None:
    sample_id = row.get(schema.id_column, "").strip()
    if not sample_id:
        return
    if sample_id in seen_ids:
        errors.append(f"Row {row_index}: duplicate id '{sample_id}'")
    seen_ids.add(sample_id)


def _validate_image_exists(
    row: dict[str, str],
    row_index: int,
    schema: DatasetSchemaConfig,
    errors: list[str],
) -> None:
    """Check that a declared image path points at a file that exists.

    Without this a config validates cleanly and then fails partway through
    feature extraction, after the model has already been loaded.
    """
    if schema.image_column is None:
        return

    raw_path = row.get(schema.image_column, "").strip()
    if not raw_path:
        return

    manifest_dir = str(schema.path.parent)
    if not Path(resolve_image_path(manifest_dir, raw_path)).exists():
        errors.append(f"Row {row_index}: image not found: {raw_path}")


def _validate_allowed_value(
    row: dict[str, str],
    row_index: int,
    column: str,
    allowed_values: tuple[str, ...],
    errors: list[str],
) -> None:
    value = row.get(column, "").strip()
    if value and value not in allowed_values:
        allowed = ", ".join(allowed_values)
        errors.append(f"Row {row_index}: invalid {column} '{value}', expected {allowed}")
