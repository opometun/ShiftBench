"""Reading image and mask paths out of a ShiftBench CSV manifest.

Paths are resolved relative to the manifest file, so a manifest stays portable
as long as it sits next to the files it points at.
"""

from __future__ import annotations

import csv
import os

from shiftbench.config import DatasetSchemaConfig

# Note: Fallbacks are only relevant for non-configured experiments
# since we strictly require input_column and label_column in configured experiments
FALLBACK_IMAGE_COLUMNS = ("img_path", "image_path", "image", "input_path", "input")
FALLBACK_MASK_COLUMNS = ("mask_path", "mask", "label_path", "label")


def resolve_image_path(manifest_dir: str, raw_path: str) -> str:
    """Resolve one manifest cell into an absolute file path.

    Shared with schema validation so that what validation checks for is
    exactly what extraction later opens.

    Manifest cells may use either '/' or '\\' as a separator (the tracked
    study CSVs were generated on Windows), so backslashes are normalized to
    the platform separator before joining. This assumes paths never contain
    a literal backslash in a filename, which holds for this project.
    """
    normalized = raw_path.strip().replace("\\", os.sep)
    image_path = os.path.expanduser(normalized)
    if not os.path.isabs(image_path):
        image_path = os.path.join(manifest_dir, image_path)
    return image_path


def load_image_paths_from_config(
    schema: DatasetSchemaConfig,
    split: str | None = None,
) -> list[str]:
    """Read image paths using a configured dataset schema.

    This is the path that makes input_column load-bearing: the column named in
    the TOML is the column actually read, with no guessing in between.

    Raises:
        ValueError: If the schema declares no input_column and the manifest
            has no recognizable one either.
    """
    return load_image_paths(str(schema.path), schema.input_column, split, schema.split_column)


def load_mask_paths_from_config(
    schema: DatasetSchemaConfig,
    split: str | None = None,
) -> list[str]:
    """Read semantic-mask paths using a configured dataset schema.

    Raises:
        ValueError: If the schema declares no label_column and the manifest
            has no recognizable one either.
    """
    return load_mask_paths(str(schema.path), schema.label_column, split, schema.split_column)


def load_image_paths(
    input_manifest: str,
    image_column: str | None = None,
    split: str | None = None,
    split_column: str = "split",
) -> list[str]:
    """Read absolute image paths from a CSV manifest.

    Args:
        input_manifest: Path to the CSV manifest.
        image_column: Column holding image paths. When omitted, falls back to
            guessing among FALLBACK_IMAGE_COLUMNS, which is only safe for
            manifests written outside a configured experiment.
        split: If not None, only paths for this split are loaded.
        split_column: Column holding split information.

    Returns:
        Absolute image paths, in manifest row order.

    Raises:
        ValueError: If the manifest is empty, no image column is found, or a
            row is missing its path.
    """
    return _load_column_paths(
        input_manifest, image_column, FALLBACK_IMAGE_COLUMNS, "image", split, split_column
    )


def load_mask_paths(
    input_manifest: str, 
    mask_column: str,
    split: str | None = None,
    split_column: str = "split",
) -> list[str]:
    """Read absolute semantic-mask paths from a CSV manifest.

    Args:
        input_manifest: Path to the CSV manifest.
        mask_column: Column holding mask paths. When omitted, falls back to
            guessing among FALLBACK_MASK_COLUMNS, which is only safe for
            manifests written outside a configured experiment.
        split: If not None, only paths for this split are loaded.
        split_column: Column holding split information.

    Returns:
        Absolute mask paths, in manifest row order.

    Raises:
        ValueError: If the manifest is empty, no mask column is found, or a row
            is missing its path.
    """
    return _load_column_paths(
        input_manifest, mask_column, FALLBACK_MASK_COLUMNS, "mask", split, split_column
    )


def _load_column_paths(
    input_manifest: str,
    column: str | None,
    fallback_columns: tuple[str, ...] | None,
    kind: str,
    split: str | None = None,
    split_column: str = "split",
) -> list[str]:
    manifest_path = os.path.abspath(os.path.expanduser(input_manifest))
    manifest_dir = os.path.dirname(manifest_path)

    with open(manifest_path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input manifest is empty: {input_manifest}") from exc

        column_index = _resolve_column(
            header, column, fallback_columns, input_manifest, kind
        )

        if split is not None:
            if not split_column:
                raise ValueError("split_column must be provided when split filtering is used")
            split_index = _resolve_column(
                header, split_column, None, input_manifest, "split",
            )

        paths = []
        for row_number, row in enumerate(reader, start=2):
            # If split filtering is requested
            if (split is not None) and (row[split_index] != split):
                continue
            # Get path
            if column_index >= len(row) or not row[column_index].strip():
                raise ValueError(f"Row {row_number}: missing {kind} path")
            paths.append(resolve_image_path(manifest_dir, row[column_index]))

        if not paths:
            raise ValueError(f"No rows matched the request in manifest '{input_manifest}'.")
    return paths


def _resolve_column(
    header: list[str],
    column: str | None,
    fallback_columns: tuple[str, ...] | None,
    input_manifest: str,
    kind: str,
) -> int:
    column_by_name = {name.strip(): index for index, name in enumerate(header)}

    if column is not None:
        if column not in column_by_name:
            raise ValueError(f"Manifest {input_manifest} has no column '{column}'")
        return column_by_name[column]

    for candidate in fallback_columns or ():
        if candidate in column_by_name:
            return column_by_name[candidate]

    expected = ", ".join(fallback_columns or ())
    raise ValueError(f"Input manifest must contain an {kind} path column: {expected}")
