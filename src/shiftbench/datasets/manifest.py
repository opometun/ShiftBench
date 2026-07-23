"""Reading image and mask paths out of a ShiftBench CSV manifest.

Paths are resolved relative to the manifest file, so a manifest stays portable
as long as it sits next to the files it points at.
"""

from __future__ import annotations

import csv
import os

from shiftbench.config import DatasetSchemaConfig

FALLBACK_IMAGE_COLUMNS = ("image_path", "file_path", "filepath", "path", "image")


def resolve_image_path(manifest_dir: str, raw_path: str) -> str:
    """Resolve one manifest cell into an absolute file path.

    Shared with schema validation so that what validation checks for is
    exactly what extraction later opens.
    """
    image_path = os.path.expanduser(raw_path.strip())
    if not os.path.isabs(image_path):
        image_path = os.path.join(manifest_dir, image_path)
    return image_path


def load_image_paths_from_config(schema: DatasetSchemaConfig) -> list[str]:
    """Read image paths using a configured dataset schema.

    This is the path that makes image_column load-bearing: the column named in
    the TOML is the column actually read, with no guessing in between.

    Raises:
        ValueError: If the schema declares no image_column and the manifest
            has no recognizable one either.
    """
    return load_image_paths(str(schema.path), schema.image_column)


def load_mask_paths_from_config(schema: DatasetSchemaConfig) -> list[str]:
    """Read semantic-mask paths using a configured dataset schema.

    Raises:
        ValueError: If the schema declares no mask_column.
    """
    if schema.mask_column is None:
        msg = "Dataset schema declares no mask_column"
        raise ValueError(msg)
    return load_mask_paths(str(schema.path), schema.mask_column)


def load_image_paths(
    input_manifest: str,
    image_column: str | None = None,
) -> list[str]:
    """Read absolute image paths from a CSV manifest.

    Args:
        input_manifest: Path to the CSV manifest.
        image_column: Column holding image paths. When omitted, falls back to
            guessing among FALLBACK_IMAGE_COLUMNS, which is only safe for
            manifests written outside a configured experiment.

    Returns:
        Absolute image paths, in manifest row order.

    Raises:
        ValueError: If the manifest is empty, no image column is found, or a
            row is missing its path.
    """
    return _load_column_paths(
        input_manifest, image_column, FALLBACK_IMAGE_COLUMNS, "image"
    )


def load_mask_paths(input_manifest: str, mask_column: str) -> list[str]:
    """Read absolute semantic-mask paths from a CSV manifest.

    Unlike images, masks have no fallback guessing: the column must be named
    explicitly, normally via the config's mask_column.

    Args:
        input_manifest: Path to the CSV manifest.
        mask_column: Column holding mask paths.

    Returns:
        Absolute mask paths, in manifest row order.

    Raises:
        ValueError: If the manifest is empty, the column is absent, or a row
            is missing its path.
    """
    return _load_column_paths(input_manifest, mask_column, None, "mask")


def _load_column_paths(
    input_manifest: str,
    column: str | None,
    fallback_columns: tuple[str, ...] | None,
    kind: str,
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

        paths = []
        for row_number, row in enumerate(reader, start=2):
            if column_index >= len(row) or not row[column_index].strip():
                raise ValueError(f"Row {row_number}: missing {kind} path")
            paths.append(resolve_image_path(manifest_dir, row[column_index]))

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
