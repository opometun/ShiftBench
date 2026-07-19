"""Reading image paths out of a ShiftBench CSV manifest.

Paths are resolved relative to the manifest file, so a manifest stays portable
as long as it sits next to the images it points at.
"""

from __future__ import annotations

import csv
import os

from shiftbench.config import DatasetSchemaConfig

FALLBACK_IMAGE_COLUMNS = ("image_path", "file_path", "filepath", "path", "image")


def resolve_image_path(manifest_dir: str, raw_path: str) -> str:
    """Resolve one manifest cell into an absolute image path.

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
    manifest_path = os.path.abspath(os.path.expanduser(input_manifest))
    manifest_dir = os.path.dirname(manifest_path)

    with open(manifest_path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input manifest is empty: {input_manifest}") from exc

        column_index = _resolve_image_column(header, image_column, input_manifest)

        image_paths = []
        for row_number, row in enumerate(reader, start=2):
            if column_index >= len(row):
                raise ValueError(f"Row {row_number}: missing image path")
            if not row[column_index].strip():
                raise ValueError(f"Row {row_number}: missing image path")
            image_paths.append(resolve_image_path(manifest_dir, row[column_index]))

    return image_paths


def _resolve_image_column(
    header: list[str],
    image_column: str | None,
    input_manifest: str,
) -> int:
    column_by_name = {name.strip(): index for index, name in enumerate(header)}

    if image_column is not None:
        if image_column not in column_by_name:
            raise ValueError(
                f"Manifest {input_manifest} has no column '{image_column}'"
            )
        return column_by_name[image_column]

    for candidate in FALLBACK_IMAGE_COLUMNS:
        if candidate in column_by_name:
            return column_by_name[candidate]

    expected = ", ".join(FALLBACK_IMAGE_COLUMNS)
    raise ValueError(f"Input manifest must contain an image path column: {expected}")
