from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.config import DatasetSchemaConfig
from shiftbench.datasets.manifest import (
    FALLBACK_IMAGE_COLUMNS,
    load_image_paths,
    load_image_paths_from_config,
    load_mask_paths,
    load_mask_paths_from_config,
)


def _schema(
    path: Path,
    image_column: str | None,
    mask_column: str | None = None,
    num_classes: int | None = None,
) -> DatasetSchemaConfig:
    return DatasetSchemaConfig(
        path=path,
        required_columns=("sample_id", "split", "source", "label", "text"),
        id_column="sample_id",
        split_column="split",
        source_column="source",
        label_column=mask_column,
        input_column=image_column,
        allowed_splits=("train",),
        allowed_sources=("real",),
        num_classes=num_classes,
    )


class ManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def _write(self, name: str, body: str) -> Path:
        path = self.directory / name
        path.write_text(body.lstrip(), encoding="utf-8")
        return path

    def test_finds_column_by_fallback_guessing(self) -> None:
        manifest = self._write(
            "guess.csv",
            """
id,image_path
0,a.png
1,b.png
""",
        )

        paths = load_image_paths(str(manifest))

        self.assertEqual([os.path.basename(path) for path in paths], ["a.png", "b.png"])

    def test_accepts_every_documented_fallback_name(self) -> None:
        for column in FALLBACK_IMAGE_COLUMNS:
            with self.subTest(column=column):
                manifest = self._write(f"{column}.csv", f"id,{column}\n0,a.png\n")

                paths = load_image_paths(str(manifest))

                self.assertEqual(len(paths), 1)

    def test_explicit_column_outside_the_fallback_list(self) -> None:
        manifest = self._write("explicit.csv", "id,picture_uri\n0,a.png\n")

        paths = load_image_paths(str(manifest), "picture_uri")

        self.assertEqual(os.path.basename(paths[0]), "a.png")
        # The same file is unreadable without being told the column.
        with self.assertRaises(ValueError):
            load_image_paths(str(manifest))

    def test_explicit_column_that_is_absent_raises(self) -> None:
        manifest = self._write("guess.csv", "id,image_path\n0,a.png\n")

        with self.assertRaises(ValueError) as caught:
            load_image_paths(str(manifest), "nope")

        self.assertIn("nope", str(caught.exception))

    def test_no_image_column_raises(self) -> None:
        manifest = self._write("text.csv", "sample_id,label,text\n0,positive,hello\n")

        with self.assertRaises(ValueError) as caught:
            load_image_paths(str(manifest))

        self.assertIn("image path column", str(caught.exception))

    def test_empty_manifest_raises(self) -> None:
        manifest = self._write("empty.csv", "")

        with self.assertRaises(ValueError) as caught:
            load_image_paths(str(manifest))

        self.assertIn("empty", str(caught.exception))

    def test_relative_paths_resolve_against_the_manifest_directory(self) -> None:
        nested = self.directory / "nested"
        nested.mkdir()
        manifest = nested / "m.csv"
        manifest.write_text("id,image_path\n0,img/a.png\n", encoding="utf-8")

        paths = load_image_paths(str(manifest))

        self.assertEqual(paths[0], str(nested / "img" / "a.png"))

    def test_absolute_paths_are_left_alone(self) -> None:
        absolute = self.directory / "elsewhere" / "a.png"
        manifest = self._write("abs.csv", f"id,image_path\n0,{absolute}\n")

        paths = load_image_paths(str(manifest))

        self.assertEqual(paths[0], str(absolute))

    def test_blank_path_raises_with_row_number(self) -> None:
        manifest = self._write("blank.csv", "id,image_path\n0,a.png\n1,\n")

        with self.assertRaises(ValueError) as caught:
            load_image_paths(str(manifest))

        self.assertIn("Row 3", str(caught.exception))

    def test_short_row_raises_with_row_number(self) -> None:
        manifest = self._write("short.csv", "id,image_path\n0,a.png\n1\n")

        with self.assertRaises(ValueError) as caught:
            load_image_paths(str(manifest))

        self.assertIn("Row 3", str(caught.exception))

    def test_preserves_row_order(self) -> None:
        rows = "\n".join(f"{i},img{i}.png" for i in range(10))
        manifest = self._write("order.csv", f"id,image_path\n{rows}\n")

        paths = load_image_paths(str(manifest))

        self.assertEqual(
            [os.path.basename(path) for path in paths],
            [f"img{i}.png" for i in range(10)],
        )

    def test_windows_backslash_paths_resolve(self) -> None:
        manifest = self._write("win.csv", "id,image_path\n0,img\\a.png\n")

        paths = load_image_paths(str(manifest))

        self.assertEqual(paths[0], str(self.directory / "img" / "a.png"))


class ManifestFromConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def test_configured_column_is_the_column_actually_read(self) -> None:
        manifest = self.directory / "m.csv"
        # Two plausible columns: a decoy the fallback would grab first, and
        # the one the config names. The config must win.
        manifest.write_text(
            "sample_id,image_path,scene_uri\n0,decoy.png,real.png\n",
            encoding="utf-8",
        )

        paths = load_image_paths_from_config(_schema(manifest, "scene_uri"))

        self.assertEqual(os.path.basename(paths[0]), "real.png")

    def test_falls_back_to_guessing_when_config_names_no_column(self) -> None:
        manifest = self.directory / "m.csv"
        manifest.write_text("sample_id,image_path\n0,a.png\n", encoding="utf-8")

        paths = load_image_paths_from_config(_schema(manifest, None))

        self.assertEqual(os.path.basename(paths[0]), "a.png")

    def test_configured_column_missing_from_manifest_raises(self) -> None:
        manifest = self.directory / "m.csv"
        manifest.write_text("sample_id,image_path\n0,a.png\n", encoding="utf-8")

        with self.assertRaises(ValueError) as caught:
            load_image_paths_from_config(_schema(manifest, "scene_uri"))

        self.assertIn("scene_uri", str(caught.exception))


class MaskManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def _write(self, body: str) -> Path:
        path = self.directory / "m.csv"
        path.write_text(body.lstrip(), encoding="utf-8")
        return path

    def test_explicit_column_resolves_relative_paths(self) -> None:
        manifest = self._write("id,seg_path\n0,masks/a.png\n1,masks/b.png\n")

        paths = load_mask_paths(str(manifest), "seg_path")

        self.assertEqual(
            paths,
            [str(self.directory / "masks" / "a.png"),
             str(self.directory / "masks" / "b.png")],
        )

    def test_there_is_no_fallback_guessing_for_masks(self) -> None:
        # 'mask' style names are never guessed; the column must be named.
        manifest = self._write("id,mask_path\n0,a.png\n")

        with self.assertRaises(ValueError) as caught:
            load_mask_paths(str(manifest), "wrong_name")

        self.assertIn("wrong_name", str(caught.exception))

    def test_blank_cell_raises_with_row_number(self) -> None:
        manifest = self._write("id,seg_path\n0,a.png\n1,\n")

        with self.assertRaises(ValueError) as caught:
            load_mask_paths(str(manifest), "seg_path")

        self.assertIn("Row 3", str(caught.exception))
        self.assertIn("mask", str(caught.exception))

    def test_from_config_uses_the_configured_column(self) -> None:
        manifest = self._write("id,seg_path\n0,a.png\n")

        paths = load_mask_paths_from_config(
            _schema(manifest, None, mask_column="seg_path", num_classes=19)
        )

        self.assertEqual(os.path.basename(paths[0]), "a.png")

    def test_from_config_without_mask_column_raises(self) -> None:
        manifest = self._write("id,seg_path\n0,a.png\n")

        with self.assertRaises(ValueError) as caught:
            load_mask_paths_from_config(_schema(manifest, None))

        self.assertIn("mask_column", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
