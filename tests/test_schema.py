from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.config import DatasetSchemaConfig, load_experiment_config
from shiftbench.datasets.schema import validate_csv_dataset


class DatasetSchemaTest(unittest.TestCase):
    def test_validates_tiny_sample_dataset(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "smoke.toml")
        result = validate_csv_dataset(config.dataset)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.rows, 4)
        self.assertEqual(result.split_counts["train"], 2)
        self.assertEqual(result.source_counts["synthetic"], 2)

    def test_reports_duplicate_ids(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "smoke.toml")
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "duplicate.csv"
            data_path.write_text(
                """
sample_id,split,source,label,text
same,train,real,positive,First item.
same,test,synthetic,negative,Second item.
""".lstrip(),
                encoding="utf-8",
            )
            schema = config.dataset.__class__(
                path=data_path,
                required_columns=config.dataset.required_columns,
                id_column=config.dataset.id_column,
                split_column=config.dataset.split_column,
                source_column=config.dataset.source_column,
                label_column=config.dataset.label_column,
                text_column=config.dataset.text_column,
                allowed_splits=config.dataset.allowed_splits,
                allowed_sources=config.dataset.allowed_sources,
            )

            result = validate_csv_dataset(schema)

        self.assertFalse(result.is_valid)
        self.assertIn("duplicate id", result.errors[0])


class ImageDatasetValidationTest(unittest.TestCase):
    """Validation must catch broken image paths before the model is loaded."""

    def _schema(self, path: Path) -> DatasetSchemaConfig:
        return DatasetSchemaConfig(
            path=path,
            required_columns=("sample_id", "split", "source", "label", "image_path"),
            id_column="sample_id",
            split_column="split",
            source_column="source",
            label_column="label",
            allowed_splits=("train",),
            allowed_sources=("real",),
            image_column="image_path",
        )

    def test_accepts_images_that_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(b"fake")
            manifest = root / "m.csv"
            manifest.write_text(
                "sample_id,split,source,label,image_path\n0,train,real,x,a.png\n",
                encoding="utf-8",
            )

            result = validate_csv_dataset(self._schema(manifest))

        self.assertTrue(result.is_valid, result.errors)

    def test_reports_images_that_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "present.png").write_bytes(b"fake")
            manifest = root / "m.csv"
            manifest.write_text(
                "sample_id,split,source,label,image_path\n"
                "0,train,real,x,present.png\n"
                "1,train,real,x,gone.png\n",
                encoding="utf-8",
            )

            result = validate_csv_dataset(self._schema(manifest))

        self.assertFalse(result.is_valid)
        self.assertIn("Row 3", result.errors[0])
        self.assertIn("gone.png", result.errors[0])

    def test_text_only_dataset_skips_the_image_check(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "smoke.toml")

        result = validate_csv_dataset(config.dataset)

        self.assertIsNone(config.dataset.image_column)
        self.assertTrue(result.is_valid, result.errors)


if __name__ == "__main__":
    unittest.main()
