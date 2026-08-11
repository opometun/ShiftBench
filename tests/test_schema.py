from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.config import DatasetSchemaConfig, load_experiment_config 
from shiftbench.datasets.schema import validate_csv_dataset


def _helper_create_manifest(config, directory, rows):
    """Helper function to simulate a manifest."""
    root = Path(directory)

    # Create fake files for each row
    for sid, _, _ in rows:
        (root / f"{sid}_img.png").write_bytes(b"fake")
        (root / f"{sid}_mask.png").write_bytes(b"fake")

    # Build CSV
    csv = "sample_id,split,source,label,input\n"
    for sid, split, source in rows:
        csv += (
            f"{sid},{split},{source},"
            f"{sid}_mask.png,{sid}_img.png\n"
        )

    data_path = root / "dataset.csv"
    data_path.write_text(csv, encoding="utf-8")

    # Build schema
    schema = config.dataset.__class__(
        path=data_path,
        required_columns=config.dataset.required_columns,
        id_column=config.dataset.id_column,
        split_column=config.dataset.split_column,
        source_column=config.dataset.source_column,
        label_column=config.dataset.label_column,
        input_column=config.dataset.input_column,
        allowed_splits=config.dataset.allowed_splits,
        allowed_sources=config.dataset.allowed_sources,
        num_classes=config.dataset.num_classes,
        description_column=config.dataset.description_column,
    )

    return schema


class DatasetSchemaTest(unittest.TestCase):
    def test_validates_tiny_sample_dataset(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "smoke.toml")
        with tempfile.TemporaryDirectory() as directory:
            text = [("1", "train", "real"), ("2", "test", "synthetic")]
            schema = _helper_create_manifest(config, directory, text)
            result = validate_csv_dataset(schema)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.rows, 2)
        self.assertEqual(result.split_counts["train"], 1)
        self.assertEqual(result.source_counts["synthetic"], 1)

    def test_reports_duplicate_ids(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "smoke.toml")
        with tempfile.TemporaryDirectory() as directory:
            text = [("same", "train", "real"), ("same", "test", "synthetic")]
            schema = _helper_create_manifest(config, directory, text)
            result = validate_csv_dataset(schema)

        self.assertFalse(result.is_valid)
        self.assertIn("duplicate id", result.errors[0])


class ImageDatasetValidationTest(unittest.TestCase):
    """Validation must catch broken image paths before the model is loaded."""

    def _schema(
        self,
        path: Path,
        num_classes: int | None = None,
    ) -> DatasetSchemaConfig:
        required = ["sample_id", "split", "source", "seg_path", "image_path"]
        return DatasetSchemaConfig(
            path=path,
            required_columns=tuple(required),
            id_column="sample_id",
            split_column="split",
            source_column="source",
            label_column="seg_path",
            allowed_splits=("train",),
            allowed_sources=("real",),
            input_column="image_path",
            num_classes=num_classes,
        )

    def test_accepts_images_that_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "i.png").write_bytes(b"fake")
            (root / "m.png").write_bytes(b"fake")
            manifest = root / "m.csv"
            manifest.write_text(
                "sample_id,split,source,seg_path,image_path\n0,train,real,m.png,i.png\n",
                encoding="utf-8",
            )

            result = validate_csv_dataset(self._schema(manifest))

        self.assertTrue(result.is_valid, result.errors)

    def test_reports_images_that_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "present.png").write_bytes(b"fake")
            (root / "m1.png").write_bytes(b"fake")
            (root / "m2.png").write_bytes(b"fake")
            manifest = root / "m.csv"
            manifest.write_text(
                "sample_id,split,source,seg_path,image_path\n"
                "0,train,real,m1.png,present.png\n"
                "1,train,real,m2.png,gone.png\n",
                encoding="utf-8",
            )

            result = validate_csv_dataset(self._schema(manifest))

        self.assertFalse(result.is_valid)
        self.assertIn("Row 3", result.errors[0])
        self.assertIn("gone.png", result.errors[0])

    def test_reports_masks_that_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(b"fake")
            (root / "a_mask.png").write_bytes(b"fake")
            manifest = root / "m.csv"
            manifest.write_text(
                "sample_id,split,source,image_path,seg_path\n"
                "0,train,real,a.png,a_mask.png\n"
                "1,train,real,a.png,gone_mask.png\n",
                encoding="utf-8",
            )

            result = validate_csv_dataset(
                self._schema(manifest, num_classes=19)
            )

        self.assertFalse(result.is_valid)
        self.assertIn("Row 3", result.errors[0])
        self.assertIn("gone_mask.png", result.errors[0])

    def test_accepts_masks_that_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(b"fake")
            (root / "a_mask.png").write_bytes(b"fake")
            manifest = root / "m.csv"
            manifest.write_text(
                "sample_id,split,source,image_path,seg_path\n"
                "0,train,real,a.png,a_mask.png\n",
                encoding="utf-8",
            )

            result = validate_csv_dataset(
                self._schema(manifest, num_classes=19)
            )

        self.assertTrue(result.is_valid, result.errors)


if __name__ == "__main__":
    unittest.main()
