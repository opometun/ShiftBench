from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from shiftbench.experiments.run import run_experiment
from shiftbench.metrics import compute_covariance, compute_mean


class SmokeExperimentTest(unittest.TestCase):
    def test_run_writes_structured_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(
                config_path=PROJECT_ROOT / "configs" / "smoke.toml",
                output_root=Path(directory),
            )

            metrics = json.loads((run_dir / "metrics.json").read_text())
            validation = json.loads((run_dir / "validation.json").read_text())
            config_copy_exists = (run_dir / "config.toml").exists()
            text_log_exists = (run_dir / "logs.txt").exists()

        self.assertEqual(metrics["seed"], 42)
        self.assertEqual(metrics["dataset_rows"], 4)
        self.assertTrue(validation["is_valid"])
        self.assertTrue(config_copy_exists)
        self.assertTrue(text_log_exists)

    def test_module_cli_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "shiftbench.experiments.run",
                    "--config",
                    str(PROJECT_ROOT / "configs" / "smoke.toml"),
                    "--output-root",
                    directory,
                ],
                check=False,
                capture_output=True,
                env=env,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Run artifacts written to:", result.stdout)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)


class ShiftMetricsRunTest(unittest.TestCase):
    """Distances must land inside the run directory contract, not beside it."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def _write_stats(self, name: str, shift: float, encoder: str) -> Path:
        rng = np.random.default_rng(abs(hash(name)) % 1000)
        features = (rng.normal(0.0, 1.0, (200, 8)) + shift).astype(np.float32)
        path = self.directory / f"{name}.npz"
        np.savez(
            path,
            mu=compute_mean(features),
            sigma=compute_covariance(features),
            encoder=encoder,
            model=f"model-{encoder}",
        )
        return path

    def _write_config(self, stats_a: Path, stats_b: Path, metrics: str) -> Path:
        config_path = self.directory / "shift.toml"
        sample = PROJECT_ROOT / "data" / "sample" / "tiny_shiftbench.csv"
        config_path.write_text(
            f"""
[experiment]
name = "shift"
seed = 7
output_root = "runs"

[dataset]
path = "{sample}"
required_columns = ["sample_id", "split", "source", "label", "text"]
id_column = "sample_id"
split_column = "split"
source_column = "source"
label_column = "label"
text_column = "text"
allowed_splits = ["train", "validation", "test"]
allowed_sources = ["real", "synthetic"]

[shift]
stats_a = "{stats_a}"
stats_b = "{stats_b}"
metrics = [{metrics}]
""".strip(),
            encoding="utf-8",
        )
        return config_path

    def test_distances_are_written_into_metrics_json(self) -> None:
        stats_a = self._write_stats("a", 0.0, "dinov2")
        stats_b = self._write_stats("b", 2.0, "dinov2")
        config_path = self._write_config(stats_a, stats_b, '"centroid", "frechet"')

        with tempfile.TemporaryDirectory() as output_root:
            run_dir = run_experiment(config_path, Path(output_root))
            metrics = json.loads((run_dir / "metrics.json").read_text())

        shift = metrics["shift"]
        self.assertEqual(shift["encoder"], "dinov2")
        self.assertEqual(shift["model"], "model-dinov2")
        self.assertGreater(shift["distances"]["centroid"], 1.0)
        self.assertGreater(shift["distances"]["frechet"], 1.0)
        # The whole point: provenance travels with the numbers.
        self.assertIn("stats_a", shift)

    def test_only_requested_metrics_are_computed(self) -> None:
        stats_a = self._write_stats("a", 0.0, "dinov2")
        stats_b = self._write_stats("b", 2.0, "dinov2")
        config_path = self._write_config(stats_a, stats_b, '"centroid"')

        with tempfile.TemporaryDirectory() as output_root:
            run_dir = run_experiment(config_path, Path(output_root))
            metrics = json.loads((run_dir / "metrics.json").read_text())

        self.assertEqual(list(metrics["shift"]["distances"]), ["centroid"])

    def test_encoder_mismatch_fails_the_run_and_is_logged(self) -> None:
        stats_a = self._write_stats("a", 0.0, "dinov2")
        stats_b = self._write_stats("b", 2.0, "streetclip")
        config_path = self._write_config(stats_a, stats_b, '"frechet"')

        with tempfile.TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(ValueError, "different encoders"):
                run_experiment(config_path, Path(output_root))

            run_dir = next(Path(output_root).iterdir())
            log = json.loads((run_dir / "logs.json").read_text())

        self.assertEqual(log["status"], "failed")
        self.assertFalse((run_dir / "metrics.json").exists())

    def test_unreadable_stats_fail_the_run(self) -> None:
        stats_a = self._write_stats("a", 0.0, "dinov2")
        missing = self.directory / "does_not_exist.npz"
        config_path = self._write_config(stats_a, missing, '"centroid"')

        with tempfile.TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(ValueError, "Could not read shift stats"):
                run_experiment(config_path, Path(output_root))

    def test_runs_without_a_shift_section_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as output_root:
            run_dir = run_experiment(
                PROJECT_ROOT / "configs" / "smoke.toml",
                Path(output_root),
            )
            metrics = json.loads((run_dir / "metrics.json").read_text())

        self.assertNotIn("shift", metrics)


if __name__ == "__main__":
    unittest.main()
