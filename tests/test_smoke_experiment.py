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

from shiftbench.experiments.run import run_experiment


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


if __name__ == "__main__":
    unittest.main()
