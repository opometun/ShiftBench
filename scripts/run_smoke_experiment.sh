#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON:-python3}" -m shiftbench.experiments.run --config configs/smoke.toml
