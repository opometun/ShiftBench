"""Wiring tests for the encoder registry and SADGE's delegation to it.

torch/transformers/PIL are optional dependencies that the core test
environment does not have, so the assertions run in a subprocess with
lightweight stand-ins injected into sys.modules. This verifies wiring —
registry entries, pooling choice, delegation, model ids — not real model
behavior, which needs the '[features]' extra and a checkpoint download.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STUB_SCRIPT = r"""
import sys, types, contextlib
import numpy as np


class T:
    def __init__(s, a): s.arr = np.asarray(a, dtype="float32")
    def __getitem__(s, i): return T(s.arr[i])
    def __matmul__(s, o): return T(s.arr @ o.arr)
    def to(s, d): return s
    def numpy(s): return s.arr
    def item(s): return float(s.arr)


class _InferenceMode:
    # Real torch.inference_mode() is both a context manager and a decorator.
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __call__(self, fn): return fn


torch = types.ModuleType("torch")
torch.device = lambda x: f"dev:{x}"
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
torch.inference_mode = _InferenceMode
torch.Tensor = T
fn = types.ModuleType("torch.nn.functional")
fn.normalize = lambda t, p=2, dim=-1: T(t.arr / np.linalg.norm(t.arr, axis=dim, keepdims=True))
nn = types.ModuleType("torch.nn")
nn.functional = fn
torch.nn = nn
sys.modules["torch"] = torch
sys.modules["torch.nn"] = nn
sys.modules["torch.nn.functional"] = fn

LOADED = {}


class Proc:
    @classmethod
    def from_pretrained(cls, name):
        LOADED["proc"] = name
        return cls()
    def __call__(s, images, return_tensors=None):
        n = len(images)
        return types.SimpleNamespace(to=lambda d: {"pixel_values": T(np.zeros((n, 3, 4, 4)))})


class Model:
    @classmethod
    def from_pretrained(cls, name):
        LOADED["model"] = name
        return cls()
    def to(s, d): return s
    def parameters(s): return []
    def eval(s): return s
    def __call__(s, **kw):
        n = kw["pixel_values"].arr.shape[0]
        hidden = np.arange(n * 5 * 8, dtype="float32").reshape(n, 5, 8)
        pooled = np.stack([np.full(8, i + 1.0, dtype="float32") for i in range(n)])
        return types.SimpleNamespace(last_hidden_state=T(hidden), pooler_output=T(pooled))


class Clip(Model):
    def get_image_features(s, **kw):
        return T(np.ones((kw["pixel_values"].arr.shape[0], 8), dtype="float32"))


tf = types.ModuleType("transformers")
tf.AutoImageProcessor = tf.AutoProcessor = Proc
tf.AutoModel = Model
tf.CLIPModel = Clip
sys.modules["transformers"] = tf

pil = types.ModuleType("PIL")


class Img:
    def __enter__(s): return s
    def __exit__(s, *a): return False
    def convert(s, m): return s


pil.Image = types.SimpleNamespace(open=lambda p: Img())
sys.modules["PIL"] = pil

cv2 = types.ModuleType("cv2")
cv2.error = Exception
sys.modules["cv2"] = cv2

sys.path.insert(0, "src")

from shiftbench.features.registry import ENCODERS, get_encoder

assert set(ENCODERS) == {"dinov2", "dinov3", "streetclip"}, sorted(ENCODERS)

dinov3 = get_encoder("dinov3")
assert dinov3.default_model_name == "facebook/dinov3-vitl16-pretrain-lvd1689m"
proc, model = dinov3.load("dev:cpu", dinov3.default_model_name)
feats = dinov3.extract(proc, model, [object(), object()], "dev:cpu")
# The stub's pooler_output rows are constant 1.0 / 2.0 while last_hidden_state
# is an arange — matching rows proves dinov3 pools pooler_output, not CLS.
assert feats.arr.shape == (2, 8), feats.arr.shape
assert feats.arr[0, 0] == 1.0 and feats.arr[1, 0] == 2.0, feats.arr

try:
    get_encoder("nope")
    raise AssertionError("expected ValueError")
except ValueError as error:
    message = str(error)
    assert "dinov2" in message and "dinov3" in message and "streetclip" in message

from shiftbench.shift_quantification_metrics.representation_based.sadge.metrics.embedding import (
    DinoV3Metric,
)

metric = DinoV3Metric("dev:cpu")
value = metric.compute("real.png", "syn.png")
# Stub embeddings are parallel vectors, so cosine similarity must be exactly 1.
assert abs(value - 1.0) < 1e-6, value
assert LOADED["model"] == "facebook/dinov3-vitl16-pretrain-lvd1689m", LOADED
assert metric.higher_is_better is True

# An uninitialized MASt3R submodule must produce instructions, not a traceback.
from pathlib import Path as _Path
from shiftbench.shift_quantification_metrics.representation_based.sadge.metrics import (
    geometry,
)

geometry.MAST3R_DIR = _Path("/nonexistent/mast3r")
try:
    geometry.GeoGapMetric("dev:cpu")._load()
    raise AssertionError("expected RuntimeError")
except RuntimeError as error:
    assert "submodule update --init --recursive" in str(error), error

print("REGISTRY_STUB_OK")
"""


class FeaturesRegistryWiringTest(unittest.TestCase):
    def test_three_encoders_and_sadge_delegation(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", STUB_SCRIPT],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("REGISTRY_STUB_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
