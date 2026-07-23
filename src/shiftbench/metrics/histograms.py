"""Histogram summaries of image and mask sets, compared with JS distance.

Same summarize-then-compare shape as the Gaussian path, with an averaged
histogram as the per-dataset summary instead of (mu, sigma). Each summary here
reproduces exactly what the corresponding quantify_* function in
shift_quantification_metrics computes internally, so going through the summary
artifact gives the same number as calling that function directly — the
equivalence is pinned by tests.

Imports of the metric implementations are deferred into the functions: the
image summaries need cv2/scikit-image (the 'image' extra), and importing this
module must stay possible without them.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shiftbench.metrics.distances import js_distance


def js_compare(summary_a: Mapping[str, Any], summary_b: Mapping[str, Any]) -> float:
    """JS distance between two stored histogram summaries."""
    return float(
        js_distance(np.asarray(summary_a["hist"]), np.asarray(summary_b["hist"]))
    )


def color_summary(images: list, **_: Any) -> dict[str, np.ndarray]:
    """Mean HSV histogram of a dataset, as quantify_color_shift computes it."""
    from shiftbench.shift_quantification_metrics.image_based.color import (
        hsv_histogram,
    )

    return {"hist": np.vstack([hsv_histogram(img) for img in images]).mean(axis=0)}


def texture_summary(images: list, **_: Any) -> dict[str, np.ndarray]:
    """Mean multiscale-LBP histogram, as quantify_texture_shift computes it."""
    from shiftbench.shift_quantification_metrics.image_based.texture import (
        multiscale_lbp,
    )

    configs = [(8, 1), (16, 2)]
    return {
        "hist": np.vstack(
            [multiscale_lbp(img, configs=configs) for img in images]
        ).mean(axis=0)
    }


def class_frequency_summary(masks: list, num_classes: int) -> dict[str, np.ndarray]:
    """Mean class-frequency distribution of a mask set."""
    from shiftbench.shift_quantification_metrics.label_based.class_frequency import (
        class_frequency,
    )

    return {
        "hist": np.vstack(
            [class_frequency(mask, num_classes) for mask in masks]
        ).mean(axis=0)
    }


def class_presence_summary(masks: list, num_classes: int) -> dict[str, np.ndarray]:
    """Normalized per-image class presence counts of a mask set."""
    from shiftbench.shift_quantification_metrics.label_based.class_presence import (
        class_presence_frequency,
    )

    counts = class_presence_frequency(masks, num_classes)
    return {"hist": counts / counts.sum()}


def scene_complexity_summary(masks: list, num_classes: int) -> dict[str, np.ndarray]:
    """Distribution of distinct-class counts per image of a mask set."""
    from shiftbench.shift_quantification_metrics.label_based.scene_complexity import (
        scene_complexity_histogram,
    )

    return {"hist": scene_complexity_histogram(masks, num_classes)}
