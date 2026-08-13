"""Distribution-shift metrics, addressable by name across all metric shapes.

Two layers, per docs/integration-plan.md Phase 0:

- The METRICS registry is the unifying layer: every metric has a name and
  returns one scalar shift between dataset A and dataset B.
- The summary artifact (SUMMARIES, the .npz files) is an optimization below
  it, used only by metrics that can summarize each dataset independently —
  Gaussian stats for embeddings, averaged histograms for images/masks.
  Pairwise metrics (SADGE, registered in a later phase) declare summary=None
  and never touch that path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from shiftbench.metrics.distances import js_distance
from shiftbench.metrics.gaussian import (
    CHUNK_SIZE,
    centroid_distance,
    compute_covariance,
    compute_mean,
    frechet_distance,
    gaussian_summary,
)
from shiftbench.metrics.histograms import (
    class_frequency_summary,
    class_presence_summary,
    color_summary,
    js_compare,
    scene_complexity_summary,
    texture_summary,
)
from shiftbench.metrics.pairwise import SADGE_PARAMS, sadge_pairwise

__all__ = [
    "CHUNK_SIZE",
    "METRICS",
    "SUMMARIES",
    "Metric",
    "Summary",
    "centroid_distance",
    "compute_covariance",
    "compute_mean",
    "frechet_distance",
    "gaussian_summary",
    "get_metric",
    "get_summary",
    "js_distance",
]


@dataclass(frozen=True)
class Summary:
    """One way of reducing a dataset to a small stored artifact.

    Attributes:
        name: Kind recorded in the .npz ('gaussian', 'color', ...).
        input: Raw data it consumes: 'embeddings' | 'images' | 'masks'.
        make: (dataset, **params) -> {key: array} to store in the artifact.
        default_params: Hyperparameters baked into make, recorded alongside
            the artifact so two summaries are only comparable when they used
            the same settings.
    """

    name: str
    input: str
    make: Callable
    default_params: dict


@dataclass(frozen=True)
class Metric:
    """One shift metric: a named scalar comparison of two datasets.

    Attributes:
        name: Short name used on the command line and in run artifacts.
        input: Raw data the metric ultimately needs:
            'embeddings' | 'images' | 'masks' | 'image_dirs'.
        summary: Name of the Summary kind it compares, or None for pairwise
            metrics that cannot summarize one dataset alone.
        compare: (summary_a, summary_b) -> float over loaded summary
            artifacts, or None for pairwise metrics.
        pairwise: (list_of_dirs_a, list_of_dirs_b) -> float for pairwise metrics, 
            None for summarizing ones.
        params: Fixed settings recorded next to the metric's value in run
            artifacts.
    """

    name: str
    input: str
    summary: str | None
    compare: Callable | None
    pairwise: Callable | None = None
    params: dict = field(default_factory=dict)


def _frechet_compare(summary_a, summary_b) -> float:
    return frechet_distance(
        summary_a["mu"], summary_a["sigma"], summary_b["mu"], summary_b["sigma"]
    )


def _centroid_compare(summary_a, summary_b) -> float:
    return centroid_distance(summary_a["mu"], summary_b["mu"])


SUMMARIES: dict[str, Summary] = {
    "gaussian": Summary(
        name="gaussian",
        input="embeddings",
        make=lambda embeddings, **_: gaussian_summary(embeddings),
        default_params={},
    ),
    "color": Summary(
        name="color",
        input="images",
        make=color_summary,
        default_params={"h_bins": 32, "s_bins": 16, "v_bins": 16},
    ),
    "texture": Summary(
        name="texture",
        input="images",
        make=texture_summary,
        default_params={"configs": [[8, 1], [16, 2]]},
    ),
    "class_frequency": Summary(
        name="class_frequency",
        input="masks",
        make=class_frequency_summary,
        default_params={},
    ),
    "class_presence": Summary(
        name="class_presence",
        input="masks",
        make=class_presence_summary,
        default_params={},
    ),
    "scene_complexity": Summary(
        name="scene_complexity",
        input="masks",
        make=scene_complexity_summary,
        default_params={},
    ),
}


METRICS: dict[str, Metric] = {
    "centroid": Metric(
        name="centroid", input="embeddings", summary="gaussian",
        compare=_centroid_compare,
    ),
    "frechet": Metric(
        name="frechet", input="embeddings", summary="gaussian",
        compare=_frechet_compare,
    ),
    "color_js": Metric(
        name="color_js", input="images", summary="color", compare=js_compare,
    ),
    "texture_js": Metric(
        name="texture_js", input="images", summary="texture", compare=js_compare,
    ),
    "class_frequency_js": Metric(
        name="class_frequency_js", input="masks", summary="class_frequency",
        compare=js_compare,
    ),
    "class_presence_js": Metric(
        name="class_presence_js", input="masks", summary="class_presence",
        compare=js_compare,
    ),
    "scene_complexity_js": Metric(
        name="scene_complexity_js", input="masks", summary="scene_complexity",
        compare=js_compare,
    ),
    "sadge": Metric(
        name="sadge", input="image_dirs", summary=None, compare=None,
        pairwise=sadge_pairwise, params=SADGE_PARAMS,
    ),
}


def get_metric(name: str) -> Metric:
    """Look up a metric by name.

    Raises:
        ValueError: If the name is not a registered metric.
    """
    try:
        return METRICS[name]
    except KeyError:
        available = ", ".join(sorted(METRICS))
        raise ValueError(f"Unknown metric '{name}'. Available: {available}") from None


def get_summary(name: str) -> Summary:
    """Look up a summary kind by name.

    Raises:
        ValueError: If the name is not a registered summary.
    """
    try:
        return SUMMARIES[name]
    except KeyError:
        available = ", ".join(sorted(SUMMARIES))
        raise ValueError(f"Unknown summary '{name}'. Available: {available}") from None
