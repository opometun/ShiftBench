#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

METRICS = [
    "centroid_dinov2",
    "centroid_streetclip",
    "frechet_dinov2",
    "frechet_streetclip",
    "color_js",
    "texture_js",
    "class_frequency_js",
    "class_presence_js",
    "scene_complexity_js",
    "sadge",
    "sadge_appearance",
    "sadge_geometry",
]

LABELS = {
    "color_js": r"$\text{js}_{\text{color}}$",
    "texture_js": r"$\text{js}_{\text{texture}}$",
    "class_frequency_js": r"$\text{js}_{\text{classFrequency}}$",
    "class_presence_js": r"$\text{js}_{\text{classPresence}}$",
    "scene_complexity_js": r"$\text{js}_{\text{sceneComplexity}}$",
    "frechet_dinov2": r"$\text{fréchet}_{\text{dinov2}}$",
    "centroid_dinov2": r"$\text{centroid}_{\text{dinov2}}$",
    "frechet_streetclip": r"$\text{fréchet}_{\text{streetclip}}$",
    "centroid_streetclip": r"$\text{centroid}_{\text{streetclip}}$",
    "sadge": r"$\text{SADGE}$",
    "sadge_appearance": r"$\overline{A}_{\text{SADGE}}$",
    "sadge_geometry": r"$\overline{G}_{\text{SADGE}}$",
}

SIMILARITY = {
    "sadge",
    "sadge_appearance",
    "sadge_geometry",
}

OUT_OF_RANGE = {"sadge"}

ARCH = {
    "segformer": "SegFormer",
    "deeplabv3plus": "DeepLabV3+",
}

DISTANCE_COLOUR = "#00A6A6"
SIMILARITY_COLOUR = "#B12F7D"

X_MIN = -1.32
X_MAX = 1.16
VALUE_OFFSET = 0.045


def load(model):
    path = Path(f"results/shift/shift_json/correlation_{model}.json")

    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("target") != "miou":
        raise SystemExit(
            f"{path} must contain target='miou'."
        )

    return data


def draw(axis, payload, qualified):
    lookup = {
        row["metric"]: row
        for row in payload["correlations"]
    }

    missing = [
        metric
        for metric in METRICS
        if metric not in lookup
    ]

    if missing:
        raise SystemExit(f"Missing metrics: {missing}")

    rows = sorted(
        (lookup[metric] for metric in METRICS),
        key=lambda row: row["predictive"],
    )

    values = [row["predictive"] for row in rows]

    colours = [
        SIMILARITY_COLOUR
        if row["metric"] in SIMILARITY
        else DISTANCE_COLOUR
        for row in rows
    ]

    bars = axis.barh(
        [LABELS[row["metric"]] for row in rows],
        values,
        color=colours,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )

    if qualified:
        for bar, row in zip(bars, rows):
            if row["metric"] in OUT_OF_RANGE:
                bar.set_hatch("//")
                bar.set_edgecolor("#3A3A3A")
                bar.set_linewidth(0.8)

    axis.axvline(0, color="#333333", linewidth=0.9, zorder=3)

    axis.set_xlim(X_MIN, X_MAX)
    axis.set_xlabel(
        "Sign-adjusted Spearman correlation with test mIoU"
    )

    axis.set_title(
        ARCH[payload["model"]],
        fontsize=11,
        fontweight="bold",
        pad=8,
    )

    axis.grid(axis="x", color="#DADADA", linewidth=0.7, zorder=0)
    axis.spines[["top", "right", "left"]].set_visible(False)

    axis.tick_params(axis="y", length=0, labelsize=9)
    axis.tick_params(axis="x", labelsize=8.5)
    axis.xaxis.set_major_locator(MultipleLocator(0.25))

    for bar, value in zip(bars, values):
        if value < 0:
            label_x = value - VALUE_OFFSET
            horizontal_alignment = "right"
            label = f"−{abs(value):.2f}"
        else:
            label_x = value + VALUE_OFFSET
            horizontal_alignment = "left"
            label = f"{value:.2f}"

        axis.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha=horizontal_alignment,
            fontsize=8.5,
            color="#222222",
            clip_on=False,
            zorder=4,
        )


def save(fig, path):
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main():
    analyses = {
        model: load(model)
        for model in ARCH
    }

    out = Path("results/figures/figure2")
    out.mkdir(parents=True, exist_ok=True)

    for qualified in (False, True):
        suffix = (
            "sadge_qualified"
            if qualified
            else "standard"
        )

        caption = (
            "Distances are scored positively when they correlate "
            "negatively with mIoU; similarities are scored positively "
            "when they correlate positively with mIoU. Higher is better. "
            "Correlations are computed using mean test mIoU across "
            "six seeds (42–47)."
        )

        if qualified:
            caption += (
                " Hatched: fused SADGE is evaluated outside the range "
                "assumed by its calibration."
            )

        for model, payload in analyses.items():
            fig, axis = plt.subplots(figsize=(7.8, 5.7))

            draw(axis, payload, qualified)

            fig.text(
                0.5,
                0.01,
                caption,
                ha="center",
                fontsize=7.5,
                wrap=True,
            )

            fig.tight_layout(rect=(0.02, 0.055, 0.99, 1))

            save(
                fig,
                out / f"figure2_{model}_miou_{suffix}.png",
            )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(15.2, 5.9),
            sharex=True,
        )

        for axis, payload in zip(
            axes,
            analyses.values(),
        ):
            draw(axis, payload, qualified)

        fig.text(
            0.5,
            0.01,
            caption,
            ha="center",
            fontsize=7.5,
            wrap=True,
        )

        fig.tight_layout(rect=(0.015, 0.055, 0.99, 1))

        save(
            fig,
            out / f"figure2_combined_miou_{suffix}.png",
        )

    print(f"Wrote Figure 2 PNGs to {out.resolve()}")


if __name__ == "__main__":
    main()
