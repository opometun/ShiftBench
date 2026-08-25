#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.2,
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
})


MIXES = [
    "cityscapes100",
    "cityscapes75_gta25",
    "cityscapes50_gta50",
    "cityscapes25_gta75",
    "gta100",
    "cityscapes75_synscapes25",
    "cityscapes50_synscapes50",
    "cityscapes25_synscapes75",
    "synscapes100",
]

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
    "centroid_dinov2": r"$\text{centroid}_{\text{dinov2}}$",
    "centroid_streetclip": r"$\text{centroid}_{\text{streetclip}}$",
    "frechet_dinov2": r"$\text{fréchet}_{\text{dinov2}}$",
    "frechet_streetclip": r"$\text{fréchet}_{\text{streetclip}}$",
    "color_js": r"$\text{js}_{\text{color}}$",
    "texture_js": r"$\text{js}_{\text{texture}}$",
    "class_frequency_js": r"$\text{js}_{\text{classFrequency}}$",
    "class_presence_js": r"$\text{js}_{\text{classPresence}}$",
    "scene_complexity_js": r"$\text{js}_{\text{sceneComplexity}}$",
    "sadge": r"$\text{SADGE}$",
    "sadge_appearance": r"$\hat{A}_{\text{SADGE}}$",
    "sadge_geometry": r"$\hat{G}_{\text{SADGE}}$",
}

ARCHITECTURES = {
    "segformer": {
        "label": "SegFormer",
        "marker": "s",
    },
    "deeplabv3plus": {
        "label": "DeepLabV3+",
        "marker": "o",
    },
}

REAL_COLOUR = "#222222"
GTA_COLOUR = "#E69F00"
SYNSCAPES_COLOUR = "#1976A8"


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def mixture_colour(mixture):
    if mixture == "cityscapes100":
        return to_rgba(REAL_COLOUR, 1.0)

    if "gta" in mixture:
        fraction = (
            1.0
            if mixture == "gta100"
            else int(mixture.rsplit("gta", 1)[1]) / 100
        )
        return to_rgba(GTA_COLOUR, 0.35 + 0.65 * fraction)

    fraction = (
        1.0
        if mixture == "synscapes100"
        else int(mixture.rsplit("synscapes", 1)[1]) / 100
    )

    return to_rgba(SYNSCAPES_COLOUR, 0.35 + 0.65 * fraction)


def seed42_miou(payload, mixture):
    seeds = [str(seed) for seed in payload.get("seeds_requested", [])]

    if seeds != ["42", "43", "44"]:
        raise SystemExit(
            "Expected seeds_requested to be [42, 43, 44]. "
            f"Found: {seeds}"
        )

    return float(payload["performance_per_seed"][mixture][0])


def add_legend(axis):
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            color="#222222",
            markerfacecolor="white",
            markeredgecolor="#222222",
            markersize=6.5,
            label="SegFormer",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="#222222",
            markerfacecolor="white",
            markeredgecolor="#222222",
            markersize=6.5,
            label="DeepLabV3+",
        ),
        Patch(facecolor=REAL_COLOUR, label="100% real"),
        Patch(facecolor=SYNSCAPES_COLOUR, label="Synscapes-containing"),
        Patch(facecolor=GTA_COLOUR, label="GTA-containing"),
    ]

    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=7.2,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def main():
    distances = load_json("results/shift/distances.json")

    analyses = {
        architecture: load_json(
            f"results/shift/correlation_{architecture}.json"
        )
        for architecture in ARCHITECTURES
    }

    output_directory = Path("results/figures/figure1")
    output_directory.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        fig, axis = plt.subplots(figsize=(4.6, 3.55))
        correlation_notes = []

        for architecture, payload in analyses.items():
            style = ARCHITECTURES[architecture]

            x_values = [
                float(distances[mixture][metric])
                for mixture in MIXES
            ]

            y_values = [
                seed42_miou(payload, mixture)
                for mixture in MIXES
            ]

            for mixture, x_value, y_value in zip(
                MIXES,
                x_values,
                y_values,
            ):
                axis.scatter(
                    x_value,
                    y_value,
                    s=72,
                    marker=style["marker"],
                    color=mixture_colour(mixture),
                    edgecolor="#222222",
                    linewidth=0.65,
                    zorder=3,
                )

            rho, p_value = spearmanr(x_values, y_values)

            p_text = (
                "$p<.001$"
                if p_value < 0.001
                else f"$p={p_value:.3f}$"
            )

            correlation_notes.append(
                f"{style['label']}: $\\rho$={rho:.2f}, {p_text}"
            )

        x_label = LABELS[metric]

        if metric == "sadge_geometry":
            x_label = "Raw inlier count (MASt3R)"

        axis.set_xlabel(x_label)
        axis.set_ylabel("Test mIoU")

        axis.grid(axis="y", color="#D5D5D5", linewidth=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(
            axis="both",
            direction="out",
            length=3,
            width=0.7,
        )

        axis.text(
            0.035,
            0.035,
            "\n".join(correlation_notes),
            transform=axis.transAxes,
            fontsize=7.3,
            ha="left",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "#999999",
                "alpha": 0.94,
                "pad": 2.5,
            },
            zorder=5,
        )

        add_legend(axis)

        if metric == "sadge":
            footer = (
                "Fused SADGE is evaluated outside the range assumed by its "
                "calibration. Spearman p-values are uncorrected across 12 "
                "metric-wise tests. Correlations shown here use seed 42."
            )
        else:
            footer = (
                "Spearman p-values are uncorrected across 12 metric-wise tests. "
                "Correlations shown here use seed 42."
            )

        fig.subplots_adjust(
            left=0.15,
            right=0.98,
            top=0.97,
            bottom=0.31,
        )

        fig.text(
            0.5,
            0.025,
            footer,
            ha="center",
            va="bottom",
            fontsize=6.2,
            wrap=True,
        )

        output_path = (
            output_directory
            / f"figure1_seed42_{metric}_miou.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(fig)
        print(f"Wrote {output_path.resolve()}")

    print(
        f"Created {len(METRICS)} individual Figure 1 PNG panels "
        f"for seed 42 in {output_directory.resolve()}."
    )


if __name__ == "__main__":
    main()