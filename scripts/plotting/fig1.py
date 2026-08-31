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
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
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

ARCH = {
    "segformer": ("SegFormer", "s"),
    "deeplabv3plus": ("DeepLabV3+", "o"),
}

REAL = "#222222"
GTA = "#E69F00"
SYNSCAPES = "#1976A8"


def load(path):
    path = Path(path)

    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def mixture_colour(mixture):
    if mixture == "cityscapes100":
        return to_rgba(REAL, 1.0)

    if "gta" in mixture:
        fraction = (
            1.0
            if mixture == "gta100"
            else int(mixture.rsplit("gta", 1)[1]) / 100
        )
        return to_rgba(GTA, 0.35 + 0.65 * fraction)

    fraction = (
        1.0
        if mixture == "synscapes100"
        else int(mixture.rsplit("synscapes", 1)[1]) / 100
    )

    return to_rgba(SYNSCAPES, 0.35 + 0.65 * fraction)


def main():
    distances = load("results/shift/distances.json")

    analyses = {
        model: load(f"results/shift/shift_json/correlation_{model}.json")
        for model in ARCH
    }

    for model, payload in analyses.items():
        seeds = [str(seed) for seed in payload.get("seeds_requested", [])]

        if not seeds or seeds[0] != "42":
            raise SystemExit(
                f"{model}: seed 42 must be first. Found: {seeds}"
            )

        if payload.get("target") != "miou":
            raise SystemExit(
                f"{model}: expected target='miou'."
            )

    out = Path("results/figures/figure1")
    out.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        fig, axis = plt.subplots(figsize=(5.2, 4.05))
        notes = []

        for model, payload in analyses.items():
            model_name, marker = ARCH[model]

            x_values = [
                float(distances[mixture][metric])
                for mixture in MIXES
            ]

            y_values = [
                float(payload["performance_per_seed"][mixture][0])
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
                    s=84,
                    marker=marker,
                    color=mixture_colour(mixture),
                    edgecolor="#222222",
                    linewidth=0.7,
                    zorder=3,
                )

            rho, p_value = spearmanr(x_values, y_values)

            p_text = (
                "$p<.001$"
                if p_value < 0.001
                else f"$p={p_value:.3f}$"
            )

            notes.append(
                f"{model_name}: $\\rho$={rho:.2f}, {p_text}"
            )

        axis.set_xlabel(LABELS[metric])
        axis.set_ylabel("Test mIoU")

        axis.grid(axis="y", color="#D5D5D5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=3, width=0.7)

        if metric in ("sadge_appearance", "sadge_geometry"):
            text_x, ha = 0.965, "right"
        else:
            text_x, ha = 0.035, "left"

        axis.text(
            text_x,
            0.035,
            "\n".join(notes),
            transform=axis.transAxes,
            fontsize=8.3,
            va="bottom",
            ha=ha,
            bbox={
                "facecolor": "white",
                "edgecolor": "#999999",
                "alpha": 0.94,
                "pad": 3,
            },
        )
        
        legend_handles = [
            Line2D(
                [0], [0],
                marker="s",
                linestyle="None",
                color=REAL,
                markerfacecolor="white",
                markeredgecolor=REAL,
                markersize=7,
                label="SegFormer",
            ),
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                color=REAL,
                markerfacecolor="white",
                markeredgecolor=REAL,
                markersize=7,
                label="DeepLabV3+",
            ),
            Patch(facecolor=REAL, label="100% real"),
            Patch(facecolor=SYNSCAPES, label="Synscapes-containing"),
            Patch(facecolor=GTA, label="GTA-containing"),
        ]

        axis.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=3,
            frameon=False,
        )

        footer = (
            "Spearman p-values are uncorrected across 12 "
            "metric-wise tests. Correlations use seed 42."
        )

        if metric == "sadge":
            footer = (
                "Fused SADGE is evaluated outside the range assumed "
                "by its calibration. " + footer
            )

        fig.subplots_adjust(
            left=0.16,
            right=0.98,
            top=0.97,
            bottom=0.32,
        )

        fig.text(
            0.5,
            0.025,
            footer,
            ha="center",
            va="bottom",
            fontsize=7.2,
            wrap=True,
        )

        path = out / f"figure1_seed42_{metric}_miou.png"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(fig)
        print(f"Wrote {path.resolve()}")

    print(f"Created {len(METRICS)} Figure 1 PNGs in {out.resolve()}")


if __name__ == "__main__":
    main()