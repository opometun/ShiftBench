#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


LABELS = {
    "color_js": r"$\mathregular{js}_{\mathregular{color}}$",
    "texture_js": r"$\mathregular{js}_{\mathregular{texture}}$",
    "class_frequency_js": r"$\mathregular{js}_{\mathregular{classFrequency}}$",
    "class_presence_js": r"$\mathregular{js}_{\mathregular{classPresence}}$",
    "scene_complexity_js": r"$\mathregular{js}_{\mathregular{sceneComplexity}}$",
    "frechet_dinov2": r"$\mathregular{fréchet}_{\mathregular{dinov2}}$",
    "centroid_dinov2": r"$\mathregular{centroid}_{\mathregular{dinov2}}$",
    "frechet_streetclip": r"$\mathregular{fréchet}_{\mathregular{streetclip}}$",
    "centroid_streetclip": r"$\mathregular{centroid}_{\mathregular{streetclip}}$",
    "sadge": r"$\mathregular{SADGE}$",
    "sadge_appearance": r"$\hat{A}_{\mathregular{SADGE}}$",
    "sadge_geometry": r"$\hat{G}_{\mathregular{SADGE}}$",
}

ARCH = {
    "segformer": "SegFormer",
    "deeplabv3plus": "DeepLabV3+",
}

METRIC_ORDER = [
    "color_js",
    "texture_js",
    "class_frequency_js",
    "class_presence_js",
    "scene_complexity_js",
    "frechet_dinov2",
    "centroid_dinov2",
    "frechet_streetclip",
    "centroid_streetclip",
    "sadge",
    "sadge_appearance",
    "sadge_geometry",
]


def load(key):
    path = Path(f"results/shift/correlation_{key}.json")

    if not path.is_file():
        raise SystemExit(
            f"Missing input: {path}\n"
            "Download/copy the precomputed correlation JSON files into "
            "results/shift/ before running this plotting script."
        )

    return json.loads(path.read_text())


def draw(ax, payload, qualified):
    entries = payload["within_ratio"]["metrics"]
    metrics = [metric for metric in METRIC_ORDER if metric in entries]
    ratios = [pair["ratio"] for pair in payload["within_ratio"]["pairs"]]

    colours = {
        True: "#2E7D32",
        False: "#C62828",
        None: "#BDBDBD",
    }

    for row, metric in enumerate(metrics):
        for col, call in enumerate(entries[metric]["calls"]):
            unavailable = call.get("unavailable", False)
            correct = None if unavailable else call.get("correct")

            if unavailable:
                colour = "#EEEEEE"
                symbol = "?"
                text_colour = "#333333"
            elif correct is True:
                colour = colours[True]
                symbol = "✓"
                text_colour = "white"
            elif correct is False:
                colour = colours[False]
                symbol = "×"
                text_colour = "white"
            else:
                colour = colours[None]
                symbol = "–"
                text_colour = "#333333"

            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=colour,
                    edgecolor="white",
                    linewidth=1.2,
                )
            )

            ax.text(
                col,
                row,
                symbol,
                ha="center",
                va="center",
                fontsize=14,
                weight="bold",
                fontfamily="DejaVu Sans",
                color=text_colour,
            )

        if qualified and metric == "sadge":
            ax.add_patch(
                Rectangle(
                    (-0.5, row - 0.5),
                    len(ratios),
                    1,
                    fill=False,
                    hatch="//",
                    edgecolor="white",
                    linewidth=0,
                )
            )

    ax.set(
        xlim=(-0.5, len(ratios) - 0.5),
        ylim=(len(metrics) - 0.5, -0.5),
        xticks=range(len(ratios)),
        xticklabels=ratios,
        yticks=range(len(metrics)),
        yticklabels=[LABELS[metric] for metric in metrics],
    )

    ax.set_xlabel("Synthetic fraction")
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title(ARCH[payload["model"]], weight="bold")


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    payloads = {key: load(key) for key in ARCH}

    out = Path("results/figures/figure3")
    out.mkdir(parents=True, exist_ok=True)

    legend = [
        Patch(facecolor="#2E7D32", label="Correct"),
        Patch(facecolor="#C62828", label="Incorrect"),
        Patch(facecolor="#BDBDBD", label="Tie: not scored"),
        Patch(facecolor="#EEEEEE", label="Unavailable: metric has no value"),
    ]

    for qualified in (False, True):
        suffix = "sadge_qualified" if qualified else "standard"

        extra = (
            [
                Patch(
                    facecolor="white",
                    hatch="//",
                    edgecolor="#666666",
                    label="Fused SADGE: outside calibrated range",
                )
            ]
            if qualified
            else []
        )

        for key, payload in payloads.items():
            fig, ax = plt.subplots(figsize=(7.3, 5.8))
            draw(ax, payload, qualified)

            ax.legend(
                handles=legend + extra,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.28),
                frameon=False,
                ncol=2,
                fontsize=7.5,
            )

            fig.tight_layout()
            save(fig, out / f"figure3_{key}_miou_{suffix}.png")

        fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.9))

        for ax, payload in zip(axes, payloads.values()):
            draw(ax, payload, qualified)

        fig.legend(
            handles=legend + extra,
            loc="lower center",
            frameon=False,
            ncol=5,
            fontsize=7.5,
        )

        fig.tight_layout(rect=(0, 0.07, 1, 1))
        save(fig, out / f"figure3_combined_miou_{suffix}.png")


if __name__ == "__main__":
    main()