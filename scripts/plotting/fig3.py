#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})


LABELS = {
    "color_js": r"$\mathrm{JS}_{\mathrm{color}}$",
    "texture_js": r"$\mathrm{JS}_{\mathrm{texture}}$",
    "class_frequency_js": r"$\mathrm{JS}_{\mathrm{classFrequency}}$",
    "class_presence_js": r"$\mathrm{JS}_{\mathrm{classPresence}}$",
    "scene_complexity_js": r"$\mathrm{JS}_{\mathrm{sceneComplexity}}$",
    "frechet_dinov2": r"$\mathrm{Fr\acute{e}chet}_{\mathrm{DINOv2}}$",
    "centroid_dinov2": r"$\mathrm{Centroid}_{\mathrm{DINOv2}}$",
    "frechet_streetclip": r"$\mathrm{Fr\acute{e}chet}_{\mathrm{StreetCLIP}}$",
    "centroid_streetclip": r"$\mathrm{Centroid}_{\mathrm{StreetCLIP}}$",
    "sadge": r"$\mathrm{SADGE}$",
    "sadge_appearance": r"$\overline{A}_{\mathrm{SADGE}}$",
    "sadge_geometry": r"$\overline{G}_{\mathrm{SADGE}}$",
}

ARCH = {
    "segformer": "SegFormer",
    "deeplabv3plus": "DeepLabV3+",
}

OUT_OF_RANGE = {"sadge"}


def load(model):
    path = Path(f"results/shift/correlation_{model}.json")

    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("target") != "miou":
        raise SystemExit(
            f"{path} must contain target='miou'."
        )

    return data


def draw(axis, payload, qualified):
    entries = payload["within_ratio"]["metrics"]

    metrics = sorted(
        entries,
        key=lambda metric: (
            -(
                entries[metric]["accuracy"]
                if entries[metric]["accuracy"] is not None
                else -1
            ),
            LABELS[metric],
        ),
    )

    ratios = [
        pair["ratio"]
        for pair in payload["within_ratio"]["pairs"]
    ]

    for row, metric in enumerate(metrics):
        for col, call in enumerate(entries[metric]["calls"]):
            correct = call.get("correct")

            if correct is True:
                colour = "#2E7D32"
                symbol = "✓"
                text_colour = "white"
            elif correct is False:
                colour = "#C62828"
                symbol = "×"
                text_colour = "white"
            else:
                colour = "#BDBDBD"
                symbol = "–"
                text_colour = "#333333"

            axis.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=colour,
                    edgecolor="white",
                    linewidth=1.2,
                )
            )

            axis.text(
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
            axis.add_patch(
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

    axis.set(
        xlim=(-0.5, len(ratios) - 0.5),
        ylim=(len(metrics) - 0.5, -0.5),
        xticks=range(len(ratios)),
        xticklabels=ratios,
        yticks=range(len(metrics)),
        yticklabels=[LABELS[metric] for metric in metrics],
    )

    axis.set_xlabel("Synthetic fraction")
    axis.tick_params(axis="y", length=0, labelsize=9)
    axis.tick_params(axis="x", labelsize=9)
    axis.set_title(ARCH[payload["model"]], weight="bold")


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

    print(f"Wrote Figure 3 PNGs to {out.resolve()}")


if __name__ == "__main__":
    main()
