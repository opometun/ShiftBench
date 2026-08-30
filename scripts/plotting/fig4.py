#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

ARCH = {
    "segformer": ("SegFormer", "s", "-"),
    "deeplabv3plus": ("DeepLabV3+", "o", ":"),
}

SERIES = [
    (
        "Synscapes",
        [
            "cityscapes100",
            "cityscapes75_synscapes25",
            "cityscapes50_synscapes50",
            "cityscapes25_synscapes75",
            "synscapes100",
        ],
        "#1976A8",
    ),
    (
        "GTA-V",
        [
            "cityscapes100",
            "cityscapes75_gta25",
            "cityscapes50_gta50",
            "cityscapes25_gta75",
            "gta100",
        ],
        "#E69F00",
    ),
]

RATIOS = np.array([0, 25, 50, 75, 100])


def load(key):
    path = Path(f"results/shift/correlation_{key}.json")

    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")

    return json.loads(path.read_text())


def mean_ci(values, ddof):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    sd = values.std(ddof=ddof)

    half_width = (
        t.ppf(0.975, len(values) - 1)
        * sd
        / np.sqrt(len(values))
    )

    return mean, half_width


def main():
    payloads = {key: load(key) for key in ARCH}

    all_values = [
        value
        for payload in payloads.values()
        for values in payload["performance_per_seed"].values()
        for value in values
    ]

    low = min(all_values)
    high = max(all_values)
    padding = max(0.01, 0.06 * (high - low))

    out = Path("results/figures/figure4")
    out.mkdir(parents=True, exist_ok=True)

    # Generate figure with ddof=1 (95% t confidence intervals)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for key, payload in payloads.items():
        model, marker, linestyle = ARCH[key]

        for source, mixtures, colour in SERIES:
            values = [
                mean_ci(
                    payload["performance_per_seed"][mixture],
                    ddof=1,
                )
                for mixture in mixtures
            ]

            means = np.array([result[0] for result in values])
            ci = np.array([result[1] for result in values])

            ax.plot(
                RATIOS,
                means,
                color=colour,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=7,
                label=f"{source}, {model}",
            )

            ax.fill_between(
                RATIOS,
                means - ci,
                means + ci,
                color=colour,
                alpha=0.13,
            )

    ax.set(
        xlim=(-5, 105),
        ylim=(low - padding, high + padding),
        xticks=RATIOS,
        xlabel="Synthetic images in training mixture (%)",
        ylabel="Test mIoU",
    )

    ax.grid(color="#D9D9D9", linewidth=0.7)
    ax.legend(frameon=False, loc="lower left", fontsize=9)

    fig.text(
        0.5,
        0.01,
        "Points show means across six training seeds; "
        "shaded regions show two-sided 95% t confidence intervals.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    fig.savefig(
        out / "figure4_miou_ci95.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Generate alternative figure with ddof=0 (±1 SD)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for key, payload in payloads.items():
        model, marker, linestyle = ARCH[key]

        for source, mixtures, colour in SERIES:
            values = [
                mean_ci(
                    payload["performance_per_seed"][mixture],
                    ddof=0,
                )
                for mixture in mixtures
            ]

            means = np.array([result[0] for result in values])
            sd = np.array([result[1] for result in values])

            ax.plot(
                RATIOS,
                means,
                color=colour,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=7,
                label=f"{source}, {model}",
            )

            ax.fill_between(
                RATIOS,
                means - sd,
                means + sd,
                color=colour,
                alpha=0.13,
            )

    ax.set(
        xlim=(-5, 105),
        ylim=(low - padding, high + padding),
        xticks=RATIOS,
        xlabel="Synthetic images in training mixture (%)",
        ylabel="Test mIoU",
    )

    ax.grid(color="#D9D9D9", linewidth=0.7)
    ax.legend(frameon=False, loc="lower left", fontsize=9)

    fig.text(
        0.5,
        0.01,
        "Points show means across six training seeds; "
        "shaded regions show ±1 standard deviation.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    fig.savefig(
        out / "figure4_miou_sd.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Wrote Figure 4 PNGs to {out.resolve()}")


if __name__ == "__main__":
    main()
