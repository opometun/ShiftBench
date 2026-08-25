"""Plot the summaries of the original (i.e. 100%) datasets Cityscapes (train and test), Synscapes, and GTA-V used in the study.
Use this script to replicate the data analysis plots.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from shiftbench.metrics.gaussian import centroid_distance

# -------------------------------------------------------------------
# CITYSCAPES LABELS
# -------------------------------------------------------------------
LABELS = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
    5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation',
    9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
    14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle',
}

# -------------------------------------------------------------------
# COLOR / STYLE CONFIGURATION
# -------------------------------------------------------------------
DATASET_ORDER = ["cityscapes_test", "cityscapes_train", "synscapes_train", "gta_train"]
TRAIN_SETS = ["cityscapes_train", "synscapes_train", "gta_train"]

DATASET_COLORS = {
    "cityscapes_train": "black",
    "cityscapes_test": "grey",
    "synscapes_train": "#0072B2",
    "gta_train": "#D55E00",
}
DATASET_MARKERS = {
    "cityscapes_test": "o", "cityscapes_train": "o",
    "synscapes_train": "s", "gta_train": "^",
}
LINESTYLES = {
    "cityscapes_train": "-", "synscapes_train": "--", "gta_train": ":",
}
FILL_ALPHA = {"cityscapes_test": 0.25}
LEGEND_LABELS = {
    "cityscapes_test": "Cityscapes (test)",
    "cityscapes_train": "Cityscapes (train)",
    "synscapes_train": "Synscapes (train)",
    "gta_train": "GTA-V (train)",
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
})

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
BASE = "results/shift/summaries"
FILES = {
    "cityscapes_test": {
        "class_frequency": f"{BASE}/inference_class_frequency.npz",
        "class_presence": f"{BASE}/inference_class_presence.npz",
        "color": f"{BASE}/inference_color.npz",
        "texture": f"{BASE}/inference_texture.npz",
        "scene_complexity": f"{BASE}/inference_scene_complexity.npz",
        "dinov2": f"{BASE}/inference_gaussian_dinov2.npz",
        "streetclip": f"{BASE}/inference_gaussian_streetclip.npz",
    },
    "cityscapes_train": {
        "class_frequency": f"{BASE}/cityscapes100_class_frequency.npz",
        "class_presence": f"{BASE}/cityscapes100_class_presence.npz",
        "color": f"{BASE}/cityscapes100_color.npz",
        "texture": f"{BASE}/cityscapes100_texture.npz",
        "scene_complexity": f"{BASE}/cityscapes100_scene_complexity.npz",
        "dinov2": f"{BASE}/cityscapes100_gaussian_dinov2.npz",
        "streetclip": f"{BASE}/cityscapes100_gaussian_streetclip.npz",
    },
    "synscapes_train": {
        "class_frequency": f"{BASE}/synscapes100_class_frequency.npz",
        "class_presence": f"{BASE}/synscapes100_class_presence.npz",
        "color": f"{BASE}/synscapes100_color.npz",
        "texture": f"{BASE}/synscapes100_texture.npz",
        "scene_complexity": f"{BASE}/synscapes100_scene_complexity.npz",
        "dinov2": f"{BASE}/synscapes100_gaussian_dinov2.npz",
        "streetclip": f"{BASE}/synscapes100_gaussian_streetclip.npz",
    },
    "gta_train": {
        "class_frequency": f"{BASE}/gta100_class_frequency.npz",
        "class_presence": f"{BASE}/gta100_class_presence.npz",
        "color": f"{BASE}/gta100_color.npz",
        "texture": f"{BASE}/gta100_texture.npz",
        "scene_complexity": f"{BASE}/gta100_scene_complexity.npz",
        "dinov2": f"{BASE}/gta100_gaussian_dinov2.npz",
        "streetclip": f"{BASE}/gta100_gaussian_streetclip.npz",
    },
}

SAVE_PATH = "results/shift/figures/orig_dataset_analysis"

# -------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------
def save_fig(figure, filename: str, base_path: str = SAVE_PATH, pad: float = 0.3, h_pad: float | None = None):
    """Save figure in a fixed layout."""
    full_path = os.path.join(base_path, filename) if base_path else filename
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    kwargs = {"pad": pad}
    if h_pad is not None:
        kwargs["h_pad"] = h_pad

    figure.tight_layout(**kwargs)
    figure.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()


def step_plot_legend():
    """Prepare legend handles for step plot (scene complexity, color)."""
    handles = [
        mpatches.Patch(
            facecolor=DATASET_COLORS["cityscapes_test"],
            alpha=FILL_ALPHA["cityscapes_test"],
            label=LEGEND_LABELS["cityscapes_test"],
        )
    ]
    for dataset in TRAIN_SETS:
        handles.append(
            mlines.Line2D(
                [], [], color=DATASET_COLORS[dataset], linestyle=LINESTYLES[dataset],
                linewidth=1.3, label=LEGEND_LABELS[dataset],
            )
        )
    return handles


def cluster_dodge_offsets(hist_dict, order:np.ndarray, num_rows:int, log_scale:bool, collision_thresh:float, dodge:float):
    """Assign small vertical offsets so close dots don't overlap when plotted 
    (isolated points get zero offset).

    Returns: {dataset: array of length num_rows} of y-offsets.
    """
    y_offsets = {d: np.zeros(num_rows) for d in DATASET_ORDER}

    for row in range(num_rows):
        vals = np.array([hist_dict[d][order][row] for d in DATASET_ORDER])
        metric = np.log10(np.clip(vals, 1e-12, None)) if log_scale else vals
        idx_by_val = np.argsort(metric)

        clusters = [[idx_by_val[0]]]
        for idx in idx_by_val[1:]:
            if metric[idx] - metric[clusters[-1][-1]] < collision_thresh:
                clusters[-1].append(idx)
            else:
                clusters.append([idx])

        for cluster in clusters:
            k = len(cluster)
            if k == 1:
                continue
            local_offsets = (np.arange(k) - (k - 1) / 2) * dodge
            for pos, dataset_idx in zip(local_offsets, cluster):
                y_offsets[DATASET_ORDER[dataset_idx]][row] = pos

    return y_offsets


def sorted_class_order(hist_dict, labels:dict, num_classes:int):
    """Sort classes by descending mean value across datasets."""
    mean_val = np.mean([hist_dict[d][:num_classes] for d in hist_dict], axis=0)
    order = np.argsort(mean_val)[::-1]
    class_names = [labels[i] for i in order]
    return order, class_names


def plot_dodged_dots(hist_dict, labels, num_classes, xlabel, filename,
                       log_scale=True, collision_thresh=0.06, dodge=0.13):
    """
    Generic Cleveland dot plot with collision-aware dodge: guide line spans
    min-max per row, markers dodge only where values nearly coincide.
    Used for class_frequency and class_presence — same visual logic, only
    the input histogram, x-label, and log/linear scale differ.
    """
    order, class_names = sorted_class_order(hist_dict, labels, num_classes)
    y = np.arange(num_classes)

    plt.figure(figsize=(3.5, 5.5))

    for row in y:
        vals = [hist_dict[d][order][row] for d in DATASET_ORDER]
        plt.plot([min(vals), max(vals)], [row, row], color="lightgrey", linewidth=1, zorder=1)

    y_offsets = cluster_dodge_offsets(hist_dict, order, num_classes, log_scale,
                                        collision_thresh, dodge)

    for dataset in DATASET_ORDER:
        values = hist_dict[dataset][order]
        plt.scatter(
            values, y + y_offsets[dataset],
            color=DATASET_COLORS[dataset], marker=DATASET_MARKERS[dataset],
            s=20, label=LEGEND_LABELS[dataset], zorder=3,
            edgecolors="white", linewidths=0.4,
        )

    plt.yticks(y, class_names)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel + (" (log scale)" if log_scale else ""))
    if log_scale:
        plt.xscale("log")

    plt.grid(axis="x", which="major", alpha=0.3, linewidth=0.5)
    plt.legend(loc="upper left", frameon=True, handlelength=1.5)
    save_fig(plt, filename)


def plot_scene_complexity_hist(hist_dict) -> None:
    """Plot the scene complexity histograms of all datasets."""
    num_bins = len(next(iter(hist_dict.values())))
    x = np.arange(num_bins)

    plt.figure(figsize=(3.5, 2.5))
    plt.fill_between(
        x, hist_dict["cityscapes_test"], step="mid",
        color=DATASET_COLORS["cityscapes_test"], alpha=FILL_ALPHA["cityscapes_test"],
        linewidth=0, zorder=1,
    )
    for dataset in TRAIN_SETS:
        plt.step(
            x, hist_dict[dataset], where="mid", color=DATASET_COLORS[dataset],
            linestyle=LINESTYLES[dataset], linewidth=1.3, zorder=2,
        )

    plt.xlabel("number of distinct classes per sample")
    plt.ylabel("proportion of samples in dataset")
    plt.xticks(x[::2])
    plt.xlim(x[0] - 0.5, x[-1] + 0.5)
    plt.legend(handles=step_plot_legend(), loc="upper left", frameon=True, handlelength=1.8)
    save_fig(plt, "scene_complexity.png")


def _split_hsv_hist(hist, h_bins=32, s_bins=16, v_bins=16):
    """Split the HSV histogram into three separate histograms."""
    h = hist[:h_bins]
    s = hist[h_bins:h_bins + s_bins]
    v = hist[h_bins + s_bins:h_bins + s_bins + v_bins]
    return h, s, v

def plot_hsv_histograms(hist_dict, h_bins=32, s_bins=16, v_bins=16):
    """Plot the color histograms of all datasets."""
    channels = [("Hue", h_bins, (0, 180)), ("Saturation", s_bins, (0, 256)), ("Value", v_bins, (0, 256))]
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.5))

    for ax, (name, n_bins, (lo, hi)) in zip(axes, channels):
        bin_centers = (np.linspace(lo, hi, n_bins + 1)[:-1] + np.linspace(lo, hi, n_bins + 1)[1:]) / 2

        h, s, v = _split_hsv_hist(hist_dict["cityscapes_test"], h_bins, s_bins, v_bins)
        values = {"Hue": h, "Saturation": s, "Value": v}[name]
        ax.fill_between(bin_centers, values, step="mid", color=DATASET_COLORS["cityscapes_test"],
                          alpha=FILL_ALPHA["cityscapes_test"], linewidth=0, zorder=1)

        for dataset in TRAIN_SETS:
            h, s, v = _split_hsv_hist(hist_dict[dataset], h_bins, s_bins, v_bins)
            values = {"Hue": h, "Saturation": s, "Value": v}[name]
            ax.step(bin_centers, values, where="mid", color=DATASET_COLORS[dataset],
                     linestyle=LINESTYLES[dataset], linewidth=1.3, zorder=2)

        ax.set_title(name, fontsize=8, pad=3)
        ax.set_xlim(lo, hi)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    axes[1].legend(handles=step_plot_legend(), loc="upper right", fontsize=6,
                    frameon=True, handlelength=1.8)
    save_fig(fig, "color_hsv.png", h_pad=1.2)


def plot_lbp_histograms(hist_dict, configs):
    """Plot the texture histograms of all datasets."""
    n_bins_per_config = [P + 2 for P, R in configs]
    offsets = np.cumsum([0] + n_bins_per_config)
    fig, axes = plt.subplots(len(configs), 1, figsize=(3.5, 2.6 * len(configs)))
    if len(configs) == 1:
        axes = [axes]

    for ax, (P, R), start, n_bins in zip(axes, configs, offsets, n_bins_per_config):
        x = np.arange(n_bins)
        bin_labels = [str(i) for i in range(P + 1)] + ["n.u."]
        bar_width = 0.8 / len(DATASET_ORDER)

        for i, dataset in enumerate(DATASET_ORDER):
            values = hist_dict[dataset][start:start + n_bins]
            offset = (i - (len(DATASET_ORDER) - 1) / 2) * bar_width
            ax.bar(x + offset, values, width=bar_width, color=DATASET_COLORS[dataset],
                    label=LEGEND_LABELS[dataset])

        ax.axvline(x[-1] - 0.5, color="grey", linewidth=0.6, linestyle=":")
        ax.set_title(f"P={P}, R={R}", fontsize=8, pad=3)
        ax.set_xlim(x[0] - 0.75, x[-1] + 0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=6 if n_bins <= 10 else 5)
        ax.set_ylabel("density", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    axes[0].legend(fontsize=6, loc="upper left", frameon=True,
                    framealpha=0.9, facecolor="white", handlelength=1.5)
    save_fig(fig, "texture_lbp.png", h_pad=1.2)


def plot_mean_shift(
    mus: dict, reference: str = "cityscapes_test",
    embedding_models: list = ("dinov2", "streetclip"),
    save_path: str = SAVE_PATH,
):
    """Plot the shift in distribution means using cosine distance."""
    datasets = [d for d in DATASET_ORDER if d != reference]
    x = np.arange(len(datasets)) * 0.6
    bar_width = 0.4

    fig, axes = plt.subplots(1, len(embedding_models), figsize=(3.5, 1.8), sharey=False)
    if len(embedding_models) == 1:
        axes = [axes]

    for ax, model in zip(axes, embedding_models):
        vals = [
            centroid_distance(mus[model][reference], mus[model][dataset])
            for dataset in datasets
        ]

        bars = ax.bar(
            x, vals, width=bar_width,
            color=[DATASET_COLORS[d] for d in datasets],
            edgecolor="black", linewidth=0.4,
        )

        ax.set_title(model, fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.grid(axis="y", alpha=0.3, linewidth=0.4)

    axes[0].set_ylabel("centroid distance\nfrom Cityscapes (test)", fontsize=6)
    handles = [plt.Rectangle((0,0),1,1, color=DATASET_COLORS[d]) for d in datasets]
    labels = [LEGEND_LABELS[d] for d in datasets]

    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.15), ncol=len(datasets),
               fontsize=6, frameon=False)

    plt.tight_layout(w_pad=1.0)
    plt.savefig(f"{save_path}/mean_shift_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_covariance_spectrum(
    sigmas: dict, 
    embedding_models: list = ("dinov2", "streetclip"),
    top_k: int = 50, 
    save_path: str = SAVE_PATH,
):
    """Plot the Eigenvalue spectrum per dataset."""
    fig, axes = plt.subplots(len(embedding_models), 1, figsize=(3.5, 1.8 * len(embedding_models)))
    if len(embedding_models) == 1:
        axes = [axes]

    for ax, model in zip(axes, embedding_models):
        for dataset in DATASET_ORDER:
            eigvals = np.linalg.eigvalsh(sigmas[model][dataset])[::-1][:top_k]
            ax.plot(
                np.arange(top_k), eigvals,
                color=DATASET_COLORS[dataset],
                linestyle=LINESTYLES.get(dataset, ":"),
                linewidth=1.3, label=LEGEND_LABELS[dataset],
            )
        ax.set_yscale("log")
        ax.set_title(model, fontsize=8, pad=3)
        ax.set_ylabel("eigenvalue (log)", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    axes[-1].set_xlabel("eigenvalue rank")
    axes[0].legend(fontsize=6, loc="upper right")
    plt.tight_layout(h_pad=1.2)
    plt.savefig(f"{save_path}/covariance_spectrum_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------
# MAIN CODE
# -------------------------------------------------------------------

def main() -> None:
    metrics = ["class_frequency", "class_presence", "color", "texture", "scene_complexity"]
    histograms = {metric: {} for metric in metrics}
    for dataset, metric_files in FILES.items():
        for metric, path in metric_files.items():
            if metric not in metrics:
                continue
            data = np.load(path, allow_pickle=True)
            histograms[metric][dataset] = data["hist"]

    plot_hsv_histograms(histograms["color"])

    plot_lbp_histograms(histograms["texture"], configs=[(8, 1), (16, 2)])
    
    plot_dodged_dots(
        histograms["class_frequency"], LABELS, num_classes=19,
        xlabel="proportion of pixels", filename="class_frequency.png",
        log_scale=True, collision_thresh=0.06, dodge=0.13,
    )

    plot_dodged_dots(
        histograms["class_presence"], LABELS, num_classes=19,
        xlabel="class presence in dataset\n(relative to other classes)",
        filename="class_presence.png",
        log_scale=False, collision_thresh=0.002, dodge=0.13,
    )

    plot_scene_complexity_hist(histograms["scene_complexity"])

    metrics = ["dinov2", "streetclip"]
    mus = {metric: {} for metric in metrics}
    sigmas = {metric: {} for metric in metrics}
    for dataset, metric_files in FILES.items():
        for metric, path in metric_files.items():
            if metric not in metrics:
                continue
            data = np.load(path, allow_pickle=True)
            mus[metric][dataset] = data["mu"]
            sigmas[metric][dataset] = data["sigma"]

    plot_mean_shift(mus, reference="cityscapes_test")
    plot_covariance_spectrum(sigmas)


if __name__ == "__main__":
    main()