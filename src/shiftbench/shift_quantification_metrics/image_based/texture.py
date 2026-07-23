"""This file contains all functions relevant for the texture-based shift quantification."""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern

from shiftbench.shift_quantification_metrics.distances import js_distance


def lbp_histogram(img:np.ndarray, n_neighbors:int=8, radius:int=1) -> np.ndarray:
    """Compute a normalized Local Binary Pattern histogram from a grayscale image."""
    lbp = local_binary_pattern(img, P=n_neighbors, R=radius, method='uniform')
    n_bins = n_neighbors + 2
    hist, _ = np.histogram(lbp.ravel(),
                           bins=n_bins,
                           range=(0, n_bins),
                           density=True)
    return hist


def multiscale_lbp(img:np.ndarray, configs:list[tuple[int,int]]) -> np.ndarray:
    """Compute multiscale LBP histograms for different (P,R) pairs."""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    all_hists = []
    for P, R in configs:
        hist = lbp_histogram(img, n_neighbors=P, radius=R)
        all_hists.append(hist)

    return np.concatenate(all_hists)


def quantify_texture_shift(train_ds:list, inference_ds:list, configs:list=[(8,1), (16,2)]) -> float:
    """Compute JS-based texture distribution shift between two datasets."""
    # Get image-level feature vectors
    train_img_features = np.vstack([multiscale_lbp(img, configs=configs) for img in train_ds])   # (n_images, n_bins) array of per-image histograms -> n_bins is sum over all P+2
    inference_img_features = np.vstack([multiscale_lbp(img, configs=configs) for img in inference_ds])

    # Get dataset-level feature vectors
    train_ds_features = train_img_features.mean(axis=0)
    inference_ds_features = inference_img_features.mean(axis=0)

    # JS-distance 
    distribution_shift = js_distance(a=train_ds_features, b=inference_ds_features)
    return distribution_shift
