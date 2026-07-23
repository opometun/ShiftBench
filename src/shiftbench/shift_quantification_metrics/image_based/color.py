"""This file contains all functions relevant for the color-based shift quantification."""

import numpy as np
import cv2

from shiftbench.shift_quantification_metrics.distances import js_distance


def hsv_histogram(img:np.ndarray, h_bins:int=32, s_bins:int=16, v_bins:int=16) -> np.ndarray:
    """
    Compute normalized HSV histograms for an RGB image.
    Returns a concatenated feature vector: [H | S | V].
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    hist_H, _ = np.histogram(H.ravel(), bins=h_bins, range=(0, 180), density=True)
    hist_S, _ = np.histogram(S.ravel(), bins=s_bins, range=(0, 256), density=True)
    hist_V, _ = np.histogram(V.ravel(), bins=v_bins, range=(0, 256), density=True)

    return np.concatenate([hist_H, hist_S, hist_V])


def quantify_color_shift(train_ds:list, inference_ds:list) -> float:
    """Compute JS-based color distribution shift between two datasets."""
    # Get image-level feature vectors
    train_img_features = np.vstack([hsv_histogram(img) for img in train_ds])
    inference_img_features = np.vstack([hsv_histogram(img) for img in inference_ds])

    # Get dataset-level feature vectors
    train_ds_features = train_img_features.mean(axis=0)
    inference_ds_features = inference_img_features.mean(axis=0)

    # JS-distance
    distribution_shift = js_distance(a=train_ds_features, b=inference_ds_features)
    return distribution_shift
