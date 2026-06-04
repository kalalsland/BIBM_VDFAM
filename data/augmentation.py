"""Data augmentation utilities for class-imbalanced multimodal feature data.

Provides three SMOTE-based strategies and a weighted sampler builder.

Note: This module operates on *pre-encoded* feature vectors (numpy arrays),
not raw images or text.
"""

from typing import Tuple

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import WeightedRandomSampler


# ----------------------------------------------------------------- SMOTE ---


def smote_tabular_copy(
    tabular: np.ndarray,
    image: np.ndarray,
    text: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SMOTE on tabular features; copy nearest-neighbour image/text for synthetic rows.

    Synthetic tabular samples are generated via SMOTE.  For each synthetic
    row the nearest original tabular sample is located and its image/text
    features are copied verbatim so all modalities remain aligned.

    Returns:
        Resampled ``(tabular, image, text, labels)``.
    """
    sm = SMOTE(random_state=random_state)
    X_tab_res, y_res = sm.fit_resample(tabular, labels)

    nn = NearestNeighbors(n_neighbors=1).fit(tabular)
    idxs = nn.kneighbors(X_tab_res, return_distance=False).reshape(-1)

    return X_tab_res, image[idxs], text[idxs], y_res


def smote_image_copy(
    tabular: np.ndarray,
    image: np.ndarray,
    text: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SMOTE on image features; copy nearest-neighbour tabular/text for synthetic rows."""
    image_flat = image.reshape(image.shape[0], -1)
    sm = SMOTE(random_state=random_state)
    X_img_res, y_res = sm.fit_resample(image_flat, labels)

    nn = NearestNeighbors(n_neighbors=1).fit(image_flat)
    idxs = nn.kneighbors(X_img_res, return_distance=False).reshape(-1)
    X_img_res = X_img_res.reshape(-1, *image.shape[1:])

    return tabular[idxs], X_img_res, text[idxs], y_res


# -------------------------------------------------------- weighted sampler --


def build_sampler_for_labels(labels: np.ndarray) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` that up-samples the minority class."""
    class_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1.0 / class_count
    samples_weight = torch.from_numpy(
        np.array([weight[t] for t in labels])
    ).double()
    return WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)


# ------------------------------------------------------- apply by config --


def apply_smote(
    tabular: np.ndarray,
    image: np.ndarray,
    text: np.ndarray,
    labels: np.ndarray,
    smote_method: str,
    random_state: int = 42,
):
    """Apply the SMOTE strategy named in *smote_method*.

    Args:
        tabular, image, text, labels: Training split arrays.
        smote_method: One of ``"tabular_copy"``, ``"image_copy"``, ``"weighted"``,
                      ``"none"``.
        random_state: RNG seed for SMOTE.

    Returns:
        ``(tabular, image, text, labels, sampler)`` where *sampler* is a
        ``WeightedRandomSampler`` (when ``smote_method=="weighted"``) or ``None``.
    """
    sampler = None

    if smote_method == "tabular_copy":
        tabular, image, text, labels = smote_tabular_copy(
            tabular, image, text, labels, random_state=random_state
        )
        print("SMOTE(tabular_copy) applied.")
    elif smote_method == "image_copy":
        tabular, image, text, labels = smote_image_copy(
            tabular, image, text, labels, random_state=random_state
        )
        print("SMOTE(image_copy) applied.")
    elif smote_method == "weighted":
        sampler = build_sampler_for_labels(labels)
        print("Using WeightedRandomSampler for class balance.")
    # "none" → do nothing

    return tabular, image, text, labels, sampler
