"""Load pre-encoded multimodal features from Excel files.

Raw images and clinical notes are private; this module only handles the
already-encoded feature representations produced by the separate encoder
scripts (encoder/image_encoder, encoder/text_encoder, encoder/label_encoder).
"""

import ast
from typing import Tuple

import numpy as np
import pandas as pd


def load_data(
    clinic_path: str,
    image_feat_path: str,
    text_feat_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and merge the three modality feature files.

    Args:
        clinic_path:     Path to clinic Excel file (contains tabular features + label).
        image_feat_path: Path to image feature Excel file (column ``image_features``).
        text_feat_path:  Path to text feature Excel file  (column ``text_features``).

    Returns:
        Tuple ``(tabular, image, text, labels)`` as numpy arrays.
        - tabular: (N, D_tab)  – all columns between 'id' and the last three
        - image:   (N, D_img)  – stacked image feature vectors
        - text:    (N, D_txt)  – stacked text feature vectors
        - labels:  (N,)        – integer class labels (0 / 1)
    """
    clinic_data = pd.read_excel(clinic_path)
    image_feats = pd.read_excel(image_feat_path)
    text_feats = pd.read_excel(text_feat_path)

    merged = clinic_data.merge(image_feats, on="id", how="inner")
    merged = merged.merge(text_feats, on="id", how="inner")

    # Feature columns stored as stringified Python lists → parse them.
    merged["image_features"] = merged["image_features"].apply(ast.literal_eval)
    merged["text_features"] = merged["text_features"].apply(ast.literal_eval)

    # Tabular: everything between 'id' (col 0) and the last three columns
    # (image_features, text_features, label).
    tabular_features = merged.iloc[:, 1:-3].values
    image_features = np.vstack(merged["image_features"].values)
    text_features = np.vstack(merged["text_features"].values)
    labels = merged["label"].values.astype(int)

    return tabular_features, image_features, text_features, labels
