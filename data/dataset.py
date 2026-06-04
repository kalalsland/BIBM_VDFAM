"""PyTorch Dataset for pre-encoded multimodal stroke prognosis data.

Data are already encoded as feature vectors (tabular, image, text).
StandardScaler normalization is applied per modality; scalers fitted on the
training split are passed to val/test splits to prevent leakage.
"""

from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from ..config import CONFIG


class MultimodalDataset(Dataset):
    """Dataset wrapping pre-encoded tabular / image / text feature arrays.

    Args:
        tabular:        (N, D_tab) tabular features.
        image:          (N, D_img) image features.
        text:           (N, D_txt) text features.
        labels:         (N,)       integer class labels.
        tabular_scaler: Fitted ``StandardScaler`` for tabular.  ``None`` → fit new.
        image_scaler:   Fitted ``StandardScaler`` for image.    ``None`` → fit new.
        text_scaler:    Fitted ``StandardScaler`` for text.     ``None`` → fit new.
        is_train:       Whether this split is training (enables Gaussian noise).
    """

    def __init__(
        self,
        tabular: np.ndarray,
        image: np.ndarray,
        text: np.ndarray,
        labels: np.ndarray,
        tabular_scaler: Optional[StandardScaler] = None,
        image_scaler: Optional[StandardScaler] = None,
        text_scaler: Optional[StandardScaler] = None,
        is_train: bool = True,
    ) -> None:
        super().__init__()

        self.tabular_scaler, self.tabular = self._scale(tabular, tabular_scaler)
        self.image_scaler, self.image = self._scale(image, image_scaler)
        self.text_scaler, self.text = self._scale(text, text_scaler)

        self.labels = labels
        self.is_train = is_train

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _scale(
        X: np.ndarray, scaler: Optional[StandardScaler]
    ):
        if scaler is None:
            scaler = StandardScaler()
            return scaler, scaler.fit_transform(X)
        return scaler, scaler.transform(X)

    # -------------------------------------------------------------- interface
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        tab = self.tabular[idx].copy()
        img = self.image[idx].copy()
        txt = self.text[idx].copy()

        if self.is_train and CONFIG["use_gaussian_noise"] and np.random.rand() > 0.5:
            tab += np.random.normal(0, 0.01, tab.shape)
            img += np.random.normal(0, 0.01, img.shape)
            txt += np.random.normal(0, 0.01, txt.shape)

        return (
            torch.tensor(tab, dtype=torch.float32),
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(txt, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
