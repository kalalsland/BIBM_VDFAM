"""OursFusion: Vision-Guided Dual Alignment Fusion model.

Architecture:
  1. VisionGuidedTextEncoder   – cross-modal interaction in hidden space
  2. Low-rank projection heads  – image_projection, text_projection
  3. Alignment MLPs             – symmetric cosine calibration loss
  4. Fusion + Classifier        – concat reduce → [tabular ‖ fused] → MLP

The model is called ``OursFusion`` to align with the ablation study naming
convention used throughout this project.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ----------------------------------------------- VisionGuidedTextEncoder --


class VisionGuidedTextEncoder(nn.Module):
    """Vision-guided cross-modal encoder.

    Maps image and text features into a shared hidden space via a small
    Transformer encoder, allowing bidirectional interaction.

    Forward returns:
        E_img_encoded: (B, hidden_dim) – image token after cross-modal update.
        E_txt_encoded: (B, hidden_dim) – text  token after cross-modal update.
        V_txt:         (B, hidden_dim) – text token mapped back to vision space.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.vision_embed = nn.Linear(vision_dim, hidden_dim)
        self.text_embed = nn.Linear(text_dim, hidden_dim)

        self.mlp_v2t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_t2v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, vision_feats: Tensor, text_feats: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        v_img = self.vision_embed(vision_feats)
        e_txt = self.text_embed(text_feats)

        e_img = self.mlp_v2t(v_img)

        combined = torch.stack([e_img, e_txt], dim=1)  # (B, 2, D)
        encoded = self.encoder(combined)               # (B, 2, D)

        e_img_encoded = encoded[:, 0]
        e_txt_encoded = encoded[:, 1]
        v_txt = self.mlp_t2v(e_txt_encoded)

        return e_img_encoded, e_txt_encoded, v_txt


# ---------------------------------------------------------------- OursFusion --


class OursFusion(nn.Module):
    """Three-modality fusion model (tabular + image + text).

    Args:
        tabular_dim: Dimension of tabular features.
        image_dim:   Dimension of image features.
        text_dim:    Dimension of text features.
        cfg:         Config dict with keys:
                     ``hidden_dim``, ``projection_dim``, ``num_heads``,
                     ``dropout_rate``.
    """

    def __init__(
        self,
        tabular_dim: int,
        image_dim: int,
        text_dim: int,
        cfg: Dict,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.tabular_fc = nn.Identity()

        self.vision_guided_encoder = VisionGuidedTextEncoder(
            vision_dim=image_dim,
            text_dim=text_dim,
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout_rate"],
        )

        self.image_projection = nn.Linear(cfg["hidden_dim"], cfg["projection_dim"])
        self.text_projection = nn.Linear(cfg["hidden_dim"], cfg["projection_dim"])

        self.fusion_reduction = nn.Sequential(
            nn.Linear(2 * cfg["projection_dim"], cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout_rate"]),
            nn.Linear(cfg["hidden_dim"], tabular_dim),
            nn.ReLU(),
            nn.Dropout(cfg["dropout_rate"]),
        )

        classifier_input_dim = 2 * tabular_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(cfg["dropout_rate"]),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(cfg["dropout_rate"]),
            nn.Linear(128, 2),
        )

        # Learnable temperature (softplus ensures positive)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Projection calibration MLPs (hidden space)
        self.mlp_image_to_text = nn.Sequential(
            nn.Linear(cfg["hidden_dim"], cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout_rate"]),
            nn.Linear(cfg["hidden_dim"], cfg["hidden_dim"]),
        )
        self.mlp_text_to_image = nn.Sequential(
            nn.Linear(cfg["hidden_dim"], cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout_rate"]),
            nn.Linear(cfg["hidden_dim"], cfg["hidden_dim"]),
        )

    def forward(
        self,
        tabular: Tensor,
        image: Tensor,
        text: Tensor,
        compute_contrastive: bool = True,
        compute_alignment: bool = True,
        compute_cl: bool = True,
        compute_align: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            logits:            (B, 2)
            contrastive_loss:  scalar
            alignment_loss:    scalar
        """
        tabular_out = self.tabular_fc(tabular)

        e_img_encoded, e_txt_encoded, _ = self.vision_guided_encoder(image, text)

        image_proj = self.image_projection(e_img_encoded)
        text_proj = self.text_projection(e_txt_encoded)

        contrastive_loss = torch.tensor(0.0, device=image_proj.device)
        if compute_contrastive and compute_cl:
            contrastive_loss = self._contrastive_loss(image_proj, text_proj)

        alignment_loss = torch.tensor(0.0, device=image_proj.device)
        if compute_alignment and compute_align:
            img2txt = self.mlp_image_to_text(e_img_encoded)
            txt2img = self.mlp_text_to_image(e_txt_encoded)
            loss_i2t = 1.0 - F.cosine_similarity(img2txt, e_txt_encoded, dim=-1).mean()
            loss_t2i = 1.0 - F.cosine_similarity(txt2img, e_img_encoded, dim=-1).mean()
            alignment_loss = 0.5 * (loss_i2t + loss_t2i)

        fused = torch.cat([image_proj, text_proj], dim=1)
        fused_reduced = self.fusion_reduction(fused)
        combined = torch.cat([tabular_out, fused_reduced], dim=1)
        logits = self.classifier(combined)

        return logits, contrastive_loss, alignment_loss

    def _contrastive_loss(self, z_img: Tensor, z_txt: Tensor) -> Tensor:
        """InfoNCE-style symmetric contrastive loss."""
        z_img = F.normalize(z_img, dim=1)
        z_txt = F.normalize(z_txt, dim=1)
        temp = F.softplus(self.logit_scale) + 1e-6
        logits = (z_img @ z_txt.t()) / temp
        labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i + loss_t)
