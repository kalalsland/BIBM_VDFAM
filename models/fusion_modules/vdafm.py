"""VDAFM (Vision-Guided Dual Alignment Fusion Module).

This module is a refactored, research-oriented implementation extracted from the
legacy prototype in `MMF/base_fusion.py`.

The goal of VDAFM is to *align* and *fuse* vision/text representations before
combining them with tabular features for prognosis prediction.

Key ideas (as used in this codebase)
-----------------------------------
1. Vision-guided text encoding
   - Map image features and text features into a shared hidden space.
   - Use a lightweight Transformer encoder over a 2-token sequence
     `[E_img, E_txt]` to allow cross-modal interaction.

2. Low-rank projection (projection head)
   - Project both modalities from `hidden_dim -> projection_dim`.
   - In practice, choosing `projection_dim << hidden_dim` acts as an explicit
     low-rank bottleneck:

       Z_img = P_img(E_img_encoded),   Z_txt = P_txt(E_txt_encoded)

     where `P_*` are linear maps with output dimension `r = projection_dim`.

3. Projection calibration (alignment MLPs)
   - Learn small MLPs to calibrate each modality toward the other modality's
     encoded representation:

       hat_E_img2txt = g_img(E_img_encoded)
       hat_E_txt2img = g_txt(E_txt_encoded)

     The alignment loss is symmetric cosine distance:

       L_align = 0.5 * (1 - cos(hat_E_img2txt, E_txt_encoded))
              + 0.5 * (1 - cos(hat_E_txt2img, E_img_encoded))

4. Contrastive learning (optional)
   - Use an InfoNCE-style loss on the projected features (Z_img, Z_txt).

This file focuses on the *model definition* and keeps training/data concerns
outside, complying with `PROJECT_STRUCTURE_PLAN.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FusionStrategy = Literal[
    "concat_reduce_concat",
    "ca_fusion_concat",
    "add_reduce_concat",
    "add_reduce_add",
    "separate_reduce_concat",
    "separate_reduce_add",
]


@dataclass(frozen=True)
class VDAFMConfig:
    """Configuration for VDAFM.

    Attributes:
        image_dim: Input image feature dimension.
        text_dim: Input text feature dimension.
        tabular_dim: Input tabular feature dimension.
        hidden_dim: Shared hidden dimension used for cross-modal interaction.
        projection_dim: Bottleneck dimension r for low-rank projection heads.
        num_heads: Number of attention heads for the Transformer encoder.
        dropout: Dropout probability used in MLPs/Transformer.
        fusion_strategy: How to fuse (tabular, vision, text) into classifier input.
        num_classes: Output classes.
    """

    image_dim: int
    text_dim: int
    tabular_dim: int

    hidden_dim: int = 512
    projection_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1
    num_transformer_layers: int = 1  # must match ours_fusion.py (1 layer)

    fusion_strategy: FusionStrategy = "concat_reduce_concat"
    num_classes: int = 2


class VisionGuidedTextEncoder(nn.Module):
    """Vision-guided text encoder used inside VDAFM.

    This encoder follows the legacy design:

    1) Embed image and text into a shared hidden space.
    2) Produce an image-conditioned token `E_img` via an MLP.
    3) Stack the pair `[E_img, E_txt]` as a 2-token sequence and feed it into a
       Transformer encoder to exchange information.

    Mathematically, for a batch (B):

        V_img = W_v x_img
        E_txt = W_t x_txt
        E_img = f_v2t(V_img)

        [E_img^*, E_txt^*] = Transformer([E_img, E_txt])

    and an auxiliary mapping back to "vision space":

        V_txt = f_t2v(E_txt^*)

    Returns:
        E_img_encoded: (B, hidden_dim)
        E_txt_encoded: (B, hidden_dim)
        V_txt: (B, hidden_dim)
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, vision_feats: Tensor, text_feats: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            vision_feats: Image feature tensor of shape (B, vision_dim).
            text_feats: Text feature tensor of shape (B, text_dim).

        Returns:
            Tuple of (E_img_encoded, E_txt_encoded, V_txt), each of shape (B, hidden_dim).

        Raises:
            ValueError: If the input tensors do not have expected ranks.
        """

        if vision_feats.ndim != 2 or text_feats.ndim != 2:
            raise ValueError(
                "vision_feats and text_feats must be rank-2 tensors: (batch, feature_dim)."
            )

        v_img = self.vision_embed(vision_feats)
        e_txt = self.text_embed(text_feats)

        e_img = self.mlp_v2t(v_img)

        # (B, 2, D): token-0 is image-conditioned token, token-1 is text token.
        combined = torch.stack([e_img, e_txt], dim=1)
        encoded = self.encoder(combined)

        e_img_encoded = encoded[:, 0]
        e_txt_encoded = encoded[:, 1]

        v_txt = self.mlp_t2v(e_txt_encoded)
        return e_img_encoded, e_txt_encoded, v_txt


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between image and text tokens."""

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # NOTE: legacy code concatenated two attended tokens (2*hidden_dim) but projected with
        # a (hidden_dim -> hidden_dim) linear layer, which is a bug.
        self.fusion_proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, image_feats: Tensor, text_feats: Tensor) -> Tensor:
        """Fuse image and text by symmetric cross-attention.

        Args:
            image_feats: (B, image_dim)
            text_feats: (B, text_dim)

        Returns:
            Fused tensor of shape (B, hidden_dim).
        """

        if image_feats.ndim != 2 or text_feats.ndim != 2:
            raise ValueError("image_feats/text_feats must be (B, D).")

        image_tok = self.image_proj(image_feats).unsqueeze(1)  # (B, 1, D)
        text_tok = self.text_proj(text_feats).unsqueeze(1)  # (B, 1, D)

        # image queries text
        fused_image, _ = self.cross_attention(query=image_tok, key=text_tok, value=text_tok)
        # text queries image
        fused_text, _ = self.cross_attention(query=text_tok, key=image_tok, value=image_tok)

        fused = torch.cat([fused_image.squeeze(1), fused_text.squeeze(1)], dim=1)
        return self.fusion_proj(fused)


class VDAFM(nn.Module):
    """Vision-Guided Dual Alignment Fusion Module (VDAFM).

    This is the refactored *fusion model* that consumes three modalities:

    - tabular: (B, tabular_dim)
    - image:   (B, image_dim)
    - text:    (B, text_dim)

    and outputs:

    - logits: (B, num_classes)
    - contrastive_loss: scalar tensor (optional)
    - alignment_loss: scalar tensor (optional)

    The computation is:

    1) Cross-modal interaction in hidden space via VisionGuidedTextEncoder.
    2) Low-rank projection to `projection_dim` for contrastive learning.
    3) Projection calibration (alignment) in hidden space via symmetric cosine losses.
    4) Feature fusion according to `fusion_strategy`, then classification.
    """

    def __init__(self, cfg: VDAFMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.tabular_dim <= 0 or cfg.image_dim <= 0 or cfg.text_dim <= 0:
            raise ValueError("tabular_dim/image_dim/text_dim must be positive.")

        self.vision_guided_encoder = VisionGuidedTextEncoder(
            vision_dim=cfg.image_dim,
            text_dim=cfg.text_dim,
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            num_layers=cfg.num_transformer_layers,
        )

        self.cross_attention_fusion = CrossAttentionFusion(
            image_dim=cfg.hidden_dim,
            text_dim=cfg.hidden_dim,
            hidden_dim=cfg.projection_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )

        # Low-rank projections (bottleneck r = projection_dim)
        self.image_projection = nn.Linear(cfg.hidden_dim, cfg.projection_dim)
        self.text_projection = nn.Linear(cfg.hidden_dim, cfg.projection_dim)

        # Temperature parameter for contrastive learning
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Projection calibration MLPs (in hidden space)
        self.mlp_image_to_text = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.mlp_text_to_image = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        # fusion reductions are modules (not created inside forward)
        # concat_reduce_concat concatenates two projection_dim vectors (2*proj_dim input),
        # so we use a two-layer MLP with a hidden_dim bottleneck to better absorb the
        # richer joint representation before compressing to tabular_dim.
        self.fusion_reduce_2proj_to_tab = nn.Sequential(
            nn.Linear(2 * cfg.projection_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.tabular_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.fusion_reduce_proj_to_tab = nn.Sequential(
            nn.Linear(cfg.projection_dim, cfg.tabular_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        if cfg.fusion_strategy in {"concat_reduce_concat", "ca_fusion_concat", "add_reduce_concat"}:
            classifier_in = 2 * cfg.tabular_dim
        elif cfg.fusion_strategy in {"add_reduce_add", "separate_reduce_add"}:
            classifier_in = cfg.tabular_dim
        elif cfg.fusion_strategy == "separate_reduce_concat":
            classifier_in = 3 * cfg.tabular_dim
        else:
            raise ValueError(f"Unknown fusion_strategy: {cfg.fusion_strategy}")

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, cfg.num_classes),
        )

    def forward(
        self,
        tabular: Tensor,
        image: Tensor,
        text: Tensor,
        *,
        compute_contrastive: bool = True,
        compute_alignment: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            tabular: Tabular features (B, tabular_dim).
            image: Image features (B, image_dim).
            text: Text features (B, text_dim).
            compute_contrastive: Whether to compute contrastive loss.
            compute_alignment: Whether to compute alignment loss.

        Returns:
            logits: (B, num_classes)
            contrastive_loss: scalar tensor (0 if disabled)
            alignment_loss: scalar tensor (0 if disabled)

        Raises:
            ValueError: If shapes are incompatible.
        """

        if tabular.ndim != 2 or image.ndim != 2 or text.ndim != 2:
            raise ValueError("tabular/image/text must be rank-2 tensors.")
        if tabular.shape[1] != self.cfg.tabular_dim:
            raise ValueError(
                f"tabular feature dim mismatch: expected {self.cfg.tabular_dim}, got {tabular.shape[1]}"
            )
        if image.shape[1] != self.cfg.image_dim:
            raise ValueError(
                f"image feature dim mismatch: expected {self.cfg.image_dim}, got {image.shape[1]}"
            )
        if text.shape[1] != self.cfg.text_dim:
            raise ValueError(
                f"text feature dim mismatch: expected {self.cfg.text_dim}, got {text.shape[1]}"
            )

        e_img_encoded, e_txt_encoded, _ = self.vision_guided_encoder(image, text)

        z_img = self.image_projection(e_img_encoded)
        z_txt = self.text_projection(e_txt_encoded)

        contrastive_loss = torch.zeros((), device=z_img.device)
        if compute_contrastive:
            contrastive_loss = self.contrastive_loss(z_img, z_txt)

        alignment_loss = torch.zeros((), device=z_img.device)
        if compute_alignment:
            alignment_loss = self.alignment_loss(e_img_encoded, e_txt_encoded)

        fused = self._fuse(tabular, z_img, z_txt, e_img_encoded, e_txt_encoded)
        logits = self.classifier(fused)
        return logits, contrastive_loss, alignment_loss

    def alignment_loss(self, e_img: Tensor, e_txt: Tensor) -> Tensor:
        """Symmetric projection calibration loss (cosine distance).

        This is the "projection calibration" step mentioned in the task.

        We do NOT align the low-rank projections directly; instead, we learn
        calibration maps in hidden space so the model can adjust each modality
        toward the other without collapsing everything into the bottleneck.

        Args:
            e_img: Encoded image token in hidden space (B, hidden_dim).
            e_txt: Encoded text token in hidden space (B, hidden_dim).

        Returns:
            Scalar tensor alignment loss.
        """

        img_to_txt = self.mlp_image_to_text(e_img)
        txt_to_img = self.mlp_text_to_image(e_txt)

        loss_img_to_txt = 1.0 - F.cosine_similarity(img_to_txt, e_txt, dim=-1).mean()
        loss_txt_to_img = 1.0 - F.cosine_similarity(txt_to_img, e_img, dim=-1).mean()
        return 0.5 * (loss_img_to_txt + loss_txt_to_img)

    def contrastive_loss(self, z_img: Tensor, z_txt: Tensor) -> Tensor:
        """InfoNCE-style contrastive loss on low-rank projected features.

        The learnable temperature is constrained positive by `softplus`.

        Args:
            z_img: Projected image features (B, r).
            z_txt: Projected text features (B, r).

        Returns:
            Scalar tensor loss.
        """

        z_img = F.normalize(z_img, dim=1)
        z_txt = F.normalize(z_txt, dim=1)

        temperature = F.softplus(self.logit_scale) + 1e-6
        logits = (z_img @ z_txt.t()) / temperature
        labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i + loss_t)

    def _fuse(
        self,
        tabular: Tensor,
        z_img: Tensor,
        z_txt: Tensor,
        e_img: Tensor,
        e_txt: Tensor,
    ) -> Tensor:
        """Fuse modalities according to the configured fusion strategy."""

        s = self.cfg.fusion_strategy
        if s == "concat_reduce_concat":
            fused_reduced = self.fusion_reduce_2proj_to_tab(torch.cat([z_img, z_txt], dim=1))
            return torch.cat([tabular, fused_reduced], dim=1)

        if s == "ca_fusion_concat":
            # Use hidden tokens, fuse via cross-attention -> projection_dim, then reduce to tabular_dim
            ca_fused = self.cross_attention_fusion(e_img, e_txt)  # (B, projection_dim)
            fused_reduced = self.fusion_reduce_proj_to_tab(ca_fused)
            return torch.cat([tabular, fused_reduced], dim=1)

        if s == "add_reduce_concat":
            fused_reduced = self.fusion_reduce_proj_to_tab(z_img + z_txt)
            return torch.cat([tabular, fused_reduced], dim=1)

        if s == "add_reduce_add":
            fused_reduced = self.fusion_reduce_proj_to_tab(z_img + z_txt)
            return tabular + fused_reduced

        if s == "separate_reduce_concat":
            img_reduced = self.fusion_reduce_proj_to_tab(z_img)
            txt_reduced = self.fusion_reduce_proj_to_tab(z_txt)
            return torch.cat([tabular, img_reduced, txt_reduced], dim=1)

        if s == "separate_reduce_add":
            img_reduced = self.fusion_reduce_proj_to_tab(z_img)
            txt_reduced = self.fusion_reduce_proj_to_tab(z_txt)
            return tabular + img_reduced + txt_reduced

        raise ValueError(f"Unknown fusion_strategy: {s}")
