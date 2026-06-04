"""
Stroke Multimodal Prognosis
===========================

A PyTorch-based framework for multimodal stroke prognosis prediction using
pre-encoded clinical tabular, image, and text features.

Primary model: OursFusion (Vision-Guided Dual Alignment Fusion, VDAFM).

Run training:
    python -m stroke_multimodal_prognosis.main
"""

__version__ = "2.0.0"

from .config import CONFIG
from .models.ours_fusion import OursFusion, VisionGuidedTextEncoder

__all__ = [
    "CONFIG",
    "OursFusion",
    "VisionGuidedTextEncoder",
]
