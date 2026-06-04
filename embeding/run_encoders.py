"""Orchestrator: run all three feature encoders in sequence.

This script calls, in order:
  1. label_encoder.residual_mlp  — clinical tabular → clinic_data_1.xlsx
  2. image_encoder.resnet3d      — NIfTI MRI        → image_feature.xlsx
  3. text_encoder.biobert        — radiology text   → medical_text_features_by_qianwen_best.xlsx

All three output files land in  data/encoded/  by default (configurable via
env variables; see .env.example).

Prerequisites
-------------
- Fill in .env (copy from .env.example) with your paths and API keys.
- pip install -r requirements.txt
- Download MedicalNet weights and place them at MEDICALNET_WEIGHTS.

Usage
-----
    # From the repository root:
    python -m stroke_multimodal_prognosis.embeding.run_encoders

    # Or run individual encoders:
    python -m stroke_multimodal_prognosis.embeding.label_encoder.residual_mlp
    python -m stroke_multimodal_prognosis.embeding.image_encoder.resnet3d
    python -m stroke_multimodal_prognosis.embeding.text_encoder.biobert
"""

import os
import sys
from pathlib import Path

# ── Load .env if available ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parents[2] / ".env"
    if _env.exists():
        load_dotenv(_env)
        print(f"[run_encoders] Loaded .env from {_env}")
    else:
        print(f"[run_encoders] No .env found at {_env} — using process environment.")
except ImportError:
    print("[run_encoders] python-dotenv not installed; using process environment.")

# ── Default output paths ──────────────────────────────────────────────────────
_ENCODED_DIR = Path(__file__).parents[2] / "data" / "encoded"

_clinic_output = os.environ.get("CLINIC_ENCODED_OUTPUT",  str(_ENCODED_DIR / "clinic_data_1.xlsx"))
_image_output  = os.environ.get("IMAGE_FEATURE_OUTPUT",   str(_ENCODED_DIR / "image_feature.xlsx"))
_text_output   = os.environ.get("TEXT_FEATURE_OUTPUT",    str(_ENCODED_DIR / "medical_text_features_by_qianwen_best.xlsx"))

# ── Required environment variables ───────────────────────────────────────────
_REQUIRED = {
    "CLINIC_RAW_PATH":    "Path to raw clinical Excel file (id, features…, label).",
    "RAW_NII_DIR":        "Directory containing raw .nii MRI files.",
    "MEDICALNET_WEIGHTS": "Path to MedicalNet ResNet-50 checkpoint (.pth).",
    "TEXT_QIANWEN_DIR":   "Directory containing .txt radiology reports.",
}

_missing = [k for k, _ in _REQUIRED.items() if not os.environ.get(k)]
if _missing:
    print("\n[run_encoders] ERROR — missing required environment variables:\n")
    for key in _missing:
        print(f"  {key}  — {_REQUIRED[key]}")
    print("\nCopy .env.example to .env and fill in the values, then re-run.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Tabular encoder (ResidualMLP)
# ═══════════════════════════════════════════════════════════════════════════════

def run_label_encoder():
    print("\n" + "=" * 60)
    print("Step 1/3 — Tabular encoder (ResidualMLP)")
    print("=" * 60)
    from .label_encoder.residual_mlp import encode_and_save
    encode_and_save(
        clinic_path=os.environ["CLINIC_RAW_PATH"],
        output_excel=_clinic_output,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2 — Image encoder (MedicalNet 3D ResNet-50)
# ═══════════════════════════════════════════════════════════════════════════════

def run_image_encoder():
    print("\n" + "=" * 60)
    print("Step 2/3 — Image encoder (MedicalNet 3D ResNet-50)")
    print("=" * 60)
    from .image_encoder.resnet3d import extract_and_save
    extract_and_save(
        nii_dir=os.environ["RAW_NII_DIR"],
        weights_path=os.environ["MEDICALNET_WEIGHTS"],
        output_excel=_image_output,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Text encoder (BioBERT)
# ═══════════════════════════════════════════════════════════════════════════════

def run_text_encoder():
    print("\n" + "=" * 60)
    print("Step 3/3 — Text encoder (BioBERT v1.1 PubMed)")
    print("=" * 60)
    from .text_encoder.biobert import extract_and_save
    extract_and_save(
        text_dir=os.environ["TEXT_QIANWEN_DIR"],
        local_model_dir=os.environ.get("BIOBERT_MODEL_DIR") or None,
        output_excel=_text_output,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_label_encoder()
    run_image_encoder()
    run_text_encoder()

    print("\n" + "=" * 60)
    print("All encoders finished.  Encoded features written to:")
    print(f"  Tabular : {_clinic_output}")
    print(f"  Image   : {_image_output}")
    print(f"  Text    : {_text_output}")
    print("=" * 60)
    print("You can now run training:  python main.py")
