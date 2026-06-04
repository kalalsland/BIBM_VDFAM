"""Download and cache the Google MedGemma-4B-IT model locally.

Run this once before generate_text_medgemma.py to pre-fetch the model
weights.  Requires a Hugging Face account and acceptance of the MedGemma
licence at https://huggingface.co/google/medgemma-4b-it.

Configuration (environment variables / .env)
--------------------------------------------
HF_TOKEN           Your Hugging Face access token (required).
MEDGEMMA_MODEL_DIR Local directory to save the model weights.
                   Default: E:/pretrained/medgemma-4b-it

Usage
-----
    python -m stroke_multimodal_prognosis.pre.download_medgemma
    # or
    python stroke_multimodal_prognosis/pre/download_medgemma.py
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
except ImportError:
    pass

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
MODEL_DIR = os.environ.get("MEDGEMMA_MODEL_DIR", "E:/pretrained/medgemma-4b-it")
MODEL_ID  = "google/medgemma-4b-it"

if not HF_TOKEN:
    print(
        "[download_medgemma] ERROR: HF_TOKEN is not set.\n"
        "Copy .env.example to .env and add your Hugging Face token.\n"
        "Accept the MedGemma licence at: https://huggingface.co/google/medgemma-4b-it"
    )
    sys.exit(1)

if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from huggingface_hub import login

    login(token=HF_TOKEN)

    model_path     = os.path.join(MODEL_DIR, "model")
    processor_path = os.path.join(MODEL_DIR, "processor")

    if Path(model_path).exists() and Path(processor_path).exists():
        print(f"[download_medgemma] Model already cached at {MODEL_DIR}. Nothing to do.")
        sys.exit(0)

    print(f"[download_medgemma] Downloading {MODEL_ID} → {MODEL_DIR} …")
    os.makedirs(MODEL_DIR, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",   # save in fp32-compatible format first
        token=HF_TOKEN,
    )

    processor.save_pretrained(processor_path)
    model.save_pretrained(model_path)
    print(f"[download_medgemma] Saved to {MODEL_DIR}")
