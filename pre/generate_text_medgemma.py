"""Generate radiology descriptions for severe stroke cases via MedGemma-4B-IT.

Uses the Google MedGemma-4B-IT multimodal model (run locally) to produce
detailed descriptions of catastrophic ischemic stroke findings from axial
MRI PNG slices.  Designed as a complement to generate_text_qianwen.py for
cases where >70 % of brain parenchyma is involved.

Configuration (environment variables / .env)
--------------------------------------------
HF_TOKEN              Hugging Face access token (required; model is gated).
MEDGEMMA_MODEL_DIR    Local cache directory for MedGemma weights.
                      Auto-downloaded from HuggingFace on first run.
MRI_SLICES_DIR        Directory of patient PNG slice folders.
TEXT_MEDGEMMA_DIR     Output directory for generated .txt files.
MEDGEMMA_DESCRIPTIONS  Descriptions per patient (default 1).
MEDGEMMA_MAX_WORKERS  Concurrent workers (default 1 — large model).

Usage
-----
    python -m stroke_multimodal_prognosis.pre.generate_text_medgemma
    # or
    python stroke_multimodal_prognosis/pre/generate_text_medgemma.py
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ── Load .env if available ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parents[2] / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

# ── Configuration from environment ───────────────────────────────────────────
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
MODEL_NAME    = "google/medgemma-4b-it"
MODEL_DIR     = os.environ.get("MEDGEMMA_MODEL_DIR", "")
INPUT_DIR     = os.environ.get("MRI_SLICES_DIR", "")
OUTPUT_DIR    = os.environ.get("TEXT_MEDGEMMA_DIR", "")
DESCRIPTIONS  = int(os.environ.get("MEDGEMMA_DESCRIPTIONS", "1"))
MAX_WORKERS   = int(os.environ.get("MEDGEMMA_MAX_WORKERS", "1"))
RETRY_LIMIT   = 5
REQUEST_DELAY = 2.0
MODALITY      = "MRI"

if not HF_TOKEN:
    print(
        "[medgemma] ERROR: HF_TOKEN is not set.\n"
        "Copy .env.example to .env and add your Hugging Face token.\n"
        "Also accept the MedGemma licence at: "
        "https://huggingface.co/google/medgemma-4b-it"
    )
    sys.exit(1)

if not INPUT_DIR or not OUTPUT_DIR:
    print("[medgemma] ERROR: MRI_SLICES_DIR and TEXT_MEDGEMMA_DIR must be set.")
    sys.exit(1)

# ── Prompt templates (severe / catastrophic cases) ────────────────────────────
_PROMPTS = [
    (
        "Describe ONLY the visible pathological findings in this axial {m} scan "
        "showing extensive ischemic stroke. Focus on: "
        "1. Global signal abnormalities (e.g., diffuse DWI hyperintensity). "
        "2. Loss of gray-white matter differentiation. "
        "3. Mass effect (midline shift in mm). "
        "4. Vascular territory involvement (pan-hemispheric vs. bilateral). "
        "5. Estimated affected brain volume (>70 %)."
    ),
    (
        "As a neuroradiologist, analyse this critical axial {m} scan with extensive ischemic injury. "
        "• Characterise the global diffusion restriction pattern (confluent hyperintensity on DWI b1000). "
        "• Describe haemorrhagic transformation in core infarct zones. "
        "• Note brainstem involvement and posterior fossa effects. "
        "• Quantify cerebral oedema using ASPECTS (score ≤3). "
        "• Identify early signs of herniation."
    ),
    (
        "Generate a critical findings report for this massive ischemic stroke on axial {m}: "
        "1. Core infarct volume estimation (>100 mL). "
        "2. Cytotoxic vs. vasogenic oedema distribution. "
        "3. Vascular flow void assessment (absent MCA flow voids). "
        "4. Cortical ribbon sign in all lobes. "
        "5. Life-threatening complications."
    ),
]

_IRRELEVANT_KW = [
    "i'm sorry", "i cannot", "as an ai", "not shown", "unable to determine",
    "without clinical", "not visible", "speculate", "assume",
    "normal", "unremarkable", "no significant", "no acute", "clear", "healthy",
]
_SEVERE_KW = [
    "extensive", "massive", "global", "pan-cerebral", "herniation",
    "midline shift", "brainstem", "edema", "compression", "effacement",
    "cytotoxic", "vasogenic", "ischemic", "infarct", "territory",
]


def _is_relevant(text: str) -> bool:
    low = text.lower()
    if any(k in low for k in _SEVERE_KW):
        return True
    return not any(k in low for k in _IRRELEVANT_KW)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading (lazy — happens once inside the main process)
# ═══════════════════════════════════════════════════════════════════════════════

_pipe = None  # transformers pipeline instance


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from transformers import pipeline as hf_pipeline

    model_source = MODEL_DIR if (MODEL_DIR and Path(MODEL_DIR).exists()) else MODEL_NAME

    _pipe = hf_pipeline(
        "image-text-to-text",
        model=model_source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )

    if MODEL_DIR and not Path(MODEL_DIR).exists():
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        _pipe.model.save_pretrained(MODEL_DIR)
        _pipe.tokenizer.save_pretrained(MODEL_DIR)
        print(f"[medgemma] Model saved to {MODEL_DIR}")

    print("[medgemma] Model ready.")
    return _pipe


# ═══════════════════════════════════════════════════════════════════════════════
#  Core generation
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_output(raw) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        gt = raw.get("generated_text", "")
        if isinstance(gt, list) and gt:
            last = gt[-1]
            return last.get("content", str(last)) if isinstance(last, dict) else str(last)
        return str(gt)
    if isinstance(raw, list) and raw:
        return _parse_output(raw[0])
    return str(raw)


def _generate_one(patient_id: str, image_path: str) -> str | None:
    try:
        img = Image.open(image_path)
    except Exception as exc:
        print(f"[medgemma] Image load failed ({patient_id}): {exc}")
        return None

    prompt = random.choice(_PROMPTS).format(m=MODALITY)
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are an expert neuroradiologist. "
                    "Describe ONLY what is visible in the image. "
                    "Do not speculate beyond the image content."
                ),
            }],
        },
        {
            "role": "user",
            "content": [
                {"type": "text",  "text": prompt},
                {"type": "image", "image": img},
            ],
        },
    ]

    pipe = _get_pipe()
    for attempt in range(RETRY_LIMIT):
        try:
            out = pipe(text=messages, max_new_tokens=768, return_full_text=False)
            text = _parse_output(out)
            if _is_relevant(text):
                return text
        except Exception as exc:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(REQUEST_DELAY * (2 ** attempt))
            else:
                return f"[ERROR] {exc}"
    return f"[ERROR] Could not generate relevant content for {patient_id}"


def process_patient(patient_dir: str) -> str:
    pid = os.path.basename(patient_dir)
    out_file = os.path.join(OUTPUT_DIR, f"{pid}.txt")

    if os.path.exists(out_file):
        return f"Skipped (exists): {pid}"

    axial = os.path.join(patient_dir, "axial.png")
    if not os.path.exists(axial):
        return f"Failed (missing axial.png): {pid}"

    texts = []
    for _ in range(DESCRIPTIONS):
        t = _generate_one(pid, axial)
        if t and not t.startswith("[ERROR]"):
            texts.append(t)
        time.sleep(REQUEST_DELAY)

    if not texts:
        return f"Failed (no valid responses): {pid}"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"===== Patient: {pid} | Model: {MODEL_NAME} =====\n\n")
        for i, desc in enumerate(texts, 1):
            f.write(f"===== 描述 {i}/{DESCRIPTIONS} =====\n{desc}\n\n")
    return f"OK: {pid} ({len(texts)} descriptions)"


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch runner
# ═══════════════════════════════════════════════════════════════════════════════

def batch_generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Initialise model once in the main thread
    _get_pipe()

    patient_dirs = [
        os.path.join(INPUT_DIR, d)
        for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ]
    if not patient_dirs:
        print("[medgemma] No patient directories found.")
        return

    print(f"[medgemma] Processing {len(patient_dirs)} patients …")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_patient, d): d for d in patient_dirs}
        with tqdm(total=len(patient_dirs), desc="MedGemma") as bar:
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                bar.update(1)
                bar.set_postfix_str(res[-60:])

    ok   = sum(1 for r in results if r.startswith("OK"))
    skip = sum(1 for r in results if r.startswith("Skip"))
    fail = sum(1 for r in results if r.startswith("Failed"))

    summary = {
        "total": len(patient_dirs),
        "ok": ok, "skipped": skip, "failed": fail,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[medgemma] Done — OK: {ok}, Skipped: {skip}, Failed: {fail}")


if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[medgemma] GPU: {total_gb:.1f} GB")
        if total_gb < 16:
            print("[medgemma] WARNING: <16 GB VRAM — MedGemma-4B-IT may OOM.")
    batch_generate()
