"""Generate radiology text descriptions from MRI PNG slices via Qianwen VLM.

Calls the Alibaba DashScope qwen-vl-max API with axial MRI PNG images and
structured neuroradiology prompts to produce 10 complementary descriptions
per patient.  Outputs one .txt file per patient in TEXT_QIANWEN_DIR.

Configuration (environment variables / .env)
--------------------------------------------
QIANWEN_API_KEY        Your DashScope API key (required).
MRI_SLICES_DIR         Directory of patient PNG slice folders (axial.png, …).
TEXT_QIANWEN_DIR       Output directory for generated .txt files.
QIANWEN_DESCRIPTIONS   Number of descriptions per patient (default 10).
QIANWEN_MAX_WORKERS    Concurrent API threads (default 1).
QIANWEN_REQUEST_DELAY  Seconds between requests (default 2.0).

Usage
-----
    python -m stroke_multimodal_prognosis.pre.generate_text_qianwen
    # or
    python stroke_multimodal_prognosis/pre/generate_text_qianwen.py
"""

import base64
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
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
API_BASE  = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
API_KEY   = os.environ.get("QIANWEN_API_KEY", "")
MODEL     = "qwen-vl-max"

INPUT_DIR          = os.environ.get("MRI_SLICES_DIR", "")
OUTPUT_DIR         = os.environ.get("TEXT_QIANWEN_DIR", "")
DESCRIPTIONS       = int(os.environ.get("QIANWEN_DESCRIPTIONS", "10"))
MAX_WORKERS        = int(os.environ.get("QIANWEN_MAX_WORKERS", "1"))
REQUEST_DELAY      = float(os.environ.get("QIANWEN_REQUEST_DELAY", "2.0"))
RETRY_LIMIT        = 3

if not API_KEY:
    print(
        "[generate_text_qianwen] ERROR: QIANWEN_API_KEY is not set.\n"
        "Copy .env.example to .env and add your DashScope API key."
    )
    sys.exit(1)

if not INPUT_DIR or not OUTPUT_DIR:
    print(
        "[generate_text_qianwen] ERROR: MRI_SLICES_DIR and TEXT_QIANWEN_DIR must be set.\n"
        "See .env.example for details."
    )
    sys.exit(1)

# ── Prompt templates ──────────────────────────────────────────────────────────
_PROMPTS = [
    (
        "You are a neuroradiologist specialising in stroke medicine. "
        "Analyse the provided axial DWI MRI image of this acute ischemic stroke patient "
        "and provide a structured assessment. "
        "1. Infarct Core Volume (mL): provide a numerical estimate. "
        "2. ASPECTS Score (0-10): for MCA territory involvement. "
        "3. Key Structures Involved: list specific brain regions and infer the occluded vessel. "
        "4. Mass Effect: quantify midline shift in mm if present. "
        "5. Estimated NIHSS Range: probable score range based on findings. "
        "Use professional terminology. Keep the response under 250 tokens."
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Core functions
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_image(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as exc:
        print(f"[qianwen] Image encode failed: {image_path} — {exc}")
        return None


def _call_api(b64_img: str, prompt: str) -> str:
    payload = {
        "model": MODEL,
        "input": {
            "messages": [{
                "role": "user",
                "content": [
                    {"image": f"data:image/png;base64,{b64_img}"},
                    {"text": prompt},
                ],
            }]
        },
        "parameters": {"max_tokens": 250, "top_p": 0.8, "temperature": 0.7},
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    resp = requests.post(API_BASE, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    content = result["output"]["choices"][0]["message"]["content"]
    if isinstance(content, list) and content:
        return content[0].get("text", str(content))
    return str(content)


def _generate_one(patient_id: str, axial_path: str) -> str | None:
    b64 = _encode_image(axial_path)
    if not b64:
        return None
    prompt = random.choice(_PROMPTS)
    for attempt in range(RETRY_LIMIT):
        try:
            return _call_api(b64, prompt)
        except Exception as exc:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(REQUEST_DELAY * (2 ** attempt))
            else:
                print(f"[qianwen] API failed for {patient_id}: {exc}")
                return f"[ERROR] {exc}"
    return None


def process_patient(patient_dir: str) -> str:
    """Generate DESCRIPTIONS text reports for one patient and save to file."""
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
        for i, desc in enumerate(texts, 1):
            f.write(f"===== 描述 {i}/{DESCRIPTIONS} =====\n{desc}\n\n")
    return f"OK: {pid} ({len(texts)} descriptions)"


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch runner
# ═══════════════════════════════════════════════════════════════════════════════

def batch_generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    patient_dirs = [
        os.path.join(INPUT_DIR, d)
        for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ]
    if not patient_dirs:
        print("[qianwen] No patient directories found.")
        return

    print(f"[qianwen] Processing {len(patient_dirs)} patients  "
          f"({DESCRIPTIONS} descriptions each, {MAX_WORKERS} worker(s)) …")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_patient, d): d for d in patient_dirs}
        with tqdm(total=len(patient_dirs), desc="Generating") as bar:
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                bar.update(1)
                bar.set_postfix_str(res[-60:])

    ok    = sum(1 for r in results if r.startswith("OK"))
    skip  = sum(1 for r in results if r.startswith("Skip"))
    fail  = sum(1 for r in results if r.startswith("Failed"))
    print(f"\n[qianwen] Done — OK: {ok}, Skipped: {skip}, Failed: {fail}")


if __name__ == "__main__":
    batch_generate()
