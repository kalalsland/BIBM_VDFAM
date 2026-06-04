"""Filter radiology text files and flag low-quality descriptions.

Scans a directory of .txt files and identifies ones that are too short or
lack real imaging findings.  Writes an invalid-files list to a report file.

Configuration (environment variables / .env)
--------------------------------------------
TEXT_FILTER_DIR      Directory of .txt radiology description files.
INVALID_FILES_REPORT Output path for the list of invalid filenames.
                     Default: <TEXT_FILTER_DIR>/invalid_files.txt

Usage
-----
    python -m stroke_multimodal_prognosis.pre.filter_quality
    # or
    python stroke_multimodal_prognosis/pre/filter_quality.py
"""

import os
import re
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Quality criteria
# ═══════════════════════════════════════════════════════════════════════════════

_INVALID_KW = [
    "not able to be analyzed",
    "no specific findings",
    "normal",
    "unremarkable",
    "not visible",
    "not identified",
    "no significant",
    "no acute",
]

_VALID_CRITERIA = [
    # 3-D measurement in mm
    lambda t: re.search(r'\b\d+\.?\d*\s*[x×]\s*\d+\.?\d*\s*[x×]\s*\d+\.?\d*\s*mm\b', t),
    # volume in mL
    lambda t: re.search(r'\b\d+\.?\d*\s*ml\b', t),
    # imaging abbreviations
    lambda t: re.search(r'\b(adc|aspects|tmax|dwi|flair|swi)\b', t),
    # ADC values in scientific notation
    lambda t: re.search(r'\b\d+\.?\d*\s*[×x]\s*10\^?[-]?[³3]', t),
    # ASPECTS fraction
    lambda t: (re.search(r'\b\d+/\d+\b', t) and 'aspects' in t),
    # lesion + location
    lambda t: (
        any(kw in t for kw in ["lesion", "edema", "infarct", "ischemic", "occlusion"])
        and re.search(r'\bin\b.*\b(territory|capsule|brainstem)\b', t)
    ),
    # DWI + ADC co-occurrence
    lambda t: (
        re.search(r'hyperintensity.*(dwi|t2)', t)
        and re.search(r'hypointensity.*(adc|swi)', t)
    ),
]


def is_valid(text: str) -> bool:
    low = text.lower()
    if len(low.split()) < 30:
        return False
    if any(kw in low for kw in _INVALID_KW):
        return False
    return any(criterion(low) for criterion in _VALID_CRITERIA)


def find_invalid(folder: str) -> list[str]:
    invalid = []
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(folder, fname)
        try:
            text = Path(fpath).read_text(encoding="utf-8")
            if not is_valid(text):
                invalid.append(fname)
        except Exception as exc:
            print(f"[filter_quality] Error reading {fname}: {exc}")
            invalid.append(fname)
    return invalid


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _folder = os.environ.get("TEXT_FILTER_DIR", os.environ.get("TEXT_MEDGEMMA_DIR", ""))
    if not _folder:
        print("[filter_quality] ERROR: Set TEXT_FILTER_DIR (or TEXT_MEDGEMMA_DIR).")
        sys.exit(1)

    _default_report = os.path.join(_folder, "invalid_files.txt")
    _report = os.environ.get("INVALID_FILES_REPORT", _default_report)

    invalid = find_invalid(_folder)

    if invalid:
        print(f"[filter_quality] Found {len(invalid)} invalid files:")
        for f in invalid:
            print(f"  - {f}")
        with open(_report, "w", encoding="utf-8") as out:
            out.write("Invalid description files:\n")
            for f in invalid:
                out.write(f"{f}\n")
        print(f"[filter_quality] Report saved → {_report}")
    else:
        print("[filter_quality] No invalid files found.")
