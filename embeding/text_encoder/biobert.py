"""BioBERT v1.1 PubMed text encoder.

Encodes radiology text reports into dense feature vectors (768-dim last-layer
CLS, or up to 3072-dim with last-4-layer concatenation) and saves the results
to an Excel file consumed by the training pipeline.

Configuration (via environment variables or .env file)
------------------------------------------------------
BIOBERT_MODEL_DIR    Local directory for cached BioBERT weights.
                     Auto-downloaded from HuggingFace on first run if absent.
                     HuggingFace model: monologg/biobert_v1.1_pubmed
TEXT_QIANWEN_DIR     Directory containing .txt radiology reports.
TEXT_FEATURE_OUTPUT  Path for the output Excel file.
                     Default: <package>/data/encoded/
                             medical_text_features_by_qianwen_best.xlsx

Usage
-----
    python -m stroke_multimodal_prognosis.embeding.text_encoder.biobert
"""

import os
import random
import re
import warnings
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# ── Load .env if available ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parents[3] / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

_ENCODED_DIR = Path(__file__).parents[3] / "data" / "encoded"
_DEFAULT_OUTPUT = str(_ENCODED_DIR / "medical_text_features_by_qianwen_best.xlsx")
_HF_MODEL_ID = "monologg/biobert_v1.1_pubmed"


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BioBERTConfig:
    """All tunable options for the text encoder in one place."""
    model_name: str = _HF_MODEL_ID
    local_model_dir: Optional[str] = None   # overridden from env at runtime
    max_length: int = 256

    # Text segmentation — description blocks are separated by this pattern
    delimiter: str = r"===== 描述 \d+/\d+ ====="
    segment_selection: str = "first"        # "first" | "last" | "random"

    # Phrase filtering
    use_whitelist: bool = True
    use_blacklist: bool = True
    whitelist_phrases: List[str] = field(default_factory=list)
    blacklist_phrases: List[str] = field(default_factory=list)

    # Prompt engineering
    use_prompt_engineering: bool = True
    prompt_selection: str = "random"        # "first" | "last" | "random"
    prompts: List[str] = field(default_factory=list)

    # CLS pooling strategy
    cls_pooling: str = "last4_avg"          # "last" | "last4_avg" | "last4_cat"
    output_precision: int = 6

    def __post_init__(self):
        if not self.whitelist_phrases:
            self.whitelist_phrases = [
                "infarct core volume", "volume estimate", "ml based", "visible extent",
                "aspects score", "dwi-aspects", "large infarct", "moderate-sized infarct",
                "mca territory", "anterior circulation",
                "lentiform nucleus", "insular ribbon", "frontal cortex", "parietal cortex",
                "m3 cortex", "adjacent cortex", "superior frontal gyrus", "postcentral gyrus",
                "m2 branch", "m1 segment", "proximal mca", "occlusion",
                "mass effect", "no significant mass effect",
                "nihss range", "nihss score", "neurological deficit",
                "restricted diffusion", "diffusion restriction", "dwi mri",
                "acute ischemic stroke", "clinical correlation",
            ]
        if not self.blacklist_phrases:
            self.blacklist_phrases = [
                "without direct visualization", "assumed from", "based on classic",
                "general framework", "illustration", "step-by-step", "outlines",
                "framework for", "approach to", "common scenario", "structured approach",
                "systematic assessment", "the following reasoning", "note:",
                "appropriate consideration for uncertainty",
                "reflecting the clinical presentation",
                "further clinical correlation", "this assessment reflects",
                "this assessment is based on", "based on the observed",
                "based on the provided image",
            ]
        if not self.prompts:
            self.prompts = [
                "Predicting NIHSS score from DWI MRI findings of acute ischemic stroke: ",
                "Estimating neurological deficit severity (NIHSS) based on infarct characteristics: ",
                "Correlation between MCA territory infarction and NIHSS score prediction: ",
                "Prognostic assessment from DWI-ASPECTS and infarct volume measurements: ",
                "Predicting clinical outcome based on lentiform nucleus and insular ribbon involvement: ",
                "NIHSS score estimation from acute ischemic stroke location and extent: ",
                "Radiological predictors of stroke severity using ASPECTS and infarct volume: ",
                "MRI-based prognosis prediction for moderate to severe ischemic stroke: ",
                "Early DWI MRI biomarkers for NIHSS score estimation in acute stroke: ",
                "Estimating clinical deficit from cortical and subcortical infarct patterns: ",
            ]


# ═══════════════════════════════════════════════════════════════════════════════
#  Text processor
# ═══════════════════════════════════════════════════════════════════════════════

class TextProcessor:
    """Parse, filter and encode radiology text files."""

    _SENT_RE = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')

    def __init__(self, config: BioBERTConfig, tokenizer: BertTokenizer):
        self.cfg = config
        self.tokenizer = tokenizer
        self.file_paths: List[Path] = []

    def load_directory(self, data_dir: str, extensions=(".txt",)) -> int:
        self.file_paths = []
        for ext in extensions:
            self.file_paths.extend(sorted(Path(data_dir).glob(f"*{ext}")))
        return len(self.file_paths)

    def _split(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(self.cfg.delimiter, text) if s.strip()]

    def _select(self, segments: List[str]) -> str:
        if not segments:
            return ""
        sel = self.cfg.segment_selection
        if sel == "first":
            return segments[0]
        if sel == "last":
            return segments[-1]
        return random.choice(segments)

    def _apply_whitelist(self, text: str) -> str:
        if not self.cfg.use_whitelist:
            return text
        sentences = self._SENT_RE.split(text)
        kept = [s for s in sentences
                if any(p.lower() in s.lower() for p in self.cfg.whitelist_phrases)]
        return " ".join(kept) if kept else " ".join(sentences[:2])

    def _apply_blacklist(self, text: str) -> str:
        if not self.cfg.use_blacklist:
            return text
        sentences = self._SENT_RE.split(text)
        kept = [s for s in sentences
                if not any(p.lower() in s.lower() for p in self.cfg.blacklist_phrases)]
        return " ".join(kept) if kept else text

    def _add_prompt(self, text: str) -> str:
        if not self.cfg.use_prompt_engineering or not self.cfg.prompts:
            return text
        sel = self.cfg.prompt_selection
        prompt = (self.cfg.prompts[0] if sel == "first"
                  else self.cfg.prompts[-1] if sel == "last"
                  else random.choice(self.cfg.prompts))
        return prompt + text

    def preprocess(self, text_path: Path):
        try:
            full = text_path.read_text(encoding="utf-8")
            segments = self._split(full)
            if not segments:
                warnings.warn(f"[biobert] No segments found: {text_path}")
                return None
            text = self._select(segments)
            text = self._apply_blacklist(text)
            text = self._apply_whitelist(text)
            text = self._add_prompt(text)
            if len(text) > 1000:
                text = text[:1000] + " [TRUNCATED]"
            return self.tokenizer(
                text,
                max_length=self.cfg.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        except Exception as exc:
            warnings.warn(f"[biobert] Failed {text_path}: {exc}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self.encodings[idx].items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature-extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_cls(outputs, pooling: str) -> torch.Tensor:
    if pooling == "last4_cat":
        layers = outputs.hidden_states[-4:]
        return torch.cat([l[:, 0, :] for l in layers], dim=1)
    if pooling == "last4_avg":
        layers = outputs.hidden_states[-4:]
        return torch.stack([l[:, 0, :] for l in layers]).mean(dim=0)
    return outputs.last_hidden_state[:, 0, :]


def extract_and_save(
    text_dir: str,
    local_model_dir: Optional[str],
    output_excel: str,
    config: Optional[BioBERTConfig] = None,
    batch_size: int = 8,
):
    """Encode all .txt files in *text_dir* and write feature vectors to Excel."""
    if config is None:
        config = BioBERTConfig(local_model_dir=local_model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[biobert] Device: {device}")

    model_source = local_model_dir if (local_model_dir and Path(local_model_dir).exists()) else config.model_name
    print(f"[biobert] Loading model from: {model_source}")
    tokenizer = BertTokenizer.from_pretrained(model_source)
    need_hidden = config.cls_pooling in ("last4_avg", "last4_cat")
    model = BertModel.from_pretrained(model_source, output_hidden_states=need_hidden).to(device).eval()

    if local_model_dir and not Path(local_model_dir).exists():
        Path(local_model_dir).mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(local_model_dir)
        model.save_pretrained(local_model_dir)
        print(f"[biobert] Saved model to {local_model_dir}")

    processor = TextProcessor(config, tokenizer)
    n = processor.load_directory(text_dir)
    print(f"[biobert] Found {n} text files.")

    encodings, paths = [], []
    for p in tqdm(processor.file_paths, desc="Pre-processing text"):
        enc = processor.preprocess(p)
        if enc is not None:
            encodings.append(enc)
            paths.append(p)

    if not encodings:
        raise RuntimeError("[biobert] No valid text files found.")

    features = []
    ds = TextDataset(encodings)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    prec_str = "1." + "0" * config.output_precision

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            ids_ = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(ids_, attention_mask=mask, output_hidden_states=need_hidden)
            cls = _extract_cls(out, config.cls_pooling).cpu().numpy().astype(np.float64)
            for vec in cls:
                rounded = np.vectorize(
                    lambda x: float(Decimal(str(x)).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP))
                )(vec)
                features.append(", ".join(f"{v:.{config.output_precision}f}" for v in rounded))

    ids = [p.stem for p in paths]
    df = pd.DataFrame({"id": ids, "text_features": features})
    Path(output_excel).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel, index=False)
    print(f"[biobert] Saved {len(df)} feature vectors → {output_excel}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _text_dir   = os.environ.get("TEXT_QIANWEN_DIR", "")
    _model_dir  = os.environ.get("BIOBERT_MODEL_DIR", "")
    _output     = os.environ.get("TEXT_FEATURE_OUTPUT", _DEFAULT_OUTPUT)

    if not _text_dir:
        raise EnvironmentError(
            "Set TEXT_QIANWEN_DIR to the directory containing .txt radiology reports. "
            "See .env.example for details."
        )

    extract_and_save(
        text_dir=_text_dir,
        local_model_dir=_model_dir or None,
        output_excel=_output,
    )
