"""MedicalNet 3D ResNet-50 image encoder.

Encodes NIfTI MRI volumes into 2048-dimensional feature vectors using the
MedicalNet pre-trained ResNet-50 backbone and saves the results to an Excel
file that the training pipeline can consume.

Configuration (via environment variables or .env file)
------------------------------------------------------
MEDICALNET_WEIGHTS   Path to the MedicalNet ResNet-50 checkpoint (.pth).
                     Download: https://github.com/Tencent/MedicalNet
RAW_NII_DIR          Directory containing raw .nii / .nii.gz files.
IMAGE_FEATURE_OUTPUT Path for the output Excel file.
                     Default: <package>/data/encoded/image_feature.xlsx

Usage
-----
    python -m stroke_multimodal_prognosis.embeding.image_encoder.resnet3d
"""

import os
import warnings
from collections import OrderedDict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from tqdm import tqdm

# ── Load .env if available ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parents[3] / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

# ── Default output path ───────────────────────────────────────────────────────
_ENCODED_DIR = Path(__file__).parents[3] / "data" / "encoded"
_DEFAULT_OUTPUT = str(_ENCODED_DIR / "image_feature.xlsx")


# ═══════════════════════════════════════════════════════════════════════════════
#  Model definition — adapted from MedicalNet
# ═══════════════════════════════════════════════════════════════════════════════

class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3D(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        features = self.avgpool(x).view(x.size(0), -1)
        return features  # (B, 2048)


def load_medicalnet_weights(model: ResNet3D, pretrained_path: str) -> ResNet3D:
    """Load MedicalNet checkpoint, stripping common prefixes."""
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    clean = OrderedDict()
    for k, v in state.items():
        k = k.removeprefix("module.").removeprefix("net.")
        clean[k] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"[resnet3d] Missing keys ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"[resnet3d] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
    print("[resnet3d] MedicalNet weights loaded.")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Pre-processing
# ═══════════════════════════════════════════════════════════════════════════════

class MedicalImageProcessor:
    """Load and normalise NIfTI files for MedicalNet inference."""

    def __init__(self, target_shape=(64, 224, 224)):
        self.target_shape = target_shape

    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
        arr = np.clip(arr, p1, p99)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return ((arr - 0.5) / 0.5).astype(np.float32)

    def preprocess(self, nii_path: str):
        try:
            data = nib.load(nii_path).get_fdata()
            data = self._normalise(data)
            factors = [t / s for t, s in zip(self.target_shape, data.shape)]
            data = zoom(data, factors, order=1)
            return torch.from_numpy(data).float().unsqueeze(0)  # (1, D, H, W)
        except Exception as exc:
            warnings.warn(f"[resnet3d] Skipping {nii_path}: {exc}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature-extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def extract_and_save(
    nii_dir: str,
    weights_path: str,
    output_excel: str,
    target_shape=(64, 224, 224),
    batch_size: int = 2,
):
    """Extract 2048-dim features from all NIfTI files in *nii_dir* and save."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[resnet3d] Device: {device}")

    model = ResNet3D(Bottleneck3D, [3, 4, 6, 3]).to(device)
    model = load_medicalnet_weights(model, weights_path)
    model.eval()

    processor = MedicalImageProcessor(target_shape)
    nii_files = sorted(Path(nii_dir).rglob("*.nii")) + sorted(Path(nii_dir).rglob("*.nii.gz"))
    print(f"[resnet3d] Found {len(nii_files)} NIfTI files.")

    tensors, paths = [], []
    for p in tqdm(nii_files, desc="Pre-processing"):
        if p.stat().st_size == 0:
            continue
        t = processor.preprocess(str(p))
        if t is not None:
            tensors.append(t)
            paths.append(p)

    if not tensors:
        raise RuntimeError("[resnet3d] No valid NIfTI files found.")

    feature_rows = []
    for i in tqdm(range(0, len(tensors), batch_size), desc="Extracting features"):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)  # (B, 1, D, H, W)
        with torch.no_grad():
            feats = model(batch).cpu().numpy().astype(np.float64)
        for vec in feats:
            rounded = [
                float(Decimal(str(x)).quantize(Decimal("1.000000"), rounding=ROUND_HALF_UP))
                for x in vec
            ]
            feature_rows.append(", ".join(f"{v:.6f}" for v in rounded))

    # Derive patient IDs from filenames (strip extension)
    ids = [p.stem.replace(".nii", "") for p in paths]

    df = pd.DataFrame({
        "id": ids,
        "image_features": feature_rows,
    })
    Path(output_excel).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel, index=False)
    print(f"[resnet3d] Saved {len(df)} feature vectors → {output_excel}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _weights = os.environ.get("MEDICALNET_WEIGHTS", "")
    _nii_dir = os.environ.get("RAW_NII_DIR", "")
    _output  = os.environ.get("IMAGE_FEATURE_OUTPUT", _DEFAULT_OUTPUT)

    if not _weights:
        raise EnvironmentError(
            "Set MEDICALNET_WEIGHTS to the path of the MedicalNet ResNet-50 "
            "checkpoint. See .env.example for details."
        )
    if not _nii_dir:
        raise EnvironmentError(
            "Set RAW_NII_DIR to the directory containing .nii files. "
            "See .env.example for details."
        )

    extract_and_save(
        nii_dir=_nii_dir,
        weights_path=_weights,
        output_excel=_output,
    )
