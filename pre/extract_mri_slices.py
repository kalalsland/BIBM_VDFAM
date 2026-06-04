"""Extract axial, coronal, and sagittal PNG slices from NIfTI MRI files.

Reads .nii / .nii.gz files from RAW_NII_DIR, produces three high-resolution
PNGs per patient (axial.png, coronal.png, sagittal.png) in MRI_SLICES_DIR/<id>/.

Configuration (environment variables / .env)
--------------------------------------------
RAW_NII_DIR      Directory containing .nii files (one per patient).
MRI_SLICES_DIR   Output base directory; a sub-folder is created per patient.

Usage
-----
    python -m stroke_multimodal_prognosis.pre.extract_mri_slices
    # or
    python stroke_multimodal_prognosis/pre/extract_mri_slices.py
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

# ── Load .env if available ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parents[2] / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Core functions
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_slice(data: np.ndarray) -> np.ndarray:
    """Percentile clip + gamma correction → uint8."""
    vmin, vmax = np.percentile(data, 5), np.percentile(data, 95)
    clipped = np.clip((data - vmin) / (vmax - vmin + 1e-8), 0, 1)
    gamma = np.power(clipped, 1 / 1.2) * 255
    return gamma.astype(np.uint8)


def _extract_slice(volume: np.ndarray, axis: int, index: int, scale: float) -> np.ndarray:
    """Extract a 2-D slice along *axis* at *index*, rotate and upscale."""
    if axis == 0:
        sl = volume[index, :, :]
    elif axis == 1:
        sl = volume[:, index, :]
    else:
        sl = volume[:, :, index]

    if axis in (0, 1):
        sl = np.rot90(sl)

    norm = _normalise_slice(sl)
    img = Image.fromarray(norm)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    return np.array(img.resize((new_w, new_h), Image.LANCZOS))


def generate_mri_slices(
    nii_path: str,
    output_dir: str,
    scale_factor: float = 2.0,
) -> bool:
    """Extract three orthogonal slices from a NIfTI file and save as PNG."""
    try:
        data = nib.load(nii_path).get_fdata()
        os.makedirs(output_dir, exist_ok=True)

        slices = {
            "axial":    _extract_slice(data, 2, data.shape[2] // 2, scale_factor),
            "coronal":  _extract_slice(data, 1, data.shape[1] // 2, scale_factor),
            "sagittal": _extract_slice(data, 0, data.shape[0] // 2, scale_factor),
        }
        for name, arr in slices.items():
            Image.fromarray(arr).save(
                os.path.join(output_dir, f"{name}.png"),
                format="PNG",
                compress_level=0,
            )
        return True
    except Exception as exc:
        print(f"[extract_mri_slices] Error on {nii_path}: {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _raw_dir    = os.environ.get("RAW_NII_DIR", "")
    _slices_dir = os.environ.get("MRI_SLICES_DIR", "")
    _scale      = float(os.environ.get("MRI_SCALE_FACTOR", "2.0"))

    if not _raw_dir:
        raise EnvironmentError("Set RAW_NII_DIR. See .env.example for details.")
    if not _slices_dir:
        raise EnvironmentError("Set MRI_SLICES_DIR. See .env.example for details.")

    nii_files = list(Path(_raw_dir).glob("*.nii")) + list(Path(_raw_dir).glob("*.nii.gz"))
    print(f"[extract_mri_slices] Found {len(nii_files)} NIfTI files in {_raw_dir}")
    print(f"[extract_mri_slices] Scale factor: {_scale}x")

    ok = err = 0
    for nii in nii_files:
        patient_id = nii.stem.replace(".nii", "")
        out_folder = os.path.join(_slices_dir, patient_id)
        if generate_mri_slices(str(nii), out_folder, scale_factor=_scale):
            ok += 1
        else:
            err += 1

    print(f"[extract_mri_slices] Done — success: {ok}, failed: {err} / {len(nii_files)}")
