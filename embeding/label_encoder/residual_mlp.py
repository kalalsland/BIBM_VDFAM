"""Residual MLP label / tabular encoder.

Projects raw clinical tabular features from an Excel file into a higher-
dimensional space (default 256-dim) via a two-layer residual MLP and saves
the result as a new Excel file consumed by the training pipeline.

Configuration (via environment variables or .env file)
------------------------------------------------------
CLINIC_RAW_PATH      Path to the raw clinical Excel file.
                     Expected columns: id (col 0), features (cols 1…-2),
                     label (last col).
CLINIC_ENCODED_OUTPUT  Output path for the encoded Excel file.
                       Default: <package>/data/encoded/clinic_data_1.xlsx

Usage
-----
    python -m stroke_multimodal_prognosis.embeding.label_encoder.residual_mlp
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ── Load .env if available ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parents[3] / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

_ENCODED_DIR = Path(__file__).parents[3] / "data" / "encoded"
_DEFAULT_OUTPUT = str(_ENCODED_DIR / "clinic_data_1.xlsx")


# ═══════════════════════════════════════════════════════════════════════════════
#  Model definition
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualMLP(nn.Module):
    """Two-layer residual MLP that projects input_dim → output_dim (default 256).

    A skip connection is added when input_dim == output_dim.
    """

    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        if identity.shape[1] == out.shape[1]:
            out = out + identity
        return self.relu(out)


# ═══════════════════════════════════════════════════════════════════════════════
#  Encoding pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def encode_and_save(
    clinic_path: str,
    output_excel: str,
    output_dim: int = 256,
    sheet_name: str = "Sheet1",
):
    """Read clinic Excel, project features with ResidualMLP, save output Excel.

    The clinic Excel is expected to have:
        col 0     — patient id
        cols 1…-2 — clinical features
        last col  — label
    """
    df = pd.read_excel(clinic_path, sheet_name=sheet_name)

    patient_ids = df.iloc[:, 0]
    features    = df.iloc[:, 1:-1]
    labels      = df.iloc[:, -1]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features).astype(np.float32)
    x = torch.from_numpy(scaled)

    model = ResidualMLP(input_dim=features.shape[1], output_dim=output_dim)
    with torch.no_grad():
        projected = model(x).numpy()

    out_df = pd.DataFrame(
        projected,
        columns=[f"feature_{i}" for i in range(projected.shape[1])],
    )
    out_df.insert(0, "id", patient_ids.values)
    out_df["label"] = labels.values

    Path(output_excel).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(output_excel, index=False)
    print(
        f"[residual_mlp] {features.shape[1]}-dim → {projected.shape[1]}-dim | "
        f"Saved {len(out_df)} rows → {output_excel}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _clinic_path = os.environ.get("CLINIC_RAW_PATH", "")
    _output      = os.environ.get("CLINIC_ENCODED_OUTPUT", _DEFAULT_OUTPUT)

    if not _clinic_path:
        raise EnvironmentError(
            "Set CLINIC_RAW_PATH to the raw clinical Excel file. "
            "See .env.example for details."
        )

    encode_and_save(clinic_path=_clinic_path, output_excel=_output)
