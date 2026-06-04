"""embeding – feature-extraction encoders for stroke multimodal prognosis.

Sub-packages
------------
image_encoder   MedicalNet 3D ResNet-50 — encodes NIfTI MRI volumes.
text_encoder    BioBERT v1.1 PubMed     — encodes radiology text reports.
label_encoder   Residual MLP            — projects clinical tabular features.

Usage
-----
Run ``python -m stroke_multimodal_prognosis.embeding.run_encoders`` to
execute all three encoders in sequence and write the encoded feature Excel
files to ``data/encoded/``.
"""
