"""Tes ekspor CSV metrik dan PNG confusion matrix (FR-14)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from PIL import Image

from core.ekspor import confusion_matrix_png, metrik_ke_csv, metrik_ke_dataframe
from core.konstanta import KELAS, METODE, NAMA_METODE
from core.metrics import hitung_metrik


def test_metrik_ke_dataframe_dan_csv() -> None:
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 1, 1, 1, 2, 3, 3, 3]
    metrik = {m: hitung_metrik(y_true, y_pred) for m in METODE}
    df = metrik_ke_dataframe(metrik)
    assert len(df) == 3
    assert df["metode"].tolist() == [NAMA_METODE[m] for m in METODE]
    for k in KELAS:
        assert f"presisi_{k}" in df.columns
        assert f"recall_{k}" in df.columns
        assert f"f1_{k}" in df.columns
        assert f"dukungan_{k}" in df.columns
    teks_csv = metrik_ke_csv(metrik)
    df_baca = pd.read_csv(io.StringIO(teks_csv))
    assert list(df_baca.columns) == list(df.columns)
    assert len(df_baca) == 3


def test_confusion_matrix_png_menghasilkan_gambar_valid() -> None:
    cm = np.array([[291, 69, 38, 2], [2, 392, 3, 3], [0, 0, 400, 0], [0, 7, 0, 393]])
    data_png = confusion_matrix_png(cm, "Confusion Matrix Grid Search", "N = 1.600, data uji")
    assert data_png.startswith(b"\x89PNG\r\n\x1a\n")
    img = Image.open(io.BytesIO(data_png))
    assert img.format == "PNG"
    assert img.size[0] > 400 and img.size[1] > 400
