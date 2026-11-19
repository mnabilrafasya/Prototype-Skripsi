"""Tes ekspor CSV metrik dan PNG confusion matrix (FR-14)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from PIL import Image

from core.artifacts import Artefak
from core.ekspor import (
    confusion_matrix_png,
    gradcam_delapan_contoh_png,
    gradcam_tiga_model_png,
    kurva_konvergensi_png,
    metrik_ke_csv,
    metrik_ke_dataframe,
    multi_seed_f1_png,
    sebaran_f1_trial_png,
)
from core.gradcam import HasilGradCAM
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


def test_ekspor_png_gradcam_dan_riwayat(artefak: Artefak) -> None:
    citra = np.zeros((224, 224, 3), dtype=np.float32)
    cam = np.linspace(0.0, 1.0, 224 * 224, dtype=np.float32).reshape(224, 224)
    per_metode = {
        m: HasilGradCAM(
            metode=m,
            kelas_prediksi=0,
            probabilitas_prediksi=0.95,
            probabilitas_semua=np.array([0.95, 0.02, 0.02, 0.01], dtype=np.float32),
            citra_asli=citra,
            cam_resized=cam,
            overlay=citra,
        )
        for m in METODE
    }
    for data_png in (
        gradcam_tiga_model_png("glioma/Te-gl_16.jpg", 0, per_metode),
        gradcam_delapan_contoh_png([(137, "glioma/Te-gl_16.jpg", 0, per_metode)]),
        kurva_konvergensi_png(artefak.trial),
        sebaran_f1_trial_png(artefak.trial),
        multi_seed_f1_png(artefak.multi_seed, artefak.meta.seeds_used),
    ):
        assert data_png.startswith(b"\x89PNG\r\n\x1a\n")
        img = Image.open(io.BytesIO(data_png))
        assert img.format == "PNG"

