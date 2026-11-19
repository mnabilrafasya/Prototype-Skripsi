"""Tes halaman Streamlit Tahap 4 (halaman 4 dan 5) dengan AppTest."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from core.artifacts import Artefak
from core.data import indeks_contoh_laporan
from core.formatting import format_angka
from core.hasil import HasilBatch, WaktuModel, kumpulkan_info
from core.konstanta import METODE
from ui import sumber

AKAR = Path(__file__).resolve().parents[1]


def _buat_hasil_uji(artefak: Artefak, indeks: np.ndarray) -> HasilBatch:
    m = artefak.manifest
    rng = np.random.default_rng(42)
    return HasilBatch(
        indeks=indeks,
        metode=METODE,
        prediksi={k: m[f"pred_{k}"].to_numpy()[indeks].astype(np.int64) for k in METODE},
        proba_prediksi={k: rng.uniform(0.6, 0.99, len(indeks)).astype(np.float64) for k in METODE},
        waktu={k: WaktuModel(len(indeks), 10.0, 9.0) for k in METODE},
        info=kumpulkan_info(16, "semua data uji"),
    )


def test_halaman_5_tidak_mengimpor_pustaka_pelatihan_atau_uji_statistik() -> None:
    jalur = AKAR / "pages" / "5_Riwayat_dan_MultiSeed.py"
    pohon = ast.parse(jalur.read_text(encoding="utf-8"), filename=str(jalur))
    terlarang = {"torch", "torchvision", "scipy", "statsmodels", "sklearn"}
    for simpul in ast.walk(pohon):
        if isinstance(simpul, ast.Import):
            for n in simpul.names:
                assert n.name.split(".")[0] not in terlarang
        elif isinstance(simpul, ast.ImportFrom) and simpul.module:
            assert simpul.module.split(".")[0] not in terlarang


def test_halaman_5_menampilkan_riwayat_dan_multiseed_dari_artefak() -> None:
    at = AppTest.from_file(str(AKAR / "pages" / "5_Riwayat_dan_MultiSeed.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Riwayat Pencarian dan Multi-Seed"
    assert any("tidak dihitung ulang di aplikasi" in i.value for i in at.info)
    # 5 tabel: waktu pencarian, konfigurasi terbaik, riwayat trial, ringkasan multi-seed, per seed, wilcoxon -> 6 tabel
    assert len(at.dataframe) == 6
    df_waktu = at.dataframe[0].value
    assert "Waktu rata-rata per trial — total/jumlah trial (detik)" in df_waktu.columns
    assert "Rata-rata kolom training_time_sec (detik)" in df_waktu.columns
    df_ms = at.dataframe[3].value
    assert "Simpangan baku F1-score macro (ddof = 1)" in df_ms.columns
    df_wil = at.dataframe[5].value
    assert len(df_wil) == 3


def test_halaman_4_tombol_8_contoh_dan_daftar_salah_klasifikasi(artefak: Artefak) -> None:
    at = AppTest.from_file(str(AKAR / "pages" / "4_Analisis_Kesalahan.py"), default_timeout=120)
    at.session_state[sumber.KUNCI_HASIL] = _buat_hasil_uji(artefak, np.arange(1600))
    at.session_state[sumber.KUNCI_SUMBER_HASIL] = "dijalankan pada sesi ini"
    at.run()
    assert not at.exception
    assert at.title[0].value == "Analisis Kesalahan dan Grad-CAM"
    # Tabel daftar salah klasifikasi + tabel info Grad-CAM citra salah terpilih
    assert len(at.dataframe) == 2

    # Klik tombol "Muat 8 contoh laporan" dan pastikan urutannya persis sama dengan notebook
    at.button(key="tombol_muat_8_contoh").click().run()
    assert not at.exception
    df_8 = at.dataframe[0].value
    harapan_indeks = [format_angka(int(i)) for i in indeks_contoh_laporan()]
    assert df_8["Indeks manifest"].tolist() == harapan_indeks
    assert df_8["Indeks manifest"].tolist() != sorted(harapan_indeks)
