"""Konstanta tetap penelitian (bukan hasil penelitian).

Isi berkas ini bersumber dari spesifikasi dan lampiran_kode_inti.ipynb:
urutan kelas, parameter praproses, kunci metode, dan sitasi dataset.
"""

from __future__ import annotations

from typing import Final

# Urutan indeks kelas tetap: 0 glioma, 1 meningioma, 2 notumor, 3 pituitary.
KELAS: Final[tuple[str, ...]] = ("glioma", "meningioma", "notumor", "pituitary")

# Praproses inferensi (eval_transform di notebook, sel 3).
UKURAN_CITRA: Final[int] = 224
MEAN_IMAGENET: Final[tuple[float, ...]] = (0.485, 0.456, 0.406)
STD_IMAGENET: Final[tuple[float, ...]] = (0.229, 0.224, 0.225)

# Jumlah data uji bawaan (folder Testing: 400 per kelas x 4 kelas).
JUMLAH_DATA_UJI: Final[int] = 1600

# Kunci metode persis seperti pada artefak.
METODE: Final[tuple[str, ...]] = ("grid_search", "random_search", "optuna_tpe")

# Nama tampilan metode persis seperti pada laporan dan mcnemar_acuan.csv.
NAMA_METODE: Final[dict[str, str]] = {
    "grid_search": "Grid Search",
    "random_search": "Random Search",
    "optuna_tpe": "Optuna (TPE)",
}

# Tiga pasangan uji, urutan sama dengan PAIRS di notebook (sel 12).
PASANGAN: Final[tuple[tuple[str, str], ...]] = (
    ("grid_search", "random_search"),
    ("grid_search", "optuna_tpe"),
    ("random_search", "optuna_tpe"),
)

# Sitasi dataset: persis entri Daftar Pustaka laporan (versi 2 dipakai penelitian).
# Dikonfirmasi pengguna pada Tahap 2.
SITASI_DATASET: Final[str] = (
    "Masoud Nickparvar. (2026). Brain Tumor MRI Dataset (Version 2) [Data set]. "
    "Kaggle. https://doi.org/10.34740/KAGGLE/DSV/14832123"
)

# Kriteria penerimaan FR-01: total parameter tiap model (EfficientNetB0 + head baru).
JUMLAH_PARAMETER_MODEL: Final[int] = 4_336_512

# Ambang status FR-13.
TOLERANSI_SELISIH_AKURASI: Final[float] = 0.002
MAKS_PREDIKSI_BERBEDA: Final[int] = 8

# Seed tetap pemilihan sampel (FR-03, FR-10).
SEED_SAMPEL: Final[int] = 42
JUMLAH_CONTOH_LAPORAN: Final[int] = 8
