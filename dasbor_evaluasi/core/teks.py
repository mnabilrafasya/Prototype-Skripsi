"""Teks penelitian yang dikutip dari laporan skripsi (bagian 1.3 dan 1.5).

Kalimat disalin apa adanya dari LAPORAN_SKRIPSI.docx; hanya spasi berlebih akibat
kode sitasi Zotero yang dirapikan. Tidak ada angka hasil penelitian di sini.
"""

from __future__ import annotations

from typing import Final

JUDUL_SKRIPSI: Final[str] = (
    "Perbandingan Grid Search, Random Search, dan Optuna (TPE) dalam Optimasi "
    "Hyperparameter EfficientNetB0 untuk Klasifikasi Tumor Otak pada Citra MRI"
)

# Laporan, bagian 1.3 Tujuan Penelitian.
TUJUAN_PENELITIAN: Final[tuple[str, ...]] = (
    "Membandingkan performa model EfficientNetB0 yang dioptimasi menggunakan Grid Search, "
    "Random Search, dan Optuna (TPE) berdasarkan metrik akurasi dan F1-score pada data "
    "validasi maupun data uji.",
    "Menganalisis efisiensi komputasi ketiga metode optimasi hyperparameter dalam hal total "
    "waktu pencarian dan waktu per trial.",
    "Mengevaluasi konsistensi generalisasi model dengan membandingkan performa pada data "
    "validasi dan data uji, serta mengidentifikasi kemungkinan faktor penyebab kesenjangan "
    "performa yang teramati antara kedua data tersebut.",
    "Mengidentifikasi pola konfigurasi hyperparameter yang dipilih oleh masing-masing metode "
    "sebagai konfigurasi optimal.",
    "Menganalisis performa model per kelas tumor untuk mengidentifikasi kelas yang paling sulit "
    "diklasifikasikan, didukung dengan analisis kualitatif menggunakan Grad-CAM "
    "(Selvaraju et al., 2017).",
    "Memvalidasi kestabilan statistik dari perbandingan antarmetode melalui pengujian dengan "
    "beberapa random seed berbeda.",
)

# Laporan, bagian 1.5 Batasan Masalah.
BATASAN_PENELITIAN: Final[tuple[str, ...]] = (
    "Dataset yang digunakan terbatas pada Brain Tumor MRI Dataset (Masoud Nickparvar, 2026) "
    "yang terdiri dari 4 kelas (glioma, meningioma, notumor, pituitary), sehingga generalisasi "
    "hasil terbatas pada karakteristik dataset ini.",
    "Arsitektur model dibatasi hanya pada EfficientNetB0 pretrained ImageNet sebagai backbone "
    "tetap (Tan & Le, 2019); penelitian ini tidak membandingkan arsitektur CNN lain.",
    "Ruang pencarian hyperparameter dibatasi pada lima parameter yang divariasikan: learning "
    "rate, ukuran batch, jenis optimizer (Adam dan SGD), dropout rate, dan jumlah blok backbone "
    "yang di-unfreeze (n_unfreeze). Parameter jumlah unit dense layer (256) dan weight decay "
    "(1×10⁻⁴) ditetapkan konstan (fixed) dan tidak termasuk dalam ruang pencarian, begitu pula "
    "hyperparameter lain seperti jenis scheduler dan strategi augmentasi data.",
    "Anggaran percobaan dibatasi pada 48 kombinasi/trial untuk masing-masing metode agar "
    "perbandingan bersifat adil.",
    "Pelatihan setiap konfigurasi dibatasi maksimal 10 epoch dengan mekanisme early stopping "
    "berdasarkan F1-score validasi (patience = 3), bukan hingga model mencapai konvergensi penuh.",
    "Proses pencarian hyperparameter (48 trial per metode) dilakukan dengan seed tunggal (42) "
    "mengingat keterbatasan sumber daya komputasi untuk mengulang seluruh proses pencarian; "
    "validasi tambahan dengan variasi seed dilakukan secara terpisah dan hanya pada tahap "
    "pelatihan ulang konfigurasi terbaik (bukan pada seluruh proses pencarian), mengikuti "
    "pertimbangan efisiensi yang lazim pada penelitian dengan sumber daya terbatas "
    "(Picard, 2023; Bouthillier et al., 2021)",
)

PERNYATAAN_PROTOTIPE: Final[str] = (
    "Aplikasi ini adalah prototipe penelitian untuk evaluasi batch pada data uji bawaan. "
    "Aplikasi ini bukan alat diagnosis medis dan tidak boleh dipakai untuk keputusan klinis."
)

# Nama tampilan kunci konfigurasi di meta.json (kunci yang tidak ada di sini
# ditampilkan apa adanya).
NAMA_KONFIGURASI: Final[dict[str, str]] = {
    "lr": "learning rate",
    "batch_size": "ukuran batch",
    "optimizer": "optimizer",
    "weight_decay": "weight decay",
    "dropout_rate": "dropout rate",
    "dense_units": "dense units",
    "n_unfreeze": "n_unfreeze",
}
