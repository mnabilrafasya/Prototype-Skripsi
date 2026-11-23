"""Halaman 1: Ringkasan Penelitian (FR-15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.formatting import format_angka, format_angka_ringkas
from core.konstanta import (
    JUMLAH_LATIH,
    JUMLAH_TRAINING,
    JUMLAH_VALIDASI,
    KELAS,
    METODE,
    NAMA_METODE,
    SITASI_DATASET,
)
from core.teks import (
    BATASAN_PENELITIAN,
    JUDUL_SKRIPSI,
    NAMA_KONFIGURASI,
    PERNYATAAN_PROTOTIPE,
    TUJUAN_PENELITIAN,
)
from ui import sumber, tema
from ui.tampilan import tampilkan_tabel

tema.judul("Ringkasan Penelitian")
a = sumber.artefak()
meta = a.meta

with st.container(border=True):
    st.markdown(f"### {JUDUL_SKRIPSI}")
    st.write("Program Studi Teknik Informatika, Fakultas Ilmu Komputer, Universitas Sriwijaya, 2026.")

st.subheader("Dataset dan Pembagiannya")
st.write(f"Dataset: {SITASI_DATASET}")
st.write(
    "Kelas dan urutan indeks tetap: "
    + ", ".join(f"{i} {k}" for i, k in enumerate(KELAS))
    + "."
)

jumlah_uji = np.bincount(a.manifest["label"].to_numpy(), minlength=len(KELAS))
total_uji = int(jumlah_uji.sum())
total_dataset = JUMLAH_TRAINING + total_uji

k1, k2, k3, k4 = st.columns(4)
k1.metric("Data latih (80% Training)", format_angka(JUMLAH_LATIH))
k2.metric("Data validasi (20% Training)", format_angka(JUMLAH_VALIDASI))
k3.metric("Data uji (dipakai prototipe)", format_angka(total_uji))
k4.metric("Total dataset", format_angka(total_dataset))

tampilkan_tabel(
    pd.DataFrame(
        {
            "Subset": ["Data latih", "Data validasi", "Data uji"],
            "Sumber": [
                f"Folder Training ({format_angka(JUMLAH_TRAINING)} citra), 80%",
                f"Folder Training ({format_angka(JUMLAH_TRAINING)} citra), 20%",
                "Folder Testing",
            ],
            "Jumlah citra": [
                format_angka(JUMLAH_LATIH),
                format_angka(JUMLAH_VALIDASI),
                format_angka(total_uji),
            ],
            "Keterangan": [
                "Tidak ada di prototipe",
                "Tidak ada di prototipe",
                "Per kelas: "
                + ", ".join(f"{k} {format_angka(int(n))}" for k, n in zip(KELAS, jumlah_uji))
                + ". Dipakai prototipe.",
            ],
        }
    )
)
st.caption(
    f"Folder Training dipecah stratified 80:20 dengan SPLIT_SEED = {meta.split_seed}. "
    f"Seed pelatihan ulang konfigurasi terbaik: {meta.train_seed}. "
    "Prototipe hanya memakai data uji bawaan; tidak ada unggahan dataset."
)

st.subheader("Konfigurasi Terbaik Hyperparameter")
kunci = list(meta.best_configs[METODE[0]])
konfigurasi = pd.DataFrame(
    [
        {"Metode": NAMA_METODE[m]}
        | {
            NAMA_KONFIGURASI.get(k, k): (
                format_angka_ringkas(v) if isinstance(v, (int, float)) else str(v)
            )
            for k, v in meta.best_configs[m].items()
        }
        for m in METODE
    ],
    columns=["Metode", *(NAMA_KONFIGURASI.get(k, k) for k in kunci)],
)
tampilkan_tabel(konfigurasi)
st.caption("Sumber: meta.json. Nilai numerik ditampilkan dengan paling banyak 6 desimal.")

with st.expander("Tujuan Penelitian (6 butir — BAB I 1.3)", expanded=False):
    st.markdown("\n".join(f"{i}. {t}" for i, t in enumerate(TUJUAN_PENELITIAN, start=1)))

with st.expander("Batasan Penelitian (6 butir — BAB I 1.5)", expanded=False):
    st.markdown("\n".join(f"{i}. {t}" for i, t in enumerate(BATASAN_PENELITIAN, start=1)))

st.subheader("Pernyataan")
with st.container(border=True):
    st.markdown(f"**{PERNYATAAN_PROTOTIPE}**")

tema.footer()
