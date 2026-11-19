"""Titik masuk aplikasi multipage Streamlit (urutan halaman sesuai spesifikasi)."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Evaluasi Optimasi Hyperparameter EfficientNetB0",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigasi = st.navigation(
    [
        st.Page("pages/1_Ringkasan.py", title="Ringkasan Penelitian", default=True),
        st.Page("pages/2_Prediksi_Batch.py", title="Prediksi Batch"),
        st.Page("pages/3_Evaluasi.py", title="Evaluasi dan Perbandingan"),
        st.Page("pages/4_Analisis_Kesalahan.py", title="Analisis Kesalahan dan Grad-CAM"),
        st.Page("pages/5_Riwayat_dan_MultiSeed.py", title="Riwayat Pencarian dan Multi-Seed"),
    ],
    position="sidebar",
)
navigasi.run()
