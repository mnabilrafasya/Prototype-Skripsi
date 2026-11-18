"""Pembantu tampilan: tabel berformat Indonesia, subjudul grafik, dan peta panas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.formatting import format_angka, format_metrik
from core.hasil import HasilBatch
from core.konstanta import KELAS, NAMA_METODE, WARNA_AKSEN, WARNA_METODE
from core.metrics import MetrikModel
from ui import tema


def subjudul(hasil: HasilBatch) -> str:
    """Subjudul grafik: jumlah sampel dan sumber data, mis. 'N = 1.600, data uji'."""
    teks = f"N = {format_angka(hasil.n)}, data uji"
    return teks if hasil.penuh else f"{teks} ({hasil.info['sampel']})"


def penanda_metode(metode: str) -> str:
    """Penanda persegi berwarna tetap untuk metode (teks tetap gelap agar kontras tinggi)."""
    return f'<span style="color:{WARNA_METODE[metode]}">&#9632;</span> {NAMA_METODE[metode]}'


def tampilkan_tabel(df: pd.DataFrame, tinggi: int | None = None) -> None:
    """Tampilkan tabel string. Mode Presentasi memakai tabel HTML agar huruf ikut membesar."""
    if tema.presentasi_aktif() and tinggi is None:
        st.table(df, hide_index=True)
    else:
        st.dataframe(df, hide_index=True, width="stretch", height=tinggi or "auto")


def peta_panas(matriks: np.ndarray, metode: str, hasil: HasilBatch) -> go.Figure:
    """Peta panas confusion matrix (angka di sel); skala satu warna aksen."""
    ukuran = 18 if tema.presentasi_aktif() else 13
    teks = [[format_angka(int(v)) for v in baris] for baris in matriks]
    fig = go.Figure(
        go.Heatmap(
            z=matriks,
            x=list(KELAS),
            y=list(KELAS),
            text=teks,
            texttemplate="%{text}",
            colorscale=[[0.0, "#FFFFFF"], [1.0, WARNA_AKSEN]],
            showscale=False,
            hovertemplate="Kelas asli: %{y}<br>Kelas prediksi: %{x}<br>Jumlah: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Confusion Matrix {NAMA_METODE[metode]}",
            subtitle=dict(text=subjudul(hasil)),
        ),
        xaxis=dict(title="Kelas prediksi", side="bottom"),
        yaxis=dict(title="Kelas asli", autorange="reversed"),
        separators=",.",
        font=dict(size=ukuran, color="#111111"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=80, b=10),
        height=460,
    )
    return fig


def grafik_perbandingan_metrik(
    metrik: dict[str, MetrikModel], hasil: HasilBatch
) -> go.Figure:
    """Grafik batang perbandingan akurasi, F1-score macro, dan F1-score per kelas."""
    ukuran = 17 if tema.presentasi_aktif() else 12
    kategori = ["Akurasi", "F1-score macro", *(f"F1 {k}" for k in KELAS)]
    fig = go.Figure()
    for m, km in metrik.items():
        nilai = [
            km.accuracy,
            km.f1_macro,
            *(float(km.per_kelas.loc[k, "f1"]) for k in KELAS),
        ]
        teks = [format_metrik(v) for v in nilai]
        fig.add_trace(
            go.Bar(
                name=NAMA_METODE[m],
                x=kategori,
                y=nilai,
                text=teks,
                textposition="outside",
                marker_color=WARNA_METODE[m],
                hovertemplate=(
                    f"Metode: {NAMA_METODE[m]}<br>"
                    "Metrik: %{x}<br>Nilai: %{text}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=dict(
            text="Perbandingan Metrik Ketiga Model",
            subtitle=dict(text=subjudul(hasil)),
        ),
        xaxis=dict(title="Metrik evaluasi"),
        yaxis=dict(title="Nilai metrik", range=[0.0, 1.08], tickformat=".4f"),
        barmode="group",
        separators=",.",
        font=dict(size=ukuran, color="#111111"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(title="Metode", orientation="h", y=1.02, yanchor="bottom", x=0.0),
        margin=dict(l=10, r=10, t=100, b=10),
        height=460,
    )
    return fig

