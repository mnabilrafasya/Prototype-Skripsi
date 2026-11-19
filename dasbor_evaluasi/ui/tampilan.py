"""Pembantu tampilan: tabel berformat Indonesia, subjudul grafik, dan peta panas."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.formatting import format_angka, format_metrik
from core.hasil import HasilBatch
from core.konstanta import KELAS, METODE, NAMA_METODE, WARNA_AKSEN, WARNA_METODE
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


def grafik_konvergensi(trial: dict[str, pd.DataFrame]) -> go.Figure:
    """Kurva konvergensi (best_so_far) ketiga metode dalam satu grafik (FR-11)."""
    ukuran = 17 if tema.presentasi_aktif() else 12
    n_trial = len(next(iter(trial.values())))
    simbol = {"grid_search": "circle", "random_search": "square", "optuna_tpe": "triangle-up"}
    fig = go.Figure()
    for m in METODE:
        df = trial[m]
        teks = [format_metrik(float(v)) for v in df["best_so_far"]]
        fig.add_trace(
            go.Scatter(
                name=NAMA_METODE[m],
                x=df["trial"],
                y=df["best_so_far"],
                mode="lines+markers",
                marker=dict(symbol=simbol[m], size=7, color=WARNA_METODE[m]),
                line=dict(color=WARNA_METODE[m], width=2.5),
                text=teks,
                hovertemplate=(
                    f"Metode: {NAMA_METODE[m]}<br>"
                    "Trial ke-%{x}<br>F1 terbaik sejauh ini: %{text}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=dict(
            text="Kurva Konvergensi Optimasi Hyperparameter",
            subtitle=dict(
                text=f"N = {format_angka(n_trial)} trial per metode, data validasi (artefak)"
            ),
        ),
        xaxis=dict(title="Trial ke-n", dtick=4),
        yaxis=dict(title="F1-score macro validasi terbaik sejauh ini", tickformat=".4f"),
        separators=",.",
        font=dict(size=ukuran, color="#111111"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(title="Metode", orientation="h", y=1.02, yanchor="bottom", x=0.0),
        margin=dict(l=10, r=10, t=100, b=10),
        height=440,
    )
    return fig


def grafik_sebaran_trial(
    trial: dict[str, pd.DataFrame], n_startup_optuna: int = 10
) -> go.Figure:
    """Sebaran F1-score validasi per trial dan best_so_far untuk ketiga metode (FR-11)."""
    ukuran = 16 if tema.presentasi_aktif() else 12
    n_trial = len(next(iter(trial.values())))
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=[NAMA_METODE[m] for m in METODE],
    )
    for kol, m in enumerate(METODE, start=1):
        df = trial[m]
        teks_f1 = [format_metrik(float(v)) for v in df["val_f1"]]
        fig.add_trace(
            go.Scatter(
                name=f"F1 trial ({NAMA_METODE[m]})",
                x=df["trial"],
                y=df["val_f1"],
                mode="markers",
                marker=dict(color=WARNA_METODE[m], size=7, opacity=0.8),
                text=teks_f1,
                hovertemplate=(
                    f"Metode: {NAMA_METODE[m]}<br>"
                    "Trial ke-%{x}<br>F1 validasi: %{text}<extra></extra>"
                ),
            ),
            row=1,
            col=kol,
        )
        fig.add_trace(
            go.Scatter(
                name=f"Terbaik sejauh ini ({NAMA_METODE[m]})",
                x=df["trial"],
                y=df["best_so_far"],
                mode="lines",
                line=dict(color=WARNA_AKSEN, width=2, shape="hv"),
                showlegend=(kol == 1),
                hoverinfo="skip",
            ),
            row=1,
            col=kol,
        )
        if m == "optuna_tpe":
            fig.add_vrect(
                x0=0.5,
                x1=n_startup_optuna + 0.5,
                fillcolor="#CCCCCC",
                opacity=0.25,
                line_width=1,
                line_dash="dash",
                line_color="#666666",
                row=1,
                col=kol,
            )
        fig.update_xaxes(title_text="Trial ke-n", row=1, col=kol)
    fig.update_yaxes(title_text="F1-score macro validasi", tickformat=".4f", row=1, col=1)
    fig.update_layout(
        title=dict(
            text="Sebaran F1-score Validasi per Trial",
            subtitle=dict(
                text=(
                    f"N = {format_angka(n_trial)} trial per metode, data validasi (artefak; "
                    f"area abu-abu Optuna = {n_startup_optuna} trial awal acak)"
                )
            ),
        ),
        separators=",.",
        font=dict(size=ukuran, color="#111111"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", y=-0.22, x=0.0),
        margin=dict(l=10, r=10, t=100, b=60),
        height=460,
    )
    return fig


def grafik_multi_seed(multi_seed: pd.DataFrame, seeds: Sequence[int]) -> go.Figure:
    """F1-score macro per seed per metode dari multi_seed_hasil.csv (FR-12)."""
    ukuran = 17 if tema.presentasi_aktif() else 12
    label_seed = [str(s) for s in seeds]
    fig = go.Figure()
    for m in METODE:
        sub = multi_seed[multi_seed["method"] == m].set_index("seed")
        nilai = [float(sub.loc[int(s), "f1_macro"]) for s in seeds]
        teks = [format_metrik(v) for v in nilai]
        fig.add_trace(
            go.Bar(
                name=NAMA_METODE[m],
                x=label_seed,
                y=nilai,
                text=teks,
                textposition="outside",
                marker_color=WARNA_METODE[m],
                hovertemplate=(
                    f"Metode: {NAMA_METODE[m]}<br>"
                    "Seed: %{x}<br>F1-score macro: %{text}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=dict(
            text="F1-score Macro per Seed pada Setiap Metode",
            subtitle=dict(
                text=(
                    f"N = {format_angka(len(seeds))} seed per metode, data uji "
                    "(dihitung di notebook, artefak)"
                )
            ),
        ),
        xaxis=dict(title="Seed pelatihan ulang (TRAIN_SEED)", type="category"),
        yaxis=dict(title="F1-score macro data uji", range=[0.85, 0.96], tickformat=".4f"),
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


