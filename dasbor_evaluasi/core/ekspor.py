"""Ekspor metrik ke CSV dan confusion matrix ke PNG (FR-14), tanpa streamlit.

CSV memakai titik desimal (format data); tampilan di aplikasi memakai koma.
PNG dibuat dengan matplotlib (deviasi yang disetujui dari Plotly+kaleido).
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from core.formatting import format_angka  # noqa: E402
from core.konstanta import KELAS, NAMA_METODE, WARNA_AKSEN  # noqa: E402
from core.metrics import MetrikModel  # noqa: E402

PETA_WARNA = LinearSegmentedColormap.from_list("aksen", ["#FFFFFF", WARNA_AKSEN])


def metrik_ke_dataframe(metrik: dict[str, MetrikModel]) -> pd.DataFrame:
    """Satu baris per model: n, accuracy, f1_macro, dan presisi/recall/f1/dukungan per kelas."""
    baris = []
    for m, k in metrik.items():
        b: dict[str, object] = {
            "metode": NAMA_METODE[m],
            "n": k.n,
            "accuracy": k.accuracy,
            "f1_macro": k.f1_macro,
        }
        for kelas in KELAS:
            b[f"presisi_{kelas}"] = k.per_kelas.loc[kelas, "presisi"]
            b[f"recall_{kelas}"] = k.per_kelas.loc[kelas, "recall"]
            b[f"f1_{kelas}"] = k.per_kelas.loc[kelas, "f1"]
            b[f"dukungan_{kelas}"] = int(k.per_kelas.loc[kelas, "dukungan"])
        baris.append(b)
    return pd.DataFrame(baris)


def metrik_ke_csv(metrik: dict[str, MetrikModel]) -> str:
    """CSV metrik semua model (UTF-8, titik desimal)."""
    return metrik_ke_dataframe(metrik).to_csv(index=False, float_format="%.10g", lineterminator="\n")


def confusion_matrix_png(
    matriks: np.ndarray, judul: str, subjudul: str, ukuran_font: int = 12
) -> bytes:
    """Gambar confusion matrix berlabel Indonesia (angka di tiap sel) sebagai bytes PNG."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150)
    try:
        ax.imshow(matriks, cmap=PETA_WARNA, vmin=0)
        ax.set_xticks(range(len(KELAS)), labels=KELAS, fontsize=ukuran_font)
        ax.set_yticks(range(len(KELAS)), labels=KELAS, fontsize=ukuran_font)
        ax.set_xlabel("Kelas prediksi", fontsize=ukuran_font)
        ax.set_ylabel("Kelas asli", fontsize=ukuran_font)
        puncak = matriks.max() if matriks.size else 0
        for i in range(matriks.shape[0]):
            for j in range(matriks.shape[1]):
                ax.text(
                    j,
                    i,
                    format_angka(int(matriks[i, j])),
                    ha="center",
                    va="center",
                    fontsize=ukuran_font,
                    color="#FFFFFF" if puncak and matriks[i, j] > puncak / 2 else "#111111",
                )
        fig.suptitle(judul, fontsize=ukuran_font + 2, fontweight="bold", x=0.5, y=0.98)
        ax.set_title(subjudul, fontsize=ukuran_font - 1, color="#333333")
        fig.tight_layout()
        penyangga = io.BytesIO()
        fig.savefig(penyangga, format="png", facecolor="#FFFFFF")
        return penyangga.getvalue()
    finally:
        plt.close(fig)
