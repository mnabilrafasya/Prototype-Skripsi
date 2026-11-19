"""Ekspor metrik ke CSV serta confusion matrix, Grad-CAM, dan grafik ke PNG (FR-14).

CSV memakai titik desimal (format data); tampilan di aplikasi memakai koma.
PNG dibuat dengan matplotlib (deviasi yang disetujui dari Plotly+kaleido).
Modul ini tidak mengimpor streamlit.
"""

from __future__ import annotations

from collections.abc import Sequence
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from core.formatting import format_angka, format_metrik  # noqa: E402
from core.gradcam import HasilGradCAM  # noqa: E402
from core.konstanta import KELAS, METODE, NAMA_METODE, WARNA_AKSEN, WARNA_METODE  # noqa: E402
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


def _simpan_fig(fig: plt.Figure) -> bytes:
    penyangga = io.BytesIO()
    fig.savefig(penyangga, format="png", facecolor="#FFFFFF", bbox_inches="tight")
    return penyangga.getvalue()


def gradcam_tiga_model_png(
    berkas: str,
    label_asli: int,
    hasil_per_metode: dict[str, HasilGradCAM],
    ukuran_font: int = 10,
) -> bytes:
    """Gambar PNG Grad-CAM satu citra untuk ketiga model berdampingan (3 baris x 3 kolom)."""
    metode_urut = [m for m in METODE if m in hasil_per_metode]
    fig, axes = plt.subplots(
        3, len(metode_urut), figsize=(3.6 * len(metode_urut), 9.2), dpi=150, squeeze=False
    )
    try:
        nama_asli = KELAS[label_asli]
        fig.suptitle(
            f"Perbandingan Grad-CAM Ketiga Model — {berkas} (Label asli: {nama_asli})",
            fontsize=ukuran_font + 2,
            fontweight="bold",
            color=WARNA_AKSEN,
            y=0.99,
        )
        label_baris = ("Citra asli", "Peta panas Grad-CAM", "Overlay Grad-CAM (alpha = 0,45)")
        for j, m in enumerate(metode_urut):
            h = hasil_per_metode[m]
            pred_nama = KELAS[h.kelas_prediksi]
            status = "Benar" if h.kelas_prediksi == label_asli else "Salah"
            judul_kolom = (
                f"{NAMA_METODE[m]}\n"
                f"Prediksi: {pred_nama} ({format_metrik(h.probabilitas_prediksi)}) — {status}"
            )
            axes[0, j].set_title(
                judul_kolom, fontsize=ukuran_font, fontweight="bold", color=WARNA_METODE[m]
            )
            axes[0, j].imshow(h.citra_asli)
            axes[1, j].imshow(h.cam_resized, cmap="jet", vmin=0.0, vmax=1.0)
            axes[2, j].imshow(h.overlay)
            for i in range(3):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                if j == 0:
                    axes[i, j].set_ylabel(label_baris[i], fontsize=ukuran_font)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        return _simpan_fig(fig)
    finally:
        plt.close(fig)


def gradcam_delapan_contoh_png(
    daftar_contoh: Sequence[tuple[int, str, int, dict[str, HasilGradCAM]]],
    ukuran_font: int = 9,
) -> bytes:
    """Gambar PNG 8 contoh laporan berdampingan (8 baris x 4 kolom: Asli + 3 Overlay)."""
    n_baris = len(daftar_contoh)
    fig, axes = plt.subplots(n_baris, 4, figsize=(12.5, 2.7 * n_baris), dpi=150, squeeze=False)
    try:
        fig.suptitle(
            "Grad-CAM 8 Citra Contoh Laporan (Seed 42, Urutan Identik Notebook)",
            fontsize=ukuran_font + 3,
            fontweight="bold",
            color=WARNA_AKSEN,
            y=0.995,
        )
        for i, (indeks, berkas, label_asli, per_metode) in enumerate(daftar_contoh):
            citra_asli = next(iter(per_metode.values())).citra_asli
            axes[i, 0].imshow(citra_asli)
            axes[i, 0].set_title(
                f"#{i + 1} (idx {format_angka(indeks)}): {berkas}\nLabel asli: {KELAS[label_asli]}",
                fontsize=ukuran_font,
                fontweight="bold",
            )
            axes[i, 0].axis("off")
            for j, m in enumerate(METODE, start=1):
                h = per_metode[m]
                status = "Benar" if h.kelas_prediksi == label_asli else "Salah"
                axes[i, j].imshow(h.overlay)
                axes[i, j].set_title(
                    f"{NAMA_METODE[m]}\n"
                    f"Pred: {KELAS[h.kelas_prediksi]} ({format_metrik(h.probabilitas_prediksi)}) — {status}",
                    fontsize=ukuran_font,
                    color=WARNA_METODE[m],
                    fontweight="bold",
                )
                axes[i, j].axis("off")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        return _simpan_fig(fig)
    finally:
        plt.close(fig)


def kurva_konvergensi_png(
    trial: dict[str, pd.DataFrame], ukuran_font: int = 11
) -> bytes:
    """Gambar PNG kurva konvergensi (best_so_far) ketiga metode dalam satu grafik."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
    try:
        n_trial = len(next(iter(trial.values())))
        penanda = {"grid_search": "o", "random_search": "s", "optuna_tpe": "^"}
        for m in METODE:
            df = trial[m]
            ax.plot(
                df["trial"],
                df["best_so_far"],
                label=NAMA_METODE[m],
                color=WARNA_METODE[m],
                marker=penanda[m],
                markersize=4,
                linewidth=2,
            )
        ax.set_xlabel("Trial ke-n", fontsize=ukuran_font)
        ax.set_ylabel("F1-score macro validasi terbaik sejauh ini", fontsize=ukuran_font)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_metrik(v)))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=ukuran_font - 1)
        fig.suptitle(
            "Kurva Konvergensi Optimasi Hyperparameter",
            fontsize=ukuran_font + 2,
            fontweight="bold",
            color=WARNA_AKSEN,
            y=0.98,
        )
        ax.set_title(
            f"N = {format_angka(n_trial)} trial per metode, data validasi (artefak)",
            fontsize=ukuran_font - 1,
            color="#333333",
        )
        fig.tight_layout()
        return _simpan_fig(fig)
    finally:
        plt.close(fig)


def sebaran_f1_trial_png(
    trial: dict[str, pd.DataFrame], n_startup_optuna: int = 10, ukuran_font: int = 10
) -> bytes:
    """Gambar PNG sebaran F1-score macro validasi per trial untuk ketiga metode."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=150, sharey=True)
    try:
        n_trial = len(next(iter(trial.values())))
        for ax, m in zip(axes, METODE):
            df = trial[m]
            x = df["trial"].to_numpy()
            y = df["val_f1"].to_numpy()
            ax.scatter(
                x, y, s=26, alpha=0.75, color=WARNA_METODE[m], label="F1 validasi tiap trial"
            )
            ax.step(
                x,
                df["best_so_far"].to_numpy(),
                where="post",
                color=WARNA_AKSEN,
                linewidth=1.8,
                label="Terbaik sejauh ini",
            )
            if m == "optuna_tpe":
                ax.axvspan(0.5, n_startup_optuna + 0.5, color="#CCCCCC", alpha=0.25)
                ax.axvline(
                    n_startup_optuna + 0.5,
                    color="#666666",
                    linestyle="--",
                    linewidth=1,
                    label=f"Batas startup acak ({n_startup_optuna} trial)",
                )
            ax.set_title(NAMA_METODE[m], fontsize=ukuran_font + 1, fontweight="bold")
            ax.set_xlabel("Trial ke-n", fontsize=ukuran_font)
            ax.grid(alpha=0.3)
            ax.legend(loc="lower right", fontsize=ukuran_font - 2)
        axes[0].set_ylabel("F1-score macro validasi", fontsize=ukuran_font)
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_metrik(v)))
        fig.suptitle(
            f"Sebaran F1-score Validasi per Trial (N = {format_angka(n_trial)} trial per metode, data validasi)",
            fontsize=ukuran_font + 2,
            fontweight="bold",
            color=WARNA_AKSEN,
            y=1.01,
        )
        fig.tight_layout()
        return _simpan_fig(fig)
    finally:
        plt.close(fig)


def multi_seed_f1_png(
    multi_seed: pd.DataFrame, seeds: Sequence[int], ukuran_font: int = 11
) -> bytes:
    """Gambar PNG F1-score macro per seed per metode dari multi_seed_hasil.csv."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=150)
    try:
        label_seed = [str(s) for s in seeds]
        x = np.arange(len(seeds))
        lebar = 0.25
        for idx_m, m in enumerate(METODE):
            sub = multi_seed[multi_seed["method"] == m].set_index("seed")
            nilai = [float(sub.loc[int(s), "f1_macro"]) for s in seeds]
            ax.bar(
                x + (idx_m - 1) * lebar,
                nilai,
                width=lebar,
                label=NAMA_METODE[m],
                color=WARNA_METODE[m],
            )
        ax.set_xticks(x, labels=label_seed, fontsize=ukuran_font)
        ax.set_xlabel("Seed pelatihan ulang (TRAIN_SEED)", fontsize=ukuran_font)
        ax.set_ylabel("F1-score macro data uji", fontsize=ukuran_font)
        ax.set_ylim(0.85, 0.96)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_metrik(v)))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=ukuran_font - 1)
        fig.suptitle(
            "F1-score Macro per Seed pada Setiap Metode",
            fontsize=ukuran_font + 2,
            fontweight="bold",
            color=WARNA_AKSEN,
            y=0.98,
        )
        ax.set_title(
            f"N = {format_angka(len(seeds))} seed per metode, data uji (dihitung di notebook, artefak)",
            fontsize=ukuran_font - 1,
            color="#333333",
        )
        fig.tight_layout()
        return _simpan_fig(fig)
    finally:
        plt.close(fig)

