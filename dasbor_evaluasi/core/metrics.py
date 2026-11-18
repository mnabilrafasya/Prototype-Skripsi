"""Metrik klasifikasi dan perbandingan dengan nilai acuan (FR-07, FR-13).

Rumus memakai scikit-learn persis seperti ``evaluate_on_test`` di notebook:
``accuracy_score``, ``f1_score(average='macro')`` (tanpa argumen ``labels``),
``classification_report``, dan ``confusion_matrix``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from core.artifacts import HasilAcuan
from core.konstanta import (
    KELAS,
    MAKS_PREDIKSI_BERBEDA,
    TOLERANSI_SELISIH_AKURASI,
)

STATUS_SESUAI = "SESUAI"
STATUS_TIDAK_SESUAI = "TIDAK SESUAI"


@dataclass(frozen=True)
class MetrikModel:
    """Metrik satu model pada satu himpunan sampel."""

    n: int
    accuracy: float
    f1_macro: float
    per_kelas: pd.DataFrame  # indeks: kelas; kolom: presisi, recall, f1, dukungan
    confusion_matrix: np.ndarray  # (4, 4); baris: label asli, kolom: prediksi
    laporan: str  # classification_report teks


def hitung_metrik(
    y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray
) -> MetrikModel:
    """Hitung metrik; label tetap 0..3 sehingga matriks selalu 4x4.

    Catatan: ``f1_macro`` memakai ``f1_score(average='macro')`` seperti notebook,
    yaitu rata-rata atas kelas yang muncul pada y_true atau y_pred. Pada data uji
    penuh ini sama dengan rata-rata keempat kelas.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true dan y_pred harus tidak kosong dan sama panjang "
            f"({len(y_true)} dan {len(y_pred)})"
        )
    label = list(range(len(KELAS)))
    laporan_dict = classification_report(
        y_true,
        y_pred,
        labels=label,
        target_names=list(KELAS),
        output_dict=True,
        zero_division=0,
    )
    per_kelas = pd.DataFrame(
        {
            "presisi": [laporan_dict[k]["precision"] for k in KELAS],
            "recall": [laporan_dict[k]["recall"] for k in KELAS],
            "f1": [laporan_dict[k]["f1-score"] for k in KELAS],
            "dukungan": [int(laporan_dict[k]["support"]) for k in KELAS],
        },
        index=pd.Index(KELAS, name="kelas"),
    )
    return MetrikModel(
        n=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro")),
        per_kelas=per_kelas,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=label),
        laporan=classification_report(
            y_true,
            y_pred,
            labels=label,
            target_names=list(KELAS),
            zero_division=0,
        ),
    )


@dataclass(frozen=True)
class PerbandinganAcuan:
    """Hasil FR-13 untuk satu model."""

    selisih_akurasi: float  # aplikasi - acuan
    selisih_f1_macro: float  # aplikasi - acuan
    prediksi_berbeda: int  # terhadap test_manifest_prediksi.csv
    indeks_berbeda: tuple[int, ...]
    confusion_matrix_sama: bool
    status: str
    rincian: tuple[str, ...]  # alasan bila TIDAK SESUAI


def bandingkan_dengan_acuan(
    metrik: MetrikModel,
    acuan: HasilAcuan,
    prediksi: Sequence[int] | np.ndarray,
    prediksi_manifest: Sequence[int] | np.ndarray,
) -> PerbandinganAcuan:
    """Bandingkan metrik dan prediksi aplikasi dengan artefak (data uji penuh).

    SESUAI bila ``|selisih akurasi| <= TOLERANSI_SELISIH_AKURASI`` dan prediksi
    berbeda ``<= MAKS_PREDIKSI_BERBEDA``; selain itu TIDAK SESUAI beserta rincian.
    """
    prediksi = np.asarray(prediksi)
    manifest = np.asarray(prediksi_manifest)
    if len(prediksi) != len(manifest):
        raise ValueError(
            f"panjang prediksi aplikasi ({len(prediksi)}) berbeda dari manifest ({len(manifest)})"
        )
    berbeda = np.flatnonzero(prediksi != manifest)
    selisih_akurasi = metrik.accuracy - acuan.accuracy
    selisih_f1 = metrik.f1_macro - acuan.f1_macro
    rincian: list[str] = []
    if abs(selisih_akurasi) > TOLERANSI_SELISIH_AKURASI:
        rincian.append(
            f"selisih akurasi {selisih_akurasi:+.6f} melebihi ambang {TOLERANSI_SELISIH_AKURASI}"
        )
    if len(berbeda) > MAKS_PREDIKSI_BERBEDA:
        rincian.append(
            f"{len(berbeda)} prediksi berbeda dari manifest, melebihi ambang {MAKS_PREDIKSI_BERBEDA}"
        )
    return PerbandinganAcuan(
        selisih_akurasi=selisih_akurasi,
        selisih_f1_macro=selisih_f1,
        prediksi_berbeda=len(berbeda),
        indeks_berbeda=tuple(int(i) for i in berbeda),
        confusion_matrix_sama=bool(np.array_equal(metrik.confusion_matrix, acuan.confusion_matrix)),
        status=STATUS_SESUAI if not rincian else STATUS_TIDAK_SESUAI,
        rincian=tuple(rincian),
    )
