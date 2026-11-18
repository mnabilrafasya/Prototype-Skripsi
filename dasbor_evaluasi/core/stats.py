"""Uji McNemar exact dengan koreksi Holm (FR-08).

Mengikuti bagian McNemar di lampiran_kode_inti.ipynb (sel 12): tabel
[[sama-sama benar, hanya A benar], [hanya B benar, sama-sama salah]],
``statsmodels.stats.contingency_tables.mcnemar(table, exact=True)``, p = 1,0
bila jumlah diskordan nol, lalu ``multipletests(method='holm', alpha=0.05)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from core.konstanta import METODE, NAMA_METODE, PASANGAN

ALPHA = 0.05
CATATAN_TANPA_DISKORDAN = (
    "Jumlah sampel diskordan nol; p-value ditetapkan 1,0 (tidak ada perbedaan prediksi benar/salah)."
)


@dataclass(frozen=True)
class HasilMcNemar:
    """Hasil McNemar satu pasangan (sebelum koreksi Holm)."""

    n: int
    sama_benar: int
    hanya_a: int
    hanya_b: int
    sama_salah: int
    selisih_akurasi: float  # akurasi A - akurasi B
    p_value: float
    catatan: str  # kosong bila ada sampel diskordan


def uji_mcnemar(benar_a: Sequence[bool] | np.ndarray, benar_b: Sequence[bool] | np.ndarray) -> HasilMcNemar:
    """McNemar exact untuk dua vektor boolean 'prediksi benar' pada sampel yang sama."""
    a = np.asarray(benar_a, dtype=bool)
    b = np.asarray(benar_b, dtype=bool)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("benar_a dan benar_b harus vektor 1 dimensi tidak kosong dan sama panjang")
    sama_benar = int(np.sum(a & b))
    hanya_a = int(np.sum(a & ~b))
    hanya_b = int(np.sum(~a & b))
    sama_salah = int(np.sum(~a & ~b))
    tabel = [[sama_benar, hanya_a], [hanya_b, sama_salah]]
    if hanya_a + hanya_b > 0:
        p = float(mcnemar(tabel, exact=True).pvalue)
        catatan = ""
    else:
        p = 1.0
        catatan = CATATAN_TANPA_DISKORDAN
    return HasilMcNemar(
        n=len(a),
        sama_benar=sama_benar,
        hanya_a=hanya_a,
        hanya_b=hanya_b,
        sama_salah=sama_salah,
        selisih_akurasi=float(a.mean() - b.mean()),
        p_value=p,
        catatan=catatan,
    )


def mcnemar_tiga_pasangan(
    y_true: Sequence[int] | np.ndarray,
    prediksi: Mapping[str, Sequence[int] | np.ndarray],
) -> pd.DataFrame:
    """McNemar untuk tiga pasangan (PASANGAN) dengan koreksi Holm.

    ``prediksi`` harus memuat ketiga metode dengan panjang sama dengan ``y_true``.
    Kolom hasil memakai nama internal yang sama dengan ``muat_mcnemar`` ditambah
    ``n`` dan ``catatan``.
    """
    y = np.asarray(y_true)
    kurang = [m for m in METODE if m not in prediksi]
    if kurang:
        raise ValueError(f"prediksi metode berikut tidak ada: {kurang}")
    for m in METODE:
        if len(prediksi[m]) != len(y):
            raise ValueError(f"panjang prediksi '{m}' ({len(prediksi[m])}) berbeda dari label ({len(y)})")
    benar = {m: np.asarray(prediksi[m]) == y for m in METODE}
    hasil = [uji_mcnemar(benar[a], benar[b]) for a, b in PASANGAN]
    _, p_holm, _, _ = multipletests([h.p_value for h in hasil], alpha=ALPHA, method="holm")
    return pd.DataFrame(
        {
            "pasangan": [f"{NAMA_METODE[a]} vs {NAMA_METODE[b]}" for a, b in PASANGAN],
            "metode_a": [a for a, _ in PASANGAN],
            "metode_b": [b for _, b in PASANGAN],
            "n": [h.n for h in hasil],
            "sama_benar": [h.sama_benar for h in hasil],
            "hanya_a": [h.hanya_a for h in hasil],
            "hanya_b": [h.hanya_b for h in hasil],
            "sama_salah": [h.sama_salah for h in hasil],
            "selisih_akurasi": [h.selisih_akurasi for h in hasil],
            "p_value": [h.p_value for h in hasil],
            "p_holm": [float(p) for p in p_holm],
            "signifikan": ["YA" if p < ALPHA else "tidak" for p in p_holm],
            "catatan": [h.catatan for h in hasil],
        }
    )
