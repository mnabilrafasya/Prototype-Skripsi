"""Eksekusi batch beberapa model pada sampel yang sama (FR-05), tanpa streamlit."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
from torch import nn
from torchvision import transforms

from core.data import DataUji
from core.hasil import HasilBatch, WaktuModel, kumpulkan_info
from core.inference import jalankan_inferensi
from core.konstanta import METODE

# kemajuan(metode, selesai_semua, total_semua, detik_berjalan)
Kemajuan = Callable[[str, int, int, float], None]


def jalankan_batch(
    data_uji: DataUji,
    indeks: np.ndarray,
    metode: Sequence[str],
    ambil_model: Callable[[str], nn.Module],
    transform: transforms.Compose,
    ukuran_batch: int,
    deskripsi_sampel: str,
    kemajuan: Kemajuan | None = None,
) -> HasilBatch:
    """Jalankan tiap model pada ``indeks`` yang sama, berurutan menurut ``METODE``.

    Satu model diproses penuh sebelum model berikutnya. Hanya hasil lengkap yang
    dikembalikan, sehingga semua model selalu memakai sampel yang sama. Pembatalan
    dilakukan pemanggil dengan menghentikan eksekusi di dalam ``kemajuan``.
    """
    urut = tuple(m for m in METODE if m in metode)
    if not urut or len(urut) != len(set(metode)):
        raise ValueError(f"metode tidak valid: {list(metode)}; pilihan {METODE}")
    jalur = data_uji.jalur(indeks)
    nama = [data_uji.path[int(i)] for i in indeks]
    total_semua = len(jalur) * len(urut)
    mulai = time.perf_counter()
    prediksi: dict[str, np.ndarray] = {}
    proba: dict[str, np.ndarray] = {}
    waktu: dict[str, WaktuModel] = {}
    selesai_sebelumnya = 0
    for m in urut:
        def lapor(selesai: int, _total: int, m: str = m, dasar: int = selesai_sebelumnya) -> None:
            if kemajuan is not None:
                kemajuan(m, dasar + selesai, total_semua, time.perf_counter() - mulai)

        h = jalankan_inferensi(
            ambil_model(m), jalur, transform, ukuran_batch, nama, kemajuan=lapor
        )
        prediksi[m] = h.prediksi
        proba[m] = h.proba_prediksi
        waktu[m] = WaktuModel(h.n_selesai, h.waktu_total_detik, h.waktu_forward_detik)
        selesai_sebelumnya += h.n_selesai
    return HasilBatch(
        indeks=np.asarray(indeks, dtype=np.int64),
        metode=urut,
        prediksi=prediksi,
        proba_prediksi=proba,
        waktu=waktu,
        info=kumpulkan_info(ukuran_batch, deskripsi_sampel),
    )
