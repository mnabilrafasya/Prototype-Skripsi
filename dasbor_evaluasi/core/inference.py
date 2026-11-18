"""Inferensi batch deterministik di CPU (FR-05, FR-09)."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from core.data import DatasetUji
from core.konstanta import KELAS

UKURAN_BATCH_DIIZINKAN: tuple[int, ...] = (8, 16, 32)


@dataclass(frozen=True)
class HasilInferensi:
    """Hasil inferensi satu model pada urutan sampel yang diberikan."""

    prediksi: np.ndarray  # (n,) int64, argmax logit
    proba: np.ndarray  # (n, jumlah_kelas) float32, softmax logit
    n_selesai: int
    waktu_total_detik: float  # baca citra + praproses + forward
    waktu_forward_detik: float  # hanya forward model

    @property
    def proba_prediksi(self) -> np.ndarray:
        """Probabilitas kelas yang diprediksi, per sampel."""
        return self.proba[np.arange(len(self.prediksi)), self.prediksi]

    @property
    def throughput(self) -> float:
        """Citra per detik berdasarkan waktu total."""
        return self.n_selesai / self.waktu_total_detik if self.waktu_total_detik > 0 else 0.0


def jalankan_inferensi(
    model: nn.Module,
    jalur: Sequence[Path],
    transform: transforms.Compose,
    ukuran_batch: int,
    nama: Sequence[str] | None = None,
    kemajuan: Callable[[int, int], None] | None = None,
) -> HasilInferensi:
    """Jalankan model pada ``jalur`` sesuai urutannya (tanpa shuffle).

    ``kemajuan(selesai, total)`` dipanggil setelah tiap batch (antarmuka membatalkan
    proses dengan menghentikan eksekusi pada panggilan ini). Tensor citra tidak
    ditahan; hanya prediksi dan probabilitas.
    """
    if ukuran_batch not in UKURAN_BATCH_DIIZINKAN:
        raise ValueError(
            f"ukuran batch harus salah satu dari {UKURAN_BATCH_DIIZINKAN}, diterima {ukuran_batch}"
        )
    model.eval()
    muatan = DataLoader(
        DatasetUji(jalur, transform, nama),
        batch_size=ukuran_batch,
        shuffle=False,
        num_workers=0,
    )
    total = len(jalur)
    prediksi: list[np.ndarray] = []
    proba: list[np.ndarray] = []
    selesai = 0
    waktu_forward = 0.0
    awal = time.perf_counter()
    iterator = iter(muatan)
    with torch.no_grad():
        while selesai < total:
            batch = next(iterator)
            t0 = time.perf_counter()
            logit = model(batch)
            waktu_forward += time.perf_counter() - t0
            prediksi.append(logit.argmax(dim=1).numpy())
            proba.append(torch.softmax(logit, dim=1).numpy())
            selesai += batch.shape[0]
            if kemajuan is not None:
                kemajuan(selesai, total)
    waktu_total = time.perf_counter() - awal
    return HasilInferensi(
        prediksi=np.concatenate(prediksi).astype(np.int64) if prediksi else np.empty(0, np.int64),
        proba=(
            np.concatenate(proba).astype(np.float32)
            if proba
            else np.empty((0, len(KELAS)), np.float32)
        ),
        n_selesai=selesai,
        waktu_total_detik=waktu_total,
        waktu_forward_detik=waktu_forward,
    )
