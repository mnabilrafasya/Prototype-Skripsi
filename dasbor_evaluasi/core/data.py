"""Data uji bawaan: validasi folder, pemilihan sampel (FR-03), dan praproses (FR-04).

Modul ini tidak mengimpor streamlit. Praproses identik dengan ``eval_transform``
di lampiran_kode_inti.ipynb (sel 3) dan pembacaan citra identik dengan
``BrainTumorDataset`` (sel 4): ``Image.open(path).convert('RGB')``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

from core.konstanta import (
    JUMLAH_CONTOH_LAPORAN,
    JUMLAH_DATA_UJI,
    KELAS,
    SEED_SAMPEL,
)

# Filter ekstensi yang sama dengan collect_paths di notebook (sel 2).
EKSTENSI_GAMBAR: tuple[str, ...] = (".jpg", ".jpeg", ".png")

MODE_SEMUA = "semua"
MODE_ACAK = "acak"
MODE_SEIMBANG = "seimbang"
MODE_SAMPEL: tuple[str, ...] = (MODE_SEMUA, MODE_ACAK, MODE_SEIMBANG)


class DataUjiError(Exception):
    """Galat data uji; pesan menyebut jumlah atau berkas yang bermasalah."""


@dataclass(frozen=True)
class DataUji:
    """Data uji bawaan dengan urutan sama dengan test_manifest_prediksi.csv."""

    folder: Path
    path: tuple[str, ...]  # relatif: kelas/nama_berkas
    label: np.ndarray  # int64, indeks kelas

    def __len__(self) -> int:
        return len(self.path)

    def jalur(self, indeks: Sequence[int] | np.ndarray) -> list[Path]:
        """Jalur absolut untuk indeks yang diberikan (urutan indeks dipertahankan)."""
        return [self.folder / self.path[int(i)] for i in indeks]


def muat_data_uji(folder: str | Path, manifest: pd.DataFrame) -> DataUji:
    """Validasi folder data_uji/ terhadap manifest lalu kembalikan ``DataUji``.

    Memeriksa: folder dan subfolder kelas ada, tidak ada berkas bukan gambar,
    jumlah citra tepat ``JUMLAH_DATA_UJI``, dan semua path manifest ada.
    Seluruh masalah dikumpulkan dalam satu pesan.
    """
    akar = Path(folder)
    if not akar.is_dir():
        raise DataUjiError(f"Folder data uji tidak ditemukan: {akar}")

    masalah: list[str] = []
    jumlah: dict[str, int] = {}
    bukan_gambar: list[str] = []
    for kelas in KELAS:
        sub = akar / kelas
        if not sub.is_dir():
            masalah.append(f"Subfolder kelas '{kelas}' tidak ditemukan di {akar}")
            jumlah[kelas] = 0
            continue
        berkas = [p for p in sub.iterdir() if p.is_file()]
        gambar = [p for p in berkas if p.suffix.lower() in EKSTENSI_GAMBAR]
        bukan_gambar += [f"{kelas}/{p.name}" for p in berkas if p not in gambar]
        jumlah[kelas] = len(gambar)

    if bukan_gambar:
        contoh = ", ".join(sorted(bukan_gambar)[:5])
        masalah.append(
            f"Ditemukan {len(bukan_gambar)} berkas bukan gambar "
            f"(ekstensi selain {', '.join(EKSTENSI_GAMBAR)}), contoh: {contoh}"
        )
    total = sum(jumlah.values())
    if total != JUMLAH_DATA_UJI:
        rincian = ", ".join(f"{k}: {v}" for k, v in jumlah.items())
        masalah.append(
            f"Jumlah citra ditemukan {total}, seharusnya {JUMLAH_DATA_UJI} ({rincian})"
        )

    path = tuple(manifest["path"].astype(str))
    hilang = [p for p in path if not (akar / p).is_file()]
    if hilang:
        masalah.append(
            f"{len(hilang)} berkas pada test_manifest_prediksi.csv tidak ada di "
            f"{akar}, contoh: {', '.join(hilang[:5])}"
        )
    if masalah:
        raise DataUjiError("\n".join(masalah))

    return DataUji(
        folder=akar,
        path=path,
        label=manifest["label"].to_numpy(dtype=np.int64),
    )


def pilih_indeks(
    label: np.ndarray,
    mode: str,
    n: int | None = None,
    seed: int = SEED_SAMPEL,
) -> np.ndarray:
    """Pilih indeks sampel (FR-03); hasil selalu terurut menurut urutan manifest.

    - ``semua``: seluruh indeks.
    - ``acak``: ``default_rng(seed).choice(N, n, replace=False)``, diurutkan.
    - ``seimbang``: ``n`` per kelas; tiap kelas diacak dengan ``default_rng(seed)``
      baru (seed sama per kelas), lalu seluruh indeks diurutkan.
    """
    jumlah = len(label)
    if mode == MODE_SEMUA:
        return np.arange(jumlah, dtype=np.int64)
    if mode == MODE_ACAK:
        if n is None or not 1 <= n <= jumlah:
            raise ValueError(f"n untuk mode acak harus 1..{jumlah}, diterima {n!r}")
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(jumlah, size=n, replace=False)).astype(np.int64)
    if mode == MODE_SEIMBANG:
        per_kelas = [np.flatnonzero(label == k) for k in range(len(KELAS))]
        maksimum = min(len(p) for p in per_kelas)
        if n is None or not 1 <= n <= maksimum:
            raise ValueError(f"n per kelas harus 1..{maksimum}, diterima {n!r}")
        terpilih = []
        for indeks_kelas in per_kelas:
            rng = np.random.default_rng(seed)
            terpilih.append(rng.choice(indeks_kelas, size=n, replace=False))
        return np.sort(np.concatenate(terpilih)).astype(np.int64)
    raise ValueError(f"mode sampel tidak dikenal: {mode!r}; pilihan {MODE_SAMPEL}")


def indeks_contoh_laporan(
    jumlah_total: int = JUMLAH_DATA_UJI, n: int = JUMLAH_CONTOH_LAPORAN
) -> np.ndarray:
    """Delapan contoh laporan, urutan persis keluaran ``default_rng(42).choice``.

    Sama dengan ``visualize_gradcam_grid`` di notebook (sel 14); TIDAK diurutkan.
    """
    return np.random.default_rng(SEED_SAMPEL).choice(jumlah_total, size=n, replace=False)


def buat_transform(
    ukuran: int, mean: Sequence[float], std: Sequence[float]
) -> transforms.Compose:
    """Praproses inferensi: Resize, ToTensor, Normalize (tanpa augmentasi)."""
    return transforms.Compose(
        [
            transforms.Resize((ukuran, ukuran)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def baca_citra(jalur: Path, nama: str | None = None) -> Image.Image:
    """Baca citra sebagai RGB; galat menyebut nama berkas."""
    try:
        return Image.open(jalur).convert("RGB")
    except (OSError, UnidentifiedImageError) as e:
        raise DataUjiError(f"Berkas bukan gambar yang valid: {nama or jalur} ({e})") from e


class DatasetUji(Dataset):
    """Dataset PyTorch berurutan untuk inferensi; mengembalikan tensor citra."""

    def __init__(
        self,
        jalur: Sequence[Path],
        transform: transforms.Compose,
        nama: Sequence[str] | None = None,
    ) -> None:
        self.jalur = list(jalur)
        self.transform = transform
        self.nama = list(nama) if nama is not None else [p.name for p in self.jalur]

    def __len__(self) -> int:
        return len(self.jalur)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.transform(baca_citra(self.jalur[i], self.nama[i]))
