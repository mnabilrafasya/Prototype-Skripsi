"""Hasil batch: struktur data, tabel prediksi (FR-06), dan berkas hasil_batch_penuh.csv.

Berkas CSV hasil batch berisi baris metadata berawalan ``# kunci: nilai`` diikuti
tabel prediksi. Format yang sama dipakai untuk unduhan (FR-14) dan untuk Mode
Presentasi (memuat hasil aplikasi sendiri, bukan manifest notebook).
"""

from __future__ import annotations

import io
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.data import DataUji
from core.konstanta import JUMLAH_DATA_UJI, KELAS, METODE

BERKAS_HASIL_BATCH = "hasil_batch_penuh.csv"
KOLOM_TETAP: tuple[str, ...] = ("berkas", "label_asli")
STATUS_BENAR = "benar"
STATUS_SALAH = "salah"
KUNCI_INFO_WAJIB: tuple[str, ...] = (
    "tanggal",
    "perangkat",
    "versi_python",
    "versi_torch",
    "versi_streamlit",
    "ukuran_batch",
    "sampel",
)


class HasilBatchError(Exception):
    """Galat berkas hasil batch; pesan menyebut berkas dan kolom."""


@dataclass(frozen=True)
class WaktuModel:
    """Waktu inferensi satu model pada batch."""

    n: int
    total_detik: float  # baca citra + praproses + forward
    forward_detik: float  # hanya forward model

    @property
    def throughput(self) -> float:
        """Citra per detik berdasarkan waktu total."""
        return self.n / self.total_detik if self.total_detik > 0 else 0.0


@dataclass(frozen=True)
class HasilBatch:
    """Hasil batch: sampel yang sama untuk semua model yang dijalankan."""

    indeks: np.ndarray  # indeks pada urutan manifest, (n,)
    metode: tuple[str, ...]  # urutan sama dengan METODE
    prediksi: dict[str, np.ndarray]  # (n,) int64
    proba_prediksi: dict[str, np.ndarray]  # (n,) float
    waktu: dict[str, WaktuModel]
    info: dict[str, str]

    @property
    def n(self) -> int:
        """Jumlah sampel."""
        return len(self.indeks)

    @property
    def semua_model(self) -> bool:
        """True bila ketiga model dijalankan pada sampel yang sama."""
        return self.metode == METODE

    @property
    def penuh(self) -> bool:
        """True bila seluruh data uji bawaan dalam urutan manifest."""
        return self.n == JUMLAH_DATA_UJI and bool(
            np.array_equal(self.indeks, np.arange(JUMLAH_DATA_UJI))
        )


def kumpulkan_info(ukuran_batch: int, deskripsi_sampel: str) -> dict[str, str]:
    """Metadata run: tanggal, perangkat, versi python/torch/streamlit, pengaturan."""
    try:
        versi_streamlit = importlib_metadata.version("streamlit")
    except importlib_metadata.PackageNotFoundError:
        versi_streamlit = "tidak terpasang"
    return {
        "tanggal": datetime.now().astimezone().isoformat(timespec="seconds"),
        "perangkat": (
            f"{platform.processor() or platform.machine()}; {platform.system()} "
            f"{platform.release()}; CPU; threads {torch.get_num_threads()}"
        ),
        "versi_python": sys.version.split()[0],
        "versi_torch": torch.__version__,
        "versi_streamlit": versi_streamlit,
        "ukuran_batch": str(ukuran_batch),
        "sampel": deskripsi_sampel,
    }


def tabel_prediksi(data_uji: DataUji, hasil: HasilBatch) -> pd.DataFrame:
    """Tabel per citra (FR-06) dengan nama kelas; kolom model mengikuti urutan METODE.

    Kolom: berkas, label_asli, lalu untuk tiap model prediksi_<m>,
    probabilitas_<m> (kelas prediksi), dan status_<m> (benar atau salah).
    """
    label = data_uji.label[hasil.indeks]
    kolom: dict[str, object] = {
        "berkas": [data_uji.path[int(i)] for i in hasil.indeks],
        "label_asli": [KELAS[int(k)] for k in label],
    }
    for m in hasil.metode:
        kolom[f"prediksi_{m}"] = [KELAS[int(k)] for k in hasil.prediksi[m]]
        kolom[f"probabilitas_{m}"] = hasil.proba_prediksi[m].astype(float)
        kolom[f"status_{m}"] = np.where(hasil.prediksi[m] == label, STATUS_BENAR, STATUS_SALAH)
    return pd.DataFrame(kolom)


def ke_csv(data_uji: DataUji, hasil: HasilBatch) -> str:
    """Serialisasi hasil batch: baris metadata '# kunci: nilai' lalu tabel CSV."""
    baris = [f"# {k}: {v}" for k, v in hasil.info.items()]
    baris.append(f"# metode: {','.join(hasil.metode)}")
    for m in hasil.metode:
        w = hasil.waktu[m]
        baris.append(f"# waktu.{m}: {w.n},{w.total_detik!r},{w.forward_detik!r}")
    tabel = tabel_prediksi(data_uji, hasil).to_csv(index=False, float_format="%.9g", lineterminator="\n")
    return "\n".join(baris) + "\n" + tabel


def simpan_csv(jalur: Path, data_uji: DataUji, hasil: HasilBatch) -> None:
    """Tulis hasil batch ke ``jalur`` (UTF-8)."""
    jalur.write_text(ke_csv(data_uji, hasil), encoding="utf-8", newline="\n")


def dari_csv(teks: str, data_uji: DataUji, nama: str = BERKAS_HASIL_BATCH) -> HasilBatch:
    """Baca dan validasi teks hasil batch terhadap data uji bawaan.

    Memeriksa metadata wajib, himpunan kolom, path ada di manifest dan unik,
    label asli sama dengan manifest, nama kelas, probabilitas pada [0, 1], dan
    status konsisten dengan prediksi.
    """
    info: dict[str, str] = {}
    for baris in teks.splitlines():
        if not baris.startswith("#"):
            break
        kunci, _, nilai = baris[1:].strip().partition(": ")
        info[kunci] = nilai
    hilang = [k for k in (*KUNCI_INFO_WAJIB, "metode") if k not in info]
    if hilang:
        raise HasilBatchError(f"{nama}: metadata wajib tidak ada: {', '.join(hilang)}")
    metode = tuple(info["metode"].split(","))
    if not metode or any(m not in METODE for m in metode) or metode != tuple(m for m in METODE if m in metode):
        raise HasilBatchError(f"{nama}: metadata 'metode' tidak valid: {info['metode']!r}")

    df = pd.read_csv(io.StringIO(teks), comment="#")
    wajib = list(KOLOM_TETAP)
    for m in metode:
        wajib += [f"prediksi_{m}", f"probabilitas_{m}", f"status_{m}"]
    kurang = [k for k in wajib if k not in df.columns]
    asing = [k for k in df.columns if k not in wajib]
    if kurang:
        raise HasilBatchError(f"{nama}: kolom wajib tidak ada: {', '.join(kurang)}")
    if asing:
        raise HasilBatchError(f"{nama}: kolom tidak dikenal: {', '.join(asing)}")
    if len(df) == 0:
        raise HasilBatchError(f"{nama}: tidak memuat baris data")
    if df["berkas"].duplicated().any():
        raise HasilBatchError(f"{nama}: kolom 'berkas' memuat duplikat")

    posisi = {p: i for i, p in enumerate(data_uji.path)}
    tidak_ada = [p for p in df["berkas"] if p not in posisi]
    if tidak_ada:
        raise HasilBatchError(
            f"{nama}: kolom 'berkas' memuat {len(tidak_ada)} path yang tidak ada di manifest, "
            f"contoh: {tidak_ada[0]}"
        )
    indeks = np.array([posisi[p] for p in df["berkas"]], dtype=np.int64)
    nomor = {k: i for i, k in enumerate(KELAS)}
    for kolom in ["label_asli", *(f"prediksi_{m}" for m in metode)]:
        asing_kelas = sorted(set(df[kolom]) - set(KELAS))
        if asing_kelas:
            raise HasilBatchError(f"{nama}: kolom '{kolom}' memuat kelas tidak dikenal: {asing_kelas}")
    label = df["label_asli"].map(nomor).to_numpy(dtype=np.int64)
    if not np.array_equal(label, data_uji.label[indeks]):
        raise HasilBatchError(f"{nama}: kolom 'label_asli' tidak sama dengan manifest")

    prediksi: dict[str, np.ndarray] = {}
    proba: dict[str, np.ndarray] = {}
    waktu: dict[str, WaktuModel] = {}
    for m in metode:
        prediksi[m] = df[f"prediksi_{m}"].map(nomor).to_numpy(dtype=np.int64)
        p = df[f"probabilitas_{m}"]
        if not pd.api.types.is_numeric_dtype(p) or p.isna().any() or ((p < 0) | (p > 1)).any():
            raise HasilBatchError(f"{nama}: kolom 'probabilitas_{m}' harus bilangan pada [0, 1]")
        proba[m] = p.to_numpy(dtype=np.float64)
        harapan = np.where(prediksi[m] == label, STATUS_BENAR, STATUS_SALAH)
        if not np.array_equal(df[f"status_{m}"].to_numpy(dtype=str), harapan):
            raise HasilBatchError(f"{nama}: kolom 'status_{m}' tidak konsisten dengan prediksi")
        kunci_waktu = f"waktu.{m}"
        try:
            n_w, total, forward = info[kunci_waktu].split(",")
            waktu[m] = WaktuModel(int(n_w), float(total), float(forward))
        except (KeyError, ValueError) as e:
            raise HasilBatchError(f"{nama}: metadata '{kunci_waktu}' tidak valid") from e
    return HasilBatch(
        indeks=indeks,
        metode=metode,
        prediksi=prediksi,
        proba_prediksi=proba,
        waktu=waktu,
        info={k: info[k] for k in KUNCI_INFO_WAJIB},
    )


def muat_csv(jalur: Path, data_uji: DataUji) -> HasilBatch:
    """Muat hasil batch dari berkas; galat menyebut nama berkas bila tidak ada."""
    if not jalur.is_file():
        raise HasilBatchError(f"{jalur.name}: berkas tidak ditemukan di {jalur.parent}")
    return dari_csv(jalur.read_text(encoding="utf-8"), data_uji, jalur.name)
