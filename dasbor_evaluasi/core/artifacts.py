"""Pemuatan dan validasi artefak penelitian (KONTRAK ARTEFAK).

Modul ini tidak mengimpor streamlit. Semua berkas dibaca apa adanya (isi dan
urutan baris tidak diubah). Kolom atau kunci yang hilang, tidak dikenal, atau
bernilai tidak valid menghasilkan ``ArtefakError`` yang menyebut nama berkas
dan kolom/kuncinya. ``muat_artefak`` mengumpulkan semua galat sekaligus.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.konstanta import (
    JUMLAH_DATA_UJI,
    KELAS,
    MEAN_IMAGENET,
    METODE,
    NAMA_METODE,
    PASANGAN,
    STD_IMAGENET,
    UKURAN_CITRA,
)

# Kunci hyperparameter yang dibaca build_model (notebook, sel 5).
KUNCI_KONFIG_MODEL: tuple[str, ...] = ("dropout_rate", "dense_units")

BERKAS_META = "meta.json"
BERKAS_WAKTU = "waktu_pencarian.csv"
BERKAS_ACUAN = "hasil_uji_acuan.json"
BERKAS_MANIFEST = "test_manifest_prediksi.csv"
BERKAS_MCNEMAR = "mcnemar_acuan.csv"
BERKAS_MULTI_SEED = "multi_seed_hasil.csv"
BERKAS_WILCOXON = "wilcoxon.csv"
BERKAS_BOBOT: dict[str, str] = {m: f"bobot/{m}.pth" for m in METODE}
BERKAS_TRIAL: dict[str, str] = {m: f"trial_{m}.csv" for m in METODE}

# Seluruh berkas dalam kontrak: 3 bobot + meta + 3 trial + 6 berkas lain = 13.
BERKAS_WAJIB: tuple[str, ...] = (
    *BERKAS_BOBOT.values(),
    BERKAS_META,
    *BERKAS_TRIAL.values(),
    BERKAS_WAKTU,
    BERKAS_ACUAN,
    BERKAS_MANIFEST,
    BERKAS_MCNEMAR,
    BERKAS_MULTI_SEED,
    BERKAS_WILCOXON,
)

KOLOM_TRIAL_TETAP: tuple[str, ...] = (
    "trial",
    "val_acc",
    "val_f1",
    "epochs_trained",
    "training_time_sec",
    "best_so_far",
)
AWALAN_KONFIG = "config."
KOLOM_WAKTU: tuple[str, ...] = ("method", "n_trials", "total_time_sec")
KOLOM_MANIFEST: tuple[str, ...] = (
    "path",
    "label",
    *(f"pred_{m}" for m in METODE),
)
KOLOM_MULTI_SEED: tuple[str, ...] = ("method", "seed", "accuracy", "f1_macro")

# Nama kolom pada berkas -> nama internal.
KOLOM_MCNEMAR: dict[str, str] = {
    "Pasangan": "pasangan",
    "Sama-sama benar": "sama_benar",
    "A benar, B salah": "hanya_a",
    "A salah, B benar": "hanya_b",
    "Sama-sama salah": "sama_salah",
    "Selisih akurasi (A-B)": "selisih_akurasi",
    "p-value": "p_value",
    "p-value (Holm)": "p_holm",
    "Signifikan (alpha=0.05)": "signifikan",
}
KOLOM_WILCOXON: dict[str, str] = {
    "Pasangan": "pasangan",
    "A menang / B menang / seri": "menang_kalah_seri",
    "Rata-rata selisih F1 (A-B)": "rata_selisih_f1",
    "p-value": "p_value",
    "p-value (Holm)": "p_holm",
    "Signifikan (alpha=0.05)": "signifikan",
}
NILAI_SIGNIFIKAN = ("YA", "tidak")
TOLERANSI_P_VALUE = 5e-5  # acuan dibulatkan 4 desimal; dipakai tes McNemar (Tahap 2)

_POLA_MENANG = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*$")
_BALIK_NAMA = {v: k for k, v in NAMA_METODE.items()}


class ArtefakError(Exception):
    """Galat satu berkas artefak; pesan memuat nama berkas dan kolom/kunci."""

    def __init__(
        self,
        berkas: str,
        rincian: str,
        kolom: str | None = None,
        kunci: str | None = None,
    ) -> None:
        self.berkas = berkas
        self.kolom = kolom
        self.kunci = kunci
        self.rincian = rincian
        lokasi = ""
        if kolom is not None:
            lokasi = f" (kolom: {kolom})"
        elif kunci is not None:
            lokasi = f" (kunci: {kunci})"
        super().__init__(f"{berkas}{lokasi}: {rincian}")


class ArtefakGagal(Exception):
    """Kumpulan galat dari seluruh berkas artefak yang tidak valid."""

    def __init__(self, galat: Sequence[ArtefakError]) -> None:
        self.galat = list(galat)
        super().__init__(
            "Artefak tidak valid:\n" + "\n".join(f"- {g}" for g in self.galat)
        )


@dataclass(frozen=True)
class Meta:
    """Isi meta.json yang sudah divalidasi."""

    classes: tuple[str, ...]
    img_size: int
    imagenet_mean: tuple[float, ...]
    imagenet_std: tuple[float, ...]
    best_configs: dict[str, dict[str, int | float | str]]
    split_seed: int
    train_seed: int
    seeds_used: tuple[int, ...]


@dataclass(frozen=True)
class HasilAcuan:
    """Nilai acuan notebook untuk satu metode (data uji penuh, seed 42)."""

    accuracy: float
    f1_macro: float
    f1_per_class: dict[str, float]
    confusion_matrix: np.ndarray


@dataclass(frozen=True)
class Artefak:
    """Seluruh artefak yang sudah divalidasi."""

    root: Path
    meta: Meta
    bobot: dict[str, Path]
    trial: dict[str, pd.DataFrame]
    waktu: pd.DataFrame
    acuan: dict[str, HasilAcuan]
    manifest: pd.DataFrame
    mcnemar: pd.DataFrame
    multi_seed: pd.DataFrame
    wilcoxon: pd.DataFrame


# ---------------------------------------------------------------- pembantu


def _daftar(nama: Sequence[str]) -> str:
    return ", ".join(f"'{n}'" for n in nama)


def _baca_csv(root: Path, nama: str) -> pd.DataFrame:
    jalur = root / nama
    if not jalur.is_file():
        raise ArtefakError(nama, "berkas tidak ditemukan")
    try:
        df = pd.read_csv(jalur)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise ArtefakError(nama, f"tidak dapat dibaca sebagai CSV ({e})") from e
    if len(df) == 0:
        raise ArtefakError(nama, "tidak memuat baris data")
    return df


def _baca_json(root: Path, nama: str) -> Any:
    jalur = root / nama
    if not jalur.is_file():
        raise ArtefakError(nama, "berkas tidak ditemukan")
    try:
        return json.loads(jalur.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ArtefakError(nama, f"tidak dapat dibaca sebagai JSON ({e})") from e


def _periksa_kolom(
    nama: str,
    df: pd.DataFrame,
    wajib: Sequence[str],
    awalan_diizinkan: str | None = None,
) -> None:
    """Pastikan himpunan kolom sama dengan ``wajib`` (urutan tidak dipersoalkan)."""
    ada = list(df.columns)
    hilang = [k for k in wajib if k not in ada]
    if hilang:
        raise ArtefakError(nama, "kolom wajib tidak ada", kolom=_daftar(hilang))
    asing = [
        k
        for k in ada
        if k not in wajib
        and not (awalan_diizinkan is not None and k.startswith(awalan_diizinkan))
    ]
    if asing:
        raise ArtefakError(nama, "kolom tidak dikenal", kolom=_daftar(asing))


def _periksa_angka(
    nama: str,
    df: pd.DataFrame,
    kolom: str,
    *,
    bulat: bool = False,
    minimum: float | None = None,
    maksimum: float | None = None,
) -> None:
    s = df[kolom]
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        raise ArtefakError(nama, "nilai harus numerik", kolom=kolom)
    nilai = s.to_numpy(dtype=float)
    if not np.isfinite(nilai).all():
        baris = int(np.flatnonzero(~np.isfinite(nilai))[0]) + 2
        raise ArtefakError(
            nama, f"nilai kosong atau tidak hingga di baris berkas ke-{baris}", kolom=kolom
        )
    if bulat and not np.array_equal(nilai, np.round(nilai)):
        raise ArtefakError(nama, "nilai harus bilangan bulat", kolom=kolom)
    if minimum is not None and (nilai < minimum).any():
        raise ArtefakError(nama, f"ada nilai di bawah {minimum}", kolom=kolom)
    if maksimum is not None and (nilai > maksimum).any():
        raise ArtefakError(nama, f"ada nilai di atas {maksimum}", kolom=kolom)


def _periksa_teks(nama: str, df: pd.DataFrame, kolom: str) -> None:
    s = df[kolom]
    if s.isna().any() or not s.map(lambda v: isinstance(v, str) and v != "").all():
        raise ArtefakError(nama, "nilai teks tidak boleh kosong", kolom=kolom)


def _adalah_bulat(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _adalah_real(v: Any) -> bool:
    return isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(float(v))


def _periksa_kunci(nama: str, data: Any, harapan: Sequence[str], jalur: str = "") -> dict:
    """Pastikan ``data`` adalah objek JSON dengan himpunan kunci persis ``harapan``."""
    if not isinstance(data, dict):
        raise ArtefakError(nama, "harus berupa objek JSON", kunci=jalur or "(akar)")
    hilang = [k for k in harapan if k not in data]
    if hilang:
        awal = f"{jalur}." if jalur else ""
        raise ArtefakError(
            nama, "kunci wajib tidak ada", kunci=_daftar([awal + k for k in hilang])
        )
    asing = [k for k in data if k not in harapan]
    if asing:
        awal = f"{jalur}." if jalur else ""
        raise ArtefakError(
            nama, "kunci tidak dikenal", kunci=_daftar([awal + k for k in asing])
        )
    return data


def _pecah_pasangan(nama: str, nilai: str, nama_ke_kunci: dict[str, str]) -> tuple[str, str]:
    bagian = nilai.split(" vs ")
    if len(bagian) != 2 or any(b not in nama_ke_kunci for b in bagian):
        raise ArtefakError(nama, f"nama pasangan tidak dikenal: '{nilai}'", kolom="Pasangan")
    return nama_ke_kunci[bagian[0]], nama_ke_kunci[bagian[1]]


def _periksa_pasangan(
    nama: str, pasangan: Sequence[tuple[str, str]]
) -> None:
    if sorted(pasangan) != sorted(PASANGAN):
        raise ArtefakError(
            nama,
            "himpunan pasangan harus tepat tiga: "
            + _daftar([f"{a} vs {b}" for a, b in PASANGAN]),
            kolom="Pasangan",
        )


# ------------------------------------------------------------------ pemuat


def muat_meta(root: Path) -> Meta:
    """Muat dan validasi meta.json."""
    n = BERKAS_META
    data = _periksa_kunci(
        n,
        _baca_json(root, n),
        (
            "classes",
            "img_size",
            "imagenet_mean",
            "imagenet_std",
            "best_configs",
            "split_seed",
            "train_seed",
            "seeds_used",
        ),
    )
    if data["classes"] != list(KELAS):
        raise ArtefakError(
            n, f"urutan kelas harus {list(KELAS)}, ditemukan {data['classes']!r}", kunci="classes"
        )
    if not _adalah_bulat(data["img_size"]) or data["img_size"] != UKURAN_CITRA:
        raise ArtefakError(
            n, f"harus {UKURAN_CITRA}, ditemukan {data['img_size']!r}", kunci="img_size"
        )
    for kunci, acuan in (("imagenet_mean", MEAN_IMAGENET), ("imagenet_std", STD_IMAGENET)):
        v = data[kunci]
        if (
            not isinstance(v, list)
            or len(v) != len(acuan)
            or not all(_adalah_real(x) for x in v)
            or not all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(v, acuan))
        ):
            raise ArtefakError(
                n, f"harus {list(acuan)}, ditemukan {v!r}", kunci=kunci
            )
    for kunci in ("split_seed", "train_seed"):
        if not _adalah_bulat(data[kunci]):
            raise ArtefakError(n, "harus bilangan bulat", kunci=kunci)
    seeds = data["seeds_used"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(_adalah_bulat(s) for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ArtefakError(
            n, "harus daftar bilangan bulat tidak kosong tanpa duplikat", kunci="seeds_used"
        )

    konfigs = _periksa_kunci(n, data["best_configs"], METODE, "best_configs")
    himpunan_kunci: set[str] | None = None
    hasil_konfig: dict[str, dict[str, int | float | str]] = {}
    for metode in METODE:
        jalur = f"best_configs.{metode}"
        k = konfigs[metode]
        if not isinstance(k, dict) or not k:
            raise ArtefakError(n, "harus objek tidak kosong", kunci=jalur)
        kurang = [x for x in KUNCI_KONFIG_MODEL if x not in k]
        if kurang:
            raise ArtefakError(
                n,
                "kunci yang dibaca build_model tidak ada",
                kunci=_daftar([f"{jalur}.{x}" for x in kurang]),
            )
        for nama_kunci, nilai in k.items():
            if isinstance(nilai, bool) or not isinstance(nilai, (int, float, str)):
                raise ArtefakError(
                    n, "nilai harus angka atau teks", kunci=f"{jalur}.{nama_kunci}"
                )
            if isinstance(nilai, float) and not math.isfinite(nilai):
                raise ArtefakError(n, "nilai tidak hingga", kunci=f"{jalur}.{nama_kunci}")
        p = k["dropout_rate"]
        if not _adalah_real(p) or not 0.0 <= p < 1.0:
            raise ArtefakError(
                n, "harus bilangan pada rentang [0, 1)", kunci=f"{jalur}.dropout_rate"
            )
        if not _adalah_bulat(k["dense_units"]) or k["dense_units"] < 1:
            raise ArtefakError(
                n, "harus bilangan bulat positif", kunci=f"{jalur}.dense_units"
            )
        if himpunan_kunci is None:
            himpunan_kunci = set(k)
        elif set(k) != himpunan_kunci:
            raise ArtefakError(
                n,
                "himpunan kunci konfigurasi berbeda antar metode",
                kunci=_daftar(sorted(set(k) ^ himpunan_kunci)),
            )
        hasil_konfig[metode] = dict(k)

    return Meta(
        classes=tuple(data["classes"]),
        img_size=data["img_size"],
        imagenet_mean=tuple(float(x) for x in data["imagenet_mean"]),
        imagenet_std=tuple(float(x) for x in data["imagenet_std"]),
        best_configs=hasil_konfig,
        split_seed=data["split_seed"],
        train_seed=data["train_seed"],
        seeds_used=tuple(seeds),
    )


def muat_bobot(root: Path, metode: str) -> Path:
    """Pastikan berkas bobot ada dan tidak kosong; isi tidak dimuat di sini."""
    nama = BERKAS_BOBOT[metode]
    jalur = root / nama
    if not jalur.is_file():
        raise ArtefakError(nama, "berkas tidak ditemukan")
    if jalur.stat().st_size == 0:
        raise ArtefakError(nama, "berkas kosong (0 byte)")
    return jalur


def muat_trial(
    root: Path, metode: str, kunci_konfig: Sequence[str] | None = None
) -> pd.DataFrame:
    """Muat trial_<metode>.csv.

    Bila ``kunci_konfig`` diberikan (dari meta.json), kolom konfigurasi harus
    persis ``config.<kunci>`` untuk tiap kunci; bila tidak, awalan ``config.``
    cukup.
    """
    nama = BERKAS_TRIAL[metode]
    df = _baca_csv(root, nama)
    if kunci_konfig is None:
        _periksa_kolom(nama, df, KOLOM_TRIAL_TETAP, AWALAN_KONFIG)
    else:
        wajib = (*KOLOM_TRIAL_TETAP, *(AWALAN_KONFIG + k for k in kunci_konfig))
        _periksa_kolom(nama, df, wajib)
    _periksa_angka(nama, df, "trial", bulat=True, minimum=1)
    _periksa_angka(nama, df, "val_acc", minimum=0, maksimum=1)
    _periksa_angka(nama, df, "val_f1", minimum=0, maksimum=1)
    _periksa_angka(nama, df, "epochs_trained", bulat=True, minimum=1)
    _periksa_angka(nama, df, "training_time_sec", minimum=0)
    _periksa_angka(nama, df, "best_so_far", minimum=0, maksimum=1)
    for kolom in df.columns:
        if kolom.startswith(AWALAN_KONFIG) and df[kolom].isna().any():
            raise ArtefakError(nama, "nilai konfigurasi kosong", kolom=kolom)
    if not np.array_equal(df["trial"].to_numpy(), np.arange(1, len(df) + 1)):
        raise ArtefakError(
            nama, "nomor trial harus 1..n berurutan tanpa duplikat", kolom="trial"
        )
    kumulatif = np.maximum.accumulate(df["val_f1"].to_numpy(dtype=float))
    if not np.allclose(df["best_so_far"].to_numpy(dtype=float), kumulatif, rtol=0, atol=1e-9):
        raise ArtefakError(
            nama, "tidak sama dengan nilai maksimum kumulatif val_f1", kolom="best_so_far"
        )
    return df


def muat_waktu(root: Path) -> pd.DataFrame:
    """Muat waktu_pencarian.csv."""
    n = BERKAS_WAKTU
    df = _baca_csv(root, n)
    _periksa_kolom(n, df, KOLOM_WAKTU)
    _periksa_teks(n, df, "method")
    if sorted(df["method"]) != sorted(METODE):
        raise ArtefakError(
            n, f"harus memuat tepat satu baris per metode: {_daftar(METODE)}", kolom="method"
        )
    _periksa_angka(n, df, "n_trials", bulat=True, minimum=1)
    _periksa_angka(n, df, "total_time_sec", minimum=0)
    return df


def muat_acuan(root: Path) -> dict[str, HasilAcuan]:
    """Muat hasil_uji_acuan.json."""
    n = BERKAS_ACUAN
    data = _periksa_kunci(n, _baca_json(root, n), METODE)
    hasil: dict[str, HasilAcuan] = {}
    for metode in METODE:
        d = _periksa_kunci(
            n,
            data[metode],
            ("accuracy", "f1_macro", "f1_per_class", "confusion_matrix"),
            metode,
        )
        for kunci in ("accuracy", "f1_macro"):
            v = d[kunci]
            if not _adalah_real(v) or not 0.0 <= v <= 1.0:
                raise ArtefakError(
                    n, "harus bilangan pada rentang [0, 1]", kunci=f"{metode}.{kunci}"
                )
        f1k = _periksa_kunci(n, d["f1_per_class"], KELAS, f"{metode}.f1_per_class")
        for kelas in KELAS:
            if not _adalah_real(f1k[kelas]) or not 0.0 <= f1k[kelas] <= 1.0:
                raise ArtefakError(
                    n,
                    "harus bilangan pada rentang [0, 1]",
                    kunci=f"{metode}.f1_per_class.{kelas}",
                )
        cm = d["confusion_matrix"]
        ok = (
            isinstance(cm, list)
            and len(cm) == len(KELAS)
            and all(
                isinstance(b, list)
                and len(b) == len(KELAS)
                and all(_adalah_bulat(x) and x >= 0 for x in b)
                for b in cm
            )
        )
        if not ok:
            raise ArtefakError(
                n,
                f"harus matriks {len(KELAS)}x{len(KELAS)} bilangan bulat tidak negatif",
                kunci=f"{metode}.confusion_matrix",
            )
        matriks = np.array(cm, dtype=np.int64)
        if matriks.sum() == 0 or not math.isclose(
            float(np.trace(matriks)) / float(matriks.sum()), float(d["accuracy"]), abs_tol=1e-9
        ):
            raise ArtefakError(
                n,
                "accuracy tidak konsisten dengan confusion_matrix (trace/total)",
                kunci=f"{metode}.accuracy",
            )
        hasil[metode] = HasilAcuan(
            accuracy=float(d["accuracy"]),
            f1_macro=float(d["f1_macro"]),
            f1_per_class={k: float(f1k[k]) for k in KELAS},
            confusion_matrix=matriks,
        )
    return hasil


def muat_manifest(root: Path) -> pd.DataFrame:
    """Muat test_manifest_prediksi.csv; urutan baris dipertahankan."""
    n = BERKAS_MANIFEST
    df = _baca_csv(root, n)
    _periksa_kolom(n, df, KOLOM_MANIFEST)
    _periksa_teks(n, df, "path")
    for kolom in KOLOM_MANIFEST[1:]:
        _periksa_angka(n, df, kolom, bulat=True, minimum=0, maksimum=len(KELAS) - 1)
    if len(df) != JUMLAH_DATA_UJI:
        raise ArtefakError(
            n, f"harus {JUMLAH_DATA_UJI} baris, ditemukan {len(df)}", kolom="path"
        )
    if df["path"].duplicated().any():
        raise ArtefakError(n, "ada path duplikat", kolom="path")
    for i, (jalur, label) in enumerate(zip(df["path"], df["label"])):
        bagian = jalur.split("/")
        if len(bagian) != 2 or not all(bagian):
            raise ArtefakError(
                n, f"baris data ke-{i + 1}: path harus 'kelas/nama_berkas', ditemukan '{jalur}'",
                kolom="path",
            )
        if bagian[0] != KELAS[int(label)]:
            raise ArtefakError(
                n,
                f"baris data ke-{i + 1}: folder '{bagian[0]}' tidak sesuai label {int(label)} "
                f"('{KELAS[int(label)]}')",
                kolom="label",
            )
    return df


def muat_mcnemar(root: Path) -> pd.DataFrame:
    """Muat mcnemar_acuan.csv; kolom diganti ke nama internal, baris tetap."""
    n = BERKAS_MCNEMAR
    df = _baca_csv(root, n)
    _periksa_kolom(n, df, tuple(KOLOM_MCNEMAR))
    _periksa_teks(n, df, "Pasangan")
    pasangan = [_pecah_pasangan(n, p, _BALIK_NAMA) for p in df["Pasangan"]]
    _periksa_pasangan(n, pasangan)
    for kolom in ("Sama-sama benar", "A benar, B salah", "A salah, B benar", "Sama-sama salah"):
        _periksa_angka(n, df, kolom, bulat=True, minimum=0)
    _periksa_angka(n, df, "Selisih akurasi (A-B)", minimum=-1, maksimum=1)
    _periksa_angka(n, df, "p-value", minimum=0, maksimum=1)
    _periksa_angka(n, df, "p-value (Holm)", minimum=0, maksimum=1)
    if not df["Signifikan (alpha=0.05)"].isin(NILAI_SIGNIFIKAN).all():
        raise ArtefakError(
            n, f"nilai harus salah satu dari {_daftar(NILAI_SIGNIFIKAN)}",
            kolom="Signifikan (alpha=0.05)",
        )
    hasil = df.rename(columns=KOLOM_MCNEMAR)
    hasil.insert(1, "metode_a", [a for a, _ in pasangan])
    hasil.insert(2, "metode_b", [b for _, b in pasangan])
    return hasil


def muat_wilcoxon(root: Path) -> pd.DataFrame:
    """Muat wilcoxon.csv; kolom ke nama internal, 'A menang / B menang / seri' dipecah."""
    n = BERKAS_WILCOXON
    df = _baca_csv(root, n)
    _periksa_kolom(n, df, tuple(KOLOM_WILCOXON))
    _periksa_teks(n, df, "Pasangan")
    kunci_sendiri = {m: m for m in METODE}
    pasangan = [_pecah_pasangan(n, p, kunci_sendiri) for p in df["Pasangan"]]
    _periksa_pasangan(n, pasangan)
    _periksa_teks(n, df, "A menang / B menang / seri")
    tiga = []
    for v in df["A menang / B menang / seri"]:
        m = _POLA_MENANG.match(v)
        if not m:
            raise ArtefakError(
                n, f"format harus 'a / b / c' bilangan bulat, ditemukan '{v}'",
                kolom="A menang / B menang / seri",
            )
        tiga.append(tuple(int(x) for x in m.groups()))
    _periksa_angka(n, df, "Rata-rata selisih F1 (A-B)", minimum=-1, maksimum=1)
    _periksa_angka(n, df, "p-value", minimum=0, maksimum=1)
    _periksa_angka(n, df, "p-value (Holm)", minimum=0, maksimum=1)
    if not df["Signifikan (alpha=0.05)"].isin(NILAI_SIGNIFIKAN).all():
        raise ArtefakError(
            n, f"nilai harus salah satu dari {_daftar(NILAI_SIGNIFIKAN)}",
            kolom="Signifikan (alpha=0.05)",
        )
    hasil = df.rename(columns=KOLOM_WILCOXON)
    hasil.insert(1, "metode_a", [a for a, _ in pasangan])
    hasil.insert(2, "metode_b", [b for _, b in pasangan])
    hasil["a_menang"] = [t[0] for t in tiga]
    hasil["b_menang"] = [t[1] for t in tiga]
    hasil["seri"] = [t[2] for t in tiga]
    return hasil.drop(columns="menang_kalah_seri")


def muat_multi_seed(root: Path, seeds_meta: Sequence[int] | None = None) -> pd.DataFrame:
    """Muat multi_seed_hasil.csv; bila ``seeds_meta`` ada, seed tiap metode harus sama."""
    n = BERKAS_MULTI_SEED
    df = _baca_csv(root, n)
    _periksa_kolom(n, df, KOLOM_MULTI_SEED)
    _periksa_teks(n, df, "method")
    asing = sorted(set(df["method"]) - set(METODE))
    if asing:
        raise ArtefakError(n, f"metode tidak dikenal: {_daftar(asing)}", kolom="method")
    _periksa_angka(n, df, "seed", bulat=True)
    _periksa_angka(n, df, "accuracy", minimum=0, maksimum=1)
    _periksa_angka(n, df, "f1_macro", minimum=0, maksimum=1)
    if df.duplicated(subset=["method", "seed"]).any():
        raise ArtefakError(n, "ada pasangan (method, seed) duplikat", kolom="seed")
    for metode in METODE:
        seeds = set(df.loc[df["method"] == metode, "seed"].astype(int))
        if not seeds:
            raise ArtefakError(n, f"tidak ada baris untuk metode '{metode}'", kolom="method")
        if seeds_meta is not None and seeds != set(seeds_meta):
            raise ArtefakError(
                n,
                f"seed metode '{metode}' tidak sama dengan seeds_used di meta.json",
                kolom="seed",
            )
    return df


# ----------------------------------------------------------------- gabungan


def _periksa_silang(
    meta: Meta | None,
    trial: dict[str, pd.DataFrame],
    waktu: pd.DataFrame | None,
    mcnemar: pd.DataFrame | None,
    wilcoxon: pd.DataFrame | None,
) -> list[ArtefakError]:
    """Pemeriksaan konsistensi antar berkas yang sudah berhasil dimuat."""
    galat: list[ArtefakError] = []
    if waktu is not None:
        for metode, df in trial.items():
            n_waktu = int(waktu.loc[waktu["method"] == metode, "n_trials"].iloc[0])
            if n_waktu != len(df):
                galat.append(
                    ArtefakError(
                        BERKAS_WAKTU,
                        f"n_trials metode '{metode}' = {n_waktu}, tetapi {BERKAS_TRIAL[metode]} "
                        f"memuat {len(df)} trial",
                        kolom="n_trials",
                    )
                )
    if mcnemar is not None:
        total = mcnemar[["sama_benar", "hanya_a", "hanya_b", "sama_salah"]].sum(axis=1)
        salah = total[total != JUMLAH_DATA_UJI]
        if len(salah):
            galat.append(
                ArtefakError(
                    BERKAS_MCNEMAR,
                    f"jumlah empat sel tabel harus {JUMLAH_DATA_UJI}, "
                    f"ditemukan {int(salah.iloc[0])} pada baris data ke-{int(salah.index[0]) + 1}",
                    kolom="Sama-sama benar",
                )
            )
    if wilcoxon is not None and meta is not None:
        jumlah = wilcoxon[["a_menang", "b_menang", "seri"]].sum(axis=1)
        salah = jumlah[jumlah != len(meta.seeds_used)]
        if len(salah):
            galat.append(
                ArtefakError(
                    BERKAS_WILCOXON,
                    f"jumlah menang/kalah/seri harus {len(meta.seeds_used)} (jumlah seed), "
                    f"ditemukan {int(salah.iloc[0])}",
                    kolom="A menang / B menang / seri",
                )
            )
    return galat


def muat_artefak(root: str | Path) -> Artefak:
    """Muat dan validasi seluruh 13 berkas artefak; kumpulkan semua galat sekaligus."""
    akar = Path(root)
    if not akar.is_dir():
        raise ArtefakGagal([ArtefakError(str(akar), "folder artefak tidak ditemukan")])

    galat: list[ArtefakError] = []

    def coba(fungsi, *argumen):
        try:
            return fungsi(*argumen)
        except ArtefakError as e:
            galat.append(e)
            return None

    meta = coba(muat_meta, akar)
    bobot = {m: coba(muat_bobot, akar, m) for m in METODE}
    trial = {
        m: coba(
            muat_trial,
            akar,
            m,
            list(meta.best_configs[m]) if meta is not None else None,
        )
        for m in METODE
    }
    waktu = coba(muat_waktu, akar)
    acuan = coba(muat_acuan, akar)
    manifest = coba(muat_manifest, akar)
    mcnemar = coba(muat_mcnemar, akar)
    multi_seed = coba(muat_multi_seed, akar, meta.seeds_used if meta is not None else None)
    wilcoxon = coba(muat_wilcoxon, akar)

    galat.extend(
        _periksa_silang(
            meta,
            {m: d for m, d in trial.items() if d is not None},
            waktu,
            mcnemar,
            wilcoxon,
        )
    )
    if galat:
        raise ArtefakGagal(galat)
    return Artefak(
        root=akar,
        meta=meta,
        bobot=bobot,
        trial=trial,
        waktu=waktu,
        acuan=acuan,
        manifest=manifest,
        mcnemar=mcnemar,
        multi_seed=multi_seed,
        wilcoxon=wilcoxon,
    )
