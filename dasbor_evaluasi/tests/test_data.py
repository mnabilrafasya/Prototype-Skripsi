"""Tes data uji: validasi folder, pemilihan sampel, dan praproses (FR-02, FR-03, FR-04)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from core.artifacts import Artefak
from core.data import (
    DataUji,
    DataUjiError,
    DatasetUji,
    baca_citra,
    buat_transform,
    indeks_contoh_laporan,
    muat_data_uji,
    pilih_indeks,
)
from core.konstanta import JUMLAH_DATA_UJI, KELAS

AKAR_PROYEK = Path(__file__).resolve().parents[1]
FOLDER_ASLI = AKAR_PROYEK / "data_uji"


# ------------------------------------------------------------ validasi folder


def test_data_uji_asli_valid_dan_urutan_sama_dengan_manifest(
    data_uji: DataUji, artefak: Artefak
) -> None:
    assert len(data_uji) == JUMLAH_DATA_UJI
    assert list(data_uji.path) == artefak.manifest["path"].tolist()
    assert data_uji.label.tolist() == artefak.manifest["label"].tolist()


@pytest.fixture
def salinan(tmp_path: Path) -> Path:
    """Salinan data_uji dengan hard link (cepat, tidak menyalin isi)."""
    tujuan = tmp_path / "data_uji"
    shutil.copytree(FOLDER_ASLI, tujuan, copy_function=os.link)
    return tujuan


def test_folder_tidak_ada(tmp_path: Path, artefak: Artefak) -> None:
    with pytest.raises(DataUjiError, match="tidak ditemukan"):
        muat_data_uji(tmp_path / "tidak_ada", artefak.manifest)


def test_jumlah_kurang_menyebut_jumlah_ditemukan(salinan: Path, artefak: Artefak) -> None:
    (salinan / artefak.manifest["path"].iloc[0]).unlink()
    with pytest.raises(DataUjiError) as info:
        muat_data_uji(salinan, artefak.manifest)
    pesan = str(info.value)
    assert f"ditemukan {JUMLAH_DATA_UJI - 1}" in pesan
    assert "glioma: 399" in pesan


def test_jumlah_lebih_menyebut_jumlah_ditemukan(salinan: Path, artefak: Artefak) -> None:
    sumber = salinan / artefak.manifest["path"].iloc[0]
    shutil.copy(sumber, sumber.with_name("tambahan.jpg"))
    with pytest.raises(DataUjiError, match=f"ditemukan {JUMLAH_DATA_UJI + 1}"):
        muat_data_uji(salinan, artefak.manifest)


def test_berkas_bukan_gambar_ditolak(salinan: Path, artefak: Artefak) -> None:
    (salinan / "notumor" / "catatan.txt").write_text("bukan gambar")
    with pytest.raises(DataUjiError, match=r"bukan gambar.*notumor/catatan\.txt"):
        muat_data_uji(salinan, artefak.manifest)


def test_subfolder_kelas_hilang(salinan: Path, artefak: Artefak) -> None:
    shutil.rmtree(salinan / "pituitary")
    with pytest.raises(DataUjiError) as info:
        muat_data_uji(salinan, artefak.manifest)
    assert "'pituitary'" in str(info.value)
    assert "ditemukan 1200" in str(info.value)


def test_path_manifest_tidak_ada_di_folder(salinan: Path, artefak: Artefak) -> None:
    pertama = salinan / artefak.manifest["path"].iloc[0]
    pertama.rename(pertama.with_name("nama_lain.jpg"))
    with pytest.raises(DataUjiError, match="tidak ada di"):
        muat_data_uji(salinan, artefak.manifest)


def test_berkas_citra_rusak_menyebut_nama_berkas(tmp_path: Path) -> None:
    rusak = tmp_path / "rusak.jpg"
    rusak.write_text("ini bukan jpeg")
    ds = DatasetUji([rusak], buat_transform(224, (0.5,) * 3, (0.5,) * 3), ["glioma/rusak.jpg"])
    with pytest.raises(DataUjiError, match="glioma/rusak.jpg"):
        ds[0]


# ------------------------------------------------------------ pemilihan sampel


def test_semua_mengembalikan_seluruh_indeks(data_uji: DataUji) -> None:
    assert pilih_indeks(data_uji.label, "semua").tolist() == list(range(JUMLAH_DATA_UJI))


def test_acak_deterministik_terurut_dan_sama_dengan_rng_seed_42(data_uji: DataUji) -> None:
    a = pilih_indeks(data_uji.label, "acak", 200)
    b = pilih_indeks(data_uji.label, "acak", 200)
    assert a.tolist() == b.tolist()
    assert a.tolist() == sorted(np.random.default_rng(42).choice(1600, 200, replace=False).tolist())
    assert len(set(a.tolist())) == 200


@pytest.mark.parametrize("n", [0, -1, 1601])
def test_acak_n_tidak_valid(data_uji: DataUji, n: int) -> None:
    with pytest.raises(ValueError):
        pilih_indeks(data_uji.label, "acak", n)


def test_seimbang_n_per_kelas_terurut_dan_deterministik(data_uji: DataUji) -> None:
    idx = pilih_indeks(data_uji.label, "seimbang", 25)
    assert idx.tolist() == sorted(idx.tolist())
    assert np.bincount(data_uji.label[idx], minlength=4).tolist() == [25] * 4
    assert idx.tolist() == pilih_indeks(data_uji.label, "seimbang", 25).tolist()
    # tiap kelas diacak dengan default_rng(42) baru
    harapan = []
    for k in range(4):
        semua_k = np.flatnonzero(data_uji.label == k)
        harapan += np.random.default_rng(42).choice(semua_k, 25, replace=False).tolist()
    assert idx.tolist() == sorted(harapan)


def test_seimbang_maksimal_400_per_kelas(data_uji: DataUji) -> None:
    assert len(pilih_indeks(data_uji.label, "seimbang", 400)) == 1600
    with pytest.raises(ValueError):
        pilih_indeks(data_uji.label, "seimbang", 401)


def test_mode_tidak_dikenal(data_uji: DataUji) -> None:
    with pytest.raises(ValueError):
        pilih_indeks(data_uji.label, "lain", 5)


def test_contoh_laporan_urutan_persis_keluaran_rng_dan_tidak_diurutkan() -> None:
    harapan = np.random.default_rng(42).choice(1600, size=8, replace=False)
    hasil = indeks_contoh_laporan()
    assert hasil.tolist() == harapan.tolist()
    assert len(set(hasil.tolist())) == 8


# ------------------------------------------------------- praproses (FR-04)

INDEKS_CONTOH = [0, 400, 800, 1200, 1599]  # satu per kelas ditambah citra terakhir
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _transform_lampiran() -> transforms.Compose:
    """Salinan harfiah eval_transform di lampiran_kode_inti.ipynb (sel 3)."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _tensor_manual(jalur: Path) -> np.ndarray:
    """Pembanding independen: PIL resize bilinear, bagi 255, normalisasi dengan numpy."""
    gambar = Image.open(jalur).convert("RGB").resize((224, 224), Image.BILINEAR)
    larik = np.asarray(gambar, dtype=np.float64) / 255.0
    larik = (larik - np.array(MEAN)) / np.array(STD)
    return larik.transpose(2, 0, 1)


@pytest.mark.parametrize("indeks", INDEKS_CONTOH)
def test_praproses_sama_dengan_hitungan_manual_independen(
    data_uji: DataUji, artefak: Artefak, indeks: int
) -> None:
    jalur = data_uji.jalur([indeks])[0]
    pipeline = buat_transform(
        artefak.meta.img_size, artefak.meta.imagenet_mean, artefak.meta.imagenet_std
    )(baca_citra(jalur))
    manual = _tensor_manual(jalur)
    assert pipeline.shape == (3, 224, 224)
    assert pipeline.dtype == torch.float32
    selisih = float(np.max(np.abs(pipeline.numpy().astype(np.float64) - manual)))
    assert selisih <= 1e-6, f"selisih maksimum {selisih:.3e} untuk {data_uji.path[indeks]}"


@pytest.mark.parametrize("indeks", INDEKS_CONTOH)
def test_praproses_sama_dengan_transform_lampiran(
    data_uji: DataUji, artefak: Artefak, indeks: int
) -> None:
    jalur = data_uji.jalur([indeks])[0]
    pipeline = buat_transform(
        artefak.meta.img_size, artefak.meta.imagenet_mean, artefak.meta.imagenet_std
    )(baca_citra(jalur))
    lampiran = _transform_lampiran()(Image.open(jalur).convert("RGB"))
    assert float((pipeline - lampiran).abs().max()) <= 1e-6


def test_dataset_uji_menjaga_urutan_dan_mengembalikan_tensor(
    data_uji: DataUji, artefak: Artefak
) -> None:
    jalur = data_uji.jalur([5, 3, 9])
    ds = DatasetUji(
        jalur,
        buat_transform(artefak.meta.img_size, artefak.meta.imagenet_mean, artefak.meta.imagenet_std),
    )
    assert len(ds) == 3
    for i, j in enumerate(jalur):
        assert torch.equal(ds[i], buat_transform(224, MEAN, STD)(baca_citra(j)))


def test_kelas_sesuai_urutan_tetap() -> None:
    assert KELAS == ("glioma", "meningioma", "notumor", "pituitary")
