"""Tes hasil batch: tabel prediksi, serialisasi CSV, validasi, dan eksekusi batch (FR-05, FR-06, FR-14)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from core.artifacts import Artefak
from core.batch import jalankan_batch
from core.data import DataUji, buat_transform
from core.hasil import (
    HasilBatch,
    HasilBatchError,
    WaktuModel,
    dari_csv,
    kumpulkan_info,
    ke_csv,
    simpan_csv,
    muat_csv,
    tabel_prediksi,
)
from core.konstanta import KELAS, METODE


def _hasil_dari_manifest(artefak: Artefak, indeks: np.ndarray) -> HasilBatch:
    """Hasil batch buatan uji dari kolom manifest (hanya untuk menguji serialisasi)."""
    m = artefak.manifest
    rng = np.random.default_rng(0)
    return HasilBatch(
        indeks=indeks,
        metode=METODE,
        prediksi={k: m[f"pred_{k}"].to_numpy()[indeks].astype(np.int64) for k in METODE},
        proba_prediksi={k: rng.uniform(0.3, 1.0, len(indeks)).astype(np.float32) for k in METODE},
        waktu={k: WaktuModel(len(indeks), 1.5 + i, 1.25 + i) for i, k in enumerate(METODE)},
        info=kumpulkan_info(16, "uji"),
    )


@pytest.fixture(scope="module")
def hasil_kecil(artefak: Artefak) -> HasilBatch:
    return _hasil_dari_manifest(artefak, np.array([0, 1, 2, 400, 401, 800, 1200, 1599]))


@pytest.fixture(scope="module")
def hasil_penuh(artefak: Artefak) -> HasilBatch:
    return _hasil_dari_manifest(artefak, np.arange(1600))


def test_kumpulkan_info_memuat_semua_kunci_wajib() -> None:
    info = kumpulkan_info(16, "semua")
    assert {"tanggal", "perangkat", "versi_python", "versi_torch", "versi_streamlit",
            "ukuran_batch", "sampel"} <= set(info)
    assert info["versi_torch"] == torch.__version__
    assert all("\n" not in v for v in info.values())


def test_tabel_prediksi_kolom_dan_isi(data_uji: DataUji, hasil_kecil: HasilBatch, artefak: Artefak) -> None:
    t = tabel_prediksi(data_uji, hasil_kecil)
    harapan = ["berkas", "label_asli"]
    for m in METODE:
        harapan += [f"prediksi_{m}", f"probabilitas_{m}", f"status_{m}"]
    assert list(t.columns) == harapan
    assert t["berkas"].tolist() == [data_uji.path[i] for i in hasil_kecil.indeks]
    assert t["label_asli"].tolist() == [KELAS[data_uji.label[i]] for i in hasil_kecil.indeks]
    for m in METODE:
        benar = hasil_kecil.prediksi[m] == data_uji.label[hasil_kecil.indeks]
        assert t[f"status_{m}"].tolist() == ["benar" if b else "salah" for b in benar]


def test_roundtrip_csv_mempertahankan_semua(data_uji: DataUji, hasil_penuh: HasilBatch) -> None:
    kembali = dari_csv(ke_csv(data_uji, hasil_penuh), data_uji)
    assert kembali.penuh and kembali.semua_model
    assert np.array_equal(kembali.indeks, hasil_penuh.indeks)
    for m in METODE:
        assert np.array_equal(kembali.prediksi[m], hasil_penuh.prediksi[m])
        assert np.allclose(kembali.proba_prediksi[m], hasil_penuh.proba_prediksi[m], atol=1e-7, rtol=0)
        assert kembali.waktu[m] == hasil_penuh.waktu[m]
    assert kembali.info == hasil_penuh.info


def test_roundtrip_urutan_indeks_tidak_diurutkan(data_uji: DataUji, artefak: Artefak) -> None:
    h = _hasil_dari_manifest(artefak, np.array([5, 3, 1599, 0]))
    kembali = dari_csv(ke_csv(data_uji, h), data_uji)
    assert kembali.indeks.tolist() == [5, 3, 1599, 0]
    assert not kembali.penuh


def test_simpan_dan_muat_berkas(tmp_path, data_uji: DataUji, hasil_kecil: HasilBatch) -> None:
    jalur = tmp_path / "hasil.csv"
    simpan_csv(jalur, data_uji, hasil_kecil)
    assert muat_csv(jalur, data_uji).indeks.tolist() == hasil_kecil.indeks.tolist()


def test_berkas_tidak_ada_menyebut_nama(tmp_path, data_uji: DataUji) -> None:
    with pytest.raises(HasilBatchError, match="hasil_batch_penuh.csv.*tidak ditemukan"):
        muat_csv(tmp_path / "hasil_batch_penuh.csv", data_uji)


def _rusak(teks: str, lama: str, baru: str) -> str:
    assert lama in teks
    return teks.replace(lama, baru, 1)


def test_metadata_wajib_hilang(data_uji: DataUji, hasil_kecil: HasilBatch) -> None:
    teks = ke_csv(data_uji, hasil_kecil)
    baris = [b for b in teks.splitlines() if not b.startswith("# versi_torch")]
    with pytest.raises(HasilBatchError, match="versi_torch"):
        dari_csv("\n".join(baris) + "\n", data_uji)


def test_kolom_hilang_dan_asing(data_uji: DataUji, hasil_kecil: HasilBatch) -> None:
    teks = ke_csv(data_uji, hasil_kecil)
    with pytest.raises(HasilBatchError, match="kolom wajib tidak ada: prediksi_grid_search"):
        dari_csv(_rusak(teks, "prediksi_grid_search", "prediksi_x"), data_uji)
    baris = [
        (b if b.startswith("#") else (b + ",kolom_asing" if b.startswith("berkas,") else b + ",1"))
        for b in teks.splitlines()
    ]
    with pytest.raises(HasilBatchError, match="kolom tidak dikenal: kolom_asing"):
        dari_csv("\n".join(baris) + "\n", data_uji)


def test_status_tidak_konsisten(data_uji: DataUji, hasil_kecil: HasilBatch) -> None:
    teks = ke_csv(data_uji, hasil_kecil)
    ubah = teks.replace(",benar", ",XX", 1) if ",benar" in teks else teks.replace(",salah", ",benar", 1)
    with pytest.raises(HasilBatchError, match="status_"):
        dari_csv(ubah, data_uji)


def test_label_tidak_sama_dengan_manifest(data_uji: DataUji, hasil_kecil: HasilBatch) -> None:
    teks = ke_csv(data_uji, hasil_kecil)
    with pytest.raises(HasilBatchError, match="label_asli"):
        dari_csv(_rusak(teks, "glioma/Te-gl_16.jpg,glioma", "glioma/Te-gl_16.jpg,notumor"), data_uji)


def test_path_tidak_ada_di_manifest(data_uji: DataUji, hasil_kecil: HasilBatch) -> None:
    teks = ke_csv(data_uji, hasil_kecil)
    with pytest.raises(HasilBatchError, match="tidak ada di manifest"):
        dari_csv(_rusak(teks, "glioma/Te-gl_16.jpg", "glioma/tidak_ada.jpg"), data_uji)


def test_probabilitas_di_luar_rentang(data_uji: DataUji, hasil_kecil: HasilBatch) -> None:
    t = tabel_prediksi(data_uji, hasil_kecil)
    h = HasilBatch(
        hasil_kecil.indeks, hasil_kecil.metode, hasil_kecil.prediksi,
        {**hasil_kecil.proba_prediksi, "optuna_tpe": hasil_kecil.proba_prediksi["optuna_tpe"] + 1.0},
        hasil_kecil.waktu, hasil_kecil.info,
    )
    assert len(t) == 8
    with pytest.raises(HasilBatchError, match="probabilitas_optuna_tpe"):
        dari_csv(ke_csv(data_uji, h), data_uji)


# ------------------------------------------------------------ eksekusi batch


@pytest.fixture(scope="module")
def jalan_kecil(artefak: Artefak, data_uji: DataUji):
    from core.model import muat_model

    cache: dict[str, nn.Module] = {}

    def ambil(m: str) -> nn.Module:
        if m not in cache:
            k = artefak.meta.best_configs[m]
            cache[m] = muat_model(artefak.bobot[m], float(k["dropout_rate"]), int(k["dense_units"]))
        return cache[m]

    transform = buat_transform(
        artefak.meta.img_size, artefak.meta.imagenet_mean, artefak.meta.imagenet_std
    )
    return ambil, transform


def test_jalankan_batch_sampel_sama_untuk_semua_model(artefak, data_uji, jalan_kecil) -> None:
    ambil, transform = jalan_kecil
    indeks = np.array([0, 1, 400, 800, 1200, 1599, 2, 3])
    panggilan: list[tuple[str, int, int]] = []
    h = jalankan_batch(
        data_uji, indeks, METODE, ambil, transform, 8, "uji",
        kemajuan=lambda m, s, t, d: panggilan.append((m, s, t)),
    )
    assert h.metode == METODE and h.semua_model and h.n == 8 and not h.penuh
    assert h.indeks.tolist() == indeks.tolist()
    for m in METODE:
        assert h.prediksi[m].shape == (8,) and h.proba_prediksi[m].shape == (8,)
        assert h.waktu[m].n == 8 and h.waktu[m].total_detik >= h.waktu[m].forward_detik > 0
    assert [p[0] for p in panggilan] == list(METODE)
    assert [p[1] for p in panggilan] == [8, 16, 24] and {p[2] for p in panggilan} == {24}


def test_jalankan_batch_satu_model(artefak, data_uji, jalan_kecil) -> None:
    ambil, transform = jalan_kecil
    h = jalankan_batch(data_uji, np.array([0, 1]), ["optuna_tpe"], ambil, transform, 8, "uji")
    assert h.metode == ("optuna_tpe",) and not h.semua_model


def test_jalankan_batch_metode_tidak_valid(artefak, data_uji, jalan_kecil) -> None:
    ambil, transform = jalan_kecil
    with pytest.raises(ValueError):
        jalankan_batch(data_uji, np.array([0]), ["lain"], ambil, transform, 8, "uji")
    with pytest.raises(ValueError):
        jalankan_batch(data_uji, np.array([0]), [], ambil, transform, 8, "uji")
