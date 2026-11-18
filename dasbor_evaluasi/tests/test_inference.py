"""Tes inferensi batch (FR-05, FR-09) dengan model dan citra asli."""

from __future__ import annotations

import numpy as np
import pytest
from torch import nn

from core.artifacts import Artefak
from core.data import DataUji, buat_transform
from core.inference import jalankan_inferensi
from core.model import muat_model

N = 16  # jumlah citra pertama pada urutan manifest


@pytest.fixture(scope="module")
def model_grid(artefak: Artefak) -> nn.Module:
    k = artefak.meta.best_configs["grid_search"]
    return muat_model(artefak.bobot["grid_search"], float(k["dropout_rate"]), int(k["dense_units"]))


@pytest.fixture(scope="module")
def transform(artefak: Artefak):
    m = artefak.meta
    return buat_transform(m.img_size, m.imagenet_mean, m.imagenet_std)


@pytest.fixture(scope="module")
def jalur(data_uji: DataUji):
    return data_uji.jalur(range(N))


@pytest.fixture(scope="module")
def dasar(model_grid, jalur, transform):
    return jalankan_inferensi(model_grid, jalur, transform, 8)


def test_bentuk_dan_probabilitas(dasar) -> None:
    assert dasar.prediksi.shape == (N,)
    assert dasar.proba.shape == (N, 4)
    assert dasar.n_selesai == N and not dasar.dibatalkan
    assert np.allclose(dasar.proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(dasar.prediksi, dasar.proba.argmax(axis=1))
    assert np.array_equal(dasar.proba_prediksi, dasar.proba.max(axis=1))


def test_ukuran_batch_tidak_mengubah_prediksi(model_grid, jalur, transform, dasar) -> None:
    lain = jalankan_inferensi(model_grid, jalur, transform, 16)
    assert np.array_equal(lain.prediksi, dasar.prediksi)
    assert np.allclose(lain.proba, dasar.proba, atol=1e-4)


def test_urutan_sampel_dipertahankan(model_grid, jalur, transform, dasar) -> None:
    terbalik = jalankan_inferensi(model_grid, list(reversed(jalur)), transform, 8)
    assert np.array_equal(terbalik.prediksi[::-1], dasar.prediksi)
    assert np.allclose(terbalik.proba[::-1], dasar.proba, atol=1e-4)


def test_deterministik_dua_kali_jalan(model_grid, jalur, transform, dasar) -> None:
    ulang = jalankan_inferensi(model_grid, jalur, transform, 8)
    assert np.array_equal(ulang.proba, dasar.proba)


def test_kemajuan_dipanggil_tiap_batch(model_grid, jalur, transform) -> None:
    panggilan: list[tuple[int, int]] = []
    jalankan_inferensi(model_grid, jalur, transform, 8, kemajuan=lambda s, t: panggilan.append((s, t)))
    assert panggilan == [(8, N), (16, N)]


def test_pembatalan_mengembalikan_hasil_sebagian(model_grid, jalur, transform) -> None:
    keadaan = {"batch": 0}

    def batal() -> bool:
        keadaan["batch"] += 1
        return keadaan["batch"] > 1  # batalkan sebelum batch kedua

    h = jalankan_inferensi(model_grid, jalur, transform, 8, batal=batal)
    assert h.dibatalkan and h.n_selesai == 8
    assert h.prediksi.shape == (8,) and h.proba.shape == (8, 4)


def test_waktu_dan_throughput_terukur(dasar) -> None:
    assert dasar.waktu_total_detik >= dasar.waktu_forward_detik > 0
    assert dasar.throughput == pytest.approx(N / dasar.waktu_total_detik)


@pytest.mark.parametrize("ukuran", [0, 4, 64])
def test_ukuran_batch_tidak_diizinkan(model_grid, jalur, transform, ukuran: int) -> None:
    with pytest.raises(ValueError, match="ukuran batch"):
        jalankan_inferensi(model_grid, jalur, transform, ukuran)


def test_tanpa_gradien_tersisa(model_grid, jalur, transform) -> None:
    jalankan_inferensi(model_grid, jalur, transform, 8)
    assert all(p.grad is None for p in model_grid.parameters())
