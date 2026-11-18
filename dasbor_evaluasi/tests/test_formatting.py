"""Tes format angka Indonesia."""

from __future__ import annotations

import numpy as np
import pytest

from core.formatting import format_angka, format_metrik


@pytest.mark.parametrize(
    ("nilai", "desimal", "harapan"),
    [
        (1600, 0, "1.600"),
        (0, 0, "0"),
        (999, 0, "999"),
        (1000, 0, "1.000"),
        (1234567, 0, "1.234.567"),
        (-1600, 0, "-1.600"),
        (0.9812, 4, "0,9812"),
        (0.5, 4, "0,5000"),
        (1, 4, "1,0000"),
        (1600.5, 1, "1.600,5"),
        (1234567.891, 2, "1.234.567,89"),
        (-0.0044, 4, "-0,0044"),
        (0.92, 1, "0,9"),
    ],
)
def test_format_angka(nilai: float, desimal: int, harapan: str) -> None:
    assert format_angka(nilai, desimal) == harapan


def test_nol_negatif_hasil_pembulatan_tanpa_minus() -> None:
    assert format_angka(-0.00001, 4) == "0,0000"
    assert format_angka(-0.0, 2) == "0,00"


def test_tipe_numpy_diterima() -> None:
    assert format_angka(np.int64(1600)) == "1.600"
    assert format_angka(np.float64(0.9225), 4) == "0,9225"
    assert format_angka(np.float32(0.5), 2) == "0,50"


def test_desimal_bawaan_nol() -> None:
    assert format_angka(1599.6) == "1.600"


def test_format_metrik_empat_desimal() -> None:
    assert format_metrik(0.9812) == "0,9812"
    assert format_metrik(0.9225) == "0,9225"
    assert format_metrik(1) == "1,0000"
    assert format_metrik(0.0) == "0,0000"


@pytest.mark.parametrize("buruk", [True, False, "1600", None, np.bool_(True)])
def test_tipe_tidak_valid_ditolak(buruk: object) -> None:
    with pytest.raises(TypeError):
        format_angka(buruk)  # type: ignore[arg-type]


@pytest.mark.parametrize("buruk", [float("nan"), float("inf"), float("-inf")])
def test_nilai_tidak_hingga_ditolak(buruk: float) -> None:
    with pytest.raises(ValueError):
        format_angka(buruk)


@pytest.mark.parametrize("desimal", [-1, 1.5, True])
def test_desimal_tidak_valid_ditolak(desimal: object) -> None:
    with pytest.raises(ValueError):
        format_angka(1.0, desimal)  # type: ignore[arg-type]
