"""Tes McNemar: contoh hitung tangan dan paritas dengan mcnemar_acuan.csv (NFR-01)."""

from __future__ import annotations

import numpy as np
import pytest

from core.artifacts import TOLERANSI_P_VALUE, Artefak
from core.konstanta import METODE, PASANGAN
from core.stats import CATATAN_TANPA_DISKORDAN, mcnemar_tiga_pasangan, uji_mcnemar


def _vektor(sama_benar: int, hanya_a: int, hanya_b: int, sama_salah: int):
    a = [True] * sama_benar + [True] * hanya_a + [False] * hanya_b + [False] * sama_salah
    b = [True] * sama_benar + [False] * hanya_a + [True] * hanya_b + [False] * sama_salah
    return np.array(a), np.array(b)


def test_exact_hitung_tangan_1_vs_5() -> None:
    # Diskordan 6, min = 1: p = 2 * P(X <= 1), X ~ Bin(6; 0,5) = 2 * (1 + 6) / 64 = 0,21875
    a, b = _vektor(3, 1, 5, 2)
    h = uji_mcnemar(a, b)
    assert (h.n, h.sama_benar, h.hanya_a, h.hanya_b, h.sama_salah) == (11, 3, 1, 5, 2)
    assert h.p_value == pytest.approx(0.21875, abs=1e-12)
    assert h.catatan == ""
    assert h.selisih_akurasi == pytest.approx(4 / 11 - 8 / 11, abs=1e-12)


def test_exact_hitung_tangan_0_vs_4() -> None:
    # Diskordan 4, min = 0: p = 2 * P(X = 0) = 2 / 16 = 0,125
    a, b = _vektor(5, 0, 4, 1)
    assert uji_mcnemar(a, b).p_value == pytest.approx(0.125, abs=1e-12)


def test_tanpa_diskordan_p_satu_dengan_catatan() -> None:
    a, b = _vektor(7, 0, 0, 3)
    h = uji_mcnemar(a, b)
    assert h.p_value == 1.0
    assert h.catatan == CATATAN_TANPA_DISKORDAN
    assert h.selisih_akurasi == 0.0


def test_p_value_dibatasi_satu() -> None:
    a, b = _vektor(0, 3, 3, 0)  # 2 * P(X <= 3) = 84/64 -> 1,0
    assert uji_mcnemar(a, b).p_value == 1.0


def test_masukan_tidak_valid() -> None:
    with pytest.raises(ValueError):
        uji_mcnemar([True, False], [True])
    with pytest.raises(ValueError):
        uji_mcnemar([], [])


def _holm_manual(p: list[float]) -> list[float]:
    """Holm step-down: p_(i) * (m - i + 1), kumulatif maksimum, dipotong 1."""
    m = len(p)
    urut = sorted(range(m), key=lambda i: p[i])
    hasil = [0.0] * m
    berjalan = 0.0
    for rank, i in enumerate(urut):
        berjalan = max(berjalan, min(1.0, (m - rank) * p[i]))
        hasil[i] = berjalan
    return hasil


def _masukan(artefak: Artefak):
    m = artefak.manifest
    return m["label"].to_numpy(), {k: m[f"pred_{k}"].to_numpy() for k in METODE}


def test_paritas_dengan_mcnemar_acuan(artefak: Artefak) -> None:
    y, pred = _masukan(artefak)
    hasil = mcnemar_tiga_pasangan(y, pred).set_index(["metode_a", "metode_b"])
    acuan = artefak.mcnemar.set_index(["metode_a", "metode_b"])
    assert set(hasil.index) == set(PASANGAN)
    for pasangan in PASANGAN:
        h, a = hasil.loc[pasangan], acuan.loc[pasangan]
        assert h["n"] == 1600
        for kolom in ("sama_benar", "hanya_a", "hanya_b", "sama_salah"):
            assert h[kolom] == a[kolom], (pasangan, kolom)
        assert h["selisih_akurasi"] == pytest.approx(a["selisih_akurasi"], abs=TOLERANSI_P_VALUE)
        assert h["p_value"] == pytest.approx(a["p_value"], abs=TOLERANSI_P_VALUE)
        assert h["p_holm"] == pytest.approx(a["p_holm"], abs=TOLERANSI_P_VALUE)
        assert h["signifikan"] == a["signifikan"]
        assert h["pasangan"] == a["pasangan"]


def test_holm_sama_dengan_hitungan_manual(artefak: Artefak) -> None:
    y, pred = _masukan(artefak)
    hasil = mcnemar_tiga_pasangan(y, pred)
    assert hasil["p_holm"].tolist() == pytest.approx(_holm_manual(hasil["p_value"].tolist()), abs=1e-12)


def test_tiga_pasangan_pada_prediksi_identik_p_satu() -> None:
    y = np.array([0, 1, 2, 3, 0, 1])
    pred = {k: y.copy() for k in METODE}
    hasil = mcnemar_tiga_pasangan(y, pred)
    assert hasil["p_value"].tolist() == [1.0] * 3
    assert hasil["p_holm"].tolist() == [1.0] * 3
    assert all(c == CATATAN_TANPA_DISKORDAN for c in hasil["catatan"])
    assert hasil["signifikan"].tolist() == ["tidak"] * 3


def test_metode_tidak_lengkap_atau_panjang_berbeda_ditolak() -> None:
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="optuna_tpe"):
        mcnemar_tiga_pasangan(y, {"grid_search": y, "random_search": y})
    with pytest.raises(ValueError, match="panjang"):
        mcnemar_tiga_pasangan(y, {"grid_search": y, "random_search": y, "optuna_tpe": y[:1]})
