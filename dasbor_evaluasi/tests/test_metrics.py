"""Tes metrik terhadap contoh kecil yang dihitung tangan, dan logika FR-13."""

from __future__ import annotations

import numpy as np
import pytest

from core.artifacts import HasilAcuan
from core.metrics import (
    STATUS_SESUAI,
    STATUS_TIDAK_SESUAI,
    bandingkan_dengan_acuan,
    hitung_metrik,
)

# Contoh hitung tangan (8 sampel, 2 per kelas):
#   y_true = 0 0 1 1 2 2 3 3 ; y_pred = 0 1 1 1 2 3 3 3
#   matriks: [[1,1,0,0],[0,2,0,0],[0,0,1,1],[0,0,0,2]]
#   akurasi = 6/8 = 0,75
#   presisi = [1, 2/3, 1, 2/3]; recall = [1/2, 1, 1/2, 1]
#   F1      = [2/3, 4/5, 2/3, 4/5]; F1 macro = (4/3 + 8/5)/4 = 11/15
Y_TRUE = [0, 0, 1, 1, 2, 2, 3, 3]
Y_PRED = [0, 1, 1, 1, 2, 3, 3, 3]


def test_metrik_contoh_hitung_tangan() -> None:
    m = hitung_metrik(Y_TRUE, Y_PRED)
    assert m.n == 8
    assert m.accuracy == pytest.approx(0.75, abs=1e-12)
    assert m.f1_macro == pytest.approx(11 / 15, abs=1e-12)
    assert m.per_kelas["presisi"].tolist() == pytest.approx([1, 2 / 3, 1, 2 / 3], abs=1e-12)
    assert m.per_kelas["recall"].tolist() == pytest.approx([0.5, 1, 0.5, 1], abs=1e-12)
    assert m.per_kelas["f1"].tolist() == pytest.approx([2 / 3, 0.8, 2 / 3, 0.8], abs=1e-12)
    assert m.per_kelas["dukungan"].tolist() == [2, 2, 2, 2]
    assert m.confusion_matrix.tolist() == [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1], [0, 0, 0, 2]]
    assert list(m.per_kelas.index) == ["glioma", "meningioma", "notumor", "pituitary"]
    assert "accuracy" in m.laporan and "macro avg" in m.laporan


def test_matriks_tetap_4x4_walau_kelas_tidak_muncul() -> None:
    m = hitung_metrik([0, 0, 1, 1], [0, 1, 1, 1])
    assert m.confusion_matrix.shape == (4, 4)
    assert m.per_kelas["dukungan"].tolist() == [2, 2, 0, 0]
    assert m.confusion_matrix.sum() == 4


def test_prediksi_sempurna() -> None:
    m = hitung_metrik(Y_TRUE, Y_TRUE)
    assert m.accuracy == 1.0 and m.f1_macro == 1.0
    assert np.array_equal(m.confusion_matrix, 2 * np.eye(4, dtype=int))


@pytest.mark.parametrize("a, b", [([], []), ([0, 1], [0])])
def test_masukan_tidak_valid(a: list[int], b: list[int]) -> None:
    with pytest.raises(ValueError):
        hitung_metrik(a, b)


# ---------------------------------------------------------------- FR-13

N = 1600
LABEL = np.zeros(N, dtype=int)
PRED_MANIFEST = np.zeros(N, dtype=int)
PRED_MANIFEST[:100] = 1  # 100 salah, 1500 benar


def _acuan() -> HasilAcuan:
    m = hitung_metrik(LABEL, PRED_MANIFEST)
    return HasilAcuan(m.accuracy, m.f1_macro, {}, m.confusion_matrix)


def _bandingkan(pred: np.ndarray):
    return bandingkan_dengan_acuan(hitung_metrik(LABEL, pred), _acuan(), pred, PRED_MANIFEST)


def test_identik_sesuai() -> None:
    r = _bandingkan(PRED_MANIFEST.copy())
    assert r.status == STATUS_SESUAI
    assert r.prediksi_berbeda == 0 and r.selisih_akurasi == 0 and r.confusion_matrix_sama


def test_delapan_beda_tanpa_selisih_akurasi_sesuai() -> None:
    pred = PRED_MANIFEST.copy()
    pred[:8] = 2  # salah -> salah (kelas lain): akurasi tetap
    r = _bandingkan(pred)
    assert r.prediksi_berbeda == 8 and r.selisih_akurasi == 0
    assert r.status == STATUS_SESUAI
    assert not r.confusion_matrix_sama


def test_sembilan_beda_tidak_sesuai_dengan_rincian() -> None:
    pred = PRED_MANIFEST.copy()
    pred[:9] = 2
    r = _bandingkan(pred)
    assert r.status == STATUS_TIDAK_SESUAI
    assert any("9 prediksi" in s for s in r.rincian)
    assert r.indeks_berbeda == tuple(range(9))


def test_selisih_akurasi_tiga_citra_masih_sesuai() -> None:
    pred = PRED_MANIFEST.copy()
    pred[1000:1003] = 1  # 3 benar -> salah: selisih 3/1600 = 0,001875
    r = _bandingkan(pred)
    assert r.selisih_akurasi == pytest.approx(-3 / N)
    assert r.status == STATUS_SESUAI


def test_selisih_akurasi_empat_citra_tidak_sesuai() -> None:
    pred = PRED_MANIFEST.copy()
    pred[1000:1004] = 1  # selisih 4/1600 = 0,0025 > 0,002
    r = _bandingkan(pred)
    assert r.status == STATUS_TIDAK_SESUAI
    assert any("akurasi" in s for s in r.rincian)
    assert r.selisih_akurasi < 0


def test_panjang_berbeda_ditolak() -> None:
    with pytest.raises(ValueError):
        bandingkan_dengan_acuan(hitung_metrik(LABEL, PRED_MANIFEST), _acuan(), PRED_MANIFEST[:10], PRED_MANIFEST)
