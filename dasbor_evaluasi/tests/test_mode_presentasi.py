"""Tes tambahan Mode Presentasi pada Tahap 5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from core.artifacts import Artefak
from core.hasil import BERKAS_HASIL_BATCH, HasilBatch, WaktuModel, kumpulkan_info
from core.konstanta import METODE
from ui import sumber, tema

AKAR = Path(__file__).resolve().parents[1]


def test_mode_presentasi_di_halaman_ringkasan_memuat_hasil_batch_penuh() -> None:
    at = AppTest.from_file(str(AKAR / "pages" / "1_Ringkasan.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert sumber.KUNCI_HASIL not in at.session_state

    at.sidebar.button(key="tombol_presentasi").click().run()
    assert not at.exception
    assert at.session_state[tema.KUNCI_PRESENTASI] is True
    hasil: HasilBatch = at.session_state[sumber.KUNCI_HASIL]
    assert hasil.penuh and hasil.semua_model
    assert at.session_state[sumber.KUNCI_SUMBER_HASIL] == f"berkas {BERKAS_HASIL_BATCH}"


def test_mode_presentasi_mengganti_batch_parsial_dengan_batch_penuh(artefak: Artefak) -> None:
    indeks_parsial = np.arange(16, dtype=np.int64)
    m = artefak.manifest
    parsial = HasilBatch(
        indeks=indeks_parsial,
        metode=("grid_search",),
        prediksi={"grid_search": m["pred_grid_search"].to_numpy()[indeks_parsial].astype(np.int64)},
        proba_prediksi={"grid_search": np.full(16, 0.9, dtype=np.float64)},
        waktu={"grid_search": WaktuModel(16, 1.0, 0.9)},
        info=kumpulkan_info(16, "acak 16 citra, seed 42"),
    )
    at = AppTest.from_file(str(AKAR / "pages" / "3_Evaluasi.py"), default_timeout=30)
    at.session_state[sumber.KUNCI_HASIL] = parsial
    at.run()
    assert not at.exception
    assert len(at.info) == 2  # FR-08 dan FR-13 tidak aktif pada batch parsial 1 model

    # Nyalakan Mode Presentasi: otomatis memuat hasil_batch_penuh.csv sehingga FR-08 & FR-13 aktif
    at.sidebar.button(key="tombol_presentasi").click().run()
    assert not at.exception
    hasil: HasilBatch = at.session_state[sumber.KUNCI_HASIL]
    assert hasil.penuh and hasil.semua_model
    df_verif = at.table[-1].value
    assert df_verif["Status"].tolist() == ["SESUAI", "SESUAI", "SESUAI"]
