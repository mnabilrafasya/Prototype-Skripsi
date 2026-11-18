"""Tes halaman Streamlit Tahap 3 (halaman 1, 2, dan 3) dengan AppTest."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from core.artifacts import Artefak
from core.hasil import HasilBatch, WaktuModel, kumpulkan_info
from core.konstanta import METODE
from core.teks import BATASAN_PENELITIAN, JUDUL_SKRIPSI, PERNYATAAN_PROTOTIPE, TUJUAN_PENELITIAN
from ui import sumber, tema

AKAR = Path(__file__).resolve().parents[1]


def _buat_hasil_uji(
    artefak: Artefak, indeks: np.ndarray, metode: tuple[str, ...] = METODE
) -> HasilBatch:
    m = artefak.manifest
    rng = np.random.default_rng(42)
    return HasilBatch(
        indeks=indeks,
        metode=metode,
        prediksi={k: m[f"pred_{k}"].to_numpy()[indeks].astype(np.int64) for k in metode},
        proba_prediksi={k: rng.uniform(0.6, 0.99, len(indeks)).astype(np.float64) for k in metode},
        waktu={k: WaktuModel(len(indeks), 10.0, 9.0) for k in metode},
        info=kumpulkan_info(16, "semua data uji" if len(indeks) == 1600 else "acak 16 citra, seed 42"),
    )


def test_core_tidak_mengimpor_streamlit() -> None:
    for p in (AKAR / "core").glob("*.py"):
        pohon = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for simpul in ast.walk(pohon):
            if isinstance(simpul, ast.Import):
                assert all(n.name != "streamlit" for n in simpul.names), p.name
            elif isinstance(simpul, ast.ImportFrom):
                assert simpul.module != "streamlit", p.name


def test_halaman_ringkasan_memuat_seluruh_unsur_fr15() -> None:
    at = AppTest.from_file(str(AKAR / "pages" / "1_Ringkasan.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Ringkasan Penelitian"
    semua_teks = "\n".join(e.value for e in [*at.markdown, *at.caption])
    assert JUDUL_SKRIPSI in semua_teks
    for t in TUJUAN_PENELITIAN:
        assert t in semua_teks
    for b in BATASAN_PENELITIAN:
        assert b in semua_teks
    assert PERNYATAAN_PROTOTIPE in semua_teks
    assert len(at.dataframe) == 2


def test_mode_presentasi_mengubah_tabel_menjadi_st_table() -> None:
    at = AppTest.from_file(str(AKAR / "pages" / "1_Ringkasan.py"), default_timeout=30)
    at.run()
    assert not at.exception
    at.sidebar.button(key="tombol_presentasi").click().run()
    assert not at.exception
    assert at.session_state[tema.KUNCI_PRESENTASI] is True
    assert len(at.table) == 2
    at.button(key="keluar_presentasi").click().run()
    assert at.session_state[tema.KUNCI_PRESENTASI] is False


def test_halaman_prediksi_batch_tanpa_dan_dengan_hasil(artefak: Artefak) -> None:
    at = AppTest.from_file(str(AKAR / "pages" / "2_Prediksi_Batch.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Prediksi Batch"
    assert any("Belum ada hasil batch" in m.value for m in at.markdown)

    at.session_state[sumber.KUNCI_HASIL] = _buat_hasil_uji(artefak, np.arange(1600))
    at.session_state[sumber.KUNCI_SUMBER_HASIL] = "dijalankan pada sesi ini"
    at.run()
    assert not at.exception
    assert len(at.dataframe) == 2  # tabel waktu inferensi + tabel prediksi per citra
    assert len(at.dataframe[1].value) == 1600

    # Aktifkan filter "Hanya yang salah"
    at.checkbox[0].check().run()
    assert not at.exception
    assert 0 < len(at.dataframe[1].value) < 1600


def test_halaman_evaluasi_batch_penuh_sesuai_dan_parsial(artefak: Artefak) -> None:
    at = AppTest.from_file(str(AKAR / "pages" / "3_Evaluasi.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert any("Belum ada hasil batch" in m.value for m in at.markdown)

    # Batch penuh 3 model: FR-07, FR-08, FR-13 aktif dan SESUAI
    at.session_state[sumber.KUNCI_HASIL] = _buat_hasil_uji(artefak, np.arange(1600))
    at.run()
    assert not at.exception
    assert len(at.error) == 0
    df_verif = at.dataframe[-1].value
    assert df_verif["Status"].tolist() == ["SESUAI", "SESUAI", "SESUAI"]
    assert df_verif["Prediksi berbeda dari manifest"].tolist() == ["0", "0", "0"]

    # Batch parsial 1 model: FR-08 dan FR-13 menampilkan info
    at.session_state[sumber.KUNCI_HASIL] = _buat_hasil_uji(
        artefak, np.arange(16), metode=("grid_search",)
    )
    at.run()
    assert not at.exception
    assert len(at.info) == 2


def test_mode_presentasi_memuat_hasil_batch_penuh_dan_fr13_aktif() -> None:
    at = AppTest.from_file(str(AKAR / "pages" / "3_Evaluasi.py"), default_timeout=30)
    at.session_state[tema.KUNCI_PRESENTASI] = True
    at.run()
    assert not at.exception
    assert len(at.error) == 0
    df_verif = at.table[-1].value
    assert df_verif["Status"].tolist() == ["SESUAI", "SESUAI", "SESUAI"]

