"""Tes halaman Streamlit Tahap 3 (halaman 1, 2, dan 3) dengan AppTest."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from core.artifacts import Artefak
from core.hasil import BERKAS_HASIL_BATCH, HasilBatch, WaktuModel, kumpulkan_info
from core.konstanta import METODE, SITASI_DATASET
from core.teks import BATASAN_PENELITIAN, JUDUL_SKRIPSI, PERNYATAAN_PROTOTIPE, TUJUAN_PENELITIAN
from ui import sumber

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


def test_streamlit_app_berjalan_normal() -> None:
    at = AppTest.from_file(str(AKAR / "streamlit_app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Ringkasan Penelitian"


def test_halaman_ringkasan_memuat_seluruh_unsur_fr15() -> None:
    at = AppTest.from_file(str(AKAR / "pages" / "1_Ringkasan.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Ringkasan Penelitian"
    semua_teks = "\n".join(e.value for e in [*at.markdown, *at.caption])
    assert JUDUL_SKRIPSI in semua_teks
    assert SITASI_DATASET in semua_teks
    for t in TUJUAN_PENELITIAN:
        assert t in semua_teks
    for b in BATASAN_PENELITIAN:
        assert b in semua_teks
    assert PERNYATAAN_PROTOTIPE in semua_teks
    assert len(at.dataframe) == 2
    assert len(at.expander) == 2
    assert all(not e.proto.expanded for e in at.expander)


def test_halaman_prediksi_batch_tanpa_dan_dengan_hasil(artefak: Artefak) -> None:
    at = AppTest.from_file(str(AKAR / "pages" / "2_Prediksi_Batch.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Prediksi Batch"
    assert any("Belum ada hasil batch" in m.value for m in at.markdown)
    assert len(at.expander) == 1
    assert not at.expander[0].proto.expanded

    at.session_state[sumber.KUNCI_HASIL] = _buat_hasil_uji(artefak, np.arange(1600))
    at.session_state[sumber.KUNCI_SUMBER_HASIL] = "dijalankan pada sesi ini"
    at.run()
    assert not at.exception
    assert len(at.dataframe) == 2  # tabel waktu inferensi + tabel prediksi per citra
    assert len(at.dataframe[1].value) == 1600
    assert len(at.dataframe[1].value.columns) == 11
    assert at.checkbox[0].value is False

    # Aktifkan filter "Hanya yang salah"
    at.checkbox[0].check().run()
    assert not at.exception
    assert 0 < len(at.dataframe[1].value) < 1600


def test_tombol_muat_hasil_tersimpan_di_halaman_prediksi_batch() -> None:
    at = AppTest.from_file(str(AKAR / "pages" / "2_Prediksi_Batch.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert sumber.KUNCI_HASIL not in at.session_state

    at.button(key="muat_hasil_tersimpan").click().run()
    assert not at.exception
    hasil = at.session_state[sumber.KUNCI_HASIL]
    assert hasil.penuh and hasil.semua_model
    assert at.session_state[sumber.KUNCI_SUMBER_HASIL] == f"berkas {BERKAS_HASIL_BATCH}"
    assert len(at.dataframe[1].value) == 1600


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
    assert len(at.tabs) == 3
    # Ringkasan di atas, diikuti McNemar dan Verifikasi terhadap Laporan (sebelum tab detail model)
    assert "F1-score macro" in at.dataframe[0].value.columns
    assert "p-value (Holm)" in at.dataframe[1].value.columns
    df_verif = next(
        d.value for d in at.dataframe if "Prediksi berbeda dari manifest" in d.value.columns
    )
    assert df_verif["Status"].tolist() == ["SESUAI", "SESUAI", "SESUAI"]
    assert df_verif["Prediksi berbeda dari manifest"].tolist() == ["0", "0", "0"]

    # Batch parsial 1 model: FR-08 dan FR-13 menampilkan info
    at.session_state[sumber.KUNCI_HASIL] = _buat_hasil_uji(
        artefak, np.arange(16), metode=("grid_search",)
    )
    at.run()
    assert not at.exception
    assert len(at.info) == 2


def test_hasil_batch_penuh_tersimpan_valid_di_halaman_evaluasi() -> None:
    at_batch = AppTest.from_file(str(AKAR / "pages" / "2_Prediksi_Batch.py"), default_timeout=30)
    at_batch.run()
    at_batch.button(key="muat_hasil_tersimpan").click().run()
    hasil_penuh = at_batch.session_state[sumber.KUNCI_HASIL]

    at_eval = AppTest.from_file(str(AKAR / "pages" / "3_Evaluasi.py"), default_timeout=30)
    at_eval.session_state[sumber.KUNCI_HASIL] = hasil_penuh
    at_eval.run()
    assert not at_eval.exception
    assert len(at_eval.error) == 0
    df_verif = next(
        d.value for d in at_eval.dataframe if "Prediksi berbeda dari manifest" in d.value.columns
    )
    assert df_verif["Status"].tolist() == ["SESUAI", "SESUAI", "SESUAI"]
