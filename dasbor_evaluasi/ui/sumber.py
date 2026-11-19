"""Pemuatan sumber (artefak, data uji, model) dengan cache dan pesan galat spesifik."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import streamlit as st
from torch import nn

from core.artifacts import Artefak, ArtefakGagal, muat_artefak
from core.data import DataUji, DataUjiError, buat_transform, muat_data_uji
from core.gradcam import GradCAM
from core.hasil import BERKAS_HASIL_BATCH, HasilBatch, HasilBatchError, muat_csv
from core.model import ModelError, muat_model
from ui import tema

AKAR = Path(__file__).resolve().parents[1]
DIR_ARTEFAK = AKAR / "artefak"
DIR_DATA_UJI = AKAR / "data_uji"
BERKAS_PRESENTASI = DIR_ARTEFAK / BERKAS_HASIL_BATCH

KUNCI_HASIL = "hasil_batch"
KUNCI_SUMBER_HASIL = "sumber_hasil"


def berhenti_dengan_galat(pesan: str) -> NoReturn:
    """Tampilkan galat, footer, lalu hentikan halaman."""
    st.error(pesan)
    tema.footer()
    st.stop()


@st.cache_resource(show_spinner="Memuat artefak")
def _artefak_tercache() -> Artefak:
    return muat_artefak(DIR_ARTEFAK)


@st.cache_resource(show_spinner="Memeriksa data uji")
def _data_uji_tercache() -> DataUji:
    return muat_data_uji(DIR_DATA_UJI, _artefak_tercache().manifest)


@st.cache_resource(show_spinner="Memuat model dan Grad-CAM")
def _model_dan_gradcam_tercache(metode: str) -> tuple[nn.Module, GradCAM]:
    artefak = _artefak_tercache()
    konfig = artefak.meta.best_configs[metode]
    model = muat_model(
        artefak.bobot[metode], float(konfig["dropout_rate"]), int(konfig["dense_units"])
    )
    return model, GradCAM(model)


def artefak() -> Artefak:
    """Artefak tervalidasi; bila gagal, tampilkan galat (nama berkas dan kolom) lalu berhenti."""
    try:
        return _artefak_tercache()
    except ArtefakGagal as e:
        berhenti_dengan_galat(str(e))


def data_uji() -> DataUji:
    """Data uji bawaan tervalidasi; galat menyebut jumlah citra yang ditemukan."""
    artefak()
    try:
        return _data_uji_tercache()
    except DataUjiError as e:
        berhenti_dengan_galat(str(e))


def ambil_model(metode: str) -> nn.Module:
    """Model dari cache (st.cache_resource); dimuat sekali per metode."""
    try:
        return _model_dan_gradcam_tercache(metode)[0]
    except ModelError as e:
        berhenti_dengan_galat(str(e))


def ambil_gradcam(metode: str) -> GradCAM:
    """Instance tunggal GradCAM dari cache (st.cache_resource) untuk metode terkait."""
    try:
        return _model_dan_gradcam_tercache(metode)[1]
    except ModelError as e:
        berhenti_dengan_galat(str(e))


def transform():
    """Praproses inferensi dari meta.json."""
    meta = artefak().meta
    return buat_transform(meta.img_size, meta.imagenet_mean, meta.imagenet_std)


def hasil_tersimpan() -> HasilBatch | None:
    """Hasil batch di session_state; di Mode Presentasi dimuat dari artefak/hasil_batch_penuh.csv."""
    if KUNCI_HASIL not in st.session_state and st.session_state.get(tema.KUNCI_PRESENTASI, False):
        try:
            hasil = muat_csv(BERKAS_PRESENTASI, data_uji())
        except HasilBatchError as e:
            st.error(f"Mode Presentasi tidak dapat memuat hasil tersimpan. {e}")
            return None
        st.session_state[KUNCI_HASIL] = hasil
        st.session_state[KUNCI_SUMBER_HASIL] = f"berkas {BERKAS_HASIL_BATCH}"
    return st.session_state.get(KUNCI_HASIL)


def keterangan_sumber(hasil: HasilBatch) -> str:
    """Teks asal hasil: dijalankan di sesi ini atau dimuat dari berkas, beserta metadata."""
    asal = st.session_state.get(KUNCI_SUMBER_HASIL, "dijalankan pada sesi ini")
    i = hasil.info
    return (
        f"Sumber hasil: {asal}. Dihasilkan {i['tanggal']}; perangkat {i['perangkat']}; "
        f"Python {i['versi_python']}, torch {i['versi_torch']}, streamlit {i['versi_streamlit']}; "
        f"ukuran batch {i['ukuran_batch']}; sampel: {i['sampel']}."
    )
