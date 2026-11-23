"""Halaman 2: Prediksi Batch (FR-02 sampai FR-06, FR-09, ekspor CSV prediksi)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.batch import jalankan_batch
from core.data import MODE_ACAK, MODE_SEIMBANG, MODE_SEMUA, pilih_indeks
from core.formatting import format_angka, format_durasi, format_metrik
from core.hasil import STATUS_SALAH, ke_csv, tabel_prediksi
from core.inference import UKURAN_BATCH_DIIZINKAN
from core.konstanta import JUMLAH_DATA_UJI, KELAS, METODE, NAMA_METODE, SEED_SAMPEL
from ui import sumber, tema
from ui.tampilan import tampilkan_tabel

KUNCI_PROSES = "proses_batch_berjalan"
PILIHAN_SAMPEL = {
    f"Seimbang n per kelas (seed {SEED_SAMPEL})": MODE_SEIMBANG,
    f"Acak n (seed {SEED_SAMPEL})": MODE_ACAK,
    "Semua data uji": MODE_SEMUA,
}
SEMUA_MODEL = "Semua model"

tema.judul("Prediksi Batch")
data_uji = sumber.data_uji()

if st.session_state.pop(KUNCI_PROSES, False):
    st.warning("Proses batch sebelumnya dibatalkan. Hasil sebagian tidak disimpan.")

st.caption(
    f"Sumber data: data uji bawaan ({format_angka(len(data_uji))} citra, 400 per kelas) "
    "dari folder data_uji/. Tidak ada unggahan dataset."
)

# ----------------------------------------------------------------- pengaturan
st.subheader("Pengaturan Batch")
with st.container(border=True):
    kol_mode, kol_n = st.columns([2, 1])
    with kol_mode:
        pilihan = st.radio("Cakupan sampel data uji", list(PILIHAN_SAMPEL), horizontal=True)
        mode = PILIHAN_SAMPEL[pilihan]
    with kol_n:
        n_sampel: int | None = None
        if mode == MODE_SEIMBANG:
            maks = int(np.bincount(data_uji.label).min())
            n_sampel = int(st.number_input("n per kelas", 1, maks, min(10, maks), step=1))
        elif mode == MODE_ACAK:
            n_sampel = int(st.number_input("n sampel", 1, JUMLAH_DATA_UJI, 40, step=1))
        else:
            st.caption("Seluruh 1.600 citra data uji (sekitar 5 menit di CPU).")

    with st.expander("Pengaturan lanjutan (ukuran batch & pilihan model)", expanded=False):
        kol_b, kol_m = st.columns(2)
        with kol_b:
            ukuran_batch = int(
                st.selectbox("Ukuran batch inferensi", UKURAN_BATCH_DIIZINKAN, index=1)
            )
            st.caption("Ukuran batch hanya memengaruhi kecepatan dan RAM, tidak mengubah prediksi.")
        with kol_m:
            cakupan = st.radio("Model yang dijalankan", ["Ketiga model", "Satu model"])
            if cakupan == "Satu model":
                metode_dipilih = [
                    st.selectbox("Pilih model", METODE, format_func=NAMA_METODE.get)
                ]
            else:
                metode_dipilih = list(METODE)

    indeks = pilih_indeks(data_uji.label, mode, n_sampel)
    deskripsi = {
        MODE_SEMUA: "semua data uji",
        MODE_ACAK: f"acak {format_angka(n_sampel or 0)} citra, seed {SEED_SAMPEL}",
        MODE_SEIMBANG: f"seimbang {format_angka(n_sampel or 0)} per kelas, seed {SEED_SAMPEL}",
    }[mode]
    st.write(
        f"Sampel yang akan diproses: **{format_angka(len(indeks))} citra** untuk "
        f"**{len(metode_dipilih)} model** (urutan sampel identik untuk semua model)."
    )
    tb1, tb2 = st.columns([1, 1.5])
    with tb1:
        jalankan = st.button("Jalankan batch", type="primary", width="stretch")
    with tb2:
        muat_tersimpan = st.button(
            "Muat hasil batch penuh tersimpan (1.600 citra)",
            key="muat_hasil_tersimpan",
            width="stretch",
        )

if muat_tersimpan:
    sumber.muat_hasil_penuh_tersimpan()

# ------------------------------------------------------------------ eksekusi
if jalankan:
    st.session_state[KUNCI_PROSES] = True
    wadah = st.empty()
    with wadah.container():
        bar = st.progress(0.0, text="Menyiapkan model")
        st.button("Batalkan", key="batalkan")

    def lapor(metode: str, selesai: int, total: int, detik: float) -> None:
        sisa = detik * (total - selesai) / selesai if selesai else 0.0
        bar.progress(
            selesai / total,
            text=(
                f"{NAMA_METODE[metode]}: {format_angka(selesai)} dari {format_angka(total)} "
                f"pemrosesan citra. Berjalan {format_durasi(detik)}, "
                f"perkiraan sisa {format_durasi(sisa)}."
            ),
        )

    hasil_baru = jalankan_batch(
        data_uji,
        indeks,
        metode_dipilih,
        sumber.ambil_model,
        sumber.transform(),
        ukuran_batch,
        deskripsi,
        kemajuan=lapor,
    )
    wadah.empty()
    st.session_state[sumber.KUNCI_HASIL] = hasil_baru
    st.session_state[sumber.KUNCI_SUMBER_HASIL] = "dijalankan pada sesi ini"
    st.session_state[KUNCI_PROSES] = False

hasil = sumber.hasil_tersimpan()
if hasil is None:
    st.write(
        "Belum ada hasil batch. Klik **Jalankan batch** untuk inferensi langsung, atau klik "
        "**Muat hasil batch penuh tersimpan (1.600 citra)**."
    )
    tema.footer()
    st.stop()

# ---------------------------------------------------------------- ringkasan
st.subheader("Hasil Batch dan Waktu Inferensi")
r1, r2, r3 = st.columns(3)
r1.metric("Jumlah sampel (N)", format_angka(hasil.n))
r2.metric("Jumlah model", format_angka(len(hasil.metode)))
r3.metric("Ukuran batch", hasil.info["ukuran_batch"])
st.caption(sumber.keterangan_sumber(hasil))

tampilkan_tabel(
    pd.DataFrame(
        {
            "Model": [NAMA_METODE[m] for m in hasil.metode],
            "N citra": [format_angka(hasil.waktu[m].n) for m in hasil.metode],
            "Waktu total (detik)": [
                format_angka(hasil.waktu[m].total_detik, 2) for m in hasil.metode
            ],
            "Waktu forward model (detik)": [
                format_angka(hasil.waktu[m].forward_detik, 2) for m in hasil.metode
            ],
            "Throughput (citra per detik)": [
                format_angka(hasil.waktu[m].throughput, 2) for m in hasil.metode
            ],
        }
    )
)
st.caption(
    "Waktu inferensi di perangkat yang dipakai (CPU), bukan waktu pencarian hyperparameter "
    "di laporan. Waktu total mencakup pembacaan citra dan praproses."
)

# ----------------------------------------------------------- tabel per citra
st.subheader("Prediksi per Citra")
tabel = tabel_prediksi(data_uji, hasil)
f1, f2, f3 = st.columns([2, 3, 2], vertical_alignment="bottom")
with f1:
    filter_model = st.selectbox(
        "Filter model",
        [SEMUA_MODEL, *hasil.metode],
        format_func=lambda m: m if m == SEMUA_MODEL else NAMA_METODE[m],
    )
with f2:
    filter_kelas = st.multiselect("Filter kelas (label asli)", list(KELAS), default=list(KELAS))
with f3:
    hanya_salah = st.checkbox("Hanya yang salah", value=False)

metode_tampil = list(hasil.metode) if filter_model == SEMUA_MODEL else [filter_model]
topeng = tabel["label_asli"].isin(filter_kelas)
if hanya_salah:
    salah = np.zeros(len(tabel), dtype=bool)
    for m in metode_tampil:
        salah |= (tabel[f"status_{m}"] == STATUS_SALAH).to_numpy()
    topeng &= salah

# Urutan kolom konsisten: berkas, label asli, lalu (prediksi, probabilitas, status) per model.
kolom = ["berkas", "label_asli"] + [
    f"{awal}_{m}" for m in metode_tampil for awal in ("prediksi", "probabilitas", "status")
]
terfilter = tabel.loc[topeng, kolom]

tampil = terfilter.copy()
for m in metode_tampil:
    tampil[f"probabilitas_{m}"] = tampil[f"probabilitas_{m}"].map(format_metrik)
    tampil[f"status_{m}"] = tampil[f"status_{m}"].str.capitalize()
tampil = tampil.rename(
    columns={"berkas": "Berkas", "label_asli": "Label asli"}
    | {
        f"{awal}_{m}": f"{judul} {NAMA_METODE[m]}"
        for m in metode_tampil
        for awal, judul in (
            ("prediksi", "Prediksi"),
            ("probabilitas", "Probabilitas"),
            ("status", "Status"),
        )
    }
)
st.write(f"Menampilkan {format_angka(len(terfilter))} dari {format_angka(len(tabel))} baris.")
tampilkan_tabel(tampil, tinggi=420)

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "Unduh CSV tabel yang ditampilkan",
        terfilter.to_csv(index=False, float_format="%.9g", lineterminator="\n"),
        file_name="prediksi_batch_tabel.csv",
        mime="text/csv",
        on_click="ignore",
    )
with d2:
    st.download_button(
        "Unduh hasil batch lengkap (CSV dengan metadata)",
        ke_csv(data_uji, hasil),
        file_name="hasil_batch_penuh.csv" if hasil.penuh and hasil.semua_model else "hasil_batch.csv",
        mime="text/csv",
        on_click="ignore",
    )

tema.footer()
