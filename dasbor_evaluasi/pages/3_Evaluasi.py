"""Halaman 3: Evaluasi dan Perbandingan (FR-07, FR-08, FR-13, ekspor metrik dan confusion matrix)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.ekspor import confusion_matrix_png, metrik_ke_csv
from core.formatting import format_angka, format_metrik, format_selisih
from core.konstanta import (
    KELAS,
    MAKS_PREDIKSI_BERBEDA,
    NAMA_METODE,
    TOLERANSI_SELISIH_AKURASI,
)
from core.metrics import (
    STATUS_TIDAK_SESUAI,
    MetrikModel,
    bandingkan_dengan_acuan,
    hitung_metrik,
)
from core.stats import mcnemar_tiga_pasangan
from ui import sumber, tema
from ui.tampilan import (
    grafik_perbandingan_metrik,
    penanda_metode,
    peta_panas,
    subjudul,
    tampilkan_tabel,
)

tema.judul("Evaluasi dan Perbandingan")
artefak = sumber.artefak()
data_uji = sumber.data_uji()
hasil = sumber.hasil_tersimpan()

if hasil is None:
    st.write(
        "Belum ada hasil batch. Jalankan batch atau muat hasil batch penuh tersimpan "
        "(1.600 citra) terlebih dahulu di halaman Prediksi Batch."
    )
    tema.footer()
    st.stop()

y_true = data_uji.label[hasil.indeks]
metrik: dict[str, MetrikModel] = {
    m: hitung_metrik(y_true, hasil.prediksi[m]) for m in hasil.metode
}

st.write(
    f"N = {format_angka(hasil.n)} citra; sampel: {hasil.info['sampel']}; "
    f"model: {', '.join(NAMA_METODE[m] for m in hasil.metode)}."
)
st.caption(sumber.keterangan_sumber(hasil))

# -------------------------------------- Ringkasan Metrik di Atas (FR-07, FR-08)
st.subheader("Ringkasan Metrik Evaluasi")
kartu = st.columns(len(hasil.metode))
for kol, m in zip(kartu, hasil.metode):
    km = metrik[m]
    with kol:
        with st.container(border=True):
            st.markdown(f"**{penanda_metode(m)}**", unsafe_allow_html=True)
            k_acc, k_f1 = st.columns(2)
            k_acc.metric("Akurasi", format_metrik(km.accuracy))
            k_f1.metric("F1-score macro", format_metrik(km.f1_macro))

ringkas_rows = []
for m in hasil.metode:
    km = metrik[m]
    baris = {
        "Metode": NAMA_METODE[m],
        "N": format_angka(km.n),
        "Akurasi": format_metrik(km.accuracy),
        "F1-score macro": format_metrik(km.f1_macro),
    }
    for k in KELAS:
        baris[f"F1 {k}"] = format_metrik(float(km.per_kelas.loc[k, "f1"]))
    ringkas_rows.append(baris)

tampilkan_tabel(pd.DataFrame(ringkas_rows))
st.download_button(
    "Unduh CSV metrik",
    metrik_ke_csv(metrik),
    file_name="metrik_evaluasi.csv",
    mime="text/csv",
    key="unduh_csv_metrik",
    on_click="ignore",
)

if hasil.semua_model:
    st.plotly_chart(
        grafik_perbandingan_metrik(metrik, hasil),
        width="stretch",
        key="plot_perbandingan_metrik",
    )

# ------------------------------ FR-08 Uji McNemar Exact dengan Koreksi Holm
st.subheader("Perbandingan Statistik: Uji McNemar (Exact + Koreksi Holm)")
if not hasil.semua_model:
    st.info(
        "Perbandingan tiga model dan McNemar's test hanya aktif bila ketiga model "
        "dijalankan pada sampel yang sama."
    )
else:
    st.write(
        f"N sampel yang dipakai untuk perbandingan berpasangan: {format_angka(hasil.n)} citra."
    )
    mc = mcnemar_tiga_pasangan(y_true, hasil.prediksi)
    tampilkan_tabel(
        pd.DataFrame(
            {
                "Pasangan (A vs B)": mc["pasangan"],
                "N": [format_angka(int(v)) for v in mc["n"]],
                "Sama-sama benar": [format_angka(int(v)) for v in mc["sama_benar"]],
                "Hanya A benar": [format_angka(int(v)) for v in mc["hanya_a"]],
                "Hanya B benar": [format_angka(int(v)) for v in mc["hanya_b"]],
                "Sama-sama salah": [format_angka(int(v)) for v in mc["sama_salah"]],
                "Selisih akurasi (A - B)": [format_selisih(float(v)) for v in mc["selisih_akurasi"]],
                "p-value (exact)": [format_metrik(float(v)) for v in mc["p_value"]],
                "p-value (Holm)": [format_metrik(float(v)) for v in mc["p_holm"]],
                "Signifikan (alpha = 0,05)": mc["signifikan"],
            }
        )
    )
    for _, r in mc.iterrows():
        if r["catatan"]:
            st.caption(f"{r['pasangan']}: {r['catatan']}")

# --------------------------------------------- FR-13 Verifikasi terhadap laporan
st.subheader("Verifikasi terhadap Laporan")
if not hasil.penuh:
    st.info(
        "Verifikasi terhadap nilai acuan laporan (hasil_uji_acuan.json dan "
        "test_manifest_prediksi.csv) hanya aktif setelah batch penuh pada seluruh "
        "data uji bawaan (1.600 citra)."
    )
else:
    st.caption(
        "Kriteria status SESUAI: selisih akurasi absolut tidak lebih dari "
        f"{format_angka(TOLERANSI_SELISIH_AKURASI, 3)} dan jumlah prediksi yang berbeda dari "
        f"test_manifest_prediksi.csv tidak lebih dari {format_angka(MAKS_PREDIKSI_BERBEDA)} citra."
    )
    baris_verif = []
    rincian_gagal: list[tuple[str, tuple[str, ...]]] = []
    detail_beda: list[dict[str, str]] = []
    for m in hasil.metode:
        km = metrik[m]
        acuan = artefak.acuan[m]
        pred_manifest = artefak.manifest[f"pred_{m}"].to_numpy()[hasil.indeks]
        pb = bandingkan_dengan_acuan(km, acuan, hasil.prediksi[m], pred_manifest)
        baris_verif.append(
            {
                "Metode": NAMA_METODE[m],
                "Akurasi aplikasi": format_metrik(km.accuracy),
                "Akurasi acuan": format_metrik(acuan.accuracy),
                "Selisih akurasi": format_selisih(pb.selisih_akurasi),
                "F1-score macro aplikasi": format_metrik(km.f1_macro),
                "F1-score macro acuan": format_metrik(acuan.f1_macro),
                "Selisih F1-score macro": format_selisih(pb.selisih_f1_macro),
                "Prediksi berbeda dari manifest": format_angka(pb.prediksi_berbeda),
                "Confusion matrix identik": "Ya" if pb.confusion_matrix_sama else "Tidak",
                "Status": pb.status,
            }
        )
        if pb.status == STATUS_TIDAK_SESUAI:
            rincian_gagal.append((NAMA_METODE[m], pb.rincian))
        for idx in pb.indeks_berbeda:
            idx_asli = int(hasil.indeks[idx])
            detail_beda.append(
                {
                    "Metode": NAMA_METODE[m],
                    "Indeks manifest": format_angka(idx_asli),
                    "Berkas": data_uji.path[idx_asli],
                    "Label asli": KELAS[int(y_true[idx])],
                    "Prediksi manifest": KELAS[int(pred_manifest[idx])],
                    "Prediksi aplikasi": KELAS[int(hasil.prediksi[m][idx])],
                }
            )
    tampilkan_tabel(pd.DataFrame(baris_verif))
    if rincian_gagal:
        for nama_m, alasan in rincian_gagal:
            st.error(f"{nama_m}: TIDAK SESUAI ({'; '.join(alasan)}).")
    if detail_beda:
        st.write("Rincian citra dengan prediksi berbeda dari test_manifest_prediksi.csv:")
        tampilkan_tabel(pd.DataFrame(detail_beda))

# ------------------------------------ FR-07 Detail per Model dalam st.tabs
st.subheader("Detail Metrik per Kelas dan Confusion Matrix")
tabs = st.tabs([NAMA_METODE[m] for m in hasil.metode])
for tab, m in zip(tabs, hasil.metode):
    km = metrik[m]
    pk = km.per_kelas
    with tab:
        st.markdown(
            f"**{penanda_metode(m)}** — N = {format_angka(km.n)} citra | "
            f"Akurasi = {format_metrik(km.accuracy)} | "
            f"F1-score macro = {format_metrik(km.f1_macro)}",
            unsafe_allow_html=True,
        )
        tampilkan_tabel(
            pd.DataFrame(
                {
                    "Kelas": list(KELAS),
                    "Presisi": [format_metrik(float(pk.loc[k, "presisi"])) for k in KELAS],
                    "Recall": [format_metrik(float(pk.loc[k, "recall"])) for k in KELAS],
                    "F1-score": [format_metrik(float(pk.loc[k, "f1"])) for k in KELAS],
                    "Dukungan (N)": [format_angka(int(pk.loc[k, "dukungan"])) for k in KELAS],
                }
            )
        )

        kiri, kanan = st.columns([1, 1.15])
        with kiri:
            st.caption("Tabel angka confusion matrix (baris: kelas asli, kolom: kelas prediksi)")
            cm_df = pd.DataFrame(
                {"Kelas asli": list(KELAS)}
                | {
                    f"Prediksi {k}": [
                        format_angka(int(km.confusion_matrix[i, j])) for i in range(len(KELAS))
                    ]
                    for j, k in enumerate(KELAS)
                }
            )
            tampilkan_tabel(cm_df)
            png_bytes = confusion_matrix_png(
                km.confusion_matrix,
                judul=f"Confusion Matrix {NAMA_METODE[m]}",
                subjudul=subjudul(hasil),
                ukuran_font=12,
            )
            st.download_button(
                f"Unduh PNG confusion matrix ({NAMA_METODE[m]})",
                png_bytes,
                file_name=f"confusion_matrix_{m}.png",
                mime="image/png",
                key=f"unduh_cm_{m}",
                on_click="ignore",
            )
        with kanan:
            st.plotly_chart(
                peta_panas(km.confusion_matrix, m, hasil),
                width="stretch",
                key=f"cm_plot_{m}",
            )

tema.footer()
