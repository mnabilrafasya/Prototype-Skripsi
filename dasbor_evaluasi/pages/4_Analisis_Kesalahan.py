"""Halaman 4: Analisis Kesalahan dan Grad-CAM (FR-10, ekspor PNG Grad-CAM FR-14)."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.data import baca_citra, indeks_contoh_laporan
from core.ekspor import gradcam_delapan_contoh_png, gradcam_tiga_model_png
from core.formatting import format_angka, format_metrik
from core.gradcam import ALPHA_OVERLAY, HasilGradCAM, hitung_gradcam_citra
from core.hasil import STATUS_SALAH, tabel_prediksi
from core.konstanta import KELAS, METODE, NAMA_METODE, SEED_SAMPEL
from ui import sumber, tema
from ui.tampilan import penanda_metode, tampilkan_tabel

KUNCI_CONTOH_LAPORAN = "delapan_contoh_laporan_dimuat"
SEMUA_MODEL_SALAH = "minimal_satu"
KETIGA_MODEL_SALAH = "ketiga_model"
SEMUA_PASANGAN = "Semua pasangan salah"


def _hitung_tiga_model(indeks_manifest: int) -> dict[str, HasilGradCAM]:
    """Hitung Grad-CAM ketiga model untuk satu citra pada indeks manifest."""
    du = sumber.data_uji()
    tf = sumber.transform()
    jalur = du.jalur([indeks_manifest])[0]
    tensor = tf(baca_citra(jalur, du.path[indeks_manifest]))
    return {
        m: hitung_gradcam_citra(sumber.ambil_gradcam(m), m, tensor, alpha=ALPHA_OVERLAY)
        for m in METODE
    }


def _jet_rgb(cam_resized: np.ndarray) -> np.ndarray:
    """Ubah peta panas 2D [0, 1] menjadi citra RGB [0, 1] dengan colormap JET OpenCV."""
    bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0


def _tampilkan_gradcam_berdampingan(
    indeks_manifest: int,
    berkas: str,
    label_asli: int,
    hasil_cam: dict[str, HasilGradCAM],
    kunci_unduh: str,
) -> None:
    """Tampilkan Grad-CAM ketiga model berdampingan beserta label asli, prediksi, dan probabilitas."""
    nama_asli = KELAS[label_asli]
    st.write(
        f"Berkas: **{berkas}** (indeks manifest: {format_angka(indeks_manifest)}) | "
        f"Label asli: **{nama_asli}** (indeks {label_asli})."
    )
    baris_info = []
    for m in METODE:
        h = hasil_cam[m]
        status = "Benar" if h.kelas_prediksi == label_asli else "Salah"
        baris = {
            "Metode": NAMA_METODE[m],
            "Label asli": nama_asli,
            "Prediksi": KELAS[h.kelas_prediksi],
            "Probabilitas kelas prediksi": format_metrik(h.probabilitas_prediksi),
            "Status": status,
        }
        for idx_k, k in enumerate(KELAS):
            baris[f"Prob. {k}"] = format_metrik(float(h.probabilitas_semua[idx_k]))
        baris_info.append(baris)
    tampilkan_tabel(pd.DataFrame(baris_info))

    kolom = st.columns(len(METODE))
    for kol, m in zip(kolom, METODE):
        h = hasil_cam[m]
        status = "Benar" if h.kelas_prediksi == label_asli else "Salah"
        with kol:
            st.markdown(f"#### {penanda_metode(m)}", unsafe_allow_html=True)
            st.write(
                f"Label asli: **{nama_asli}** | "
                f"Prediksi: **{KELAS[h.kelas_prediksi]}** "
                f"({format_metrik(h.probabilitas_prediksi)}) — **{status}**"
            )
            st.image(
                h.overlay,
                caption=f"Overlay Grad-CAM (alpha = 0,45) — {NAMA_METODE[m]}",
                width="stretch",
            )
            sub1, sub2 = st.columns(2)
            with sub1:
                st.image(h.citra_asli, caption="Citra asli", width="stretch")
            with sub2:
                st.image(
                    _jet_rgb(h.cam_resized),
                    caption="Peta panas Grad-CAM",
                    width="stretch",
                )

    png_bytes = gradcam_tiga_model_png(
        berkas=berkas,
        label_asli=label_asli,
        hasil_per_metode=hasil_cam,
        ukuran_font=11 if tema.presentasi_aktif() else 10,
    )
    nama_aman = berkas.replace("/", "_")
    st.download_button(
        "Unduh PNG Grad-CAM ketiga model (citra terpilih)",
        png_bytes,
        file_name=f"gradcam_{nama_aman}.png",
        mime="image/png",
        key=kunci_unduh,
        on_click="ignore",
    )


tema.judul("Analisis Kesalahan dan Grad-CAM")
data_uji = sumber.data_uji()
hasil = sumber.hasil_tersimpan()

# ----------------------------- 8 Contoh Laporan (urutan identik dengan notebook)
st.subheader("Delapan Citra Contoh Laporan (Seed 42)")
st.write(
    "Pemilihan memakai np.random.default_rng(42).choice(1600, size=8, replace=False) "
    "pada urutan data uji bawaan tanpa pengurutan ulang, identik dengan sel Grad-CAM di notebook."
)
if st.button("Muat 8 contoh laporan", key="tombol_muat_8_contoh", type="primary"):
    st.session_state[KUNCI_CONTOH_LAPORAN] = True

if st.session_state.get(KUNCI_CONTOH_LAPORAN, False):
    indeks_8 = indeks_contoh_laporan()
    daftar_8: list[tuple[int, str, int, dict[str, HasilGradCAM]]] = []
    baris_8: list[dict[str, str]] = []
    for urut, idx_m in enumerate(indeks_8, start=1):
        idx_int = int(idx_m)
        berkas = data_uji.path[idx_int]
        lbl = int(data_uji.label[idx_int])
        cam_m = _hitung_tiga_model(idx_int)
        daftar_8.append((idx_int, berkas, lbl, cam_m))
        b: dict[str, str] = {
            "Urutan contoh": format_angka(urut),
            "Indeks manifest": format_angka(idx_int),
            "Berkas": berkas,
            "Label asli": KELAS[lbl],
        }
        for m in METODE:
            h = cam_m[m]
            b[f"Prediksi {NAMA_METODE[m]}"] = KELAS[h.kelas_prediksi]
            b[f"Probabilitas {NAMA_METODE[m]}"] = format_metrik(h.probabilitas_prediksi)
            b[f"Status {NAMA_METODE[m]}"] = "Benar" if h.kelas_prediksi == lbl else "Salah"
        baris_8.append(b)

    tampilkan_tabel(pd.DataFrame(baris_8))
    png_8 = gradcam_delapan_contoh_png(
        daftar_8, ukuran_font=10 if tema.presentasi_aktif() else 9
    )
    st.download_button(
        "Unduh PNG Grad-CAM 8 contoh laporan",
        png_8,
        file_name=f"gradcam_8_contoh_laporan_seed_{SEED_SAMPEL}.png",
        mime="image/png",
        key="unduh_png_8_contoh",
        on_click="ignore",
    )

    pilihan_8 = st.selectbox(
        "Pilih satu dari 8 contoh laporan untuk tampilan rinci ketiga model",
        range(len(daftar_8)),
        format_func=lambda i: (
            f"Contoh #{i + 1} — idx {format_angka(daftar_8[i][0])} — "
            f"{daftar_8[i][1]} (asli: {KELAS[daftar_8[i][2]]})"
        ),
        key="pilih_contoh_8",
    )
    idx_terpilih_8, berkas_8, lbl_8, cam_8 = daftar_8[int(pilihan_8)]
    _tampilkan_gradcam_berdampingan(
        idx_terpilih_8, berkas_8, lbl_8, cam_8, kunci_unduh="unduh_satu_dari_8"
    )

# ------------------------------------ Analisis Kesalahan dari Hasil Prediksi Batch
st.subheader("Daftar Citra Salah Klasifikasi dan Grad-CAM Ketiga Model")
if hasil is None:
    st.info(
        "Belum ada hasil batch di sesi ini. Jalankan batch di halaman Prediksi Batch "
        "(atau aktifkan Mode Presentasi) untuk memfilter daftar citra salah klasifikasi."
    )
    tema.footer()
    st.stop()

st.write(
    f"Sumber daftar kesalahan: N = {format_angka(hasil.n)} citra ({hasil.info['sampel']}); "
    f"model pada batch: {', '.join(NAMA_METODE[m] for m in hasil.metode)}."
)
st.caption(sumber.keterangan_sumber(hasil))

tabel = tabel_prediksi(data_uji, hasil)
tabel.insert(0, "indeks_manifest", hasil.indeks.astype(int))

opsi_model = [SEMUA_MODEL_SALAH, *hasil.metode]
if len(hasil.metode) == len(METODE):
    opsi_model.append(KETIGA_MODEL_SALAH)


def _label_opsi_model(k: str) -> str:
    if k == SEMUA_MODEL_SALAH:
        return "Salah pada minimal satu model"
    if k == KETIGA_MODEL_SALAH:
        return "Salah pada ketiga model"
    return f"Salah pada {NAMA_METODE[k]}"


pasangan_kelas = [
    f"{asli} -> {pred}" for asli in KELAS for pred in KELAS if asli != pred
]

kol_f1, kol_f2 = st.columns(2)
with kol_f1:
    filter_model = st.selectbox(
        "Filter model",
        opsi_model,
        format_func=_label_opsi_model,
        key="filter_model_salah",
    )
with kol_f2:
    filter_pasangan = st.selectbox(
        "Filter pasangan kelas (label asli -> prediksi salah)",
        [SEMUA_PASANGAN, *pasangan_kelas],
        key="filter_pasangan_salah",
    )

if filter_model == SEMUA_MODEL_SALAH:
    metode_uji = list(hasil.metode)
    mask_salah = np.zeros(len(tabel), dtype=bool)
    for m in metode_uji:
        mask_salah |= (tabel[f"status_{m}"] == STATUS_SALAH).to_numpy()
elif filter_model == KETIGA_MODEL_SALAH:
    metode_uji = list(hasil.metode)
    mask_salah = np.ones(len(tabel), dtype=bool)
    for m in metode_uji:
        mask_salah &= (tabel[f"status_{m}"] == STATUS_SALAH).to_numpy()
else:
    metode_uji = [filter_model]
    mask_salah = (tabel[f"status_{filter_model}"] == STATUS_SALAH).to_numpy()

if filter_pasangan != SEMUA_PASANGAN:
    asli_target, pred_target = filter_pasangan.split(" -> ")
    mask_pasangan = np.zeros(len(tabel), dtype=bool)
    for m in metode_uji:
        mask_pasangan |= (
            (tabel["label_asli"] == asli_target) & (tabel[f"prediksi_{m}"] == pred_target)
        ).to_numpy()
    mask_salah &= mask_pasangan

df_salah = tabel.loc[mask_salah].reset_index(drop=True)
st.write(
    f"Ditemukan **{format_angka(len(df_salah))}** citra salah klasifikasi "
    f"dari {format_angka(len(tabel))} citra pada batch."
)

if len(df_salah) == 0:
    st.info("Tidak ada citra salah klasifikasi untuk kombinasi filter ini.")
    tema.footer()
    st.stop()

tampil_salah = pd.DataFrame(
    {
        "Indeks manifest": [format_angka(int(v)) for v in df_salah["indeks_manifest"]],
        "Berkas": df_salah["berkas"],
        "Label asli": df_salah["label_asli"],
    }
)
for m in hasil.metode:
    tampil_salah[f"Prediksi {NAMA_METODE[m]}"] = df_salah[f"prediksi_{m}"]
    tampil_salah[f"Probabilitas {NAMA_METODE[m]}"] = df_salah[f"probabilitas_{m}"].map(
        format_metrik
    )
    tampil_salah[f"Status {NAMA_METODE[m]}"] = df_salah[f"status_{m}"].str.capitalize()

tampilkan_tabel(tampil_salah, tinggi=320)

st.markdown("#### Grad-CAM Ketiga Model Berdampingan pada Citra Salah Klasifikasi")
pilihan_salah = st.selectbox(
    "Pilih citra salah klasifikasi",
    range(len(df_salah)),
    format_func=lambda i: (
        f"idx {format_angka(int(df_salah.loc[i, 'indeks_manifest']))} — "
        f"{df_salah.loc[i, 'berkas']} (asli: {df_salah.loc[i, 'label_asli']})"
    ),
    key="pilih_citra_salah",
)
baris_terpilih = df_salah.iloc[int(pilihan_salah)]
idx_manifest_salah = int(baris_terpilih["indeks_manifest"])
berkas_salah = str(baris_terpilih["berkas"])
label_salah = int(data_uji.label[idx_manifest_salah])
hasil_cam_salah = _hitung_tiga_model(idx_manifest_salah)
_tampilkan_gradcam_berdampingan(
    idx_manifest_salah,
    berkas_salah,
    label_salah,
    hasil_cam_salah,
    kunci_unduh="unduh_gradcam_salah",
)

tema.footer()
