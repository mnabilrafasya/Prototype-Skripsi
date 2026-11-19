"""Halaman 5: Riwayat Pencarian dan Multi-Seed (FR-11, FR-12, ekspor PNG FR-14).

Seluruh data pada halaman ini dibaca dari berkas artefak; aplikasi tidak
melatih ulang model dan tidak menghitung ulang uji statistik Wilcoxon.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.ekspor import (
    kurva_konvergensi_png,
    multi_seed_f1_png,
    sebaran_f1_trial_png,
)
from core.formatting import (
    format_angka,
    format_angka_ringkas,
    format_metrik,
    format_selisih,
)
from core.konstanta import METODE, NAMA_METODE
from core.teks import NAMA_KONFIGURASI
from ui import sumber, tema
from ui.tampilan import (
    grafik_konvergensi,
    grafik_multi_seed,
    grafik_sebaran_trial,
    tampilkan_tabel,
)

tema.judul("Riwayat Pencarian dan Multi-Seed")
artefak = sumber.artefak()
meta = artefak.meta

st.info("Dihitung di notebook, tidak dihitung ulang di aplikasi. Seluruh angka dibaca dari artefak.")

# ---------------------------------------------------- FR-11 Riwayat Pencarian
st.subheader("Riwayat Pencarian Hyperparameter (dari Artefak)")

st.plotly_chart(
    grafik_konvergensi(artefak.trial),
    width="stretch",
    key="plot_konvergensi",
)
st.download_button(
    "Unduh PNG kurva konvergensi",
    kurva_konvergensi_png(
        artefak.trial, ukuran_font=13 if tema.presentasi_aktif() else 11
    ),
    file_name="kurva_konvergensi.png",
    mime="image/png",
    key="unduh_png_konvergensi",
    on_click="ignore",
)

st.plotly_chart(
    grafik_sebaran_trial(artefak.trial),
    width="stretch",
    key="plot_sebaran_trial",
)
st.download_button(
    "Unduh PNG sebaran F1-score validasi per trial",
    sebaran_f1_trial_png(
        artefak.trial, ukuran_font=11 if tema.presentasi_aktif() else 10
    ),
    file_name="sebaran_f1_per_trial.png",
    mime="image/png",
    key="unduh_png_sebaran_trial",
    on_click="ignore",
)

st.markdown("#### Waktu Pencarian dan Waktu Rata-Rata per Trial")
waktu_idx = artefak.waktu.set_index("method")
baris_waktu = []
for m in METODE:
    n_t = int(waktu_idx.loc[m, "n_trials"])
    total_detik = float(waktu_idx.loc[m, "total_time_sec"])
    rata_total = total_detik / n_t
    rata_latih = float(artefak.trial[m]["training_time_sec"].mean())
    selisih_overhead = rata_total - rata_latih
    baris_waktu.append(
        {
            "Metode": NAMA_METODE[m],
            "Jumlah trial": format_angka(n_t),
            "Total waktu pencarian (detik)": format_angka(total_detik, 2),
            "Total waktu pencarian (menit)": format_angka(total_detik / 60.0, 2),
            "Waktu rata-rata per trial — total/jumlah trial (detik)": format_angka(rata_total, 2),
            "Rata-rata kolom training_time_sec (detik)": format_angka(rata_latih, 2),
            "Selisih / overhead per trial (detik)": format_selisih(selisih_overhead, 2),
        }
    )
tampilkan_tabel(pd.DataFrame(baris_waktu))
st.caption(
    "Sumber: waktu_pencarian.csv dan trial_*.csv. Waktu rata-rata per trial dihitung dari "
    "total_time_sec dibagi n_trials sesuai spesifikasi; selisih terhadap rata-rata kolom "
    "training_time_sec merupakan overhead di luar pelatihan (pembuatan DataLoader, pencatatan, "
    "dan pemilihan kandidat)."
)

st.markdown("#### Konfigurasi Terbaik Hasil Pencarian")
kunci_konfig = list(meta.best_configs[METODE[0]])
baris_konfig = []
for m in METODE:
    df_t = artefak.trial[m]
    idx_terbaik = int(df_t["val_f1"].idxmax())
    baris_terbaik = df_t.loc[idx_terbaik]
    item: dict[str, str] = {
        "Metode": NAMA_METODE[m],
        "Trial terbaik": format_angka(int(baris_terbaik["trial"])),
        "Akurasi validasi terbaik": format_metrik(float(baris_terbaik["val_acc"])),
        "F1-score macro validasi terbaik": format_metrik(float(baris_terbaik["val_f1"])),
        "Epoch dilatih": format_angka(int(baris_terbaik["epochs_trained"])),
    }
    for k in kunci_konfig:
        v = meta.best_configs[m][k]
        item[NAMA_KONFIGURASI.get(k, k)] = (
            format_angka_ringkas(v) if isinstance(v, (int, float)) else str(v)
        )
    baris_konfig.append(item)
tampilkan_tabel(pd.DataFrame(baris_konfig))
st.caption("Sumber: meta.json dan trial_*.csv.")

st.markdown("#### Tabel Riwayat Seluruh Trial per Metode")
metode_trial = st.selectbox(
    "Pilih metode untuk melihat riwayat seluruh trial",
    METODE,
    format_func=NAMA_METODE.get,
    key="pilih_metode_trial",
)
df_mentah = artefak.trial[metode_trial]
tampil_trial = pd.DataFrame(
    {
        "Trial": [format_angka(int(v)) for v in df_mentah["trial"]],
        "Akurasi validasi": [format_metrik(float(v)) for v in df_mentah["val_acc"]],
        "F1-score macro validasi": [format_metrik(float(v)) for v in df_mentah["val_f1"]],
        "Terbaik sejauh ini (best_so_far)": [
            format_metrik(float(v)) for v in df_mentah["best_so_far"]
        ],
        "Epoch dilatih": [format_angka(int(v)) for v in df_mentah["epochs_trained"]],
        "Waktu pelatihan (detik)": [
            format_angka(float(v), 2) for v in df_mentah["training_time_sec"]
        ],
    }
)
for k in kunci_konfig:
    kol_csv = f"config.{k}"
    nama_kol = NAMA_KONFIGURASI.get(k, k)
    tampil_trial[nama_kol] = [
        format_angka_ringkas(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
        for v in df_mentah[kol_csv]
    ]
tampilkan_tabel(tampil_trial, tinggi=340)

# ---------------------------------------------------- FR-12 Validasi Multi-Seed
st.subheader("Validasi Multi-Seed dan Uji Wilcoxon (dari Artefak)")
st.write(
    "**Keterangan: Dihitung di notebook, tidak dihitung ulang di aplikasi.** "
    "Data dibaca dari multi_seed_hasil.csv dan wilcoxon.csv."
)

seeds = meta.seeds_used
ms = artefak.multi_seed

st.plotly_chart(
    grafik_multi_seed(ms, seeds),
    width="stretch",
    key="plot_multi_seed",
)
st.download_button(
    "Unduh PNG grafik F1-score macro multi-seed",
    multi_seed_f1_png(ms, seeds, ukuran_font=13 if tema.presentasi_aktif() else 11),
    file_name="multi_seed_f1_macro.png",
    mime="image/png",
    key="unduh_png_multi_seed",
    on_click="ignore",
)

st.markdown("#### Ringkasan Multi-Seed (Rata-Rata dan Simpangan Baku Sampel, ddof = 1)")
baris_ringkasan_ms = []
for m in METODE:
    sub = ms[ms["method"] == m].set_index("seed").loc[list(seeds)]
    acc = sub["accuracy"].to_numpy(dtype=float)
    f1 = sub["f1_macro"].to_numpy(dtype=float)
    baris_ringkasan_ms.append(
        {
            "Metode": NAMA_METODE[m],
            "Jumlah seed": format_angka(len(seeds)),
            "Rata-rata akurasi": format_metrik(float(acc.mean())),
            "Simpangan baku akurasi (ddof = 1)": format_metrik(float(acc.std(ddof=1))),
            "Rata-rata F1-score macro": format_metrik(float(f1.mean())),
            "Simpangan baku F1-score macro (ddof = 1)": format_metrik(float(f1.std(ddof=1))),
            "Rentang F1-score macro [min, maks]": (
                f"[{format_metrik(float(f1.min()))}; {format_metrik(float(f1.max()))}]"
            ),
        }
    )
tampilkan_tabel(pd.DataFrame(baris_ringkasan_ms))

st.markdown("#### F1-score Macro dan Akurasi per Seed per Metode")
baris_per_seed = []
for s in seeds:
    b_seed: dict[str, str] = {"Seed": str(s)}
    for m in METODE:
        baris_m = ms[(ms["method"] == m) & (ms["seed"] == s)].iloc[0]
        b_seed[f"F1-score macro {NAMA_METODE[m]}"] = format_metrik(float(baris_m["f1_macro"]))
        b_seed[f"Akurasi {NAMA_METODE[m]}"] = format_metrik(float(baris_m["accuracy"]))
    baris_per_seed.append(b_seed)
tampilkan_tabel(pd.DataFrame(baris_per_seed))

st.markdown("#### Hasil Uji Wilcoxon Signed-Rank dengan Koreksi Holm (dari wilcoxon.csv)")
wi = artefak.wilcoxon
tampilkan_tabel(
    pd.DataFrame(
        {
            "Pasangan (A vs B)": [
                f"{NAMA_METODE[a]} vs {NAMA_METODE[b]}"
                for a, b in zip(wi["metode_a"], wi["metode_b"])
            ],
            "A menang / B menang / seri": [
                f"{format_angka(int(a))} / {format_angka(int(b))} / {format_angka(int(s))}"
                for a, b, s in zip(wi["a_menang"], wi["b_menang"], wi["seri"])
            ],
            "Rata-rata selisih F1 (A - B)": [
                format_selisih(float(v)) for v in wi["rata_selisih_f1"]
            ],
            "p-value": [format_metrik(float(v)) for v in wi["p_value"]],
            "p-value (Holm)": [format_metrik(float(v)) for v in wi["p_holm"]],
            "Signifikan (alpha = 0,05)": wi["signifikan"],
        }
    )
)
st.caption("Sumber: wilcoxon.csv (dihitung di notebook, tidak dihitung ulang di aplikasi).")

tema.footer()
