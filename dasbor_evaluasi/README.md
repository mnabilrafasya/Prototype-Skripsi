# Dasbor Evaluasi Optimasi Hyperparameter EfficientNetB0

Prototipe penelitian skripsi S1:
**Perbandingan Grid Search, Random Search, dan Optuna (TPE) dalam Optimasi Hyperparameter EfficientNetB0 untuk Klasifikasi Tumor Otak pada Citra MRI**
Program Studi Teknik Informatika, Fakultas Ilmu Komputer, Universitas Sriwijaya, 2026.

Aplikasi ini adalah dasbor evaluasi batch penelitian pada data uji bawaan (1.600 citra), bukan aplikasi klinis dan tidak boleh dipakai untuk diagnosis medis. Aplikasi berjalan sepenuhnya di CPU tanpa memerlukan GPU dan tanpa koneksi ke Kaggle.

---

## 1. Struktur Proyek

```text
dasbor_evaluasi/
  streamlit_app.py                  # Titik masuk aplikasi multipage Streamlit
  pages/
    1_Ringkasan.py                  # FR-15: Ringkasan Penelitian
    2_Prediksi_Batch.py             # FR-02..FR-06, FR-09, FR-14: Prediksi Batch & ekspor CSV
    3_Evaluasi.py                   # FR-07, FR-08, FR-13, FR-14: Evaluasi, McNemar, Verifikasi
    4_Analisis_Kesalahan.py         # FR-10, FR-14: Analisis Kesalahan & Grad-CAM
    5_Riwayat_dan_MultiSeed.py      # FR-11, FR-12, FR-14: Riwayat Pencarian & Multi-Seed
  core/
    konstanta.py                    # Konstanta tetap penelitian (kelas, praproses, warna, sitasi)
    teks.py                         # Kutipan tujuan (1.3), batasan (1.5), dan pernyataan prototipe
    formatting.py                   # Format angka Indonesia (titik ribuan, koma desimal)
    artifacts.py                    # Pemuat dan validator kontrak 13 berkas artefak
    data.py                         # Validator data_uji/, pemilihan sampel, dan praproses
    model.py                        # Pembangunan EfficientNetB0 (weights=None) & pemuatan .pth
    inference.py                    # Inferensi batch deterministik di CPU (num_workers=0, no_grad)
    batch.py                        # Eksekusi batch lintas model pada sampel identik
    hasil.py                        # Tabel prediksi per citra & serialisasi hasil_batch_penuh.csv
    metrics.py                      # Metrik scikit-learn & perbandingan terhadap nilai acuan (FR-13)
    stats.py                        # Uji McNemar exact dengan koreksi Holm
    gradcam.py                      # Kelas GradCAM & overlay_gradcam (identik notebook)
    ekspor.py                       # Ekspor CSV metrik serta PNG confusion matrix, Grad-CAM, grafik
  ui/
    sumber.py                       # Cache st.cache_resource untuk artefak, data uji, model, GradCAM
    tema.py                         # Tema warna (#1F3A5F), tipografi, header, dan footer
    tampilan.py                     # Tabel berformat Indonesia dan grafik interaktif Plotly
  artefak/                          # 13 berkas kontrak artefak + hasil_batch_penuh.csv
  data_uji/                         # 1.600 citra uji bawaan (400 per kelas: glioma, meningioma, notumor, pituitary)
  tests/                            # 201 tes otomatis pytest (unit + AppTest antarmuka)
  tools/
    verify_parity.py                # Skrip CLI verifikasi paritas prediksi aplikasi vs notebook
  .streamlit/
    config.toml                     # Konfigurasi tema terang Streamlit
  pytest.ini                        # Konfigurasi pytest
  requirements.txt                  # Dependensi produksi terpin (indeks PyTorch CPU)
  requirements-dev.txt              # Dependensi pengembangan/pengujian (pytest, psutil)
  README.md                         # Dokumentasi ini
```

---

## 2. Menjalankan Secara Lokal

Prasyarat: **Python 3.12** (dikembangkan dan diuji pada Python 3.12.5, Windows 11 64-bit).

### Windows (PowerShell)

```powershell
cd dasbor_evaluasi
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

### Linux / macOS (Bash)

```bash
cd dasbor_evaluasi
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run streamlit_app.py
```

---

## 3. Menjalankan Tes Otomatis dan Verifikasi Paritas

Pasang dependensi pengembangan terlebih dahulu:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

### Menjalankan seluruh tes unit dan antarmuka (`pytest`)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### Menjalankan verifikasi paritas terhadap artefak notebook (`verify_parity.py`)

```powershell
# Batch penuh (1.600 citra x 3 model)
.\.venv\Scripts\python.exe tools\verify_parity.py

# Batch penuh sekaligus memperbarui artefak/hasil_batch_penuh.csv
.\.venv\Scripts\python.exe tools\verify_parity.py --simpan-hasil

# Pengukuran cepat pada 200 citra acak (seed 42)
.\.venv\Scripts\python.exe tools\verify_parity.py --n 200
```

---

## 4. Memuat Hasil Batch Penuh Tersimpan (Untuk Presentasi Sidang)

Untuk presentasi sidang tanpa menunggu eksekusi inferensi 1.600 citra (~5 menit di CPU), halaman **Prediksi Batch** menyediakan tombol **Muat hasil batch penuh tersimpan (1.600 citra)**:

1. Memuat berkas `artefak/hasil_batch_penuh.csv` (rekaman hasil eksekusi penuh nyata aplikasi ini sendiri pada 1.600 citra beserta metadata tanggal, perangkat, versi Python/torch/streamlit, dan waktu inferensi) ke `session_state`.
2. Seluruh halaman **Evaluasi dan Perbandingan** (termasuk uji McNemar dan verifikasi FR-13) serta **Analisis Kesalahan dan Grad-CAM** langsung aktif dengan 1.600 citra.
3. Untuk demo live di depan dosen penguji, pilih cakupan kecil (mis. **Seimbang 10 per kelas** = 40 citra) lalu klik **Jalankan batch**; untuk memperbesar tampilan proyektor, gunakan tombol lipat sidebar (`<`) bawaan Streamlit dan zoom peramban (`Ctrl` + `+`).

---

## 5. Deploy ke Streamlit Community Cloud

1. Pastikan folder `dasbor_evaluasi/artefak/` (13 berkas kontrak + `hasil_batch_penuh.csv`) dan `dasbor_evaluasi/data_uji/` (1.600 citra) ikut ter-commit di repositori Git. Sebelum mempublikasikan repositori, periksa lisensi dataset Brain Tumor MRI Dataset (Masoud Nickparvar, 2026).
2. Pada dasbor Streamlit Community Cloud, buat aplikasi baru dengan pengaturan:
   - **Main file path**: `dasbor_evaluasi/streamlit_app.py`
   - **Python version** (di _Advanced settings_): pilih **3.12** (wajib diperiksa di setelan deploy agar sesuai dengan versi yang diuji).
3. Streamlit Community Cloud akan membaca `dasbor_evaluasi/requirements.txt` yang sudah memuat `--extra-index-url https://download.pytorch.org/whl/cpu` sehingga memasang `torch` dan `torchvision` versi CPU.
4. **Status verifikasi deploy**: Resolusi `--extra-index-url` dan pemakaian memori di kontainer Linux Streamlit Community Cloud berstatus **TIDAK DIVERIFIKASI** secara langsung dari mesin lokal ini; lakukan uji deploy dan periksa log build pada Streamlit Community Cloud.

---

## 6. Laporan Ukuran Kinerja dan Memori (NFR-02, NFR-03)

Diukur pada mesin pengembangan: **Intel(R) Core(TM) i7-7600U CPU @ 2.80GHz (2 inti, 2 thread PyTorch), RAM 19,8 GB, Windows 11, Python 3.12.5, `torch 2.15.0+cpu`, ukuran batch 16**:

| Skenario                                                                  |                `grid_search` |              `random_search` |                 `optuna_tpe` |               Total Waktu | RSS Proses | Puncak Working Set |
| ------------------------------------------------------------------------- | ---------------------------: | ---------------------------: | ---------------------------: | ------------------------: | ---------: | -----------------: |
| **200 citra × 3 model** (`--n 200`)                                       |  15,23 dtk (13,13 citra/dtk) |  11,64 dtk (17,19 citra/dtk) |  11,03 dtk (18,13 citra/dtk) |             **37,90 dtk** |     480 MB |         **703 MB** |
| **1.600 citra × 3 model** (`--simpan-hasil`, `2026-11-23T11:27:11+07:00`) | 108,70 dtk (14,72 citra/dtk) | 112,67 dtk (14,20 citra/dtk) | 118,74 dtk (13,48 citra/dtk) | **340,10 dtk** (5,67 mnt) |     458 MB |         **706 MB** |
| **Kelima halaman Streamlit + Grad-CAM 8 contoh + 3 model di cache**       |                            — |                            — |                            — |                         — |   620,6 MB |       **848,9 MB** |

Baris **1.600 citra × 3 model (`--simpan-hasil`)** di atas identik dengan metadata pada `artefak/hasil_batch_penuh.csv` (`waktu.grid_search` = 108,70 dtk, `waktu.random_search` = 112,67 dtk, `waktu.optuna_tpe` = 118,74 dtk; status FR-13 `SESUAI` untuk ketiga model, 0 prediksi berbeda). Seluruh pengukuran memori puncak berada di bawah batas 1 GB (NFR-02) dan di bawah ambang 900 MB (Keputusan #5).

---

## 7. Tabel Keterlacakan Spesifikasi (FR dan NFR)

| Kode       | Ringkasan Spesifikasi                                                                                                            | Berkas Implementasi                                                                                                                                        | Berkas & Fungsi Tes (`pytest`)                                                                                                                                                                                                     |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FR-01**  | Pemuatan 3 model (CPU, `weights=None`, `.pth`, `eval()`, `4.336.512` parameter, `strict=True`)                                   | `core/model.py`, `ui/sumber.py`                                                                                                                            | `tests/test_model.py` (`test_muat_model_strict_eval_cpu_dan_jumlah_parameter`, `test_tanpa_bobot_imagenet_dibangun_dengan_weights_none`, `test_kunci_hilang_ditolak`, `test_kunci_berlebih_ditolak`)                               |
| **FR-02**  | Sumber data hanya `data_uji/` (1.600 citra) sesuai urutan `test_manifest_prediksi.csv`; tanpa unggahan                           | `core/data.py`, `pages/2_Prediksi_Batch.py`                                                                                                                | `tests/test_data.py` (`test_data_uji_asli_valid_dan_urutan_sama_dengan_manifest`, `test_jumlah_kurang_menyebut_jumlah_ditemukan`, `test_jumlah_lebih_menyebut_jumlah_ditemukan`)                                                   |
| **FR-03**  | Pengaturan batch: semua / acak n (seed 42) / seimbang n per kelas (seed 42); batch 8/16/32; 1 atau 3 model; sampel identik       | `core/data.py`, `core/batch.py`, `pages/2_Prediksi_Batch.py`                                                                                               | `tests/test_data.py` (`test_acak_deterministik_terurut_dan_sama_dengan_rng_seed_42`, `test_seimbang_n_per_kelas_terurut_dan_deterministik`), `tests/test_hasil_batch.py` (`test_jalankan_batch_sampel_sama_untuk_semua_model`)     |
| **FR-04**  | Praproses identik notebook (`RGB`, `Resize(224,224)`, `ToTensor`, `Normalize` ImageNet)                                          | `core/data.py`                                                                                                                                             | `tests/test_data.py` (`test_praproses_sama_dengan_hitungan_manual_independen`, `test_praproses_sama_dengan_transform_lampiran`)                                                                                                    |
| **FR-05**  | Eksekusi batch: progress bar, estimasi sisa waktu, tombol batalkan, `session_state`, urutan deterministik                        | `core/inference.py`, `core/batch.py`, `pages/2_Prediksi_Batch.py`                                                                                          | `tests/test_inference.py` (`test_urutan_sampel_dipertahankan`, `test_deterministik_dua_kali_jalan`, `test_kemajuan_dipanggil_tiap_batch`), `tests/test_ui_tahap3.py`                                                               |
| **FR-06**  | Tabel prediksi per citra, filter model/kelas/hanya salah, unduh CSV                                                              | `core/hasil.py`, `pages/2_Prediksi_Batch.py`                                                                                                               | `tests/test_hasil_batch.py` (`test_tabel_prediksi_kolom_dan_isi`), `tests/test_ui_tahap3.py` (`test_halaman_prediksi_batch_tanpa_dan_dengan_hasil`)                                                                                |
| **FR-07**  | Metrik per model: Accuracy, F1 macro, presisi/recall/F1 per kelas, confusion matrix angka & peta panas                           | `core/metrics.py`, `ui/tampilan.py`, `pages/3_Evaluasi.py`                                                                                                 | `tests/test_metrics.py` (`test_metrik_contoh_hitung_tangan`), `tests/test_ui_tahap3.py` (`test_halaman_evaluasi_batch_penuh_sesuai_dan_parsial`)                                                                                   |
| **FR-08**  | Perbandingan 3 model & McNemar exact + Holm (aktif hanya bila 3 model pada sampel sama; catatan bila diskordan nol)              | `core/stats.py`, `pages/3_Evaluasi.py`                                                                                                                     | `tests/test_stats.py` (`test_exact_hitung_tangan_1_vs_5`, `test_tanpa_diskordan_p_satu_dengan_catatan`, `test_paritas_dengan_mcnemar_acuan`), `tests/test_ui_tahap3.py`                                                            |
| **FR-09**  | Waktu inferensi & throughput per model beserta catatan perangkat                                                                 | `core/inference.py`, `core/hasil.py`, `pages/2_Prediksi_Batch.py`                                                                                          | `tests/test_inference.py` (`test_waktu_dan_throughput_terukur`), `tests/test_ui_tahap3.py`                                                                                                                                         |
| **FR-10**  | Analisis kesalahan (filter model & pasangan kelas), Grad-CAM 3 model berdampingan, tombol `Muat 8 contoh laporan`                | `core/gradcam.py`, `core/data.py`, `pages/4_Analisis_Kesalahan.py`                                                                                         | `tests/test_gradcam.py` (`test_target_layer_dan_hook_tunggal_tidak_berulang`, `test_delapan_contoh_laporan_cocok_dengan_manifest`), `tests/test_ui_tahap4.py` (`test_halaman_4_tombol_8_contoh_dan_daftar_salah_klasifikasi`)      |
| **FR-11**  | Riwayat pencarian dari artefak: kurva konvergensi, sebaran F1 trial, konfigurasi terbaik, waktu total & rata-rata per trial      | `core/artifacts.py`, `ui/tampilan.py`, `pages/5_Riwayat_dan_MultiSeed.py`                                                                                  | `tests/test_artifacts.py`, `tests/test_ui_tahap4.py` (`test_halaman_5_menampilkan_riwayat_dan_multiseed_dari_artefak`)                                                                                                             |
| **FR-12**  | Multi-seed dari artefak: F1 per seed, mean & std sampel (`ddof=1`), Wilcoxon + Holm, label tidak dihitung ulang                  | `core/artifacts.py`, `ui/tampilan.py`, `pages/5_Riwayat_dan_MultiSeed.py`                                                                                  | `tests/test_ui_tahap4.py` (`test_halaman_5_tidak_mengimpor_pustaka_pelatihan_atau_uji_statistik`, `test_halaman_5_menampilkan_riwayat_dan_multiseed_dari_artefak`)                                                                 |
| **FR-13**  | Verifikasi terhadap `hasil_uji_acuan.json` & `test_manifest_prediksi.csv` (`SESUAI` / `TIDAK SESUAI` beserta rincian)            | `core/metrics.py`, `pages/3_Evaluasi.py`, `tools/verify_parity.py`                                                                                         | `tests/test_metrics.py` (`test_identik_sesuai`, `test_delapan_beda_tanpa_selisih_akurasi_sesuai`, `test_sembilan_beda_tidak_sesuai_dengan_rincian`, `test_selisih_akurasi_empat_citra_tidak_sesuai`)                               |
| **FR-14**  | Ekspor CSV prediksi, CSV metrik, PNG confusion matrix, PNG Grad-CAM, dan PNG grafik                                              | `core/hasil.py`, `core/ekspor.py`, `pages/2_Prediksi_Batch.py`, `pages/3_Evaluasi.py`, `pages/4_Analisis_Kesalahan.py`, `pages/5_Riwayat_dan_MultiSeed.py` | `tests/test_hasil_batch.py` (`test_roundtrip_csv_mempertahankan_semua`), `tests/test_ekspor.py` (`test_metrik_ke_dataframe_dan_csv`, `test_confusion_matrix_png_menghasilkan_gambar_valid`, `test_ekspor_png_gradcam_dan_riwayat`) |
| **FR-15**  | Ringkasan Penelitian: judul, 6 tujuan (1.3), dataset & pembagian, konfigurasi terbaik (`meta.json`), 6 batasan (1.5), pernyataan | `core/teks.py`, `pages/1_Ringkasan.py`                                                                                                                     | `tests/test_ui_tahap3.py` (`test_halaman_ringkasan_memuat_seluruh_unsur_fr15`)                                                                                                                                                     |
| **NFR-01** | Paritas & reproduktibilitas (tes otomatis praproses, metrik hitung tangan, McNemar vs acuan, loader artefak, `verify_parity.py`) | `core/*`, `tools/verify_parity.py`                                                                                                                         | Seluruh berkas di `tests/` (201 tes) + `tools/verify_parity.py` (0 prediksi berbeda dari 1.600 citra)                                                                                                                              |
| **NFR-02** | CPU saja, memori puncak < 1 GB (`torch.no_grad`, `num_workers=0`, tidak menahan seluruh tensor citra)                            | `core/model.py`, `core/inference.py`, `ui/sumber.py`                                                                                                       | `tests/test_model.py`, `tests/test_inference.py`, pengukuran `psutil` (puncak 848,9 MB < 900 MB)                                                                                                                                   |
| **NFR-03** | Ukuran kinerja 200 citra × 3 model dan 1.600 citra × 3 model terukur di mesin pengembangan                                       | `tools/verify_parity.py`                                                                                                                                   | Diukur dan dilaporkan pada Bagian 6 di atas                                                                                                                                                                                        |
| **NFR-04** | Bahasa Indonesia baku dan format angka Indonesia                                                                                 | `core/formatting.py`, `core/teks.py`, `pages/*`                                                                                                            | `tests/test_formatting.py`                                                                                                                                                                                                         |
| **NFR-05** | Validasi masukan spesifik (folder `data_uji/`, berkas bukan gambar, jumlah citra, skema artefak)                                 | `core/artifacts.py`, `core/data.py`, `core/hasil.py`                                                                                                       | `tests/test_artifacts.py`, `tests/test_data.py`, `tests/test_hasil_batch.py`                                                                                                                                                       |
| **NFR-06** | Privasi: tanpa unggahan pengguna, tidak menyimpan citra/hasil pengguna, tanpa log isi citra                                      | `pages/*`, `core/*`                                                                                                                                        | Pemeriksaan kode dan `tests/test_ui_tahap3.py`, `tests/test_ui_tahap4.py`                                                                                                                                                          |
| **NFR-07** | Keterbacaan presentasi: kontras tinggi, tata letak bersih dengan tab & expander, tanpa emoji                                     | `ui/tema.py`, `ui/tampilan.py`, `.streamlit/config.toml`                                                                                                   | `tests/test_ui_tahap3.py`, `tests/test_ui_tahap4.py`                                                                                                                                                                               |
| **NFR-08** | Kualitas kode: `core/` tidak mengimpor `streamlit`, type hints, docstring, tanpa angka hasil di-hardcode, tanpa kode mati        | `core/*`                                                                                                                                                   | `tests/test_ui_tahap3.py` (`test_core_tidak_mengimpor_streamlit`), `tests/test_ui_tahap4.py`                                                                                                                                       |
| **NFR-09** | Deploy: `requirements.txt` terpin dengan indeks CPU PyTorch, `README.md`, tanpa GPU, tanpa koneksi Kaggle                        | `requirements.txt`, `README.md`, `core/model.py`                                                                                                           | `tests/test_model.py` (`test_tanpa_bobot_imagenet_dibangun_dengan_weights_none`)                                                                                                                                                   |
