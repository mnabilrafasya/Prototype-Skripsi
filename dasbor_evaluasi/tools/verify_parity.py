"""Verifikasi paritas: prediksi aplikasi terhadap artefak notebook.

Memuat data uji bawaan, menjalankan ketiga model (CPU), lalu membandingkan dengan
``test_manifest_prediksi.csv`` dan ``hasil_uji_acuan.json`` (dan ``mcnemar_acuan.csv``),
dan mencetak tabel selisih. Status FR-13 hanya bermakna pada batch penuh.

Contoh (dari folder dasbor_evaluasi/):
    .venv\\Scripts\\python.exe tools\\verify_parity.py
    .venv\\Scripts\\python.exe tools\\verify_parity.py --n 200      # pengukuran waktu, parsial
Kode keluar: 0 bila semua SESUAI (batch penuh) atau parsial; 1 bila ada TIDAK SESUAI.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # agar paket core dapat diimpor saat dijalankan sebagai skrip

import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

from core.artifacts import ArtefakGagal, muat_artefak  # noqa: E402
from core.data import (  # noqa: E402
    MODE_ACAK,
    MODE_SEMUA,
    DataUjiError,
    buat_transform,
    muat_data_uji,
    pilih_indeks,
)
from core.inference import UKURAN_BATCH_DIIZINKAN, jalankan_inferensi  # noqa: E402
from core.konstanta import KELAS, METODE  # noqa: E402
from core.metrics import STATUS_SESUAI, bandingkan_dengan_acuan, hitung_metrik  # noqa: E402
from core.model import ModelError, hitung_parameter, muat_model  # noqa: E402
from core.stats import mcnemar_tiga_pasangan  # noqa: E402


def _memori_mb() -> tuple[float, float]:
    """(rss saat ini, puncak working set bila tersedia) dalam MB."""
    info = psutil.Process().memory_info()
    puncak = getattr(info, "peak_wset", info.rss)
    return info.rss / 2**20, puncak / 2**20


def _cetak(judul: str, df: pd.DataFrame) -> None:
    print(f"\n{judul}")
    print(df.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--artefak", type=Path, default=ROOT / "artefak")
    ap.add_argument("--data-uji", type=Path, default=ROOT / "data_uji")
    ap.add_argument("--ukuran-batch", type=int, default=16, choices=UKURAN_BATCH_DIIZINKAN)
    ap.add_argument("--n", type=int, default=None, help="sampel acak (seed 42); kosong = semua")
    args = ap.parse_args(argv)

    try:
        artefak = muat_artefak(args.artefak)
        data = muat_data_uji(args.data_uji, artefak.manifest)
    except (ArtefakGagal, DataUjiError) as e:
        print(f"GALAT: {e}", file=sys.stderr)
        return 2

    penuh = args.n is None
    indeks = pilih_indeks(data.label, MODE_SEMUA if penuh else MODE_ACAK, args.n)
    y_true = data.label[indeks]
    jalur = data.jalur(indeks)
    nama = [data.path[int(i)] for i in indeks]
    transform = buat_transform(
        artefak.meta.img_size, artefak.meta.imagenet_mean, artefak.meta.imagenet_std
    )

    print(f"Torch {torch.__version__}, perangkat CPU, threads {torch.get_num_threads()}")
    print(f"Sampel: {len(indeks)} ({'batch penuh' if penuh else 'PARSIAL, bukan dasar status FR-13'}), "
          f"ukuran batch {args.ukuran_batch}")

    # FR-01: muat ketiga model (strict, weights=None); gagal -> galat spesifik.
    model, baris_muat = {}, []
    for m in METODE:
        konfig = artefak.meta.best_configs[m]
        t0 = time.perf_counter()
        try:
            model[m] = muat_model(
                artefak.bobot[m], float(konfig["dropout_rate"]), int(konfig["dense_units"])
            )
        except ModelError as e:
            print(f"GALAT: {e}", file=sys.stderr)
            return 2
        baris_muat.append(
            {
                "metode": m,
                "dropout_rate": konfig["dropout_rate"],
                "dense_units": konfig["dense_units"],
                "parameter": hitung_parameter(model[m]),
                "strict_load": "berhasil",
                "mode_eval": not model[m].training,
                "waktu_muat_detik": round(time.perf_counter() - t0, 3),
            }
        )
    _cetak("FR-01 Pemuatan model", pd.DataFrame(baris_muat))

    hasil, baris_waktu = {}, []
    for m in METODE:
        def kemajuan(sel: int, tot: int, m: str = m) -> None:
            if sel % 160 < args.ukuran_batch or sel == tot:
                print(f"  {m}: {sel}/{tot}", flush=True)

        h = jalankan_inferensi(model[m], jalur, transform, args.ukuran_batch, nama, kemajuan)
        hasil[m] = h
        baris_waktu.append(
            {
                "metode": m,
                "n": h.n_selesai,
                "waktu_total_detik": round(h.waktu_total_detik, 2),
                "waktu_forward_detik": round(h.waktu_forward_detik, 2),
                "citra_per_detik": round(h.throughput, 2),
            }
        )

    manifest = artefak.manifest
    barisan, f1k, detail = [], [], []
    app_pred = {m: hasil[m].prediksi for m in METODE}
    semua_sesuai = True
    for m in METODE:
        metrik = hitung_metrik(y_true, app_pred[m])
        acuan = artefak.acuan[m]
        pm = manifest[f"pred_{m}"].to_numpy()[indeks]
        pb = bandingkan_dengan_acuan(metrik, acuan, app_pred[m], pm)
        semua_sesuai &= pb.status == STATUS_SESUAI
        barisan.append(
            {
                "metode": m,
                "acc_aplikasi": round(metrik.accuracy, 6),
                "acc_acuan": round(acuan.accuracy, 6),
                "selisih_acc": round(pb.selisih_akurasi, 6),
                "f1_aplikasi": round(metrik.f1_macro, 6),
                "f1_acuan": round(acuan.f1_macro, 6),
                "selisih_f1": round(pb.selisih_f1_macro, 6),
                "pred_berbeda": pb.prediksi_berbeda,
                "cm_sama": pb.confusion_matrix_sama,
                "status": pb.status if penuh else "(parsial)",
            }
        )
        for k, kelas in enumerate(KELAS):
            f1k.append(
                {
                    "metode": m,
                    "kelas": kelas,
                    "f1_aplikasi": round(float(metrik.per_kelas.loc[kelas, "f1"]), 6),
                    "f1_acuan": round(acuan.f1_per_class[kelas], 6),
                    "selisih": round(float(metrik.per_kelas.loc[kelas, "f1"]) - acuan.f1_per_class[kelas], 6),
                }
            )
        for i in pb.indeks_berbeda:
            detail.append(
                {
                    "metode": m,
                    "indeks_manifest": int(indeks[i]),
                    "berkas": nama[i],
                    "label": KELAS[int(y_true[i])],
                    "manifest": KELAS[int(pm[i])],
                    "aplikasi": KELAS[int(app_pred[m][i])],
                }
            )
        if penuh and pb.rincian:
            print(f"\nTIDAK SESUAI ({m}): " + "; ".join(pb.rincian))

    _cetak("FR-13 Metrik aplikasi vs hasil_uji_acuan.json dan test_manifest_prediksi.csv", pd.DataFrame(barisan))
    _cetak("F1-score per kelas: aplikasi vs acuan", pd.DataFrame(f1k))
    if detail:
        _cetak("Prediksi yang berbeda dari test_manifest_prediksi.csv", pd.DataFrame(detail))
    else:
        print("\nPrediksi yang berbeda dari test_manifest_prediksi.csv: tidak ada")

    mc = mcnemar_tiga_pasangan(y_true, app_pred)
    ref = artefak.mcnemar.set_index(["metode_a", "metode_b"])
    baris_mc = []
    for _, r in mc.iterrows():
        a = ref.loc[(r["metode_a"], r["metode_b"])]
        baris_mc.append(
            {
                "pasangan": r["pasangan"],
                "n": r["n"],
                "hanya_a": f"{r['hanya_a']} (acuan {a['hanya_a']})",
                "hanya_b": f"{r['hanya_b']} (acuan {a['hanya_b']})",
                "p_aplikasi": round(r["p_value"], 4),
                "p_acuan": a["p_value"],
                "p_holm_aplikasi": round(r["p_holm"], 4),
                "p_holm_acuan": a["p_holm"],
            }
        )
    _cetak("McNemar exact + Holm dari prediksi aplikasi vs mcnemar_acuan.csv", pd.DataFrame(baris_mc))

    _cetak("FR-09 Waktu inferensi di mesin ini (bukan waktu pencarian di laporan)", pd.DataFrame(baris_waktu))
    rss, puncak = _memori_mb()
    print(f"\nMemori proses: rss {rss:.0f} MB, puncak working set {puncak:.0f} MB (tiga model di memori)")

    if penuh:
        print("\nRINGKASAN FR-13: " + ("SESUAI untuk semua model" if semua_sesuai else "ADA MODEL TIDAK SESUAI"))
        return 0 if semua_sesuai else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
