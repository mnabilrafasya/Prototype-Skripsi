"""Tes pemuat artefak (core.artifacts) terhadap artefak asli dan salinan yang dirusak."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from core.artifacts import (
    BERKAS_ACUAN,
    BERKAS_BOBOT,
    BERKAS_MANIFEST,
    BERKAS_MCNEMAR,
    BERKAS_META,
    BERKAS_MULTI_SEED,
    BERKAS_TRIAL,
    BERKAS_WAJIB,
    BERKAS_WAKTU,
    BERKAS_WILCOXON,
    ArtefakError,
    ArtefakGagal,
    muat_artefak,
)
from core.hasil import BERKAS_HASIL_BATCH
from core.konstanta import JUMLAH_DATA_UJI, KELAS, METODE

AKAR_PROYEK = Path(__file__).resolve().parents[1]
ARTEFAK_ASLI = AKAR_PROYEK / "artefak"
BERKAS_CSV = [b for b in BERKAS_WAJIB if b.endswith(".csv")]


@pytest.fixture(scope="module")
def artefak_asli():
    return muat_artefak(ARTEFAK_ASLI)


@pytest.fixture
def salinan(tmp_path: Path) -> Path:
    """Salinan artefak asli untuk dirusak. Bobot diganti berkas penanda kecil karena
    pemuat hanya memeriksa keberadaan dan ukuran bobot (isi dimuat di Tahap 2)."""
    tujuan = tmp_path / "artefak"
    shutil.copytree(ARTEFAK_ASLI, tujuan, ignore=shutil.ignore_patterns("bobot"))
    (tujuan / "bobot").mkdir()
    for nama in BERKAS_BOBOT.values():
        (tujuan / nama).write_bytes(b"penanda-uji")
    return tujuan


def _ubah_csv(akar: Path, nama: str, fungsi) -> None:
    df = pd.read_csv(akar / nama)
    hasil = fungsi(df)
    (hasil if hasil is not None else df).to_csv(akar / nama, index=False)


def _ubah_json(akar: Path, nama: str, fungsi) -> None:
    data = json.loads((akar / nama).read_text(encoding="utf-8"))
    fungsi(data)
    (akar / nama).write_text(json.dumps(data), encoding="utf-8")


def _galat(akar: Path) -> list[ArtefakError]:
    with pytest.raises(ArtefakGagal) as info:
        muat_artefak(akar)
    return info.value.galat


def _ada(galat: list[ArtefakError], berkas: str, bagian: str) -> bool:
    return any(g.berkas == berkas and bagian in str(g) for g in galat)


# ------------------------------------------------------------ artefak asli


def test_kontrak_memuat_tiga_belas_berkas_unik_dan_semuanya_ada() -> None:
    assert len(BERKAS_WAJIB) == 13
    assert len(set(BERKAS_WAJIB)) == 13
    hilang = [b for b in BERKAS_WAJIB if not (ARTEFAK_ASLI / b).is_file()]
    assert hilang == []


def test_tidak_ada_berkas_selain_kontrak_di_artefak() -> None:
    ada = {
        p.relative_to(ARTEFAK_ASLI).as_posix()
        for p in ARTEFAK_ASLI.rglob("*")
        if p.is_file()
    }
    assert ada - {BERKAS_HASIL_BATCH} == set(BERKAS_WAJIB)


def test_artefak_asli_valid(artefak_asli) -> None:
    a = artefak_asli
    assert a.meta.classes == KELAS
    assert set(a.bobot) == set(METODE) and all(p.is_file() for p in a.bobot.values())
    assert set(a.trial) == set(METODE)
    assert set(a.acuan) == set(METODE)
    assert len(a.manifest) == JUMLAH_DATA_UJI


def test_jumlah_trial_sama_dengan_waktu_pencarian(artefak_asli) -> None:
    for metode in METODE:
        n = int(artefak_asli.waktu.loc[artefak_asli.waktu["method"] == metode, "n_trials"].iloc[0])
        assert len(artefak_asli.trial[metode]) == n


def test_kolom_konfigurasi_trial_sesuai_meta(artefak_asli) -> None:
    for metode in METODE:
        kolom = {c for c in artefak_asli.trial[metode].columns if c.startswith("config.")}
        assert kolom == {"config." + k for k in artefak_asli.meta.best_configs[metode]}


def test_urutan_baris_manifest_dipertahankan(artefak_asli) -> None:
    with (ARTEFAK_ASLI / BERKAS_MANIFEST).open(encoding="utf-8", newline="") as f:
        mentah = [baris["path"] for baris in csv.DictReader(f)]
    assert artefak_asli.manifest["path"].tolist() == mentah


def test_urutan_baris_trial_dan_multi_seed_dipertahankan(artefak_asli) -> None:
    for metode in METODE:
        with (ARTEFAK_ASLI / BERKAS_TRIAL[metode]).open(encoding="utf-8", newline="") as f:
            mentah = [int(b["trial"]) for b in csv.DictReader(f)]
        assert artefak_asli.trial[metode]["trial"].tolist() == mentah
    with (ARTEFAK_ASLI / BERKAS_MULTI_SEED).open(encoding="utf-8", newline="") as f:
        mentah = [(b["method"], int(b["seed"])) for b in csv.DictReader(f)]
    ms = artefak_asli.multi_seed
    assert list(zip(ms["method"], ms["seed"].astype(int))) == mentah


def test_pasangan_mcnemar_dan_wilcoxon_dipetakan_ke_kunci(artefak_asli) -> None:
    mc = artefak_asli.mcnemar
    wi = artefak_asli.wilcoxon
    harapan = {
        ("grid_search", "random_search"),
        ("grid_search", "optuna_tpe"),
        ("random_search", "optuna_tpe"),
    }
    assert set(zip(mc["metode_a"], mc["metode_b"])) == harapan
    assert set(zip(wi["metode_a"], wi["metode_b"])) == harapan


def test_wilcoxon_menang_kalah_seri_cocok_dengan_teks_berkas(artefak_asli) -> None:
    with (ARTEFAK_ASLI / BERKAS_WILCOXON).open(encoding="utf-8", newline="") as f:
        baris = list(csv.DictReader(f))
    for b, (_, r) in zip(baris, artefak_asli.wilcoxon.iterrows()):
        tiga = [int(x.strip()) for x in b["A menang / B menang / seri"].split("/")]
        assert [r["a_menang"], r["b_menang"], r["seri"]] == tiga


def test_acuan_dibaca_apa_adanya(artefak_asli) -> None:
    mentah = json.loads((ARTEFAK_ASLI / BERKAS_ACUAN).read_text(encoding="utf-8"))
    for metode in METODE:
        a = artefak_asli.acuan[metode]
        assert a.accuracy == mentah[metode]["accuracy"]
        assert a.f1_macro == mentah[metode]["f1_macro"]
        assert a.confusion_matrix.tolist() == mentah[metode]["confusion_matrix"]
        assert a.confusion_matrix.shape == (len(KELAS), len(KELAS))


# ---------------------------------------------------------- berkas hilang


@pytest.mark.parametrize("berkas", BERKAS_WAJIB)
def test_berkas_hilang_disebut_namanya(salinan: Path, berkas: str) -> None:
    (salinan / berkas).unlink()
    galat = _galat(salinan)
    assert _ada(galat, berkas, "tidak ditemukan")


def test_bobot_kosong_ditolak(salinan: Path) -> None:
    (salinan / BERKAS_BOBOT["optuna_tpe"]).write_bytes(b"")
    assert _ada(_galat(salinan), BERKAS_BOBOT["optuna_tpe"], "kosong")


def test_folder_artefak_tidak_ada(tmp_path: Path) -> None:
    with pytest.raises(ArtefakGagal) as info:
        muat_artefak(tmp_path / "tidak_ada")
    assert "tidak ditemukan" in str(info.value)


def test_semua_galat_dikumpulkan_sekaligus(salinan: Path) -> None:
    (salinan / BERKAS_WAKTU).unlink()
    (salinan / BERKAS_WILCOXON).unlink()
    galat = _galat(salinan)
    assert _ada(galat, BERKAS_WAKTU, "tidak ditemukan")
    assert _ada(galat, BERKAS_WILCOXON, "tidak ditemukan")


# ---------------------------------------------------------- kolom CSV


@pytest.mark.parametrize("berkas", BERKAS_CSV)
def test_kolom_hilang_disebut_berkas_dan_kolomnya(salinan: Path, berkas: str) -> None:
    kolom = pd.read_csv(salinan / berkas).columns[0]
    _ubah_csv(salinan, berkas, lambda df: df.rename(columns={kolom: kolom + "_x"}))
    galat = _galat(salinan)
    assert _ada(galat, berkas, f"'{kolom}'")


@pytest.mark.parametrize("berkas", BERKAS_CSV)
def test_kolom_tidak_dikenal_ditolak(salinan: Path, berkas: str) -> None:
    def tambah(df: pd.DataFrame) -> pd.DataFrame:
        df["kolom_asing"] = 1
        return df

    _ubah_csv(salinan, berkas, tambah)
    galat = _galat(salinan)
    assert _ada(galat, berkas, "kolom_asing")
    assert _ada(galat, berkas, "tidak dikenal")


def test_kolom_konfigurasi_trial_harus_sesuai_meta(salinan: Path) -> None:
    berkas = BERKAS_TRIAL["grid_search"]

    def tambah(df: pd.DataFrame) -> pd.DataFrame:
        df["config.asing"] = 1
        return df

    _ubah_csv(salinan, berkas, tambah)
    assert _ada(_galat(salinan), berkas, "config.asing")


def test_kolom_konfigurasi_trial_hilang_dibanding_meta(salinan: Path) -> None:
    berkas = BERKAS_TRIAL["random_search"]
    _ubah_csv(salinan, berkas, lambda df: df.drop(columns=["config.dropout_rate"]))
    assert _ada(_galat(salinan), berkas, "config.dropout_rate")


def test_trial_nilai_tidak_valid(salinan: Path) -> None:
    berkas = BERKAS_TRIAL["optuna_tpe"]

    def rusak(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[3, "val_f1"] = 1.5
        return df

    _ubah_csv(salinan, berkas, rusak)
    assert _ada(_galat(salinan), berkas, "val_f1")


def test_trial_nomor_tidak_berurutan(salinan: Path) -> None:
    berkas = BERKAS_TRIAL["optuna_tpe"]

    def rusak(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[2, "trial"] = 99
        return df

    _ubah_csv(salinan, berkas, rusak)
    assert _ada(_galat(salinan), berkas, "trial")


def test_best_so_far_harus_maksimum_kumulatif(salinan: Path) -> None:
    berkas = BERKAS_TRIAL["grid_search"]

    def rusak(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[len(df) - 1, "best_so_far"] = 0.0
        return df

    _ubah_csv(salinan, berkas, rusak)
    assert _ada(_galat(salinan), berkas, "best_so_far")


def test_n_trials_harus_sama_dengan_baris_trial(salinan: Path) -> None:
    _ubah_csv(salinan, BERKAS_WAKTU, lambda df: df.assign(n_trials=df["n_trials"] + 1))
    assert _ada(_galat(salinan), BERKAS_WAKTU, "n_trials")


def test_waktu_metode_tidak_lengkap(salinan: Path) -> None:
    _ubah_csv(salinan, BERKAS_WAKTU, lambda df: df.iloc[:2])
    assert _ada(_galat(salinan), BERKAS_WAKTU, "method")


# ------------------------------------------------------------ manifest


def test_manifest_jumlah_baris_salah_menyebut_jumlah_ditemukan(salinan: Path) -> None:
    _ubah_csv(salinan, BERKAS_MANIFEST, lambda df: df.iloc[:-1])
    galat = _galat(salinan)
    assert _ada(galat, BERKAS_MANIFEST, str(JUMLAH_DATA_UJI - 1))


def test_manifest_prediksi_di_luar_rentang(salinan: Path) -> None:
    def rusak(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[0, "pred_optuna_tpe"] = 4
        return df

    _ubah_csv(salinan, BERKAS_MANIFEST, rusak)
    assert _ada(_galat(salinan), BERKAS_MANIFEST, "pred_optuna_tpe")


def test_manifest_label_tidak_sesuai_folder(salinan: Path) -> None:
    def rusak(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[0, "label"] = 1
        return df

    _ubah_csv(salinan, BERKAS_MANIFEST, rusak)
    assert _ada(_galat(salinan), BERKAS_MANIFEST, "label")


def test_manifest_path_duplikat(salinan: Path) -> None:
    def rusak(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[1, "path"] = df.loc[0, "path"]
        return df

    _ubah_csv(salinan, BERKAS_MANIFEST, rusak)
    assert _ada(_galat(salinan), BERKAS_MANIFEST, "duplikat")


# ---------------------------------------------------------------- JSON


def test_meta_urutan_kelas_salah(salinan: Path) -> None:
    _ubah_json(salinan, BERKAS_META, lambda d: d.update(classes=list(reversed(d["classes"]))))
    assert _ada(_galat(salinan), BERKAS_META, "classes")


def test_meta_kunci_tidak_dikenal(salinan: Path) -> None:
    _ubah_json(salinan, BERKAS_META, lambda d: d.update(kunci_asing=1))
    assert _ada(_galat(salinan), BERKAS_META, "kunci_asing")


def test_meta_kunci_hilang(salinan: Path) -> None:
    _ubah_json(salinan, BERKAS_META, lambda d: d.pop("seeds_used"))
    assert _ada(_galat(salinan), BERKAS_META, "seeds_used")


def test_meta_img_size_salah(salinan: Path) -> None:
    _ubah_json(salinan, BERKAS_META, lambda d: d.update(img_size=256))
    assert _ada(_galat(salinan), BERKAS_META, "img_size")


def test_meta_mean_salah(salinan: Path) -> None:
    _ubah_json(salinan, BERKAS_META, lambda d: d.update(imagenet_mean=[0.5, 0.5, 0.5]))
    assert _ada(_galat(salinan), BERKAS_META, "imagenet_mean")


def test_meta_konfigurasi_tanpa_dense_units(salinan: Path) -> None:
    _ubah_json(
        salinan, BERKAS_META, lambda d: d["best_configs"]["optuna_tpe"].pop("dense_units")
    )
    assert _ada(_galat(salinan), BERKAS_META, "best_configs.optuna_tpe.dense_units")


def test_meta_dropout_di_luar_rentang(salinan: Path) -> None:
    _ubah_json(
        salinan, BERKAS_META, lambda d: d["best_configs"]["grid_search"].update(dropout_rate=1.2)
    )
    assert _ada(_galat(salinan), BERKAS_META, "best_configs.grid_search.dropout_rate")


def test_meta_metode_tidak_lengkap(salinan: Path) -> None:
    _ubah_json(salinan, BERKAS_META, lambda d: d["best_configs"].pop("random_search"))
    assert _ada(_galat(salinan), BERKAS_META, "best_configs.random_search")


def test_seed_multi_seed_harus_sama_dengan_meta(salinan: Path) -> None:
    def hapus(df: pd.DataFrame) -> pd.DataFrame:
        seed_terakhir = df.loc[df["method"] == "grid_search", "seed"].iloc[-1]
        return df[~((df["method"] == "grid_search") & (df["seed"] == seed_terakhir))]

    _ubah_csv(salinan, BERKAS_MULTI_SEED, hapus)
    assert _ada(_galat(salinan), BERKAS_MULTI_SEED, "seed")


def test_acuan_accuracy_tidak_konsisten_dengan_confusion_matrix(salinan: Path) -> None:
    _ubah_json(
        salinan, BERKAS_ACUAN, lambda d: d["grid_search"]["confusion_matrix"][0].__setitem__(0, 1)
    )
    assert _ada(_galat(salinan), BERKAS_ACUAN, "grid_search.accuracy")


def test_acuan_kunci_f1_per_kelas_hilang(salinan: Path) -> None:
    _ubah_json(salinan, BERKAS_ACUAN, lambda d: d["optuna_tpe"]["f1_per_class"].pop("notumor"))
    assert _ada(_galat(salinan), BERKAS_ACUAN, "optuna_tpe.f1_per_class.notumor")


def test_acuan_metode_tidak_dikenal(salinan: Path) -> None:
    _ubah_json(salinan, BERKAS_ACUAN, lambda d: d.update(metode_asing={}))
    assert _ada(_galat(salinan), BERKAS_ACUAN, "metode_asing")


# ---------------------------------------------------- McNemar dan Wilcoxon


def test_mcnemar_nama_pasangan_tidak_dikenal(salinan: Path) -> None:
    def rusak(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[0, "Pasangan"] = "Grid Search vs Metode Lain"
        return df

    _ubah_csv(salinan, BERKAS_MCNEMAR, rusak)
    assert _ada(_galat(salinan), BERKAS_MCNEMAR, "Pasangan")


def test_mcnemar_jumlah_sel_tidak_sama_dengan_data_uji(salinan: Path) -> None:
    def rusak(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[0, "Sama-sama benar"] += 1
        return df

    _ubah_csv(salinan, BERKAS_MCNEMAR, rusak)
    assert _ada(_galat(salinan), BERKAS_MCNEMAR, "Sama-sama benar")


def test_wilcoxon_format_menang_kalah_seri_salah(salinan: Path) -> None:
    def rusak(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[0, "A menang / B menang / seri"] = "tiga tujuh nol"
        return df

    _ubah_csv(salinan, BERKAS_WILCOXON, rusak)
    assert _ada(_galat(salinan), BERKAS_WILCOXON, "A menang / B menang / seri")


def test_wilcoxon_jumlah_tidak_sama_dengan_jumlah_seed(salinan: Path) -> None:
    def rusak(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[0, "A menang / B menang / seri"] = "1 / 1 / 1"
        return df

    _ubah_csv(salinan, BERKAS_WILCOXON, rusak)
    assert _ada(_galat(salinan), BERKAS_WILCOXON, "jumlah seed")


def test_signifikan_nilai_tidak_valid(salinan: Path) -> None:
    def rusak(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[0, "Signifikan (alpha=0.05)"] = "mungkin"
        return df

    _ubah_csv(salinan, BERKAS_MCNEMAR, rusak)
    assert _ada(_galat(salinan), BERKAS_MCNEMAR, "Signifikan (alpha=0.05)")
