"""Format angka Indonesia: titik sebagai pemisah ribuan, koma sebagai desimal."""

from __future__ import annotations

import math
from numbers import Real

DESIMAL_METRIK = 4


def format_angka(nilai: Real, desimal: int = 0) -> str:
    """Format bilangan ke gaya Indonesia, mis. 1600 -> '1.600', 0.9812 -> '0,9812'.

    Pembulatan mengikuti ``format`` bawaan Python (nilai biner persis), sama
    dengan ``round`` yang dipakai notebook. Nol negatif hasil pembulatan
    ditulis tanpa tanda minus. Galat bila bool, bukan bilangan, atau tidak hingga.
    """
    if isinstance(nilai, bool) or not isinstance(nilai, Real):
        raise TypeError(f"nilai harus bilangan, bukan {type(nilai).__name__}")
    if isinstance(desimal, bool) or not isinstance(desimal, int) or desimal < 0:
        raise ValueError(f"desimal harus bilangan bulat >= 0, diterima {desimal!r}")
    angka = float(nilai)
    if not math.isfinite(angka):
        raise ValueError(f"nilai harus hingga, diterima {nilai!r}")
    teks = f"{angka:,.{desimal}f}"
    if teks.startswith("-") and float(teks.replace(",", "")) == 0.0:
        teks = teks[1:]
    # Tukar pemisah: koma Inggris -> titik, titik Inggris -> koma (via placeholder).
    return teks.replace(",", "\0").replace(".", ",").replace("\0", ".")


def format_metrik(nilai: Real) -> str:
    """Format metrik dengan 4 desimal, mis. 0.9812 -> '0,9812'."""
    return format_angka(nilai, DESIMAL_METRIK)


def format_angka_ringkas(nilai: Real, maks_desimal: int = 6) -> str:
    """Format dengan paling banyak ``maks_desimal`` desimal; nol di belakang dibuang.

    Untuk nilai hyperparameter, mis. 0.0001 -> '0,0001', 256 -> '256'.
    """
    teks = format_angka(nilai, maks_desimal)
    if "," in teks:
        teks = teks.rstrip("0").rstrip(",")
    return "0" if teks in ("", "-0") else teks


def format_selisih(nilai: Real, desimal: int = DESIMAL_METRIK) -> str:
    """Format selisih bertanda, mis. 0.0025 -> '+0,0025', -0.0044 -> '-0,0044', 0 -> '0,0000'."""
    teks = format_angka(nilai, desimal)
    if teks.startswith("-") or float(teks.replace(".", "").replace(",", ".")) == 0.0:
        return teks
    return "+" + teks


def format_durasi(detik: Real) -> str:
    """Format durasi sebagai 'mm:ss' (atau 'jj:mm:ss' bila lebih dari satu jam)."""
    if isinstance(detik, bool) or not isinstance(detik, Real):
        raise TypeError(f"detik harus bilangan, bukan {type(detik).__name__}")
    if not math.isfinite(float(detik)) or detik < 0:
        raise ValueError(f"detik harus bilangan hingga >= 0, diterima {detik!r}")
    total = int(round(float(detik)))
    jam, sisa = divmod(total, 3600)
    menit, dtk = divmod(sisa, 60)
    return f"{jam:02d}:{menit:02d}:{dtk:02d}" if jam else f"{menit:02d}:{dtk:02d}"
