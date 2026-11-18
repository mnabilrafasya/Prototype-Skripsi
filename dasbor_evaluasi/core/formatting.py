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
