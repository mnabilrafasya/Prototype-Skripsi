"""Pembangunan dan pemuatan model EfficientNetB0 (FR-01).

Arsitektur identik dengan ``build_model`` di lampiran_kode_inti.ipynb (sel 5),
tetapi tanpa bobot ImageNet (``weights=None``); bobot dimuat dari berkas .pth.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from core.artifacts import Artefak
from core.konstanta import JUMLAH_PARAMETER_MODEL, KELAS, METODE


class ModelError(Exception):
    """Galat pemuatan model; pesan menyebut berkas dan penyebabnya."""


def bangun_model(
    dropout_rate: float, dense_units: int, jumlah_kelas: int = len(KELAS)
) -> nn.Module:
    """EfficientNetB0 (weights=None) dengan classifier head baru, di CPU."""
    model = models.efficientnet_b0(weights=None)
    masukan = model.classifier[1].in_features  # 1280 untuk B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(masukan, dense_units),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(dense_units, jumlah_kelas),
    )
    return model


def hitung_parameter(model: nn.Module) -> int:
    """Total parameter (sama dengan ``count_parameters`` di notebook)."""
    return sum(p.numel() for p in model.parameters())


def muat_model(jalur: Path, dropout_rate: float, dense_units: int) -> nn.Module:
    """Bangun model lalu muat state_dict dengan ``strict=True``; mode eval, CPU.

    Galat bila kunci missing/unexpected atau jumlah parameter bukan
    ``JUMLAH_PARAMETER_MODEL``.
    """
    model = bangun_model(dropout_rate, dense_units)
    try:
        state = torch.load(jalur, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 - torch melempar beragam tipe galat
        raise ModelError(f"{jalur.name}: gagal membaca berkas bobot ({e})") from e
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise ModelError(f"{jalur.name}: kunci state_dict tidak cocok ({e})") from e
    jumlah = hitung_parameter(model)
    if jumlah != JUMLAH_PARAMETER_MODEL:
        raise ModelError(
            f"{jalur.name}: jumlah parameter {jumlah}, seharusnya {JUMLAH_PARAMETER_MODEL}"
        )
    return model.eval()


def muat_semua_model(artefak: Artefak) -> dict[str, nn.Module]:
    """Muat ketiga model dari artefak memakai konfigurasi di meta.json."""
    hasil: dict[str, nn.Module] = {}
    for metode in METODE:
        konfig = artefak.meta.best_configs[metode]
        hasil[metode] = muat_model(
            artefak.bobot[metode],
            float(konfig["dropout_rate"]),
            int(konfig["dense_units"]),
        )
    return hasil
