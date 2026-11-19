"""Tes Grad-CAM (core.gradcam) terhadap model dan citra asli (FR-10)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from core.artifacts import Artefak
from core.data import DataUji, baca_citra, buat_transform, indeks_contoh_laporan
from core.gradcam import GradCAM, hitung_gradcam_citra, overlay_gradcam
from core.konstanta import MEAN_IMAGENET, METODE, STD_IMAGENET
from core.model import muat_model


@pytest.fixture(scope="module")
def model_dan_gradcam(artefak: Artefak) -> dict[str, tuple[nn.Module, GradCAM]]:
    hasil: dict[str, tuple[nn.Module, GradCAM]] = {}
    for m in METODE:
        k = artefak.meta.best_configs[m]
        mdl = muat_model(artefak.bobot[m], float(k["dropout_rate"]), int(k["dense_units"]))
        hasil[m] = (mdl, GradCAM(mdl))
    return hasil


def test_target_layer_dan_hook_tunggal_tidak_berulang(
    model_dan_gradcam: dict[str, tuple[nn.Module, GradCAM]],
    data_uji: DataUji,
    artefak: Artefak,
) -> None:
    tf = buat_transform(
        artefak.meta.img_size, artefak.meta.imagenet_mean, artefak.meta.imagenet_std
    )
    tensor = tf(baca_citra(data_uji.jalur([0])[0]))
    for m, (mdl, gcam) in model_dan_gradcam.items():
        assert gcam.target_layer is mdl.features[-1]
        assert len(mdl.features[-1]._forward_hooks) == 1
        assert len(mdl.features[-1]._backward_hooks) == 1
        # Panggil generate berulang kali: jumlah hook tetap 1
        hitung_gradcam_citra(gcam, m, tensor)
        hitung_gradcam_citra(gcam, m, tensor)
        assert len(mdl.features[-1]._forward_hooks) == 1
        assert len(mdl.features[-1]._backward_hooks) == 1
        # Mendaftarkan instance kedua pada model yang sama harus ditolak
        with pytest.raises(RuntimeError, match="sudah terdaftar"):
            GradCAM(mdl)


def test_paritas_gradcam_dan_overlay_terhadap_logika_lampiran(
    model_dan_gradcam: dict[str, tuple[nn.Module, GradCAM]],
    data_uji: DataUji,
    artefak: Artefak,
) -> None:
    tf = buat_transform(
        artefak.meta.img_size, artefak.meta.imagenet_mean, artefak.meta.imagenet_std
    )
    tensor = tf(baca_citra(data_uji.jalur([137])[0]))
    mdl, gcam = model_dan_gradcam["optuna_tpe"]

    inp = tensor.unsqueeze(0).clone().detach().requires_grad_(True)
    cam, pred = gcam.generate(inp)
    assert cam.shape == (7, 7)
    assert 0.0 <= float(cam.min()) <= float(cam.max()) <= 1.0
    assert pred == int(artefak.manifest["pred_optuna_tpe"].iloc[137])

    img_np, cam_resized, ov = overlay_gradcam(tensor, cam, alpha=0.45)
    assert img_np.shape == (224, 224, 3)
    assert cam_resized.shape == (224, 224)
    assert ov.shape == (224, 224, 3)

    # Hitung manual sesuai lampiran sel 14
    img_show = tensor.clone()
    for c in range(3):
        img_show[c] = img_show[c] * STD_IMAGENET[c] + MEAN_IMAGENET[c]
    manual_img = img_show.permute(1, 2, 0).numpy().clip(0, 1)
    manual_cam = cv2.resize(cam, (224, 224))
    hm = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * manual_cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
    ) / 255.0
    manual_ov = ((1 - 0.45) * manual_img + 0.45 * hm).clip(0, 1)
    assert np.allclose(ov, manual_ov, atol=1e-6)


def test_delapan_contoh_laporan_cocok_dengan_manifest(
    model_dan_gradcam: dict[str, tuple[nn.Module, GradCAM]],
    data_uji: DataUji,
    artefak: Artefak,
) -> None:
    indeks_8 = indeks_contoh_laporan()
    assert indeks_8.tolist() == [137, 1115, 1372, 1044, 691, 700, 142, 1233]
    tf = buat_transform(
        artefak.meta.img_size, artefak.meta.imagenet_mean, artefak.meta.imagenet_std
    )
    for idx in indeks_8:
        t = tf(baca_citra(data_uji.jalur([int(idx)])[0]))
        for m, (_, gcam) in model_dan_gradcam.items():
            h = hitung_gradcam_citra(gcam, m, t)
            assert h.kelas_prediksi == int(artefak.manifest[f"pred_{m}"].iloc[int(idx)])
            assert 0.0 < h.probabilitas_prediksi <= 1.0
            assert np.isclose(h.probabilitas_semua.sum(), 1.0, atol=1e-5)
