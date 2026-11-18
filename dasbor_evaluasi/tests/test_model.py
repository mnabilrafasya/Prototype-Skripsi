"""Tes pemuatan model (FR-01) terhadap bobot asli."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from core.artifacts import Artefak
from core.konstanta import JUMLAH_PARAMETER_MODEL, METODE
from core.model import ModelError, bangun_model, hitung_parameter, muat_model


@pytest.fixture(scope="module")
def model_grid(artefak: Artefak) -> nn.Module:
    k = artefak.meta.best_configs["grid_search"]
    return muat_model(artefak.bobot["grid_search"], float(k["dropout_rate"]), int(k["dense_units"]))


@pytest.mark.parametrize("metode", METODE)
def test_muat_model_strict_eval_cpu_dan_jumlah_parameter(artefak: Artefak, metode: str) -> None:
    k = artefak.meta.best_configs[metode]
    model = muat_model(artefak.bobot[metode], float(k["dropout_rate"]), int(k["dense_units"]))
    assert not model.training
    assert hitung_parameter(model) == JUMLAH_PARAMETER_MODEL
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_struktur_head_sesuai_spesifikasi(artefak: Artefak) -> None:
    k = artefak.meta.best_configs["optuna_tpe"]
    model = bangun_model(float(k["dropout_rate"]), int(k["dense_units"]))
    d1, l1, r, d2, l2 = model.classifier
    assert isinstance(d1, nn.Dropout) and d1.p == k["dropout_rate"]
    assert isinstance(l1, nn.Linear) and (l1.in_features, l1.out_features) == (1280, k["dense_units"])
    assert isinstance(r, nn.ReLU)
    assert isinstance(d2, nn.Dropout) and d2.p == k["dropout_rate"]
    assert isinstance(l2, nn.Linear) and (l2.in_features, l2.out_features) == (k["dense_units"], 4)


def test_tanpa_bobot_imagenet_dibangun_dengan_weights_none(monkeypatch: pytest.MonkeyPatch) -> None:
    dipanggil: dict = {}
    import torchvision.models as tm

    asli = tm.efficientnet_b0

    def spion(*a, **kw):
        dipanggil.update(kw)
        return asli(*a, **kw)

    monkeypatch.setattr("core.model.models.efficientnet_b0", spion)
    bangun_model(0.3, 256)
    assert dipanggil == {"weights": None}


def test_keluaran_deterministik_di_mode_eval(model_grid: nn.Module) -> None:
    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        a, b = model_grid(x), model_grid(x)
    assert a.shape == (2, 4)
    assert torch.equal(a, b)


def _simpan_state(tmp_path: Path, model: nn.Module, ubah) -> Path:
    state = {k: v.clone() for k, v in model.state_dict().items()}
    ubah(state)
    jalur = tmp_path / "rusak.pth"
    torch.save(state, jalur)
    return jalur


def test_kunci_hilang_ditolak(tmp_path: Path, model_grid: nn.Module) -> None:
    jalur = _simpan_state(tmp_path, model_grid, lambda s: s.pop("classifier.4.bias"))
    with pytest.raises(ModelError, match="rusak.pth.*classifier.4.bias"):
        muat_model(jalur, 0.5, 256)


def test_kunci_berlebih_ditolak(tmp_path: Path, model_grid: nn.Module) -> None:
    jalur = _simpan_state(tmp_path, model_grid, lambda s: s.update({"kunci_asing": torch.zeros(1)}))
    with pytest.raises(ModelError, match="kunci_asing"):
        muat_model(jalur, 0.5, 256)


def test_dense_units_tidak_cocok_ditolak(artefak: Artefak) -> None:
    with pytest.raises(ModelError, match="grid_search.pth"):
        muat_model(artefak.bobot["grid_search"], 0.5, 128)


def test_berkas_bukan_bobot_ditolak(tmp_path: Path) -> None:
    jalur = tmp_path / "bukan_bobot.pth"
    jalur.write_bytes(b"bukan bobot")
    with pytest.raises(ModelError, match="bukan_bobot.pth"):
        muat_model(jalur, 0.5, 256)
