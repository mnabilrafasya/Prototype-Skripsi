"""Grad-CAM untuk EfficientNetB0 (FR-10).

Menyalin logika kelas ``GradCAM`` dan fungsi ``overlay_gradcam`` dari
lampiran_kode_inti.ipynb (sel 14):
- target layer default: ``model.features[-1]``
- forward hook untuk aktivasi dan full backward hook untuk gradien
- bobot kanal = rata-rata gradien spasial, jumlahan berbobot, ReLU, dibagi nilai maksimum
- kelas target default = kelas prediksi
- overlay alpha = 0,45 dengan colormap JET OpenCV.
Modul ini tidak mengimpor streamlit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn

from core.konstanta import MEAN_IMAGENET, STD_IMAGENET

ALPHA_OVERLAY = 0.45


class GradCAM:
    """Grad-CAM untuk EfficientNetB0 (identik dengan lampiran_kode_inti.ipynb sel 14).

    Hook hanya didaftarkan satu kali pada saat inisialisasi instance.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.model.eval()
        self.target_layer: nn.Module = (
            target_layer if target_layer is not None else model.features[-1]
        )
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.probabilitas_terakhir: np.ndarray | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        if getattr(self.target_layer, "_gradcam_terdaftar", False):
            raise RuntimeError(
                "Hook GradCAM sudah terdaftar pada target_layer ini; "
                "gunakan satu instance GradCAM per model."
            )

        def forward_hook(
            module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            self.activations = output.detach()

        def backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor | None, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        setattr(self.target_layer, "_gradcam_terdaftar", True)

    def generate(
        self, input_tensor: torch.Tensor, target_class: int | None = None
    ) -> tuple[np.ndarray, int]:
        """Hitung heatmap Grad-CAM (H, W) ternormalisasi [0, 1] dan kelas target.

        ``input_tensor`` berukuran ``(1, 3, H, W)``. Bila ``target_class`` adalah
        ``None``, dipakai kelas prediksi ``output.argmax(dim=1).item()``.
        """
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)

        self.model.zero_grad()
        with torch.enable_grad():
            output = self.model(input_tensor)
            proba = torch.softmax(output.detach(), dim=1)[0].cpu().numpy()
            self.probabilitas_terakhir = proba

            if target_class is None:
                target_class = int(output.argmax(dim=1).item())

            score = output[0, target_class]
            score.backward()

        assert self.gradients is not None and self.activations is not None
        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling pada gradien -> bobot tiap kanal
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Jumlahan berbobot peta aktivasi
        cam = torch.zeros(
            activations.shape[1:], dtype=torch.float32, device=activations.device
        )
        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = torch.relu(cam)
        cam_np = cam.cpu().numpy()

        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        return cam_np, int(target_class)


def overlay_gradcam(
    img_tensor: torch.Tensor,
    cam: np.ndarray,
    alpha: float = ALPHA_OVERLAY,
    mean: Sequence[float] = MEAN_IMAGENET,
    std: Sequence[float] = STD_IMAGENET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Denormalisasi tensor citra ``(3, H, W)`` dan tumpangkan heatmap Grad-CAM.

    Mengembalikan ``(img_np, cam_resized, overlay)`` masing-masing dalam rentang ``[0, 1]``.
    Identik dengan ``overlay_gradcam`` di lampiran_kode_inti.ipynb (sel 14).
    """
    img_show = img_tensor.detach().clone()
    for c in range(3):
        img_show[c] = img_show[c] * float(std[c]) + float(mean[c])
    img_np = img_show.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)

    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = (1.0 - alpha) * img_np + alpha * heatmap
    overlay = overlay.clip(0.0, 1.0)

    return img_np, cam_resized, overlay


@dataclass(frozen=True)
class HasilGradCAM:
    """Hasil Grad-CAM satu model untuk satu citra."""

    metode: str
    kelas_prediksi: int
    probabilitas_prediksi: float
    probabilitas_semua: np.ndarray  # (4,)
    citra_asli: np.ndarray          # (H, W, 3) float [0, 1]
    cam_resized: np.ndarray         # (H, W) float [0, 1]
    overlay: np.ndarray             # (H, W, 3) float [0, 1]


def hitung_gradcam_citra(
    gradcam: GradCAM,
    metode: str,
    img_tensor: torch.Tensor,
    alpha: float = ALPHA_OVERLAY,
) -> HasilGradCAM:
    """Jalankan Grad-CAM satu model pada satu tensor citra ``(3, H, W)``."""
    input_batch = img_tensor.unsqueeze(0).clone().detach().requires_grad_(True)
    cam, pred_class = gradcam.generate(input_batch, target_class=None)
    assert gradcam.probabilitas_terakhir is not None
    proba = gradcam.probabilitas_terakhir.copy()
    img_np, cam_resized, overlay = overlay_gradcam(img_tensor, cam, alpha=alpha)
    return HasilGradCAM(
        metode=metode,
        kelas_prediksi=pred_class,
        probabilitas_prediksi=float(proba[pred_class]),
        probabilitas_semua=proba,
        citra_asli=img_np,
        cam_resized=cam_resized,
        overlay=overlay,
    )
