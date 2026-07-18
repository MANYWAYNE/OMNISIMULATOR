"""
OmniSimulator JPEG Degradation
Simulates image quality loss via true JPEG re-encoding or a downscale/upscale
detail-loss cycle, with optional artifact softening.

Dependency-free: uses pure torch/PIL for resizing and blurring instead of
comfy.utils.common_upscale (whose 'lanczos' support varies by ComfyUI version)
and kornia, so this node always works regardless of what's installed.
"""
import sys as _sys, os as _os
_OmniSimulator_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.realpath(__file__)), '..', '..'))
if _OmniSimulator_ROOT not in _sys.path: _sys.path.insert(0, _OmniSimulator_ROOT)
del _sys, _os, _OmniSimulator_ROOT

import math
import io
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from modules.detection_bypass.utils.torch_ops import gaussian_blur2d


def _resize(img_bchw: torch.Tensor, w: int, h: int, method: str) -> torch.Tensor:
    """Self-contained resize: F.interpolate for standard modes, PIL for lanczos."""
    if method == "lanczos":
        b, c, _, _ = img_bchw.shape
        out = torch.empty(b, c, h, w, dtype=img_bchw.dtype)
        resample = getattr(Image, "LANCZOS", None) or Image.Resampling.LANCZOS
        for i in range(b):
            arr = (img_bchw[i].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr).resize((w, h), resample)
            out[i] = torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).permute(2, 0, 1)
        return out.to(img_bchw.device)

    kwargs = {"align_corners": False} if method in ("bilinear", "bicubic") else {}
    return F.interpolate(img_bchw, size=(h, w), mode=method, **kwargs)


class OmniSimulator_JPEG_Degradation:
    """
    Applies realistic image degradation through either true JPEG compression
    or a downscale/upscale cycle, with optional artifact softening.
    """

    MODES = ["True JPEG", "Downscale/Upscale"]
    CHROMA_SUBSAMPLING_MODES = ["Standard (4:2:0 - Blotchy Color)", "High Quality (4:4:4)", "Aggressive (4:1:1)"]
    UPSCALE_METHODS = ["lanczos", "bicubic", "bilinear", "nearest"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "True JPEG"}),

                "quality": ("INT", {"default": 60, "min": 1, "max": 100, "step": 1, "tooltip": "Overall JPEG quality (1=worst, 100=best)."}),
                "chroma_subsampling": (cls.CHROMA_SUBSAMPLING_MODES, {"default": "Standard (4:2:0 - Blotchy Color)"}),

                "downscale_factor": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Factor to shrink the image by before upscaling."}),
                "upscale_method": (cls.UPSCALE_METHODS, {"default": "bicubic"}),

                "soften_artifacts": ("BOOLEAN", {"default": False, "label_on": "Soften (Rounder)", "label_off": "Sharp (Blockier)"}),
                "soften_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Amount of blur to apply to soften artifact edges."}),
            },
        }

    CATEGORY = "OmniSimulator/Utils"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "degrade"
    DESCRIPTION = """
# OmniSimulator JPEG Degradation
Simulates image quality loss.
- **True JPEG**: Authentic compression artifacts.
- **Downscale/Upscale**: Simulates resolution loss.
"""

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        image_np = image_tensor.squeeze(0).cpu().numpy()
        image_np = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(image_np, 'RGB')

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def degrade(self, image: torch.Tensor, mode: str, quality: int, chroma_subsampling: str,
                downscale_factor: float, upscale_method: str, soften_artifacts: bool, soften_sigma: float):

        # Tolerant coercion — VALIDATE_INPUTS above skips ComfyUI's own check.
        mode               = mode               if mode               in self.MODES                     else "True JPEG"
        chroma_subsampling = chroma_subsampling  if chroma_subsampling in self.CHROMA_SUBSAMPLING_MODES   else "Standard (4:2:0 - Blotchy Color)"
        upscale_method     = upscale_method      if upscale_method     in self.UPSCALE_METHODS            else "bicubic"
        quality = max(1, min(100, int(quality)))

        original_device = image.device

        if mode == "True JPEG":
            processed_images = []
            subsampling_map = {
                "High Quality (4:4:4)": 0,
                "Standard (4:2:0 - Blotchy Color)": 2,
                "Aggressive (4:1:1)": 1,
            }
            subsampling_val = subsampling_map.get(chroma_subsampling, 2)

            for i in range(image.shape[0]):
                img_pil = self.tensor_to_pil(image[i:i+1])
                buffer = io.BytesIO()
                img_pil.save(buffer, format='JPEG', quality=quality, subsampling=subsampling_val)
                buffer.seek(0)
                reloaded_pil = Image.open(buffer).convert('RGB')
                processed_images.append(self.pil_to_tensor(reloaded_pil))

            final_batch = torch.cat(processed_images, dim=0)

        elif mode == "Downscale/Upscale":
            _b, original_height, original_width, _c = image.shape

            target_w = max(1, int(original_width * downscale_factor))
            target_h = max(1, int(original_height * downscale_factor))

            img_bchw = image.permute(0, 3, 1, 2)

            # "area" is the correct/standard mode for downscaling (native to
            # F.interpolate, no external dependency needed).
            downscaled_bchw = _resize(img_bchw, target_w, target_h, "area")
            upscaled_bchw   = _resize(downscaled_bchw, original_width, original_height, upscale_method)

            final_batch = upscaled_bchw.permute(0, 2, 3, 1).clamp(0, 1)

        else:
            final_batch = image

        if soften_artifacts and soften_sigma > 0:
            final_batch_bchw = final_batch.permute(0, 3, 1, 2)
            kernel_size = 2 * math.ceil(3.0 * soften_sigma) + 1
            blurred_batch = gaussian_blur2d(final_batch_bchw, (kernel_size, kernel_size), (soften_sigma, soften_sigma))
            final_batch = blurred_batch.permute(0, 2, 3, 1)

        return (final_batch.to(original_device).clamp(0, 1),)


NODE_CLASS_MAPPINGS = {
    "OmniSimulator_JPEG_Degradation": OmniSimulator_JPEG_Degradation,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniSimulator_JPEG_Degradation": "OmniSimulator JPEG Degradation",
}