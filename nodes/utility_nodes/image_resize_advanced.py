# Filename: ComfyUI_OmniSimulator/nodes/utility_nodes/image_resize_advanced.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class OmniSimulator_ImageResizeAdvanced:
    METHODS    = ["bilinear", "bicubic", "nearest", "lanczos", "area"]
    CROP_MODES = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":  ("IMAGE",),
                "width":  ("INT", {"default": 1024, "min": 8, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 8192, "step": 8}),
                "method": (cls.METHODS,    {"default": "bilinear"}),
                "crop":   (cls.CROP_MODES, {"default": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize"
    CATEGORY = "OmniSimulator/Utils"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @staticmethod
    def _center_crop_to_aspect(image: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
        B, H, W, C = image.shape
        target_ratio, current_ratio = target_w / target_h, W / H
        if current_ratio > target_ratio:
            new_w = max(1, int(round(H * target_ratio)))
            x0 = (W - new_w) // 2
            image = image[:, :, x0:x0 + new_w, :]
        elif current_ratio < target_ratio:
            new_h = max(1, int(round(W / target_ratio)))
            y0 = (H - new_h) // 2
            image = image[:, y0:y0 + new_h, :, :]
        return image

    @staticmethod
    def _resize_lanczos_pil(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
        # torch.nn.functional.interpolate has never had a "lanczos" mode in any
        # PyTorch version, so this method is implemented via PIL specifically.
        results = []
        resample = getattr(Image, "LANCZOS", None) or Image.Resampling.LANCZOS
        for i in range(image.shape[0]):
            arr = (image[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr).resize((width, height), resample)
            results.append(torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).unsqueeze(0))
        return torch.cat(results, dim=0)

    def resize(self, image, width, height, method, crop):
        # Tolerant coercion — VALIDATE_INPUTS above already skips ComfyUI's own
        # check, so guard here in case a saved workflow has a stale value.
        method = method if method in self.METHODS else "bilinear"
        crop   = crop   if crop   in self.CROP_MODES else "disabled"
        width, height = max(8, int(width)), max(8, int(height))

        if crop == "center":
            image = self._center_crop_to_aspect(image, width, height)

        if method == "lanczos":
            return (self._resize_lanczos_pil(image, width, height),)

        t = image.permute(0, 3, 1, 2)
        # align_corners is only a valid kwarg for linear/bilinear/bicubic/trilinear;
        # passing it for 'nearest' or 'area' raises a ValueError.
        kwargs = {"align_corners": False} if method in ("bilinear", "bicubic") else {}
        t = F.interpolate(t, size=(height, width), mode=method, **kwargs)
        return (t.permute(0, 2, 3, 1).clamp(0, 1),)


NODE_CLASS_MAPPINGS = {"OmniSimulator_ImageResizeAdvanced": OmniSimulator_ImageResizeAdvanced}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_ImageResizeAdvanced": "OmniSimulator Image Resize Advanced"}