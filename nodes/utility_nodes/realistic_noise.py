"""
OmniSimulator Realistic Noise
Injects natural-looking camera sensor noise, mimicking modern smartphones.
Softer and color-dependent compared to cinematic grain, more prominent in shadows.

Dependency-free: uses a pure-torch Gaussian blur + grayscale conversion instead
of kornia, so this node always works regardless of what's pip-installed.
"""
import sys as _sys, os as _os
_OmniSimulator_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.realpath(__file__)), '..', '..'))
if _OmniSimulator_ROOT not in _sys.path: _sys.path.insert(0, _OmniSimulator_ROOT)
del _sys, _os, _OmniSimulator_ROOT

import torch

from modules.detection_bypass.utils.torch_ops import gaussian_blur2d, rgb_to_grayscale


class OmniSimulator_RealisticNoise:
    """
    Adds realistic, adjustable camera sensor noise to an image.
    Features separate controls for luma (brightness) and chroma (color) noise,
    blurring for a softer look, and highlights protection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                "luma_intensity": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Intensity of brightness noise."}),
                "chroma_intensity": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Intensity of color noise."}),
                "luma_blur_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Softness of the brightness noise. Higher values are more 'blotchy'."}),
                "chroma_blur_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "Softness of the color noise. Usually higher than luma blur."}),
                "highlights_protection": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Protects brighter areas from noise. 1.0 = full protection, 0.0 = no protection."}),
            },
        }

    CATEGORY = "OmniSimulator/Utils"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_noise"
    DESCRIPTION = """
# OmniSimulator Realistic Noise
Adds natural-looking camera sensor noise, perfect for mimicking smartphone photos.
- **Luma/Chroma Intensity**: Control brightness and color noise separately.
- **Blur Sigma**: Soften the noise to avoid a sharp, 'cinematic' look.
- **Highlights Protection**: Keeps bright areas clean, as they are in real photos.
"""

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def add_noise(self, image: torch.Tensor, seed: int, luma_intensity: float, chroma_intensity: float,
                  luma_blur_sigma: float, chroma_blur_sigma: float, highlights_protection: float):

        device = image.device
        torch.manual_seed(int(seed) % (2**63 - 1))

        batch_size, height, width, _ = image.shape
        image_bchw = image.permute(0, 3, 1, 2).to(device)

        total_noise = torch.zeros_like(image_bchw)

        if luma_intensity > 0:
            luma_noise_chw = torch.randn(batch_size, 1, height, width, device=device).repeat(1, 3, 1, 1)
            if luma_blur_sigma > 0:
                import math
                kernel_size = 2 * math.ceil(3.0 * luma_blur_sigma) + 1
                luma_noise_chw = gaussian_blur2d(luma_noise_chw, (kernel_size, kernel_size), (luma_blur_sigma, luma_blur_sigma))
            total_noise += luma_noise_chw * luma_intensity

        if chroma_intensity > 0:
            chroma_noise_chw = torch.randn(batch_size, 3, height, width, device=device)
            if chroma_blur_sigma > 0:
                import math
                kernel_size = 2 * math.ceil(3.0 * chroma_blur_sigma) + 1
                chroma_noise_chw = gaussian_blur2d(chroma_noise_chw, (kernel_size, kernel_size), (chroma_blur_sigma, chroma_blur_sigma))
            total_noise += chroma_noise_chw * chroma_intensity

        if highlights_protection > 0:
            luminance = rgb_to_grayscale(image_bchw)
            protection_mask = 1.0 - torch.clamp(
                (luminance - highlights_protection) / (1.0 - highlights_protection + 1e-6), 0.0, 1.0)
            total_noise = total_noise * protection_mask

        noisy_image_bchw = torch.clamp(image_bchw + total_noise, 0.0, 1.0)
        final_image = noisy_image_bchw.permute(0, 2, 3, 1).to(image.device)

        return (final_image,)


NODE_CLASS_MAPPINGS = {
    "OmniSimulator_RealisticNoise": OmniSimulator_RealisticNoise,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniSimulator_RealisticNoise": "OmniSimulator Realistic Noise",
}