# Filename: ComfyUI_OmniSimulator/nodes/utility_nodes/blend_colors.py
import sys as _sys, os as _os
_OmniSimulator_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.realpath(__file__)), '..', '..'))
if _OmniSimulator_ROOT not in _sys.path: _sys.path.insert(0, _OmniSimulator_ROOT)
del _sys, _os, _OmniSimulator_ROOT

import torch
import numpy as np
from modules.detection_bypass.utils.blend import blend_colors

class OmniSimulator_BlendColors:
    """
    Applies a color quantization and randomization effect by clustering similar
    colors. This can help break up the smooth, perfect gradients often found
    in AI-generated images, adding a layer of statistical noise.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Bypassed"}),
                "tolerance": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Color tolerance for clustering. Smaller values = more colors, larger values = fewer colors."
                }),
                "min_region_size": ("INT", {
                    "default": 50, "min": 1, "max": 10000,
                    "tooltip": "Minimum size in pixels for a color region to be retained."
                }),
                "max_kmeans_samples": ("INT", {
                    "default": 100000, "min": 1000, "max": 1000000,
                    "tooltip": "Maximum number of pixels to sample for the initial color clustering (for performance)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "OmniSimulator/Authenticity"

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        # Convert tensor [0, 1] float to numpy [0, 255] uint8
        if tensor.ndim == 4:
            tensor = tensor[0]
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def _numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        # Convert numpy [0, 255] uint8 back to tensor [0, 1] float
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def execute(self, image: torch.Tensor, enabled: bool, tolerance: float, min_region_size: int, max_kmeans_samples: int):
        if not enabled:
            return (image,)

        processed_images = []
        for i in range(image.shape[0]):
            img_numpy = self._tensor_to_numpy(image[i:i+1])
            
            # Call the parallelized blend_colors function from our modules
            processed_numpy = blend_colors(
                image=img_numpy,
                tolerance=tolerance,
                min_region_size=min_region_size,
                max_kmeans_samples=max_kmeans_samples
            )
            
            processed_images.append(self._numpy_to_tensor(processed_numpy))

        if not processed_images:
            return (image,)

        return (torch.cat(processed_images, dim=0),)
NODE_CLASS_MAPPINGS = {"OmniSimulator_BlendColors": OmniSimulator_BlendColors}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_BlendColors": "OmniSimulator Blend Colors"}