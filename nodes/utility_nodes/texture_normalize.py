import sys as _sys, os as _os
_OmniSimulator_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.realpath(__file__)), '..', '..'))
if _OmniSimulator_ROOT not in _sys.path: _sys.path.insert(0, _OmniSimulator_ROOT)
del _sys, _os, _OmniSimulator_ROOT

import torch
import numpy as np
from PIL import Image
import os
from modules.detection_bypass.utils.glcm_normalization import glcm_normalize
from modules.detection_bypass.utils.lbp_normalization import lbp_normalize

class OmniSimulator_Texture_Base:
    """Base class for texture normalization nodes with shared logic."""
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor is None: return None
        if tensor.ndim == 4: tensor = tensor[0]
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def _numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def _process_batch(self, image: torch.Tensor, strength: float, seed: int, ref_image: torch.Tensor, profile_path: str, process_func, **kwargs):
        if strength == 0:
            return (image,)

        ref_numpy = self._tensor_to_numpy(ref_image) if ref_image is not None else None
        
        # The underlying function will now prioritize the profile path if provided.
        # If both are provided, the profile will be used. If only ref_image is provided, it will be used.
        # If neither is provided, it will pass through.
        
        processed_images = []
        for i in range(image.shape[0]):
            img_numpy = self._tensor_to_numpy(image[i:i+1])
            
            processed_numpy = process_func(
                img_arr=img_numpy,
                ref_img_arr=ref_numpy,
                profile_path=profile_path, # Pass profile path to the core function
                strength=strength,
                seed=seed + i,
                **kwargs
            )
            processed_images.append(self._numpy_to_tensor(processed_numpy))

        return (torch.cat(processed_images, dim=0),)


class OmniSimulator_GLCM_Normalize(OmniSimulator_Texture_Base):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "distances": ("STRING", {"default": "2", "tooltip": "Pixel distance for texture comparison. '2' or '3' is a good start."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {"ref_image": ("IMAGE",), "profile_path": ("STRING", {"forceInput": True}),}
        }
    RETURN_TYPES = ("IMAGE",); FUNCTION = "execute"; CATEGORY = "OmniSimulator/Authenticity"

    def execute(self, image, strength, distances, seed, ref_image=None, profile_path=None):
        try:
            dist_list = [int(d.strip()) for d in distances.split(',')]
        except:
            dist_list = [2]
        return self._process_batch(image, strength, seed, ref_image, profile_path, glcm_normalize, distances=dist_list)

class OmniSimulator_LBP_Normalize(OmniSimulator_Texture_Base):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radius": ("INT", {"default": 3, "min": 1, "max": 10}),
                "n_points": ("INT", {"default": 24, "min": 8, "max": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {"ref_image": ("IMAGE",), "profile_path": ("STRING", {"forceInput": True}),}
        }
    RETURN_TYPES = ("IMAGE",); FUNCTION = "execute"; CATEGORY = "OmniSimulator/Authenticity"

    def execute(self, image, strength, radius, n_points, seed, ref_image=None, profile_path=None):
        return self._process_batch(image, strength, seed, ref_image, profile_path, lbp_normalize, radius=radius, n_points=n_points)
NODE_CLASS_MAPPINGS = {
    "OmniSimulator_GLCM_Normalize": OmniSimulator_GLCM_Normalize,
    "OmniSimulator_LBP_Normalize":  OmniSimulator_LBP_Normalize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniSimulator_GLCM_Normalize": "OmniSimulator GLCM Normalize",
    "OmniSimulator_LBP_Normalize":  "OmniSimulator LBP Normalize",
}