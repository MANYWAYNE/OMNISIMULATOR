# Filename: ComfyUI_OmniSimulator/nodes/utility_nodes/lens_simulation_node.py

import sys as _sys, os as _os
_OmniSimulator_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.realpath(__file__)), '..', '..'))
if _OmniSimulator_ROOT not in _sys.path: _sys.path.insert(0, _OmniSimulator_ROOT)
del _sys, _os, _OmniSimulator_ROOT

import torch
import numpy as np
from PIL import Image

# We will use the camera simulation utility, but only for specific effects
from modules.detection_bypass.camera_pipeline import simulate_camera_pipeline

class OmniSimulator_LensEffects:
    """
    Simulates the subtle, physical imperfections of a real camera lens, such as
    vignetting and chromatic aberration. This should be applied late in the
    post-processing pipeline, before final compression.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vignette_strength": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Strength of the corner darkening effect."
                }),
                "chromatic_aberration": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Strength of the color fringing on high-contrast edges."
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effects"
    CATEGORY = "OmniSimulator/Post-Processing"

    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def apply_effects(self, image: torch.Tensor, vignette_strength: float, chromatic_aberration: float, seed: int):
        if vignette_strength == 0 and chromatic_aberration == 0:
            return (image,)

        print(f"👁️ OmniSimulator Lens Effects: Applying effects.")

        processed_images = []
        for i in range(image.shape[0]):
            numpy_image = self.tensor_to_numpy(image[i:i+1])

            # --- THIS IS THE FIX ---
            # Set iso_scale to 1.0 to prevent the image from turning black.
            # All other unused effects are correctly disabled with 0 or False.
            processed_numpy_image = simulate_camera_pipeline(
                img_arr=numpy_image,
                vignette_strength=vignette_strength,
                chroma_aberr_strength=chromatic_aberration,
                seed=seed + i,
                
                # Neutral / passthrough values for unused effects
                bayer=False,
                jpeg_cycles=0,
                iso_scale=1.0, # <-- THE FIX IS HERE
                read_noise_std=0,
                hot_pixel_prob=0,
                banding_strength=0.0,
                motion_blur_kernel=1
            )
            # --- END FIX ---

            processed_tensor = self.numpy_to_tensor(processed_numpy_image)
            processed_images.append(processed_tensor)

        if not processed_images:
            return (image,)
            
        final_batch = torch.cat(processed_images, dim=0)
        
        print("OmniSimulator Lens Effects: Processing complete.")
        return (final_batch,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "OmniSimulator_LensEffects": OmniSimulator_LensEffects,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniSimulator_LensEffects": "OmniSimulator Lens Effects",
}