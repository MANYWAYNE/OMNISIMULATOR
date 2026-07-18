import sys as _sys, os as _os
_OmniSimulator_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.realpath(__file__)), '..', '..'))
if _OmniSimulator_ROOT not in _sys.path: _sys.path.insert(0, _OmniSimulator_ROOT)
del _sys, _os, _OmniSimulator_ROOT

import torch, numpy as np
from modules.detection_bypass.camera_pipeline import simulate_camera_pipeline
class OmniSimulator_Camera_Simulator:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                             "iso": ("INT", {"default": 100, "min": 20, "max": 6400}),
                             "jpeg_quality": ("INT", {"default": 0, "min": 0, "max": 5, "tooltip": "Number of JPEG re-encode cycles. 0=disabled"}),
                             "vignette": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "chromatic_aberration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1})}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "simulate"; CATEGORY = "OmniSimulator/Camera"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def simulate(self, image, seed, iso, jpeg_quality, vignette, chromatic_aberration):
        iso_scale = max(iso / 100.0, 1.0)
        read_noise = (iso / 3200.0) * 3.0
        results = []
        for i in range(image.shape[0]):
            arr = (image[i].cpu().numpy() * 255).astype(np.uint8)
            out = simulate_camera_pipeline(arr, vignette_strength=vignette, chroma_aberr_strength=chromatic_aberration,
                seed=seed+i, jpeg_cycles=jpeg_quality, iso_scale=iso_scale, read_noise_std=read_noise,
                bayer=False)  # bayer demosaic isn't exposed on this node's UI, so keep it off
            results.append(torch.from_numpy(out.astype(np.float32)/255.0).unsqueeze(0))
        return (torch.cat(results, 0),)

NODE_CLASS_MAPPINGS = {"OmniSimulator_Camera_Simulator": OmniSimulator_Camera_Simulator}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_Camera_Simulator": "OmniSimulator Camera Simulator"}