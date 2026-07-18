# ComfyUI_OmniSimulator_NODES/modules/detection_bypass/utils/perturbation.py
import numpy as np

def randomized_perturbation(img_arr: np.ndarray, magnitude_frac: float = 0.008,
                             seed: int = None) -> np.ndarray:
    """
    Apply small uniform random perturbations to each pixel.
    magnitude_frac is the fraction of 255 for the max perturbation.
    """
    rng = np.random.default_rng(seed)
    magnitude = magnitude_frac * 255.0
    noise = rng.uniform(-magnitude, magnitude, size=img_arr.shape).astype(np.float32)
    out = img_arr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)