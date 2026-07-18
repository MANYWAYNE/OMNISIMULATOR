import numpy as np
from PIL import Image

def auto_white_balance_ref(img_arr: np.ndarray, ref_arr: np.ndarray = None) -> np.ndarray:
    """
    Auto white balance. If ref_arr is provided, match channel means to reference.
    Otherwise uses grey-world assumption.
    """
    img = img_arr.astype(np.float32)
    if ref_arr is not None:
        ref = ref_arr.astype(np.float32)
        # Resize ref to match img if needed
        if ref.shape[:2] != img.shape[:2]:
            pil_ref = Image.fromarray(ref_arr)
            pil_ref = pil_ref.resize((img.shape[1], img.shape[0]), getattr(Image, "BICUBIC", None) or Image.Resampling.BICUBIC)
            ref = np.array(pil_ref).astype(np.float32)
        ref_means = ref.reshape(-1, 3).mean(axis=0)
        src_means = img.reshape(-1, 3).mean(axis=0)
    else:
        # Grey-world: normalize each channel to the global mean
        src_means = img.reshape(-1, 3).mean(axis=0)
        target = src_means.mean()
        ref_means = np.array([target, target, target])
    
    scale = ref_means / (src_means + 1e-8)
    scale = np.clip(scale, 0.1, 10.0)
    result = img * scale[np.newaxis, np.newaxis, :]
    return np.clip(result, 0, 255).astype(np.uint8)