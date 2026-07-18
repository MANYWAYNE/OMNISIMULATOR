import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image
import os

def lbp_normalize(img_arr: np.ndarray,
                  ref_img_arr: np.ndarray = None,
                  profile_path: str = None, # NEW: Added profile_path
                  radius: int = 3,
                  n_points: int = 24,
                  method: str = 'uniform',
                  strength: float = 0.9,
                  seed: int = None,
                  eps: float = 1e-8):
    
    if seed is not None: rng = np.random.default_rng(seed)
    else: rng = np.random.default_rng()

    img = np.asarray(img_arr)
    h, w, channels = img.shape
    
    img_gray = np.mean(img.astype(np.float32), axis=2)
    
    lbp_img = local_binary_pattern(img_gray, n_points, radius, method=method)
    n_bins = int(lbp_img.max() + 1)
    lbp_hist, _ = np.histogram(lbp_img.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    ref_lbp_hist = None
    
    # --- NEW LOGIC: PRIORITIZE PROFILE ---
    if profile_path and profile_path.strip():
        npz_path = f"{profile_path}.npz"
        if os.path.exists(npz_path):
            with np.load(npz_path) as stats:
                if 'lbp_hist' in stats:
                    # Load the average LBP histogram from the profile
                    avg_hist = stats['lbp_hist'].mean(axis=0)
                    # Match the bin count of the source image's LBP
                    if len(avg_hist) > n_bins:
                        ref_lbp_hist = avg_hist[:n_bins]
                    else:
                        ref_lbp_hist = np.pad(avg_hist, (0, n_bins - len(avg_hist)), 'constant')
                    ref_lbp_hist /= (ref_lbp_hist.sum() + eps) # Re-normalize

    # Fallback to live reference image
    if ref_lbp_hist is None and ref_img_arr is not None:
        ref = np.asarray(ref_img_arr)
        if ref.shape[0] != h or ref.shape[1] != w:
            ref = np.array(Image.fromarray(ref).resize((w, h), resample=Image.Resampling.BICUBIC))
        
        ref_gray = np.mean(ref.astype(np.float32), axis=2)
        ref_lbp = local_binary_pattern(ref_gray, n_points, radius, method=method)
        ref_lbp_hist, _ = np.histogram(ref_lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    # --- END NEW LOGIC ---

    if ref_lbp_hist is None:
        return img_arr

    out = np.empty_like(img, dtype=np.float32)
    
    cdf_img = np.cumsum(lbp_hist)
    cdf_ref = np.cumsum(ref_lbp_hist)
    
    mapping = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        # Find the value in reference cdf that is closest to the source cdf value
        j = np.searchsorted(cdf_ref, cdf_img[i], side='left')
        mapping[i] = j
    
    lbp_int = np.rint(lbp_img).astype(np.int32)
    mapping_per_pixel = mapping[lbp_int]
    scale_per_pixel = mapping_per_pixel / (lbp_int.astype(np.float32) + eps)
    
    noise_all = rng.normal(loc=0.0, scale=0.02 * strength, size=(h, w, channels)).astype(np.float32) * 255.0

    for c in range(channels):
        channel = img[:, :, c].astype(np.float32)
        adjusted = channel * scale_per_pixel
        blended = (1.0 - strength) * channel + strength * adjusted
        blended += noise_all[:, :, c]
        out[:, :, c] = blended

    return np.clip(out, 0, 255).astype(np.uint8)