import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import gaussian_filter
from PIL import Image
import os

def glcm_normalize(img_arr: np.ndarray,
                   ref_img_arr: np.ndarray = None,
                   profile_path: str = None, # NEW: Added profile_path
                   distances: list = [1],
                   angles: list = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                   levels: int = 256,
                   strength: float = 0.9,
                   seed: int = None,
                   eps: float = 1e-8):
    
    if seed is not None: rng = np.random.default_rng(seed)
    else: rng = np.random.default_rng()

    img = np.asarray(img_arr)
    h, w, channels = img.shape
    
    img_gray = np.mean(img.astype(np.float32), axis=2)
    img_gray_u8 = np.clip(img_gray, 0, 255).astype(np.uint8)

    glcm = graycomatrix(img_gray_u8, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    ref_contrast, ref_homogeneity = None, None

    # --- NEW LOGIC: PRIORITIZE PROFILE ---
    if profile_path and profile_path.strip():
        npz_path = f"{profile_path}.npz"
        if os.path.exists(npz_path):
            with np.load(npz_path) as stats:
                if 'glcm_props' in stats:
                    # Load the average contrast and homogeneity from the profile
                    avg_props = stats['glcm_props'].mean(axis=0)
                    ref_contrast, ref_homogeneity = avg_props[0], avg_props[1]
    
    # Fallback to live reference image if no profile was used
    if ref_contrast is None and ref_img_arr is not None:
        ref = np.asarray(ref_img_arr)
        if ref.shape[0] != h or ref.shape[1] != w:
            ref = np.array(Image.fromarray(ref).resize((w, h), resample=Image.Resampling.BICUBIC))
        
        ref_gray = np.mean(ref.astype(np.float32), axis=2)
        ref_gray_u8 = np.clip(ref_gray, 0, 255).astype(np.uint8)
        
        ref_glcm = graycomatrix(ref_gray_u8, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
        ref_contrast = graycoprops(ref_glcm, 'contrast').mean()
        ref_homogeneity = graycoprops(ref_glcm, 'homogeneity').mean()
    # --- END NEW LOGIC ---

    if ref_contrast is None:
        return img_arr # Do nothing if no reference is available

    out = np.empty_like(img, dtype=np.float32)
    
    contrast_ratio = ref_contrast / (contrast + eps)
    homogeneity_ratio = ref_homogeneity / (homogeneity + eps)
    contrast_scale = np.sqrt(contrast_ratio).astype(np.float32)
    sigma = float(np.clip(1.0 / (homogeneity_ratio + eps), 0.5, 5.0))

    noise_all = rng.normal(loc=0.0, scale=0.02 * strength, size=(h, w, channels)).astype(np.float32) * 255.0

    for c in range(channels):
        channel = img[:, :, c].astype(np.float32)
        adjusted = channel * contrast_scale
        adjusted = gaussian_filter(adjusted, sigma=(sigma, sigma))
        blended = (1.0 - strength) * channel + strength * adjusted
        blended += noise_all[:, :, c]
        out[:, :, c] = blended

    return np.clip(out, 0, 255).astype(np.uint8)