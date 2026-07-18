"""
Direct spectral matching: transfer magnitude spectrum from reference to source.
"""
import numpy as np
from PIL import Image

def direct_spectral_match(img_arr: np.ndarray, ref_arr: np.ndarray = None,
                           strength: float = 0.5, seed: int = None) -> np.ndarray:
    """
    Directly match the Fourier magnitude spectrum of img_arr to ref_arr.
    If no reference, applies 1/f normalization.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros_like(img_arr, dtype=np.float32)
    h, w = img_arr.shape[:2]

    ref_gray = None
    if ref_arr is not None:
        if ref_arr.shape[:2] != (h, w):
            pil = Image.fromarray(ref_arr)
            ref_arr = np.array(pil.resize((w, h), getattr(Image, "BICUBIC", None) or Image.Resampling.BICUBIC))
        ref_gray = np.mean(ref_arr.astype(np.float32), axis=2) if ref_arr.ndim == 3 else ref_arr.astype(np.float32)
        F_ref = np.fft.fft2(ref_gray)
        mag_ref = np.abs(np.fft.fftshift(F_ref))

    for c in range(img_arr.shape[2]):
        channel = img_arr[:, :, c].astype(np.float32)
        F = np.fft.fft2(channel)
        F_shift = np.fft.fftshift(F)
        mag_src = np.abs(F_shift)
        phase = np.angle(F_shift)

        if ref_gray is not None:
            # Transfer reference magnitude
            eps = 1e-8
            scale = (mag_ref + eps) / (mag_src + eps)
            scale = np.clip(scale, 0.1, 10.0)
            new_mag = mag_src * (1 - strength) + mag_src * scale * strength
        else:
            # 1/f normalization
            Y, X = np.mgrid[-h//2:h//2, -w//2:w//2]
            R = np.sqrt(X**2 + Y**2) + 1e-8
            target = 1.0 / R
            target = target / (target.mean() + 1e-8) * mag_src.mean()
            new_mag = mag_src * (1 - strength) + target * strength

        F_new = np.fft.ifftshift(new_mag * np.exp(1j * phase))
        ch_out = np.real(np.fft.ifft2(F_new))
        out[:, :, c] = (1 - strength) * channel + strength * ch_out

    return np.clip(out, 0, 255).astype(np.uint8)