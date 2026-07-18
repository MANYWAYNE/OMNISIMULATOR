"""
Film grain simulation. Production implementation using numpy.
API-compatible with the filmgrainer library used in pipeline_v2.
"""
import numpy as np
from PIL import Image


def process(file_in: str, file_out: str,
            scale: float = 1.0,
            src_gamma: float = 1.0,
            grain_power: float = 0.5,
            shadows: float = 0.2,
            highs: float = 0.1,
            grain_type: int = 1,
            grain_sat: float = 0.4,
            gray_scale: bool = False,
            sharpen: int = 0,
            seed: int = 0) -> None:
    """
    Add film grain to an image.

    Parameters
    ----------
    file_in     : Input image path
    file_out    : Output image path
    scale       : Scale factor for output (1.0 = original size)
    src_gamma   : Source gamma (1.0 = no adjustment)
    grain_power : Overall grain intensity [0..1]
    shadows     : Grain boost in shadow areas
    highs       : Grain boost in highlight areas
    grain_type  : 1=Gaussian, 2=Uniform
    grain_sat   : Grain saturation (colour grain fraction)
    gray_scale  : If True, produce greyscale output
    sharpen     : Sharpen passes after grain [0..3]
    seed        : Random seed
    """
    rng = np.random.default_rng(int(seed) % (2**32))
    
    pil = Image.open(file_in).convert('RGB')
    
    if scale != 1.0:
        w, h = pil.size
        pil = pil.resize((int(w * scale), int(h * scale)), getattr(Image, "LANCZOS", None) or Image.Resampling.LANCZOS)
    
    arr = np.array(pil).astype(np.float32) / 255.0
    H, W, C = arr.shape

    # Apply source gamma
    if src_gamma != 1.0:
        arr = np.power(np.clip(arr, 0, 1), src_gamma)

    # Luminance map for spatially varying grain
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Shadow/highlight weights
    shadow_w    = np.clip(1.0 - luma, 0, 1)
    highlight_w = np.clip(luma, 0, 1)
    mid_w       = 1.0 - np.abs(luma - 0.5) * 2

    spatial_w = (shadows * shadow_w + highs * highlight_w + 0.5 * mid_w)
    spatial_w = np.clip(spatial_w * grain_power, 0, 1)[:, :, np.newaxis]

    # Generate grain
    sigma = grain_power * 0.04
    if grain_type == 2:
        luma_grain = rng.uniform(-sigma, sigma, (H, W, 1))
    else:
        luma_grain = rng.normal(0, sigma, (H, W, 1))

    if grain_sat > 0:
        chroma_grain = rng.normal(0, sigma * grain_sat * 0.5, (H, W, C))
    else:
        chroma_grain = np.zeros((H, W, C))

    grain = luma_grain + chroma_grain

    # Apply spatially weighted grain
    arr = arr + grain * spatial_w

    # Sharpen
    if sharpen > 0:
        from scipy.ndimage import gaussian_filter
        for _ in range(sharpen):
            blurred = gaussian_filter(arr, sigma=0.5)
            arr = arr + (arr - blurred) * 0.5

    arr = np.clip(arr, 0, 1)

    if gray_scale:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        arr = np.stack([gray, gray, gray], axis=2)

    out = (arr * 255.0).astype(np.uint8)
    Image.fromarray(out, 'RGB').save(file_out)