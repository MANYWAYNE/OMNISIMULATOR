import numpy as np
from PIL import Image

def clahe_color_correction(img_arr: np.ndarray, clip_limit: float = 2.0,
                            tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE to the L channel (LAB space). Falls back gracefully without cv2."""
    try:
        import cv2
        lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except ImportError:
        # Fallback: simple histogram equalization on L channel via PIL
        pil = Image.fromarray(img_arr).convert('LAB')
        arr = np.array(pil).astype(np.float32)
        l = arr[:, :, 0]
        l_min, l_max = l.min(), l.max()
        if l_max > l_min:
            arr[:, :, 0] = (l - l_min) / (l_max - l_min) * 255
        return np.array(Image.fromarray(arr.astype(np.uint8), 'LAB').convert('RGB'))
    except Exception:
        return img_arr