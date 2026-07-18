import numpy as np
import re
import os
from PIL import Image

def apply_1d_lut(img_arr: np.ndarray, lut: np.ndarray, strength: float = 1.0) -> np.ndarray:
    # ... (This function is correct and remains unchanged)
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        raise ValueError("apply_1d_lut expects an HxWx3 image array")
    arr = img_arr.astype(np.float32)
    lut_arr = np.array(lut, dtype=np.float32)
    if lut_arr.ndim == 1:
        lut_arr = np.stack([lut_arr, lut_arr, lut_arr], axis=1)
    if lut_arr.shape[1] != 3:
        raise ValueError("1D LUT must have shape (N,) or (N,3)")
    N = lut_arr.shape[0]
    src_positions = np.linspace(0, 255, N)
    out = np.empty_like(arr)
    for c in range(3):
        channel = arr[..., c].ravel()
        mapped = np.interp(channel, src_positions, lut_arr[:, c])
        out[..., c] = mapped.reshape(arr.shape[0], arr.shape[1])
    out = np.clip(out, 0, 255).astype(np.uint8)
    if strength >= 1.0:
        return out
    else:
        blended = ((1.0 - strength) * img_arr.astype(np.float32) + strength * out.astype(np.float32))
        return np.clip(blended, 0, 255).astype(np.uint8)

def _trilinear_sample_lut(img_float: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Corrected trilinear interpolation for a LUT with shape (S, S, S, 3)."""
    S = lut.shape[0]
    # Scale image color values to LUT index coordinates [0, S-1]
    scaled_coords = img_float * (S - 1)
    
    # Get the integer part (floor) and fractional part of the coordinates
    coords_floor = np.floor(scaled_coords).astype(np.int32)
    coords_frac = scaled_coords - coords_floor
    
    # Clip coordinates to be within the valid LUT index range [0, S-1]
    x0, y0, z0 = np.clip(coords_floor[..., 0], 0, S - 1), np.clip(coords_floor[..., 1], 0, S - 1), np.clip(coords_floor[..., 2], 0, S - 1)
    x1, y1, z1 = np.clip(x0 + 1, 0, S - 1), np.clip(y0 + 1, 0, S - 1), np.clip(z0 + 1, 0, S - 1)

    # Get the 8 corner values from the LUT
    c000 = lut[x0, y0, z0]
    c001 = lut[x0, y0, z1]
    c010 = lut[x0, y1, z0]
    c011 = lut[x0, y1, z1]
    c100 = lut[x1, y0, z0]
    c101 = lut[x1, y0, z1]
    c110 = lut[x1, y1, z0]
    c111 = lut[x1, y1, z1]

    # Expand fractional coordinates for broadcasting
    xd, yd, zd = coords_frac[..., 0, None], coords_frac[..., 1, None], coords_frac[..., 2, None]

    # Perform trilinear interpolation
    c00 = c000 * (1 - zd) + c001 * zd
    c01 = c010 * (1 - zd) + c011 * zd
    c10 = c100 * (1 - zd) + c101 * zd
    c11 = c110 * (1 - zd) + c111 * zd
    
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd
    
    c = c0 * (1 - xd) + c1 * xd
    
    return c

def apply_3d_lut(img_arr: np.ndarray, lut3d: np.ndarray, strength: float = 1.0) -> np.ndarray:
    # ... (This function is correct and remains unchanged)
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        raise ValueError("apply_3d_lut expects an HxWx3 image array")
    img_float = img_arr.astype(np.float32) / 255.0
    sampled = _trilinear_sample_lut(img_float, lut3d)
    out = np.clip(sampled * 255.0, 0, 255).astype(np.uint8)
    if strength >= 1.0:
        return out
    else:
        blended = ((1.0 - strength) * img_arr.astype(np.float32) + strength * out.astype(np.float32))
        return np.clip(blended, 0, 255).astype(np.uint8)

def apply_lut(img_arr: np.ndarray, lut: np.ndarray, strength: float = 1.0) -> np.ndarray:
    # ... (This function is correct and remains unchanged)
    lut = np.array(lut)
    if lut.ndim == 4 and lut.shape[3] == 3:
        if lut.dtype != np.float32 and lut.max() > 1.0:
            lut = lut.astype(np.float32) / 255.0
        return apply_3d_lut(img_arr, lut, strength=strength)
    elif lut.ndim in (1, 2):
        return apply_1d_lut(img_arr, lut, strength=strength)
    else:
        raise ValueError("Unsupported LUT shape: {}".format(lut.shape))

def load_cube_lut(path: str) -> np.ndarray:
    """
    A robust parser for .cube files that handles common format variations and
    correctly orders the axes for lookup.
    """
    size = None
    data = []
    
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.upper().startswith('LUT_3D_SIZE'):
                try:
                    size = int(line.split()[-1])
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid LUT_3D_SIZE format in {path}: {line}")
                continue
            
            match = re.match(r'^(-?\d+(\.\d+)?(e-?\d+)?)\s+(-?\d+(\.\d+)?(e-?\d+)?)\s+(-?\d+(\.\d+)?(e-?\d+)?)$', line)
            if match:
                try:
                    data.append([float(c) for c in line.split()])
                except ValueError:
                    print(f"Warning: Skipping malformed data line in {path}: {line}")
                    continue

    if size is None:
        raise ValueError(f"LUT_3D_SIZE not found in .cube file: {path}")

    expected_size = size ** 3
    if len(data) != expected_size:
        raise ValueError(f"Cube LUT data length does not match size^3 (got {len(data)}, expected {expected_size})")

    # --- THE DEFINITIVE FIX ---
    # Reshape and then transpose the axes to the correct [R, G, B] order for lookup.
    lut = np.array(data, dtype=np.float32).reshape((size, size, size, 3))
    lut = np.transpose(lut, (2, 1, 0, 3))
    # --- END FIX ---
    
    if lut.max() > 1.0 + 1e-6:
        print(f"Warning: LUT values seem to be in [0, 255] range. Normalizing to [0, 1].")
        lut = lut / 255.0
        
    return np.clip(lut, 0.0, 1.0)

def load_lut(path: str) -> np.ndarray:
    # ... (This function is correct and remains unchanged)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    elif ext == '.cube':
        return load_cube_lut(path)
    else:
        try:
            with Image.open(path) as im:
                im = im.convert('RGB')
                arr = np.array(im)
            h, w = arr.shape[:2]
            if (w == 256 and h == 1) or (h == 256 and w == 1):
                lut = arr[0, :, :] if h == 1 else arr[:, 0, :]
                return lut.astype(np.float32)
            flat = arr.reshape(-1, 3).astype(np.float32)
            if flat.shape[0] <= 4096:
                return flat
            raise ValueError("Image LUT not a recognized size")
        except Exception as e:
            raise ValueError(f"Unsupported LUT file or parse error for {path}: {e}")