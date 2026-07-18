# Production version — all imports wrapped for graceful degradation

def _try(fn, label):
    try:
        return fn()
    except Exception as e:
        print(f"[OmniSimulator] ⚠️  utils.{label} import warning: {e}")
        return None

# ── Core utilities ──────────────────────────────────────────────────────────
_r = _try(lambda: __import__('importlib').import_module('.autowb', package=__name__), 'autowb')
auto_white_balance_ref = getattr(_r, 'auto_white_balance_ref', lambda img, ref=None: img) if _r else lambda img, ref=None: img

_r = _try(lambda: __import__('importlib').import_module('.clahe', package=__name__), 'clahe')
clahe_color_correction = getattr(_r, 'clahe_color_correction', lambda img, **kw: img) if _r else lambda img, **kw: img

_r = _try(lambda: __import__('importlib').import_module('.gaussian_noise', package=__name__), 'gaussian_noise')
add_gaussian_noise = getattr(_r, 'add_gaussian_noise', lambda img, **kw: img) if _r else lambda img, **kw: img

_r = _try(lambda: __import__('importlib').import_module('.perturbation', package=__name__), 'perturbation')
randomized_perturbation = getattr(_r, 'randomized_perturbation', lambda img, **kw: img) if _r else lambda img, **kw: img

_r = _try(lambda: __import__('importlib').import_module('.exif', package=__name__), 'exif')
remove_exif_pil = getattr(_r, 'remove_exif_pil', lambda img: img) if _r else lambda img: img

# ── LUT ─────────────────────────────────────────────────────────────────────
_r = _try(lambda: __import__('importlib').import_module('.color_lut', package=__name__), 'color_lut')
load_lut  = getattr(_r, 'load_lut',  None) if _r else None
apply_lut = getattr(_r, 'apply_lut', None) if _r else None

if load_lut is None:
    # Fallback minimal LUT support
    import numpy as np
    def load_lut(path):
        import re
        data = []
        with open(path, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or line.upper().startswith(('TITLE','LUT_3D_SIZE','DOMAIN')): continue
                parts = line.split()
                if len(parts) >= 3:
                    try: data.append([float(x) for x in parts[:3]])
                    except: pass
        return np.array(data, dtype=np.float32)

    def apply_lut(img_arr, lut, strength=1.0):
        return img_arr

# ── Fourier ──────────────────────────────────────────────────────────────────
_r = _try(lambda: __import__('importlib').import_module('.fourier_pipeline', package=__name__), 'fourier_pipeline')
fourier_match_spectrum = getattr(_r, 'fourier_match_spectrum', lambda img, **kw: img) if _r else lambda img, **kw: img

# ── Texture ───────────────────────────────────────────────────────────────────
_r = _try(lambda: __import__('importlib').import_module('.glcm_normalization', package=__name__), 'glcm_normalization')
glcm_normalize = getattr(_r, 'glcm_normalize', lambda img, **kw: img) if _r else lambda img, **kw: img

_r = _try(lambda: __import__('importlib').import_module('.lbp_normalization', package=__name__), 'lbp_normalization')
lbp_normalize = getattr(_r, 'lbp_normalize', lambda img, **kw: img) if _r else lambda img, **kw: img

_r = _try(lambda: __import__('importlib').import_module('.texture_utils', package=__name__), 'texture_utils')
TextureMatcher = getattr(_r, 'TextureMatcher', None) if _r else None

# ── Blend ─────────────────────────────────────────────────────────────────────
_r = _try(lambda: __import__('importlib').import_module('.blend', package=__name__), 'blend')
blend_colors = getattr(_r, 'blend_colors', lambda img, **kw: img) if _r else lambda img, **kw: img

# ── UnMarker / Spectral ───────────────────────────────────────────────────────
_r = _try(lambda: __import__('importlib').import_module('.non_semantic_attack', package=__name__), 'non_semantic_attack')
non_semantic_attack = getattr(_r, 'non_semantic_attack', lambda img, **kw: img) if _r else lambda img, **kw: img
attack_non_semantic = non_semantic_attack  # alias used by processor.py

_r = _try(lambda: __import__('importlib').import_module('.unmarker_full', package=__name__), 'unmarker_full')
if _r:
    normalize_spectrum_twostage = getattr(_r, 'normalize_spectrum_twostage', lambda img, **kw: img)
    SpectralNormalizer          = getattr(_r, 'SpectralNormalizer', None)
    attack_two_stage_unmarker   = getattr(_r, 'attack_two_stage_unmarker', lambda img, **kw: img)
else:
    normalize_spectrum_twostage = lambda img, **kw: img
    SpectralNormalizer          = None
    attack_two_stage_unmarker   = lambda img, **kw: img

# ── Direct spectral matching ──────────────────────────────────────────────────
_r = _try(lambda: __import__('importlib').import_module('.direct_spectral_matching', package=__name__), 'direct_spectral_matching')
direct_spectral_match = getattr(_r, 'direct_spectral_match', lambda img, **kw: img) if _r else lambda img, **kw: img

# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    "auto_white_balance_ref",
    "clahe_color_correction",
    "load_lut", "apply_lut",
    "remove_exif_pil",
    "fourier_match_spectrum",
    "add_gaussian_noise",
    "randomized_perturbation",
    "glcm_normalize",
    "lbp_normalize",
    "TextureMatcher",
    "blend_colors",
    "non_semantic_attack",
    "attack_non_semantic",
    "normalize_spectrum_twostage",
    "SpectralNormalizer",
    "attack_two_stage_unmarker",
    "direct_spectral_match",
]