import numpy as np
import torch

try:
    from .non_semantic_attack import non_semantic_attack
    _HAS_NSA = True
except Exception as _e:
    print(f"[OmniSimulator] unmarker_full: non_semantic_attack not available: {_e}")
    _HAS_NSA = False

PRESETS = {
    "fast":     {"iterations": 150,  "learning_rate": 4e-4, "t_lpips": 0.06, "t_l2": 5e-5},
    "balanced": {"iterations": 500,  "learning_rate": 3e-4, "t_lpips": 0.04, "t_l2": 3e-5},
    "quality":  {"iterations": 1000, "learning_rate": 2e-4, "t_lpips": 0.02, "t_l2": 1e-5},
}


def attack_two_stage_unmarker(img_np: np.ndarray, preset: str = "balanced",
                               verbose: bool = False, seed: int = None) -> np.ndarray:
    """
    Two-stage UnMarker spectral attack.
    Stage 1: Fast spectral normalization (low iterations)
    Stage 2: Fine-tune with quality preset
    """
    if not _HAS_NSA:
        print("[OmniSimulator] UnMarker not available. Returning original image.")
        return img_np

    cfg = PRESETS.get(preset, PRESETS["balanced"])
    if verbose:
        print(f"[OmniSimulator] Running UnMarker (preset={preset}, iters={cfg['iterations']})")

    if preset in ("balanced", "quality"):
        # Stage 1: fast pass
        stage1_cfg = {**PRESETS["fast"], "print_log_every_n": 50 if verbose else 0, "seed": seed}
        stage1 = non_semantic_attack(img_np, **stage1_cfg)
        # Stage 2: refinement
        stage2_cfg = {**cfg, "print_log_every_n": 100 if verbose else 0, "seed": seed}
        return non_semantic_attack(stage1, **stage2_cfg)
    else:
        cfg_run = {**cfg, "print_log_every_n": 50 if verbose else 0, "seed": seed}
        return non_semantic_attack(img_np, **cfg_run)


def normalize_spectrum_twostage(img_np: np.ndarray, strength: float = 0.5,
                                 seed: int = None) -> np.ndarray:
    """
    Simpler two-stage spectral normalization using Fourier domain.
    Works without lpips as dependency.
    """
    try:
        from .fourier_pipeline import fourier_match_spectrum
    except Exception:
        return img_np

    # Stage 1: aggressive normalization
    out = fourier_match_spectrum(img_np, mode='model', strength=strength * 0.6,
                                  randomness=0.1, phase_perturb=0.05, seed=seed)
    # Stage 2: fine pass
    out = fourier_match_spectrum(out, mode='model', strength=strength * 0.4,
                                  randomness=0.05, phase_perturb=0.02, seed=seed)
    return out


class SpectralNormalizer:
    """
    High-level interface for spectral normalization.
    Wraps both Fourier-domain and gradient-based attacks.
    """
    def __init__(self, method: str = "fourier", device: torch.device = None):
        self.method = method
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def normalize(self, img_np: np.ndarray, strength: float = 0.5,
                  preset: str = "balanced", seed: int = None,
                  verbose: bool = False) -> np.ndarray:
        if self.method == "gradient" and _HAS_NSA:
            return attack_two_stage_unmarker(img_np, preset=preset, verbose=verbose, seed=seed)
        else:
            return normalize_spectrum_twostage(img_np, strength=strength, seed=seed)

    def __call__(self, img_np: np.ndarray, **kwargs) -> np.ndarray:
        return self.normalize(img_np, **kwargs)