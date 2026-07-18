import sys as _sys, os as _os
_OmniSimulator_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.realpath(__file__)), '..', '..'))
if _OmniSimulator_ROOT not in _sys.path: _sys.path.insert(0, _OmniSimulator_ROOT)
del _sys, _os, _OmniSimulator_ROOT

import os
import math
import threading
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from modules.detection_bypass.utils.texture_utils import TextureMatcher
from modules.detection_bypass.utils.perceptual_loss import build_perceptual_model

TEXTURE_MATCHER_CACHE = {}


def _gaussian_blur2d(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Pure-torch separable Gaussian blur — drop-in replacement for
    kornia.filters.gaussian_blur2d that needs no extra package.
    x: (B, C, H, W)
    """
    if kernel_size < 1 or sigma <= 0:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma + 1e-8))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    c = x.shape[1]
    kernel = kernel_2d.expand(c, 1, kernel_size, kernel_size).contiguous()
    pad = kernel_size // 2
    return F.conv2d(x, kernel, padding=pad, groups=c)


def _optimization_worker(
    img_tensor_main: torch.Tensor,
    profile_path: str,
    iterations: int,
    learning_rate: float,
    strength: float,
    smoothness: float,   # controls blur sigma
    device: torch.device,
    result_container: list,
    exception_container: list
):
    try:
        if profile_path in TEXTURE_MATCHER_CACHE:
            texture_matcher = TEXTURE_MATCHER_CACHE[profile_path]
        else:
            texture_matcher = TextureMatcher(profile_path, device)
            TEXTURE_MATCHER_CACHE[profile_path] = texture_matcher

        # Real lpips if installed, otherwise a dependency-free built-in proxy —
        # either way this never raises, so the node always runs.
        perceptual_model, backend = build_perceptual_model(device)

        delta = torch.zeros_like(img_tensor_main, requires_grad=True)
        optimizer = optim.Adam([delta], lr=learning_rate)

        print(f"OmniSimulator Texture Engine: Starting optimization for {iterations} steps "
              f"(perceptual backend: {backend})...")
        with torch.no_grad():
            initial_loss = texture_matcher(img_tensor_main, log_stats=True)
            print(f"    - Initial Texture Loss: {initial_loss.item():.4f}")

        for i in range(iterations):
            optimizer.zero_grad()

            # Blur the delta each step so the optimizer only learns smooth,
            # low-frequency patterns instead of pixel-level noise.
            if smoothness > 0:
                kernel_size = 2 * math.ceil(2.0 * smoothness) + 1
                delta_smooth = _gaussian_blur2d(delta, kernel_size, smoothness)
            else:
                delta_smooth = delta

            perturbed_image = torch.clamp(img_tensor_main + delta_smooth, 0.0, 1.0)
            log_this_step = (i == 0) or ((i + 1) % 50 == 0) or (i == iterations - 1)

            loss_texture     = texture_matcher(perturbed_image, log_stats=log_this_step)
            loss_perceptual  = perceptual_model(perturbed_image, img_tensor_main).mean()

            total_loss = (loss_texture * 10.0) + (loss_perceptual * 1.0)

            total_loss.backward()
            optimizer.step()

            if log_this_step:
                print(f"\n  [Step {i+1}/{iterations}]")
                print(f"    - Perceptual({backend}): {loss_perceptual.item():.4f} | Total Loss: {total_loss.item():.4f}")

        with torch.no_grad():
            if smoothness > 0:
                kernel_size = 2 * math.ceil(2.0 * smoothness) + 1
                final_delta = _gaussian_blur2d(delta, kernel_size, smoothness)
            else:
                final_delta = delta

            final_image = torch.clamp(img_tensor_main + (final_delta * strength), 0.0, 1.0)

        result_container.append(final_image)
        print("OmniSimulator Texture Engine: Optimization complete.")

    except Exception as e:
        exception_container.append(e)


class OmniSimulator_Texture_Engine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "profile_base_path": ("STRING", {"forceInput": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "smoothness": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Controls the smoothness/scale of the added texture. 0.5 is fine grain, 2.0 is coarse."
                }),
                "iterations": ("INT", {"default": 200, "min": 50, "max": 1000, "step": 10}),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "OmniSimulator/Authenticity"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def execute(self, image, profile_base_path, strength, smoothness, iterations, learning_rate):
        if strength == 0:
            return (image,)
        if not profile_base_path or not str(profile_base_path).strip():
            raise ValueError("An Authenticity Profile must be connected to 'profile_base_path'.")
        npz_path = f"{profile_base_path}.npz"
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Texture profile not found at: {npz_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processed_batches = []
        for img_tensor in image:
            img_tensor_batch = img_tensor.unsqueeze(0).permute(0, 3, 1, 2).to(device)
            result_container, exception_container = [], []
            # Run in a thread to escape ComfyUI's global torch.no_grad() context,
            # which would otherwise silently disable the gradient-based optimization.
            thread = threading.Thread(
                target=_optimization_worker,
                args=(img_tensor_batch, npz_path, iterations, learning_rate, strength,
                      smoothness, device, result_container, exception_container)
            )
            thread.start()
            thread.join()
            if exception_container:
                raise exception_container[0]
            if not result_container:
                raise RuntimeError("Texture Engine optimization failed to return a result.")
            processed_tensor_bchw = result_container[0]
            processed_tensor_bhwc = processed_tensor_bchw.permute(0, 2, 3, 1).to(image.device)
            processed_batches.append(processed_tensor_bhwc)
        return (torch.cat(processed_batches, dim=0),)


NODE_CLASS_MAPPINGS = {"OmniSimulator_Texture_Engine": OmniSimulator_Texture_Engine}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_Texture_Engine": "OmniSimulator Texture Engine"}