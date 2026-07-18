import dataclasses
import os
import json
import torch
import tempfile
import time
import gc
from PIL import Image
import numpy as np

# We will import your actual processing functions here later
# from .processor import process_image 

# --- Helper functions (from your original node) ---
def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)

def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


@dataclasses.dataclass
class BypassConfig:
    """A clean, type-safe container for all our settings."""
    mode: str = "Balanced"
    unmarker_version: str = "full_balanced"  # none, simplified, full_fast, full_balanced, full_quality
    strength: float = 0.25
    profile_name: str = "Sony_A7IV_Natural"
    seed: int = 0
    debug_mode: bool = False

class BypassPipeline:
    def __init__(self, config: BypassConfig):
        self.config = config
        self.profile_dir = os.path.join(os.path.dirname(__file__), "profiles")
        self.fingerprint = self._load_profile(config.profile_name)
        print(f"✅ BypassPipeline initialized for '{config.mode}' mode with '{config.profile_name}' profile.")

    def _load_profile(self, name: str) -> dict:
        """Loads the statistical fingerprint from a JSON file."""
        profile_path = os.path.join(self.profile_dir, f"{name}.json")
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Fingerprint profile not found: {profile_path}")
        with open(profile_path, 'r') as f:
            return json.load(f)

    def _analyze_colors(self, tensor: torch.Tensor, label: str):
        """Analyze color statistics for debugging."""
        if not self.config.debug_mode:
            return

        # Convert to numpy for analysis
        img_np = (tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        # Compute channel means
        r_mean = img_np[:, :, 0].mean()
        g_mean = img_np[:, :, 1].mean()
        b_mean = img_np[:, :, 2].mean()

        # Compute channel stds
        r_std = img_np[:, :, 0].std()
        g_std = img_np[:, :, 1].std()
        b_std = img_np[:, :, 2].std()

        # RGB balance (should be close to 1.0 if balanced)
        r_g_ratio = r_mean / (g_mean + 1e-6)
        b_g_ratio = b_mean / (g_mean + 1e-6)

        print(f"\n  🔍 [{label}] Color Analysis:")
        print(f"     R: mean={r_mean:.1f}, std={r_std:.1f}")
        print(f"     G: mean={g_mean:.1f}, std={g_std:.1f}")
        print(f"     B: mean={b_mean:.1f}, std={b_std:.1f}")
        print(f"     R/G ratio: {r_g_ratio:.3f} (balanced ≈ 1.0)")
        print(f"     B/G ratio: {b_g_ratio:.3f} (balanced ≈ 1.0)")

        if b_g_ratio > 1.15:
            print(f"     ⚠️ BLUE SHIFT DETECTED! B/G ratio too high: {b_g_ratio:.3f}")
        elif b_g_ratio < 0.85:
            print(f"     ⚠️ YELLOW SHIFT DETECTED! B/G ratio too low: {b_g_ratio:.3f}")

    def _run_unmarker(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run UnMarker attack based on config.
        """
        from .utils import attack_non_semantic, attack_two_stage_unmarker

        version = self.config.unmarker_version

        if version == "none":
            print("  - Skipping UnMarker (none selected)")
            return input_tensor

        # DEBUG: Analyze input colors
        self._analyze_colors(input_tensor, "BEFORE UnMarker")

        # Convert tensor to numpy for unmarker
        img_pil = tensor_to_pil(input_tensor)
        img_np = np.array(img_pil)

        print(f"  - Running UnMarker: {version}")

        if version == "simplified":
            result_np = attack_non_semantic(
                img_np,
                iterations=500,
                learning_rate=3e-4,
            )
        elif version == "full_fast":
            result_np = attack_two_stage_unmarker(
                img_np,
                preset="fast",
                verbose=True
            )
        elif version == "full_balanced":
            result_np = attack_two_stage_unmarker(
                img_np,
                preset="balanced",
                verbose=True
            )
        elif version == "full_quality":
            result_np = attack_two_stage_unmarker(
                img_np,
                preset="quality",
                verbose=True
            )
        else:
            print(f"  ⚠️ Unknown unmarker version: {version}, skipping")
            return input_tensor

        # Convert back to tensor
        result_pil = Image.fromarray(result_np)
        result_tensor = pil_to_tensor(result_pil)

        # DEBUG: Analyze output colors
        self._analyze_colors(result_tensor, "AFTER UnMarker")

        return result_tensor

    def run(self, input_image_tensor: torch.Tensor) -> torch.Tensor:
        """The main execution pipeline."""
        current_tensor = input_image_tensor.clone()

        # DEBUG: Analyze input image
        self._analyze_colors(current_tensor, "INPUT IMAGE")

        print(f"📋 Pipeline Mode: {self.config.mode}")
        print(f"🎯 UnMarker Version: {self.config.unmarker_version}")

        if self.config.mode == "Ultra-Minimal":
            # Just run UnMarker if enabled, no other processing
            print("  ⚡ Ultra-Minimal: UnMarker only")
            current_tensor = self._run_unmarker(current_tensor)

        elif self.config.mode == "Balanced":
            # TODO: Add ISP simulation stages here (LUT, noise, etc.)
            # For now, just run UnMarker
            print("  ⚖️ Balanced: Running UnMarker")
            current_tensor = self._run_unmarker(current_tensor)

        elif self.config.mode == "Aggressive":
            # TODO: Add full pipeline with ISP + quality analysis
            # For now, just run UnMarker
            print("  🔥 Aggressive: Running UnMarker")
            current_tensor = self._run_unmarker(current_tensor)

        # DEBUG: Analyze final output
        self._analyze_colors(current_tensor, "FINAL OUTPUT")

        return current_tensor

    # TODO: Add ISP simulation methods here
    # - _run_iphone_lut()
    # - _run_iphone_noise()
    # - _run_heic_compression()
    # - _run_quality_analysis()
    # etc.