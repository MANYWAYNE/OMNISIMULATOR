import dataclasses
import os
import torch
import numpy as np
from PIL import Image
from types import SimpleNamespace
import tempfile
import time
import gc

# Use a clean, relative import to our local, patched package
from .filmgrainer_local.filmgrainer import process as process_filmgrain

# Import your existing processing functions from the same directory
from .processor import process_image

@dataclasses.dataclass
class BypassConfigV2:
    """A clean, type-safe container for all our settings."""
    mode: str = "Full Pipeline (iPhone)"
    strength: float = 0.30
    lut_path: str = ""
    seed: int = 0

class BypassPipelineV2:
    def __init__(self, config: BypassConfigV2):
        self.config = config
        print(f"V2 Pipeline Initialized: Mode='{self.config.mode}', Strength={self.config.strength:.2f}")

    def _get_base_args(self) -> SimpleNamespace:
        # Create a complete base namespace with ALL possible parameters to prevent AttributeErrors.
        # BUG FIX: Clamp seed to 32-bit range to prevent overflow in underlying libraries.
        return SimpleNamespace(
            seed=self.config.seed % (2**32), awb=False, ref=None, fft_ref=None, lut=None, lut_strength=1.0, blend=False,
            noise=False, clahe=False, fft=False, perturb=False, glcm=False, lbp=False,
            sim_camera=False, non_semantic=False,
            noise_std=0.0, clahe_clip=2.0, tile=8, cutoff=0.25, fstrength=0.9,
            randomness=0.05, phase_perturb=0.08, radial_smooth=5,
            fft_mode="auto", fft_alpha=1.0, perturb_magnitude=0.0,
            glcm_distances=[1], glcm_angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            glcm_levels=256, glcm_strength=0.9, lbp_radius=3, lbp_n_points=24, lbp_method='uniform', lbp_strength=0.9,
            no_no_bayer=True, jpeg_cycles=1, jpeg_qmin=88, jpeg_qmax=96,
            vignette_strength=0.0, chroma_strength=0.0, iso_scale=1.0,
            read_noise=2.0, hot_pixel_prob=1e-6, banding_strength=0.0, motion_blur_kernel=1,
            ns_iterations=500, ns_learning_rate=3e-4, ns_t_lpips=0.04, ns_t_l2=3e-5,
            ns_c_lpips=1e-2, ns_c_l2=0.6, ns_grad_clip=0.05,
            blend_tolerance=10.0, blend_min_region=50, blend_max_samples=100000, blend_n_jobs=None
        )

    def _apply_filmgrain(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Wrapper for our local, patched filmgrainer library."""
        pil_img = tensor_to_pil(image_tensor)
        
        s = self.config.strength
        grain_power = 0.7 + (s * 0.3)
        shadows = 0.1 + (s * 0.2)
        highs = 0.1 + (s * 0.1)
        grain_sat = 0.2 + (s * 0.4)
        
        print(f"  [V2 Pipeline] Applying film grain: Power={grain_power:.2f}, Shadows={shadows:.2f}, Sat={grain_sat:.2f}")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
            tmp_in_path, tmp_out_path = tmp_in.name, tmp_out.name
        
        try:
            pil_img.save(tmp_in_path)
            # BUG FIX: Also clamp seed here for the filmgrainer library
            process_filmgrain(
                file_in=tmp_in_path, file_out=tmp_out_path, scale=1.0, src_gamma=1.0,
                grain_power=grain_power, shadows=shadows, highs=highs, grain_type=1,
                grain_sat=grain_sat, gray_scale=False, sharpen=0, seed=self.config.seed % (2**32)
            )
            result_tensor = pil_to_tensor(Image.open(tmp_out_path))
        finally:
            del pil_img
            gc.collect()
            time.sleep(0.05)
            for f_path in [tmp_in_path, tmp_out_path]:
                if os.path.exists(f_path):
                    try:
                        os.unlink(f_path)
                    except Exception as e:
                        print(f"  - Warning: Could not delete filmgrain temp file {f_path}: {e}")
        return result_tensor

    def run(self, input_image_tensor: torch.Tensor, awb_ref_tensor: torch.Tensor = None) -> torch.Tensor:
        """The main execution pipeline, orchestrating all authenticity stages."""
        current_tensor = input_image_tensor.clone()
        
        if self.config.mode == "Full Pipeline (iPhone)":
            print("  - Executing Full Pipeline (iPhone)...")
            current_tensor = self._run_pass(current_tensor, self._get_awb_args(), awb_ref_tensor=awb_ref_tensor)
            current_tensor = self._run_pass(current_tensor, self._get_ns_args())
            current_tensor = self._run_pass(current_tensor, self._get_lut_args())
            current_tensor = self._apply_filmgrain(current_tensor)
            current_tensor = self._run_pass(current_tensor, self._get_simulation_args())
            current_tensor = self._run_pass(current_tensor, self._get_perturb_args())

        elif self.config.mode == "Ultra-Minimal (Stealth)":
            print("  - Executing Ultra-Minimal (Stealth) Pipeline...")
            current_tensor = self._run_pass(current_tensor, self._get_awb_args(), awb_ref_tensor=awb_ref_tensor)
            
            args2 = self._get_base_args(); args2.pass_name = "AI Normalizer (Minimal)"; args2.non_semantic = True
            args2.ns_iterations = 200; args2.ns_t_lpips = 0.015
            current_tensor = self._run_pass(current_tensor, args2)
            
            args3 = self._get_base_args(); args3.pass_name = "Light Camera Sim"; args3.sim_camera = True
            args3.chroma_strength = 0.4; args3.vignette_strength = 0.05; args3.jpeg_cycles = 1; args3.jpeg_qmin = 96; args3.jpeg_qmax = 99
            current_tensor = self._run_pass(current_tensor, args3)
            
            args4 = self._get_base_args(); args4.pass_name = "Final Perturbation"; args4.perturb = True
            args4.perturb_magnitude = 0.002
            current_tensor = self._run_pass(current_tensor, args4)

        return current_tensor

    def _run_pass(self, input_tensor, args, awb_ref_tensor=None, fft_ref_tensor=None):
        pass_name = getattr(args, 'pass_name', 'Unknown')
        print(f"  [V2 Pipeline] Stage: '{pass_name}'...")
        pil_img = tensor_to_pil(input_tensor)
        tmp_files_to_clean = []
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input, \
                 tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_output:
                tmp_input_path, tmp_output_path = tmp_input.name, tmp_output.name
            tmp_files_to_clean.extend([tmp_input_path, tmp_output_path])
            pil_img.save(tmp_input_path)
            args.input, args.output = tmp_input_path, tmp_output_path
            
            if awb_ref_tensor is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_awb:
                    awb_path = tmp_awb.name
                tensor_to_pil(awb_ref_tensor).save(awb_path)
                args.ref = awb_path
                tmp_files_to_clean.append(args.ref)
            
            process_image(args.input, args.output, args)
            return pil_to_tensor(Image.open(tmp_output_path))
        finally:
            del pil_img
            gc.collect()
            time.sleep(0.05)
            for f_path in tmp_files_to_clean:
                if os.path.exists(f_path):
                    try:
                        os.unlink(f_path)
                    except Exception as e:
                        print(f"  - Warning: Could not delete temp file {f_path}: {e}")
    
    def _get_awb_args(self) -> SimpleNamespace:
        args = self._get_base_args(); args.pass_name = "Auto White Balance"; args.awb = True
        return args
        
    def _get_ns_args(self) -> SimpleNamespace:
        args = self._get_base_args(); args.pass_name = "AI Normalizer (UnMarker Attack)"
        args.non_semantic = True; s = self.config.strength
        args.ns_iterations = int(200 + 300 * s); args.ns_t_lpips = 0.02 + (0.025 * s); args.ns_learning_rate = 2e-4 + (1e-4 * s)
        return args

    def _get_lut_args(self) -> SimpleNamespace:
        args = self._get_base_args(); args.pass_name = "3D LUT Color Science"
        args.lut = self.config.lut_path; args.lut_strength = 0.5 + (0.5 * self.config.strength)
        return args

    def _get_simulation_args(self) -> SimpleNamespace:
        args = self._get_base_args(); args.pass_name = "Camera Simulation"
        s = self.config.strength; args.sim_camera = True
        args.chroma_strength = 0.5 + (1.0 * s); args.vignette_strength = 0.1 + (0.3 * s)
        args.jpeg_cycles = 1 + int(s * 2); args.jpeg_qmin = int(95 - (45 * s)); args.jpeg_qmax = int(98 - (25 * s))
        if args.jpeg_qmin > args.jpeg_qmax: args.jpeg_qmin = args.jpeg_qmax
        return args

    def _get_perturb_args(self) -> SimpleNamespace:
        args = self._get_base_args(); args.pass_name = "Final Perturbation"
        args.perturb = True; args.perturb_magnitude = 0.001 + (0.007 * self.config.strength)
        return args

# Helper functions to keep file self-contained
def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4 and tensor.shape[0] == 1: tensor = tensor.squeeze(0)
    return Image.fromarray((tensor.cpu().numpy() * 255).astype(np.uint8))

def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    try:
        img_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)
    finally:
        if isinstance(pil_image, Image.Image):
            pil_image.close()