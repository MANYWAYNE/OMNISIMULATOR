# Filename: ComfyUI_OmniSimulator/nodes/utility_nodes/auto_white_balance_node.py
import torch
import numpy as np
from PIL import Image
import os

NODE_DIR = os.path.dirname(os.path.realpath(__file__))

OmniSimulator_ROOT = os.path.abspath(os.path.join(NODE_DIR, "..", ".."))

IPHONE13_REF_PATH = os.path.join(OmniSimulator_ROOT, "modules", "_refs", "iphone13.jpg")

class OmniSimulator_AutoWhiteBalance:
    """
    Corrects color cast. Can use an Authenticity Profile, a live reference image,
    a built-in iPhone 13 sample, or a simple grey-world assumption.
    """
    
    MODES = [
        "Authenticity Profile",
        "Internal Sample Image (iPhone 13)",
        "Reference Image",
        "Grey World (Balanced)", 
        "Grey World (Bright)", 
        "Grey World (Dark)", 
    ]
    
    GREY_TARGETS = {
        "Grey World (Balanced)": 128.0,
        "Grey World (Bright)": 160.0,
        "Grey World (Dark)": 96.0,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "Authenticity Profile"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "awb_ref_image": ("IMAGE",),
                "profile_path": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",); FUNCTION = "execute"; CATEGORY = "OmniSimulator/Authenticity"

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor is None: return None
        if tensor.ndim == 4: tensor = tensor[0]
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def _numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def _auto_white_balance(self, img_arr: np.ndarray, target_mean: np.ndarray) -> np.ndarray:
        img = img_arr.astype(np.float32)
        img_mean = img.reshape(-1, 3).mean(axis=0)
        if np.all(img_mean < 1e-6): return img_arr 
        scale = target_mean / (img_mean + 1e-6)
        return np.clip(img * scale, 0, 255)

    def execute(self, image: torch.Tensor, mode: str, strength: float, awb_ref_image: torch.Tensor = None, profile_path: str = None):
        if strength == 0: return (image,)
        target_mean = None

        if mode == "Authenticity Profile":
            if profile_path and profile_path.strip():
                npz_path = f"{profile_path}.npz"
                try:
                    with np.load(npz_path) as stats:
                        if 'chroma_mean' in stats:
                            target_mean = stats['chroma_mean'].mean(axis=0)
                except Exception as e:
                    print(f"AWB Warning: Could not load profile '{npz_path}'. Error: {e}")

        elif mode == "Internal Sample Image (iPhone 13)":
            if os.path.exists(IPHONE13_REF_PATH):
                ref_pil = Image.open(IPHONE13_REF_PATH)
                ref_numpy = np.array(ref_pil.convert("RGB"))
                target_mean = ref_numpy.astype(np.float32).reshape(-1, 3).mean(axis=0)
            else:
                print(f"AWB Warning: Reference image not found at {IPHONE13_REF_PATH}")

        elif mode == "Reference Image" and awb_ref_image is not None:
            ref_numpy = self._tensor_to_numpy(awb_ref_image)
            target_mean = ref_numpy.astype(np.float32).reshape(-1, 3).mean(axis=0)
        
        if target_mean is None:
            grey_val = self.GREY_TARGETS.get(mode, 128.0)
            target_mean = np.array([grey_val, grey_val, grey_val], dtype=np.float32)

        processed_images = []
        for i in range(image.shape[0]):
            orig_np = self._tensor_to_numpy(image[i:i+1])
            corr_float = self._auto_white_balance(orig_np, target_mean)
            blended = (orig_np.astype(np.float32) * (1 - strength)) + (corr_float * strength)
            processed_images.append(self._numpy_to_tensor(np.clip(blended, 0, 255).astype(np.uint8)))

        return (torch.cat(processed_images, dim=0),)