import os
import numpy as np
import torch
from PIL import Image, ImageOps


class OmniSimulator_LoadImageFromPath:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"path": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load"
    CATEGORY = "OmniSimulator/Utils"

    def load(self, path):
        # Tolerate quotes / stray whitespace / "~" that users routinely paste in.
        path = os.path.expanduser((path or "").strip().strip('"').strip("'"))
        if not path:
            raise ValueError("OmniSimulator Load Image From Path: no path provided.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"OmniSimulator Load Image From Path: image not found: {path}")

        pil = Image.open(path)
        # Honour the EXIF orientation flag so phone/camera shots load upright
        # instead of sideways (the previous version ignored orientation).
        pil = ImageOps.exif_transpose(pil)

        rgb = pil.convert("RGB")
        arr = np.asarray(rgb).astype(np.float32) / 255.0
        image = torch.from_numpy(arr).unsqueeze(0)          # [1,H,W,3]

        # Mirror ComfyUI's stock Load Image: alpha -> MASK (mask = 1 - alpha),
        # opaque zero-mask when the file has no alpha channel.
        if "A" in pil.getbands():
            alpha = np.asarray(pil.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(alpha)
        else:
            mask = torch.zeros((arr.shape[0], arr.shape[1]), dtype=torch.float32)

        return (image, mask.unsqueeze(0))                   # mask [1,H,W]


NODE_CLASS_MAPPINGS = {"OmniSimulator_LoadImageFromPath": OmniSimulator_LoadImageFromPath}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_LoadImageFromPath": "OmniSimulator Load Image From Path"}