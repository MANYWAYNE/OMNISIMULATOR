import torch
import torch.nn.functional as F
class OmniSimulator_ImageResolutionClamp:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "max_megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 32.0, "step": 0.1})}}
    RETURN_TYPES = ("IMAGE", "INT", "INT"); RETURN_NAMES = ("image", "width", "height"); FUNCTION = "clamp"; CATEGORY = "OmniSimulator/Utils"
    def clamp(self, image, max_megapixels):
        B, H, W, C = image.shape
        mp = H * W / 1_000_000
        if mp <= max_megapixels:
            return (image, W, H)
        scale = (max_megapixels / mp) ** 0.5
        nw, nh = max(64, int(W * scale // 8) * 8), max(64, int(H * scale // 8) * 8)
        out = F.interpolate(image.permute(0,3,1,2), size=(nh, nw), mode='bilinear', align_corners=False).permute(0,2,3,1)
        return (out, nw, nh)

NODE_CLASS_MAPPINGS = {"OmniSimulator_ImageResolutionClamp": OmniSimulator_ImageResolutionClamp}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_ImageResolutionClamp": "OmniSimulator Resolution Clamp"}