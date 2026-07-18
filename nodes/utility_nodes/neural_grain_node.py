import torch, numpy as np, math
class OmniSimulator_NeuralGrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                             "intensity": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.5, "step": 0.005}),
                             "size": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 10.0, "step": 0.1}),
                             "mode": (["Luminance", "Color", "Both"],)}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "apply"; CATEGORY = "OmniSimulator/Utils"
    def apply(self, image, seed, intensity, size, mode):
        if intensity == 0: return (image,)
        torch.manual_seed(seed)
        B, H, W, C = image.shape
        result = image.clone()
        if mode in ["Luminance", "Both"]:
            luma = torch.randn(B, 1, H, W)
            if size > 1:
                import torch.nn.functional as F
                ks = int(size * 2) * 2 + 1
                k = torch.ones(1,1,ks,ks) / (ks*ks)
                luma = F.conv2d(luma, k, padding=ks//2)
            result += luma.permute(0,2,3,1).expand_as(result) * intensity
        if mode in ["Color", "Both"]:
            chroma = torch.randn(B, 3, H, W) * intensity * 0.5
            result += chroma.permute(0,2,3,1)
        return (torch.clamp(result, 0, 1),)

NODE_CLASS_MAPPINGS = {"OmniSimulator_NeuralGrain": OmniSimulator_NeuralGrain}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_NeuralGrain": "OmniSimulator Neural Grain"}