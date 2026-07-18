import torch, random
class OmniSimulator_SeedGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"})}}
    RETURN_TYPES = ("INT",); RETURN_NAMES = ("seed",); FUNCTION = "get"; CATEGORY = "OmniSimulator/Utils"
    def get(self, seed): return (seed,)

NODE_CLASS_MAPPINGS = {"OmniSimulator_SeedGenerator": OmniSimulator_SeedGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_SeedGenerator": "OmniSimulator Seed Generator"}