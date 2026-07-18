import torch
class OmniSimulator_ImagePromptIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "prompts": ("STRING", {"multiline": True, "default": ""}),
                             "index": ("INT", {"default": 0, "min": 0, "max": 9999})}}
    RETURN_TYPES = ("IMAGE", "STRING", "INT"); RETURN_NAMES = ("image", "prompt", "total"); FUNCTION = "iterate"; CATEGORY = "OmniSimulator/Utils"
    def iterate(self, images, prompts, index):
        lines = [l.strip() for l in prompts.strip().splitlines() if l.strip()]
        n = images.shape[0]
        i = index % max(n, 1)
        prompt = lines[i % max(len(lines), 1)] if lines else ""
        return (images[i:i+1], prompt, n)

NODE_CLASS_MAPPINGS = {"OmniSimulator_ImagePromptIterator": OmniSimulator_ImagePromptIterator}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_ImagePromptIterator": "OmniSimulator Image Prompt Iterator"}