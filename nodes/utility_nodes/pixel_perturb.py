import torch, numpy as np
class OmniSimulator_Pixel_Perturb:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "shift_amount": ("INT", {"default": 1, "min": 0, "max": 10}),
                             "shift_probability": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001})}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "perturb"; CATEGORY = "OmniSimulator/Authenticity"
    def perturb(self, image, seed, shift_amount, shift_probability):
        if shift_probability == 0 or shift_amount == 0: return (image,)
        rng = np.random.default_rng(seed)
        results = []
        for b in range(image.shape[0]):
            arr = image[b].cpu().numpy().copy()
            H, W, C = arr.shape
            mask = rng.random((H, W)) < shift_probability
            dy = rng.integers(-shift_amount, shift_amount+1, (H, W))
            dx = rng.integers(-shift_amount, shift_amount+1, (H, W))
            ys, xs = np.where(mask)
            src_y = np.clip(ys + dy[ys, xs], 0, H-1)
            src_x = np.clip(xs + dx[ys, xs], 0, W-1)
            arr[ys, xs] = arr[src_y, src_x]
            results.append(torch.from_numpy(arr).unsqueeze(0))
        return (torch.cat(results, 0),)

NODE_CLASS_MAPPINGS = {"OmniSimulator_Pixel_Perturb": OmniSimulator_Pixel_Perturb}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_Pixel_Perturb": "OmniSimulator Pixel Perturb"}