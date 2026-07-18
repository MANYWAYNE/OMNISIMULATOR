import torch, numpy as np, os
class OmniSimulator_SpectralEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "profile_path": ("STRING", {"forceInput": True}),
                             "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "apply"; CATEGORY = "OmniSimulator/Authenticity"
    def apply(self, image, profile_path, strength):
        if strength == 0 or not profile_path: return (image,)
        npz_path = f"{profile_path}.npz" if not profile_path.endswith(".npz") else profile_path
        if not os.path.exists(npz_path):
            print(f"[OmniSimulator SpectralEngine] Profile not found: {npz_path}"); return (image,)
        results = []
        with np.load(npz_path) as data:
            for b in range(image.shape[0]):
                img = image[b].cpu().numpy()
                for ch, key in enumerate(["spectra_r","spectra_g","spectra_b"]):
                    if key not in data: continue
                    target = np.exp(np.mean(data[key], 0)); target /= target.sum() + 1e-8
                    ch_flat = img[:,:,ch].flatten()
                    src_hist, _ = np.histogram(ch_flat, bins=256, range=(0,1))
                    src_hist = src_hist.astype(np.float64); src_hist /= src_hist.sum() + 1e-8
                    src_cdf = np.cumsum(src_hist); tgt_cdf = np.cumsum(target)
                    lut = np.interp(src_cdf, tgt_cdf, np.linspace(0,1,256))
                    idx = np.clip((ch_flat * 255).astype(int), 0, 255)
                    matched = lut[idx].reshape(img[:,:,ch].shape)
                    img[:,:,ch] = img[:,:,ch] * (1 - strength) + matched * strength
                results.append(torch.from_numpy(np.clip(img, 0, 1)).unsqueeze(0))
        return (torch.cat(results, 0),)

NODE_CLASS_MAPPINGS = {"OmniSimulator_SpectralEngine": OmniSimulator_SpectralEngine}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_SpectralEngine": "OmniSimulator Spectral Engine"}