import torch, numpy as np, os
class OmniSimulator_Spectral_Normalizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "profile_path": ("STRING", {"forceInput": True}),
                             "strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "channels": (["All","R only","G only","B only"],)}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "normalize"; CATEGORY = "OmniSimulator/Authenticity"
    def normalize(self, image, profile_path, strength, channels):
        if strength == 0: return (image,)
        npz_p = f"{profile_path}.npz" if not profile_path.endswith(".npz") else profile_path
        if not os.path.exists(npz_p): return (image,)
        if channels not in ("All", "R only", "G only", "B only"): channels = "All"
        ch_map = {"All":[0,1,2],"R only":[0],"G only":[1],"B only":[2]}
        results = []
        with np.load(npz_p) as data:
            for b in range(image.shape[0]):
                img = image[b].cpu().numpy().copy()
                for ch in ch_map[channels]:
                    key = ["spectra_r","spectra_g","spectra_b"][ch]
                    if key not in data: continue
                    tgt = np.exp(np.mean(data[key], 0)); tgt /= tgt.sum()+1e-8
                    flat = img[:,:,ch].flatten()
                    h, _ = np.histogram(flat, bins=256, range=(0,1))
                    h = h.astype(np.float64); h /= h.sum()+1e-8
                    sc = np.cumsum(h); tc = np.cumsum(tgt)
                    lut = np.interp(sc, tc, np.linspace(0,1,256))
                    idx = np.clip((flat*255).astype(int),0,255)
                    img[:,:,ch] = img[:,:,ch]*(1-strength) + lut[idx].reshape(img[:,:,ch].shape)*strength
                results.append(torch.from_numpy(np.clip(img,0,1)).unsqueeze(0))
        return (torch.cat(results,0),)

NODE_CLASS_MAPPINGS = {"OmniSimulator_Spectral_Normalizer": OmniSimulator_Spectral_Normalizer}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_Spectral_Normalizer": "OmniSimulator Spectral Normalizer"}