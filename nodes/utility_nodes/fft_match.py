import torch, numpy as np, os
class OmniSimulator_FFT_Match:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01})},
                "optional": {"ref_image": ("IMAGE",), "profile_path": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "match"; CATEGORY = "OmniSimulator/Authenticity"
    def match(self, image, strength, ref_image=None, profile_path=None):
        if strength == 0: return (image,)
        target_magnitude = None
        if profile_path:
            npz_path = f"{profile_path}.npz" if not profile_path.endswith(".npz") else profile_path
            if os.path.exists(npz_path):
                with np.load(npz_path) as data:
                    if "spectra_r" in data:
                        target_magnitude = np.stack([np.exp(data["spectra_r"].mean(0)), np.exp(data["spectra_g"].mean(0)), np.exp(data["spectra_b"].mean(0))], -1)
        if target_magnitude is None and ref_image is not None:
            ref = ref_image[0].cpu().numpy()
            target_magnitude = np.abs(np.fft.fft2(ref, axes=(0,1)))
        if target_magnitude is None: return (image,)
        results = []
        for i in range(image.shape[0]):
            src = image[i].cpu().numpy()
            src_fft = np.fft.fft2(src, axes=(0,1))
            src_mag = np.abs(src_fft) + 1e-8
            src_phase = src_fft / src_mag
            h, w = src.shape[:2]
            if target_magnitude.shape[:2] != (h, w):
                from scipy.ndimage import zoom
                zh, zw = h / target_magnitude.shape[0], w / target_magnitude.shape[1]
                zoom_factors = (zh, zw, 1) if target_magnitude.ndim == 3 else (zh, zw)
                tm = zoom(target_magnitude.astype(np.float32), zoom_factors, order=1)
                # zoom() can be off by a pixel or two due to float rounding — force exact (h, w)
                th, tw = tm.shape[:2]
                if th < h or tw < w:
                    pad_spec = [(0, max(0, h - th)), (0, max(0, w - tw))] + ([(0, 0)] if tm.ndim == 3 else [])
                    tm = np.pad(tm, pad_spec, mode='edge')
                tm = tm[:h, :w]
            else: tm = target_magnitude.astype(np.float32)
            new_mag = src_mag * (1 - strength) + tm * strength
            result = np.real(np.fft.ifft2(new_mag * src_phase, axes=(0,1)))
            results.append(torch.from_numpy(np.clip(result, 0, 1).astype(np.float32)).unsqueeze(0))
        return (torch.cat(results, 0),)

NODE_CLASS_MAPPINGS = {"OmniSimulator_FFT_Match": OmniSimulator_FFT_Match}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_FFT_Match": "OmniSimulator FFT Match"}