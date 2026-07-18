import os
import json
import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths

POOL_DIR = os.path.join(folder_paths.get_input_directory(), "omni_pool")
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

DEVICE_MODELS = {
    "iPhone 12":        "iphone_12",
    "iPhone 13 Pro":    "iphone_13_pro",
    "iPhone 14 Plus":   "iphone_14_plus",
    "iPhone 14 Pro Max":"iphone_14_pro_max",
    "iPhone 15 Pro Max":"iphone_15_pro_max",
}

class Omni_AdvancedImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Batch Tensor", "Sequential"], {"default": "Batch Tensor"}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "batch_data":  ("STRING", {"default": "{}", "multiline": False}),
            },
            "optional": {
                "enable_img2img": ("BOOLEAN", {"default": True}),
                "width":  ("INT", {"default": 960, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 960, "min": 64, "max": 8192, "step": 8}),
                "aspect_label": ("STRING", {"default": "1:1"}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "LATENT", "INT", "INT", "STRING")
    RETURN_NAMES  = ("images", "latent", "width", "height", "aspect_label")
    FUNCTION      = "execute"
    CATEGORY      = "Omni"
    OUTPUT_NODE   = False

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def _blank_latent(self, w, h, b=1):
        return {"samples": torch.zeros(b, 4, h // 8, w // 8)}

    def _blank_image(self, w, h, b=1):
        return torch.zeros(b, h, w, 3)

    def _load_image(self, img_meta, fallback_w, fallback_h):
        os.makedirs(POOL_DIR, exist_ok=True)
        img_id = str(img_meta.get("id", "")).strip()
        thumb  = img_meta.get("thumbnail", "")

        path = ""
        if img_id:
            for fn in os.listdir(POOL_DIR):
                if fn.startswith(img_id) and not fn.startswith("thumb_"):
                    path = os.path.join(POOL_DIR, fn)
                    break
        if not path or not os.path.exists(path):
            path = os.path.join(POOL_DIR, thumb)
            if not os.path.exists(path):
                for root, _, files in os.walk(POOL_DIR):
                    if thumb in files: 
                        path = os.path.join(root, thumb)
                        break

        if path and os.path.exists(path):
            pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            arr = np.array(pil).astype(np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

        w = img_meta.get("width", fallback_w)
        h = img_meta.get("height", fallback_h)
        return self._blank_image(w, h, 1)

    def _resize_to_match(self, tensor, h, w):
        if tensor.shape[1] == h and tensor.shape[2] == w:
            return tensor
        import torch.nn.functional as F
        t = tensor.permute(0, 3, 1, 2)
        t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
        return t.permute(0, 2, 3, 1)

    def execute(self, mode, batch_index, batch_data,
                enable_img2img=True, width=960, height=960, aspect_label="1:1"):

        if mode not in ("Batch Tensor", "Sequential"):
            mode = "Batch Tensor"

        try:
            data = json.loads(batch_data) if batch_data and batch_data not in ("{}", "") else {}
        except Exception:
            data = {}

        if not enable_img2img:
            latents = data.get("latents", [])
            order   = data.get("order", [])
            if not latents or not order:
                return (self._blank_image(width, height), self._blank_latent(width, height), width, height, aspect_label)

            if mode == "Sequential":
                cur = 0
                for lid in order:
                    l = next((x for x in latents if x["id"] == lid), None)
                    if l is None: 
                        continue
                    rep = l.get("repeat_count", 1)
                    if cur <= batch_index < cur + rep:
                        w, h, asp = l.get("width", width), l.get("height", height), l.get("aspect_label", aspect_label)
                        return (self._blank_image(w, h), self._blank_latent(w, h), w, h, asp)
                    cur += rep
            else:
                if order:
                    l = next((x for x in latents if x["id"] == order[0]), None)
                    if l:
                        w, h, asp = l.get("width", width), l.get("height", height), l.get("aspect_label", aspect_label)
                        n = len(order)
                        return (self._blank_image(w, h, n), self._blank_latent(w, h, n), w, h, asp)
            return (self._blank_image(width, height), self._blank_latent(width, height), width, height, aspect_label)

        images_meta = data.get("images", [])
        order       = data.get("order", [])

        if not images_meta or not order:
            return (self._blank_image(width, height), self._blank_latent(width, height), width, height, aspect_label)

        def find_meta(img_id):
            return next((x for x in images_meta if str(x["id"]) == str(img_id)), None)

        if mode == "Sequential":
            cur = 0
            for img_id in order:
                meta = find_meta(img_id)
                if meta is None: 
                    continue
                rep = meta.get("repeat_count", 1)
                if cur <= batch_index < cur + rep:
                    t = self._load_image(meta, width, height)
                    h, w = t.shape[1], t.shape[2]
                    return (t, self._blank_latent(w, h), w, h, aspect_label)
                cur += rep
            meta = find_meta(order[-1]) or images_meta[-1]
            t = self._load_image(meta, width, height)
            h, w = t.shape[1], t.shape[2]
            return (t, self._blank_latent(w, h), w, h, aspect_label)

        tensors = []
        for img_id in order:
            meta = find_meta(img_id)
            if meta is None: 
                continue
            t = self._load_image(meta, width, height)
            for _ in range(meta.get("repeat_count", 1)):
                tensors.append(t)

        if not tensors:
            return (self._blank_image(width, height), self._blank_latent(width, height), width, height, aspect_label)

        ref_h, ref_w = tensors[0].shape[1], tensors[0].shape[2]
        batch = torch.cat([self._resize_to_match(t, ref_h, ref_w) for t in tensors], dim=0)
        return (batch, self._blank_latent(ref_w, ref_h, len(tensors)), ref_w, ref_h, aspect_label)


class Omni_WANAspectRatio:
    RATIOS = {
        "9:16 Portrait": (720, 1280), 
        "3:4 Portrait": (720, 960), 
        "1:1 Square": (960, 960),
        "4:3 Landscape": (960, 720), 
        "16:9 Landscape": (1280, 720)
    }
    @classmethod
    def INPUT_TYPES(cls): 
        return {"required": {"ratio": (list(cls.RATIOS.keys()),)}}
    
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "aspect_label")
    FUNCTION = "get"
    CATEGORY = "Omni"
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs): 
        return True
        
    def get(self, ratio):
        if ratio not in self.RATIOS: 
            ratio = next(iter(self.RATIOS))
        w, h = self.RATIOS[ratio]
        return (w, h, ratio.split(" ")[0])


class Omni_SDXLAspectRatio:
    RATIOS = {
        "9:16 Portrait": (768, 1344), 
        "3:4 Portrait": (896, 1152), 
        "1:1 Square": (1024, 1024),
        "4:3 Landscape": (1152, 896), 
        "16:9 Landscape": (1344, 768), 
        "21:9 Ultrawide": (1536, 640)
    }
    @classmethod
    def INPUT_TYPES(cls): 
        return {"required": {"ratio": (list(cls.RATIOS.keys()),)}}
        
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "aspect_label")
    FUNCTION = "get"
    CATEGORY = "Omni"
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs): 
        return True
        
    def get(self, ratio):
        if ratio not in self.RATIOS: 
            ratio = next(iter(self.RATIOS))
        w, h = self.RATIOS[ratio]
        return (w, h, ratio.split(" ")[0])


class Omni_ResolutionPresetsPro:
    RATIOS = {
        "1:1 Square": (1, 1),
        "3:2 Landscape": (3, 2),
        "2:3 Portrait": (2, 3),
        "4:3 Landscape": (4, 3),
        "3:4 Portrait": (3, 4),
        "16:9 Landscape": (16, 9),
        "9:16 Portrait": (9, 16),
        "4:5 Portrait": (4, 5),
        "5:4 Landscape": (5, 4)
    }
    RESOLUTIONS = {"1K": 1024, "2K": 2048, "4K": 4096}
    
    @classmethod
    def INPUT_TYPES(cls): 
        return {"required": {"ratio": (list(cls.RATIOS.keys()),), "resolution": (list(cls.RESOLUTIONS.keys()),)}}
        
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("aspect_ratio", "resolution", "width", "height", "aspect_label")
    FUNCTION = "get"
    CATEGORY = "Omni"
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs): 
        return True
        
    def get(self, ratio, resolution):
        if ratio not in self.RATIOS: 
            ratio = next(iter(self.RATIOS))
        if resolution not in self.RESOLUTIONS: 
            resolution = next(iter(self.RESOLUTIONS))
        wr, hr = self.RATIOS[ratio]
        base = self.RESOLUTIONS[resolution]
        if wr >= hr: 
            h = base
            w = int(base * wr / hr)
        else: 
            w = base
            h = int(base * hr / wr)
        w = (w // 64) * 64
        h = (h // 64) * 64
        label = ratio.split(" ")[0]
        return (label, resolution, w, h, ratio)


class Omni_iPhoneSensorProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), 
                "device_model": (list(DEVICE_MODELS.keys()),),
                "color_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "apply_icc": ("BOOLEAN", {"default": True}),
                "texture_enhance": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05})
            },
            "optional": {
                "noise_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.3, "step": 0.005}),
                "iso": ("INT", {"default": 32, "min": 20, "max": 6400, "step": 50})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Omni/Camera"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs): 
        return True

    def _hist_match(self, ch, log_spec, strength):
        tgt = np.exp(log_spec)
        tgt = tgt / (tgt.sum() + 1e-8)
        flat = ch.flatten()
        src_h, _ = np.histogram(flat, bins=256, range=(0, 1))
        src_h = src_h.astype(np.float64)
        src_h /= src_h.sum() + 1e-8
        sc = np.cumsum(src_h)
        tc = np.cumsum(tgt)
        lut = np.interp(sc, tc, np.linspace(0, 1, 256))
        idx = np.clip((flat * 255).astype(int), 0, 255)
        matched = lut[idx].reshape(ch.shape)
        return ch * (1 - strength) + matched * strength

    def apply(self, image, device_model, color_strength=0.7, apply_icc=True,
              texture_enhance=0.25, noise_amount=0.0, iso=32):
        if device_model not in DEVICE_MODELS:
            device_model = next(iter(DEVICE_MODELS))
        key  = DEVICE_MODELS[device_model]
        npz  = os.path.join(_MODELS_DIR, f"{key}.npz")
        icc  = os.path.join(_MODELS_DIR, f"{key}.icc")

        data = {}
        if os.path.exists(npz):
            try:
                with np.load(npz) as npz_file:
                    data = {k: npz_file[k] for k in npz_file.files}
            except Exception as e:
                print(f"[OmniSimulator] iPhoneSensorProfile: Could not read '{npz}': {e}")

        results = []
        for b in range(image.shape[0]):
            img = image[b].cpu().numpy().copy()
            if data and color_strength > 0:
                for ci, k in enumerate(["spectra_r", "spectra_g", "spectra_b"]):
                    if k in data: 
                        img[:, :, ci] = self._hist_match(img[:, :, ci], np.mean(data[k], 0), color_strength)
                img = np.clip(img, 0, 1)
            
            if apply_icc and os.path.exists(icc):
                try:
                    from PIL import ImageCms
                    import io as _io
                    pil = Image.fromarray((img * 255).astype(np.uint8))
                    with open(icc, 'rb') as f: 
                        icc_data = f.read()
                    srgb = ImageCms.createProfile('sRGB')
                    dev  = ImageCms.getOpenProfile(_io.BytesIO(icc_data))
                    xfm  = ImageCms.buildTransformFromOpenProfiles(srgb, dev, 'RGB', 'RGB')
                    pil  = ImageCms.applyTransform(pil, xfm)
                    img  = np.array(pil).astype(np.float32) / 255.0
                except Exception as e: 
                    print(f"[OmniSimulator] ICC error: {e}")
            
            if texture_enhance > 0 and data and 'glcm_props' in data:
                from PIL.ImageFilter import GaussianBlur
                contrast = float(np.mean(data['glcm_props'][:, 0]))
                amt = texture_enhance * np.clip(contrast / 500, 0.05, 1.5)
                pil = Image.fromarray((img * 255).astype(np.uint8))
                blr = np.array(pil.filter(GaussianBlur(1.2))).astype(np.float32) / 255
                img = np.clip(img + (img - blr) * amt, 0, 1)
            
            if noise_amount > 0 or iso > 100:
                iso_f = np.log2(max(iso, 32) / 32) * 0.003
                noise = np.random.normal(0, noise_amount + iso_f, img.shape)
                img = np.clip(img + noise, 0, 1)
            
            results.append(torch.from_numpy(img.astype(np.float32)).unsqueeze(0))
        return (torch.cat(results, 0),)


NODE_CLASS_MAPPINGS = {
    "Omni_AdvancedImageLoader":   Omni_AdvancedImageLoader,
    "Omni_WANAspectRatio":        Omni_WANAspectRatio,
    "Omni_SDXLAspectRatio":       Omni_SDXLAspectRatio,
    "Omni_ResolutionPresetsPro":  Omni_ResolutionPresetsPro,
    "Omni_iPhoneSensorProfile":   Omni_iPhoneSensorProfile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Omni_AdvancedImageLoader":   "Omni: Advanced Image Loader",
    "Omni_WANAspectRatio":        "Omni: WAN Aspect Ratio",
    "Omni_SDXLAspectRatio":       "Omni: SDXL Aspect Ratio",
    "Omni_ResolutionPresetsPro":  "Omni: Aspect Ratio Pro",
    "Omni_iPhoneSensorProfile":   "Omni: iPhone Sensor Profile",
}