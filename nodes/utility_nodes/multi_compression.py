import torch
import numpy as np
import random
import io
from PIL import Image as PILImage

def _t2np(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 4: t = t.squeeze(0)
    return (t.cpu().numpy() * 255).astype(np.uint8)

def _np2t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)

def _jpeg_rt(img: np.ndarray, quality: int) -> np.ndarray:
    pil = PILImage.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, subsampling=2, optimize=False)
    buf.seek(0)
    return np.array(PILImage.open(buf).convert("RGB"))

class OmniSimulator_MultiCompression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image":       ("IMAGE",),
            "enabled":     ("BOOLEAN",{"default":True}),
            "cycles":      ("INT",  {"default":1,"min":1,"max":10}),
            "min_quality": ("INT",  {"default":75,"min":1,"max":100}),
            "max_quality": ("INT",  {"default":75,"min":1,"max":100}),
            "seed":        ("INT",  {"default":0,"min":0,"max":0xffffffffffffffff}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "execute"; CATEGORY = "OmniSimulator/Utility"

    def execute(self,image,enabled,cycles,min_quality,max_quality,seed):
        if not enabled: return (image,)
        min_q=min(min_quality,max_quality); max_q=max(min_quality,max_quality); out=[]
        for i in range(image.shape[0]):
            img=_t2np(image[i:i+1]); rng=random.Random(seed+i)
            for _ in range(cycles): img=_jpeg_rt(img,rng.randint(min_q,max_q))
            out.append(_np2t(img))
        return (torch.cat(out, dim=0),)

NODE_CLASS_MAPPINGS = {
    "OmniSimulator_MultiCompression": OmniSimulator_MultiCompression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniSimulator_MultiCompression": "Multi Compression",
}