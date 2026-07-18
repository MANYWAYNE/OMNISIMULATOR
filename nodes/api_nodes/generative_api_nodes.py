"""Generative API base — stub for providers (Wavespeed, fal.ai)."""
import os
import io
import base64
import requests
import torch
import numpy as np
from PIL import Image

def _ws_build_t2i(engine, prompt, aspect_ratio="1:1", width=1024, height=1024,
                  resolution=None, seed=-1, num_images=1, **kw):
    return {"prompt": prompt, "aspect_ratio": aspect_ratio,
            "num_images": num_images, "seed": seed if seed >= 0 else None}

def _ws_build_i2i(engine, prompt, image_1=None, image_2=None, image_3=None,
                  image_4=None, aspect_ratio="1:1", width=1024, height=1024,
                  resolution=None, seed=-1, num_images=1, **kw):
    payload = {"prompt": prompt, "aspect_ratio": aspect_ratio,
               "num_images": num_images, "seed": seed if seed >= 0 else None}
    image_urls = []
    for img in [image_1, image_2, image_3, image_4]:
        if img is not None:
            img_np = img.cpu().numpy()
            if img_np.ndim == 4: 
                img_np = img_np[0]
            if img_np.max() <= 1.0: 
                img_np = (img_np * 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img_np).save(buf, format='JPEG', quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode()
            image_urls.append(f"data:image/jpeg;base64,{b64}")
    if image_urls:
        payload["image_urls"] = image_urls
    return payload

MODEL_CONFIG = {
    "Nano Banana Pro": {
        "providers": {
            "wavespeed.ai": {
                "t2i_endpoint": "wavespeed-ai/wan-i2v-480p",
                "i2i_endpoint": "wavespeed-ai/wan-i2v-480p-i2i",
                "build_payload": _ws_build_t2i,
            }
        }
    },
    "Nano Banana": {
        "providers": {
            "wavespeed.ai": {
                "t2i_endpoint": "wavespeed-ai/wan-i2v-480p",
                "i2i_endpoint": "wavespeed-ai/wan-i2v-480p-i2i",
                "build_payload": _ws_build_t2i,
            }
        }
    },
}

class OmniSimulator_GenerativeAPIBase:
    """Base class for generative API clients."""
    def __init__(self):
        self.api_key = ""

    def set_api_key(self, key: str):
        self.api_key = key

    def _submit_wavespeed(self, endpoint: str, payload: dict,
                          return_all_outputs: bool = False):
        url = f"https://api.wavespeed.ai/api/v3/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        if not resp.ok:
            raise Exception(f"[OmniSimulator] Wavespeed error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        outputs = data.get("data", {}).get("outputs", [])
        if not outputs:
            raise Exception("[OmniSimulator] No outputs in Wavespeed response")
        if return_all_outputs:
            return outputs
        return outputs[0]

    def _submit_fal_sync(self, endpoint: str, payload: dict, timeout: int = 120) -> str:
        url = f"https://fal.run/{endpoint}"
        headers = {"Authorization": f"Key {self.api_key}",
                   "Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if not resp.ok:
            raise Exception(f"[OmniSimulator] fal.ai error {resp.status_code}: {resp.text[:200]}")
        result = resp.json()
        if "images" in result and result["images"]:
            return result["images"][0]["url"]
        raise Exception("[OmniSimulator] No image URL in fal.ai response")

NODE_CLASS_MAPPINGS = {"OmniSimulator_GenerativeAPIBase": OmniSimulator_GenerativeAPIBase}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_GenerativeAPIBase": "Omni: Generative API Base"}