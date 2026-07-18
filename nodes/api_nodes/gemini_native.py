import torch
import numpy as np
import base64
import io
from PIL import Image
import json

try:
    import urllib.request
    import urllib.error
    _HTTP_OK = True
except ImportError:
    _HTTP_OK = False

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

SAFETY_LEVELS = [
    "Block None",
    "Block Few (High Only)",
    "Block Some (Medium+)",
    "Block Most (Low+)",
]

class OmniSimulator_GeminiNative:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "tooltip": "Your Google Gemini API key."}),
                "model": (GEMINI_MODELS, {"default": "gemini-2.5-pro"}),
                "seed": ("INT", {"default": 1111111, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                "temperature": ("FLOAT", {"default": 0.111, "min": 0.0, "max": 2.0, "step": 0.001}),
                "enable_thinking": ("BOOLEAN", {"default": True, "label_on": "Thinking Enabled", "label_off": "Thinking Disabled"}),
                "safety_level": (SAFETY_LEVELS, {"default": "Block None"}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "custom_model": ("STRING", {"default": "", "tooltip": "Optional: overrides 'model' with any Gemini model string."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "Omni/API"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def _tensor_to_b64(self, tensor: torch.Tensor) -> str:
        if tensor.ndim == 4: 
            tensor = tensor[0]
        arr = (np.clip(tensor.cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
        pil = Image.fromarray(arr, 'RGB')
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=90)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _safety_settings(self, level: str):
        harm_cats = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
        threshold_map = {
            "Block None": "BLOCK_NONE",
            "Block Few (High Only)": "BLOCK_ONLY_HIGH",
            "Block Some (Medium+)": "BLOCK_MEDIUM_AND_ABOVE",
            "Block Most (Low+)": "BLOCK_LOW_AND_ABOVE",
        }
        threshold = threshold_map.get(level, "BLOCK_NONE")
        return [{"category": c, "threshold": threshold} for c in harm_cats]

    def generate_content(self, api_key, model, seed, temperature, enable_thinking,
                         safety_level, prompt, image_1=None, image_2=None, custom_model=""):
        if not api_key or not api_key.strip():
            raise ValueError("[OmniSimulator] API key is required.")
        if not _HTTP_OK:
            raise RuntimeError("[OmniSimulator] Python's urllib is unavailable in this environment.")

        if custom_model and custom_model.strip():
            model = custom_model.strip()
        elif model not in GEMINI_MODELS:
            model = "gemini-2.5-pro"
            
        if safety_level not in SAFETY_LEVELS:
            safety_level = "Block None"

        parts = []
        for img_tensor in [image_1, image_2]:
            if img_tensor is not None:
                b64 = self._tensor_to_b64(img_tensor)
                parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})
        parts.append({"text": prompt or ""})

        config = {"temperature": temperature}
        
        has_thinking_support = any(x in model for x in ["2.5", "2.0"])
        if enable_thinking and has_thinking_support:
            config["thinkingConfig"] = {"thinkingBudget": 1024}

        payload = json.dumps({
            "contents": [{"parts": parts}],
            "generationConfig": config,
            "safetySettings": self._safety_settings(safety_level),
        }).encode('utf-8')

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method='POST')
            
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"[OmniSimulator] HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:500]}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"[OmniSimulator] Network error reaching Gemini API: {e.reason}")
        except TimeoutError:
            raise RuntimeError("[OmniSimulator] Request timed out after 120s.")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"[OmniSimulator] Could not parse Gemini's response: {e}")

        if "candidates" not in result:
            block_reason = result.get("promptFeedback", {}).get("blockReason")
            if block_reason:
                raise RuntimeError(f"[OmniSimulator] Prompt was blocked by Gemini safety filters: {block_reason}")
            raise RuntimeError(f"[OmniSimulator] Unexpected response (no candidates): {json.dumps(result)[:500]}")

        text = ""
        for cand in result.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    text += part["text"]
        return (text.strip(),)

NODE_CLASS_MAPPINGS = {"OmniSimulator_GeminiNative": OmniSimulator_GeminiNative}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_GeminiNative": "Omni: Gemini Native"}