import sys
import os
import torch
import json
import hashlib
from comfy.model_management import InterruptProcessingException
from nodes import PreviewImage
from .image_filter_messaging import send_and_wait, TimeoutResponse

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
if _ROOT not in sys.path: 
    sys.path.insert(0, _ROOT)

ASPECT_RATIOS = {
    "None (free)": None,
    "896x1200 (Nano Banana 1K)": (896, 1200),
    "3:4 (Portrait)": (3, 4),
    "4:3 (Landscape)": (4, 3),
    "2:3 (Portrait)": (2, 3),
    "3:2 (Landscape)": (3, 2),
    "9:16 (Tall)": (9, 16),
    "16:9 (Wide)": (16, 9),
    "1:1 (Square)": (1, 1),
    "4:5 (Portrait)": (4, 5),
    "5:4 (Landscape)": (5, 4),
}


class OMNI_Interactive_Crop(PreviewImage):
    RETURN_TYPES = ("IMAGE", "MASK", "STRING",)
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_data_json",)
    FUNCTION = "crop_interactively"
    CATEGORY = "OmniSimulator/Interactive"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "timeout": ("INT", {"default": 600, "min": 1, "max": 999999}),
                "cache_behavior": (["Run editor normally", "Edit previous crop", "Resend previous crop"], {
                    "tooltip": "Behavior when a cached crop for this image already exists."
                }),
                "bypass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, skips interactive UI and applies proposed_crop_json directly (or full image if not provided)."
                }),
                "lock_aspect_ratio": (
                    list(ASPECT_RATIOS.keys()),
                    {
                        "default": "3:4 (Portrait)",
                        "tooltip": "Lock crop box to specific aspect ratio. Set to 'None (free)' for free-form cropping."
                    }
                ),
            },
            "optional": {
                "mask": ("MASK",),
                "proposed_crop_json": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "uid": "UNIQUE_ID",
                "node_identifier": "NID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs):
        return True

    def crop_interactively(self, **kwargs):
        image = kwargs.get('image')
        timeout = kwargs.get('timeout')
        cache_behavior = kwargs.get('cache_behavior', 'Run editor normally')
        bypass = kwargs.get('bypass', False)
        uid = kwargs.get('uid')
        node_identifier = kwargs.get('node_identifier')
        mask = kwargs.get('mask')
        proposed_crop_json = kwargs.get('proposed_crop_json')

        if image is None or timeout is None or uid is None or node_identifier is None:
            raise ValueError("OMNI_Interactive_Crop is missing required inputs. Check connections.")

        img_height = image.shape[1]
        img_width = image.shape[2]

        if bypass:
            if proposed_crop_json and proposed_crop_json.strip():
                try:
                    crop_data = json.loads(proposed_crop_json)
                    print(f"[OmniSimulator] Crop BYPASS: Applying proposed crop directly: {crop_data}")
                    return self._apply_crop(image, mask, crop_data)
                except json.JSONDecodeError:
                    print("[OmniSimulator] Crop BYPASS: Invalid proposed_crop_json, returning full image.")
            else:
                print("[OmniSimulator] Crop BYPASS: No proposed crop, returning full image.")

            full_mask = torch.ones(1, img_height, img_width) if mask is None else mask
            crop_data = {"x": 0, "y": 0, "width": img_width, "height": img_height}
            return (image, full_mask, json.dumps(crop_data))

        cache_dir = os.path.join(_ROOT, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        hasher = hashlib.sha256()
        hasher.update(image.cpu().numpy().tobytes())
        cache_key = hasher.hexdigest()
        cache_filepath = os.path.join(cache_dir, f"{cache_key}_crop.json")

        if cache_behavior == "Resend previous crop" and os.path.exists(cache_filepath):
            print(f"[OmniSimulator] Crop Cache Hit! Resending previous crop from {cache_filepath}")
            with open(cache_filepath, 'r') as f:
                crop_data = json.load(f)
            return self._apply_crop(image, mask, crop_data)

        save_kwargs = {}
        if "prompt" in kwargs: 
            save_kwargs["prompt"] = kwargs.get("prompt")
        if "extra_pnginfo" in kwargs: 
            save_kwargs["extra_pnginfo"] = kwargs.get("extra_pnginfo")

        urls = self.save_images(images=image, **save_kwargs)['ui']['images']

        lock_aspect_ratio = kwargs.get('lock_aspect_ratio', 'None (free)')
        aspect_ratio_tuple = ASPECT_RATIOS.get(lock_aspect_ratio)

        payload = {
            "uid": uid,
            "node_identifier": node_identifier,
            "urls": urls,
            "interactive_crop": True,
        }

        if aspect_ratio_tuple is not None:
            payload["lock_aspect_ratio"] = {
                "width": aspect_ratio_tuple[0],
                "height": aspect_ratio_tuple[1],
                "label": lock_aspect_ratio
            }

        if cache_behavior == "Edit previous crop" and os.path.exists(cache_filepath):
            print(f"[OmniSimulator] Crop Cache: Loading previous crop for editing from {cache_filepath}")
            with open(cache_filepath, 'r') as f:
                payload["proposed_crop"] = json.load(f)
        elif proposed_crop_json and proposed_crop_json.strip():
            try:
                payload["proposed_crop"] = json.loads(proposed_crop_json)
            except json.JSONDecodeError:
                print("[OmniSimulator] Interactive Crop: Invalid proposed_crop_json. Ignoring.")

        print("[OmniSimulator] Interactive Crop: Waiting for user to define crop area...")
        response = send_and_wait(payload, timeout, uid, node_identifier)

        if isinstance(response, TimeoutResponse):
            print("[OmniSimulator] Interactive Crop: Timed out. Passing through original image.")
            full_mask = torch.ones_like(image[:, :, :, 0]) if mask is None else mask
            crop_data = {"x": 0, "y": 0, "width": image.shape[2], "height": image.shape[1], "status": "timeout"}
            return (image, full_mask, json.dumps(crop_data))

        crop_data = response.crop if hasattr(response, 'crop') else None

        if not crop_data:
            raise InterruptProcessingException("User cancelled the crop operation.")

        print(f"[OmniSimulator] Interactive Crop: Received crop data: {crop_data}")

        print(f"[OmniSimulator] Saving crop data to cache: {cache_filepath}")
        with open(cache_filepath, 'w') as f:
            json.dump(crop_data, f)

        return self._apply_crop(image, mask, crop_data)

    def _apply_crop(self, image, mask, crop_data):
        img_height = image.shape[1]
        img_width = image.shape[2]

        x = int(crop_data.get("x", 0))
        y = int(crop_data.get("y", 0))
        width = int(crop_data.get("width", img_width))
        height = int(crop_data.get("height", img_height))

        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width = max(1, min(width, img_width - x))
        height = max(1, min(height, img_height - y))

        if x != crop_data.get("x") or y != crop_data.get("y") or width != crop_data.get("width") or height != crop_data.get("height"):
            print(f"[OmniSimulator] Crop: Adjusted crop to fit image bounds: x={x}, y={y}, w={width}, h={height}")

        cropped_image = image[:, y:y+height, x:x+width, :]

        cropped_mask = None
        if mask is not None:
            mask_to_crop = mask
            if mask.dim() == 4:
                mask_to_crop = mask.squeeze(1)

            cropped_mask = mask_to_crop[:, y:y+height, x:x+width]
        else:
            cropped_mask = torch.ones_like(cropped_image[:, :, :, 0])

        adjusted_crop_data = {"x": x, "y": y, "width": width, "height": height}
        return (cropped_image, cropped_mask, json.dumps(adjusted_crop_data))


NODE_CLASS_MAPPINGS = {"OMNI_Interactive_Crop": OMNI_Interactive_Crop}
NODE_DISPLAY_NAME_MAPPINGS = {"OMNI_Interactive_Crop": "OMNI Interactive Crop"}