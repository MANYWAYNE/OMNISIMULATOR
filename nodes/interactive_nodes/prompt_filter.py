import sys
import os
import torch
import hashlib
import json
from nodes import PreviewImage
from comfy.model_management import InterruptProcessingException
from .image_filter_messaging import send_and_wait, Response, TimeoutResponse

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
if _ROOT not in sys.path: 
    sys.path.insert(0, _ROOT)

HIDDEN = {
    "prompt": "PROMPT",
    "extra_pnginfo": "EXTRA_PNGINFO",
    "uid": "UNIQUE_ID",
    "node_identifier": "NID",
}


class OMNI_PromptFilter(PreviewImage):
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "STRING", "INT")
    RETURN_NAMES = ("images", "prompt_list_positive", "prompt_list_negative", "seed_list", "indexes", "count")
    OUTPUT_IS_LIST = (True, True, True, True, False, False)
    INPUT_IS_LIST = True
    FUNCTION = "func"
    CATEGORY = "OmniSimulator/Interactive"
    OUTPUT_NODE = False
    DESCRIPTION = "Filter images and their associated prompts/seeds together"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt_list_positive": ("STRING", {"forceInput": True}),
                "prompt_list_negative": ("STRING", {"forceInput": True}),
                "timeout": ("INT", {"default": 600, "min": 1, "max": 9999999, "tooltip": "Timeout in seconds."}),
                "ontimeout": (["send none", "send all", "send first", "send last"], {}),
                "cache_behavior": (["Run selector normally", "Resend previous selection"], {
                    "tooltip": "Behavior when a cached selection for this image batch already exists."
                }),
            },
            "optional": {
                "seed_list": ("INT", {"forceInput": True}),
                "tip": ("STRING", {"default": "", "tooltip": "Optional tip text displayed in popup window"}),
            },
            "hidden": HIDDEN,
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs):
        return True

    def func(self, **kwargs):
        images = kwargs.get('images', [])
        prompt_list_positive = kwargs.get('prompt_list_positive', [])
        prompt_list_negative = kwargs.get('prompt_list_negative', [])
        seed_list = kwargs.get('seed_list')
        timeout = kwargs.get('timeout', [600])
        ontimeout = kwargs.get('ontimeout', ['send none'])
        cache_behavior = kwargs.get('cache_behavior', ['Run selector normally'])
        tip = kwargs.get('tip', [''])
        uid = kwargs.get('uid')
        node_identifier = kwargs.get('node_identifier')

        timeout_val = timeout[0] if isinstance(timeout, list) else timeout
        ontimeout_val = ontimeout[0] if isinstance(ontimeout, list) else ontimeout
        cache_behavior_val = cache_behavior[0] if isinstance(cache_behavior, list) else cache_behavior
        uid_val = uid[0] if isinstance(uid, list) else uid
        node_identifier_val = node_identifier[0] if isinstance(node_identifier, list) else node_identifier
        tip_val = (tip[0] if isinstance(tip, list) and tip else tip) or ""

        save_kwargs = {}
        if "prompt" in kwargs:
            p = kwargs.get("prompt")
            save_kwargs["prompt"] = p[0] if isinstance(p, list) else p
        if "extra_pnginfo" in kwargs:
            e = kwargs.get("extra_pnginfo")
            save_kwargs["extra_pnginfo"] = e[0] if isinstance(e, list) else e

        image_list = list(images) if isinstance(images, (list, tuple)) else [images]
        pos_list = list(prompt_list_positive) if isinstance(prompt_list_positive, (list, tuple)) else [prompt_list_positive]
        neg_list = list(prompt_list_negative) if isinstance(prompt_list_negative, (list, tuple)) else [prompt_list_negative]
        seeds = list(seed_list) if seed_list is not None and isinstance(seed_list, (list, tuple)) else (
            [seed_list[0]] if seed_list is not None else None
        )

        B = len(image_list)

        if B == 0:
            raise InterruptProcessingException()

        normalized_images = []
        for img in image_list:
            if img.dim() == 3: 
                img = img.unsqueeze(0)
            normalized_images.append(img)

        batch_tensor = torch.cat(normalized_images, dim=0)

        if len(pos_list) < B:
            pos_list = pos_list + [""] * (B - len(pos_list))
        if len(neg_list) < B:
            neg_list = neg_list + [""] * (B - len(neg_list))
        if seeds is None:
            seeds = [0] * B
        elif len(seeds) < B:
            seeds = seeds + [0] * (B - len(seeds))

        cache_dir = os.path.join(_ROOT, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        hasher = hashlib.sha256()
        hasher.update(batch_tensor.cpu().numpy().tobytes())
        cache_key = hasher.hexdigest()
        cache_filepath = os.path.join(cache_dir, f"{cache_key}_prompt_filter_selection.json")

        images_to_return = None

        if cache_behavior_val == "Resend previous selection" and os.path.exists(cache_filepath):
            print(f"[OmniSimulator] Using cached selection for images {cache_key[:8]}...")
            with open(cache_filepath, 'r') as f:
                cached_data = json.load(f)
            images_to_return = cached_data.get('selection', [])
        else:
            all_the_same = (B > 1 and all((batch_tensor[i] == batch_tensor[0]).all() for i in range(1, B)))
            urls = self.save_images(images=batch_tensor, **save_kwargs)['ui']['images']
            payload = {
                "uid": uid_val,
                "urls": urls,
                "allsame": all_the_same,
                "extras": ["", "", ""],
                "tip": tip_val,
                "video_frames": 1,
            }

            response: Response = send_and_wait(payload, timeout_val, uid_val, node_identifier_val)

            if isinstance(response, TimeoutResponse):
                if ontimeout_val == 'send none':
                    images_to_return = []
                elif ontimeout_val == 'send all':
                    images_to_return = list(range(B))
                elif ontimeout_val == 'send first':
                    images_to_return = [0]
                elif ontimeout_val == 'send last':
                    images_to_return = [B - 1]
            else:
                images_to_return = [int(x) for x in response.selection] if response.selection else []

            if not isinstance(response, TimeoutResponse):
                print(f"[OmniSimulator] Saving selection to cache: {cache_filepath}")
                cache_data = {'selection': images_to_return}
                with open(cache_filepath, 'w') as f:
                    json.dump(cache_data, f)

        if images_to_return is None or len(images_to_return) == 0:
            raise InterruptProcessingException()

        final_indices = images_to_return

        filtered_images = [normalized_images[int(i)] for i in final_indices]
        filtered_pos = [pos_list[int(i)] for i in final_indices]
        filtered_neg = [neg_list[int(i)] for i in final_indices]
        filtered_seeds = [seeds[int(i)] for i in final_indices]

        indexes_str = ",".join(str(int(x)) for x in final_indices)
        count = len(final_indices)

        print(f"[OmniSimulator] Selected {count} items at indices: {indexes_str}")

        return (filtered_images, filtered_pos, filtered_neg, filtered_seeds, indexes_str, count)


NODE_CLASS_MAPPINGS = {"OMNI_PromptFilter": OMNI_PromptFilter}
NODE_DISPLAY_NAME_MAPPINGS = {"OMNI_PromptFilter": "OMNI Prompt Filter"}