import sys
import os
import random
import torch
import hashlib
import json
import base64
import io
from PIL import Image
import numpy as np

from nodes import PreviewImage, LoadImage
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

def pil_to_natural_mask_tensor(pil_image):
    if pil_image.mode == 'RGBA':
        mask_np = np.array(pil_image.split()[-1]).astype(np.float32) / 255.0
    else:
        mask_np = np.array(pil_image.convert("L")).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
    return mask_tensor

def save_natural_mask_as_alpha(natural_mask_tensor, filepath):
    mask_np = natural_mask_tensor.cpu().numpy().squeeze()
    rgb = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    alpha = (mask_np * 255).astype(np.uint8)
    rgba_image = Image.fromarray(np.dstack((rgb, alpha)), 'RGBA')
    rgba_image.save(filepath, 'PNG')


class OMNI_ImageFilter(PreviewImage):
    RETURN_TYPES = ("IMAGE", "LATENT", "MASK", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "latents", "masks", "extra1", "extra2", "extra3", "indexes")
    FUNCTION = "func"
    CATEGORY = "OmniSimulator/Interactive"
    OUTPUT_NODE = False
    DESCRIPTION = "Allows you to preview images and choose which, if any to proceed with"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "images": ("IMAGE", ), 
                "timeout": ("INT", {"default": 600, "min": 1, "max": 9999999, "tooltip": "Timeout in seconds."}),
                "ontimeout": (["send none", "send all", "send first", "send last"], {}),
                "cache_behavior": (["Run selector normally", "Resend previous selection"], {
                    "tooltip": "Behavior when a cached selection for this image batch already exists."
                }),
            },
            "optional": {
                "latents": ("LATENT", {"tooltip": "Optional - if provided, will be output"}),
                "masks": ("MASK", {"tooltip": "Optional - if provided, will be output"}),
                "tip": ("STRING", {"default": "", "tooltip": "Optional - if provided, will be displayed in popup window"}),
                "extra1": ("STRING", {"default": ""}),
                "extra2": ("STRING", {"default": ""}),
                "extra3": ("STRING", {"default": ""}),
                "pick_list_start": ("INT", {"default": 0, "tooltip": "The number used in pick_list for the first image"}),
                "pick_list": ("STRING", {"default": "", "tooltip": "If a comma separated list of integers is provided, the images with these indices will be selected automatically."}),
                "video_frames": ("INT", {"default": 1, "min": 1, "tooltip": "treat each block of n images as a video"}),
            },
            "hidden": HIDDEN,
        }
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs):
        return True
    
    def func(self, images, timeout, ontimeout, cache_behavior, uid, node_identifier, tip="", extra1="", extra2="", extra3="", latents=None, masks=None, pick_list_start: int = 0, pick_list: str = "", video_frames: int = 1, **kwargs):
        e1, e2, e3 = extra1, extra2, extra3
        B = images.shape[0]
        if video_frames > B: 
            video_frames = 1

        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)

        hasher = hashlib.sha256()
        hasher.update(images.cpu().numpy().tobytes())
        cache_key = hasher.hexdigest()
        cache_filepath = os.path.join(cache_dir, f"{cache_key}_selection.json")

        images_to_return = None

        if cache_behavior == "Resend previous selection" and os.path.exists(cache_filepath):
            print(f"[OmniSimulator] Image Filter: Using cached selection for images {cache_key[:8]}...")
            with open(cache_filepath, 'r') as f:
                cached_data = json.load(f)
            images_to_return = cached_data.get('selection', [])
            e1, e2, e3 = cached_data.get('extras', [extra1, extra2, extra3])
        else:
            try:    
                images_to_return = [int(x.strip()) % B for x in pick_list.split(',')] if pick_list else []
            except Exception as e: 
                print(f"[OmniSimulator] {e} parsing pick_list - will manually select")
                images_to_return = []

            if len(images_to_return) == 0:
                all_the_same = (B and all((images[i] == images[0]).all() for i in range(1, B))) 
                urls = self.save_images(images=images, **kwargs)['ui']['images']
                payload = {"uid": uid, "urls": urls, "allsame": all_the_same, "extras": [extra1, extra2, extra3], "tip": tip, "video_frames": video_frames}

                response = send_and_wait(payload, timeout, uid, node_identifier)

                if isinstance(response, TimeoutResponse):
                    if ontimeout == 'send none':  
                        images_to_return = []
                    if ontimeout == 'send all':   
                        images_to_return = [*range(len(images) // video_frames)]
                    if ontimeout == 'send first': 
                        images_to_return = [0, ]
                    if ontimeout == 'send last':  
                        images_to_return = [(len(images) // video_frames) - 1, ]
                else:
                    e1, e2, e3 = response.get_extras([extra1, extra2, extra3])
                    images_to_return = [int(x) for x in response.selection] if response.selection else []
                
                if not isinstance(response, TimeoutResponse):
                    print(f"[OmniSimulator] Saving image selection to cache: {cache_filepath}")
                    cache_data = {'selection': images_to_return, 'extras': [e1, e2, e3]}
                    with open(cache_filepath, 'w') as f:
                        json.dump(cache_data, f)

        if images_to_return is None or len(images_to_return) == 0: 
            raise InterruptProcessingException()

        if video_frames > 1:
            final_indices = [key * video_frames + frm for key in images_to_return for frm in range(video_frames)]
        else:
            final_indices = images_to_return

        images = torch.stack(list(images[int(i)] for i in final_indices))
        latents = {"samples": torch.stack(list(latents['samples'][int(i)] for i in final_indices))} if latents is not None else None
        masks = torch.stack(list(masks[int(i)] for i in final_indices)) if masks is not None else None

        try: 
            int(pick_list_start)
        except: 
            pick_list_start = 0

        return (images, latents, masks, e1, e2, e3, ",".join(str(int(x) + int(pick_list_start)) for x in images_to_return))

    
class OMNI_TextImageFilter(PreviewImage):
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "text", "extra1", "extra2", "extra3")
    FUNCTION = "func"
    CATEGORY = "OmniSimulator/Interactive"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "image": ("IMAGE", ), 
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Bypassed"}),
                "text": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 600, "min": 1, "max": 9999999, "tooltip": "Timeout in seconds."}),
                "cache_behavior": (["Edit (Loads Saved Text)", "Bypass if Image is Cached", "Bypass & Repeat Last Text"], {
                    "tooltip": (
                        "Edit (Loads Saved Text): Always shows the popup. Loads your last saved edit for this image if available.\n\n"
                        "Bypass if Image is Cached: Skips the popup and automatically uses the saved text for this specific image if it exists. If not, the editor appears.\n\n"
                        "Bypass & Repeat Last Text: Skips the popup and automatically uses the last text you entered in this node for any image. If not used before, the editor appears."
                    )
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional - if provided, will be overlaid on image"}),
                "tip": ("STRING", {"default": "", "tooltip": "Optional - if provided, will be displayed in popup window"}),
                "extra1": ("STRING", {"default": ""}),
                "extra2": ("STRING", {"default": ""}),
                "extra3": ("STRING", {"default": ""}),
                "textareaheight": ("INT", {"default": 150, "min": 50, "max": 500, "tooltip": "Height of text area in pixels"}),
            },
            "hidden": HIDDEN,
        }
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs):
        return True
    
    def func(self, image, enabled, text, timeout, cache_behavior, uid, node_identifier, extra1="", extra2="", extra3="", mask=None, tip="", textareaheight=None, **kwargs):
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        node_memory_filepath = os.path.join(cache_dir, f"{hashlib.sha256(uid.encode('utf-8') + b'text_filter_node_memory').hexdigest()}_text_edit.json")
        image_specific_filepath = os.path.join(cache_dir, f"{hashlib.sha256(uid.encode('utf-8') + image.cpu().numpy().tobytes()).hexdigest()}_text_edit.json")

        if not enabled:
            print("[OmniSimulator] Text Filter: Bypassed. Checking for most relevant cached text...")
            if os.path.exists(node_memory_filepath):
                with open(node_memory_filepath, 'r') as f: 
                    cached_data = json.load(f)
                print(f"[OmniSimulator] Bypassed Mode: Using node's last saved text from {node_memory_filepath}")
                return (image, cached_data.get('edited_text', text), *cached_data.get('extras', [extra1, extra2, extra3]))
            if os.path.exists(image_specific_filepath):
                with open(image_specific_filepath, 'r') as f: 
                    cached_data = json.load(f)
                print(f"[OmniSimulator] Bypassed Mode: Using image-specific cached text from {image_specific_filepath}")
                return (image, cached_data.get('edited_text', text), *cached_data.get('extras', [extra1, extra2, extra3]))
            print("[OmniSimulator] Bypassed Mode: No cache found. Passing input text.")
            return (image, text, extra1, extra2, extra3)

        if cache_behavior == "Bypass & Repeat Last Text":
            if os.path.exists(node_memory_filepath):
                print(f"[OmniSimulator] Active Mode ('Repeat'): Using node memory cache from {node_memory_filepath}")
                with open(node_memory_filepath, 'r') as f: 
                    cached_data = json.load(f)
                return (image, cached_data.get('edited_text', text), *cached_data.get('extras', [extra1, extra2, extra3]))
            else:
                print("[OmniSimulator] Active Mode ('Repeat'): No node memory cache found. Proceeding to editor.")
        
        most_relevant_cached_text = None
        most_relevant_extras = [extra1, extra2, extra3]

        if os.path.exists(node_memory_filepath):
            try:
                with open(node_memory_filepath, 'r') as f: 
                    data = json.load(f)
                if data.get('source_text') == text:
                    most_relevant_cached_text = data.get('edited_text')
                    most_relevant_extras = data.get('extras', most_relevant_extras)
                    print("[OmniSimulator] Found valid, most recent text in the node's memory cache.")
            except Exception: 
                pass
        
        if most_relevant_cached_text is None and os.path.exists(image_specific_filepath):
            try:
                with open(image_specific_filepath, 'r') as f: 
                    data = json.load(f)
                if data.get('source_text') == text:
                    most_relevant_cached_text = data.get('edited_text')
                    most_relevant_extras = data.get('extras', most_relevant_extras)
                    print("[OmniSimulator] Found valid text in the image-specific cache.")
            except Exception: 
                pass

        if cache_behavior == "Bypass if Image is Cached" and most_relevant_cached_text is not None:
            print(f"[OmniSimulator] Active Mode ('Bypass'): Valid cache found. Bypassing popup.")
            return (image, most_relevant_cached_text, *most_relevant_extras)

        text_for_editor = text 
        if most_relevant_cached_text is not None:
            text_for_editor = most_relevant_cached_text
            print(f"[OmniSimulator] Active Mode ('Edit'): Pre-filling editor with most relevant cached text.")

        print("[OmniSimulator] Text Filter: Launching editor...")
        urls = self.save_images(images=image, **kwargs)['ui']['images']
        payload = {"uid": uid, "urls": urls, "text": text_for_editor, "extras": most_relevant_extras, "tip": tip}
        if textareaheight is not None: 
            payload['textareaheight'] = textareaheight
        if mask is not None: 
            payload['mask_urls'] = self.save_images(images=mask_to_image(mask), **kwargs)['ui']['images']

        response = send_and_wait(payload, timeout, uid, node_identifier)
        
        if isinstance(response, TimeoutResponse):
            return (image, text_for_editor, *most_relevant_extras)

        final_text = response.text
        e1, e2, e3 = response.get_extras(most_relevant_extras)

        new_cache_data = {
            'source_text': text,       
            'edited_text': final_text, 
            'extras': [e1, e2, e3]
        }
        
        print(f"[OmniSimulator] Saving to image-specific cache: {image_specific_filepath}")
        with open(image_specific_filepath, 'w') as f:
            json.dump(new_cache_data, f)
        
        print(f"[OmniSimulator] Updating node's persistent memory (most recent action): {node_memory_filepath}")
        with open(node_memory_filepath, 'w') as f:
            json.dump(new_cache_data, f)
            
        return (image, final_text, e1, e2, e3)


def mask_to_image(mask: torch.Tensor):
    return torch.stack([mask, mask, mask, 1.0 - mask], -1)

    
class OMNI_MaskImageFilter(PreviewImage, LoadImage):
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "mask_inverted", "extra1", "extra2", "extra3")
    FUNCTION = "func"
    CATEGORY = "OmniSimulator/Interactive"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "image": ("IMAGE", ),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Bypassed"}),
                "timeout": ("INT", {"default": 600, "min": 1, "max": 9999999, "tooltip": "Timeout in seconds."}),
                "if_no_mask": (["cancel", "send blank"], {}),
                "cache_behavior": (["Run editor normally", "Edit previous mask", "Resend previous mask"], {
                    "tooltip": "Behavior when a cached mask for this image already exists."
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "optional initial mask"}),
                "tip": ("STRING", {"default": "", "tooltip": "Optional - if provided, will be displayed in popup window"}),
                "extra1": ("STRING", {"default": ""}),
                "extra2": ("STRING", {"default": ""}),
                "extra3": ("STRING", {"default": ""}),
            },
            "hidden": HIDDEN,
        }
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return f"{random.random()}"
    
    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs): 
        return True
    
    def func(self, image, enabled, timeout, uid, if_no_mask, cache_behavior, node_identifier, mask=None, extra1="", extra2="", extra3="", tip="", **kwargs):
        if not enabled:
            print("[OmniSimulator] Mask Filter: Bypassed")
            if mask is None:
                mask = torch.zeros_like(image[:, :, :, 0])
            return (image, mask, 1.0 - mask, extra1, extra2, extra3)

        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        hasher = hashlib.sha256()
        hasher.update(image.cpu().numpy().tobytes())
        cache_key = hasher.hexdigest()
        cache_filepath = os.path.join(cache_dir, f"{cache_key}_mask.png")

        if cache_behavior == "Resend previous mask" and os.path.exists(cache_filepath):
            print(f"[OmniSimulator] Mask Cache Hit! Resending previous mask from {cache_filepath}")
            pil_mask = Image.open(cache_filepath)
            natural_mask = pil_to_natural_mask_tensor(pil_mask)
            return (image, natural_mask, 1.0 - natural_mask, extra1, extra2, extra3)

        initial_mask_for_editor = None
        target_device = image.device
        
        if mask is not None:
            initial_mask_for_editor = mask.to(target_device)
        
        if cache_behavior == "Edit previous mask" and os.path.exists(cache_filepath):
            print(f"[OmniSimulator] Mask Cache: Loading previous mask for editing from {cache_filepath}")
            pil_mask = Image.open(cache_filepath)
            initial_mask_for_editor = pil_to_natural_mask_tensor(pil_mask).to(target_device)

        if initial_mask_for_editor is not None:
            image_rgb = image[:, :, :, :3]
            alpha_channel = initial_mask_for_editor.unsqueeze(-1)
            saveable = torch.cat((image_rgb, alpha_channel), dim=-1)
        else:
            saveable = image

        urls = self.save_images(images=saveable, **kwargs)['ui']['images']
        payload = {"uid": uid, "urls": urls, "maskedit": True, "extras": [extra1, extra2, extra3], "tip": tip}
        
        print("[OmniSimulator] Mask Filter: Waiting for user interaction...")
        response = send_and_wait(payload, timeout, uid, node_identifier)
        
        if (response.masked_image):
            print(f"[OmniSimulator] Using masked_image path: {response.masked_image}")
            try:
                loaded_image, mask_from_editor = self.load_image(os.path.join('clipspace', response.masked_image) + " [input]")
                final_natural_mask = 1.0 - mask_from_editor

                print(f"[OmniSimulator] Saving corrected natural_mask to cache: {cache_filepath}")
                save_natural_mask_as_alpha(final_natural_mask, cache_filepath)

                return (loaded_image, final_natural_mask, 1.0 - final_natural_mask, *response.get_extras([extra1, extra2, extra3]))
            except FileNotFoundError:
                pass
        elif (response.masked_data):
            try:
                data = response.masked_data.split(',', 1)[-1]
                bytes_data = data.encode('utf-8')
                image_data = base64.decodebytes(bytes_data)
                data_io = io.BytesIO(image_data)
                img = Image.open(data_io)

                if 'A' in img.getbands():
                    mask_np = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                else:
                    mask_np = np.array(img.convert('L')).astype(np.float32) / 255.0

                final_natural_mask = torch.from_numpy(mask_np).unsqueeze(0)

                print(f"[OmniSimulator] Saving mask from data URL to cache: {cache_filepath}")
                save_natural_mask_as_alpha(final_natural_mask, cache_filepath)

                return (image, final_natural_mask, 1.0 - final_natural_mask, *response.get_extras([extra1, extra2, extra3]))
            except Exception as e:
                print(f"[OmniSimulator] Error processing masked_data: {e}")

        if if_no_mask == 'cancel':
            raise InterruptProcessingException()

        blank_natural_mask = torch.zeros_like(image[:, :, :, 0])
        return (image, blank_natural_mask, 1.0 - blank_natural_mask, *response.get_extras([extra1, extra2, extra3]))


NODE_CLASS_MAPPINGS = {
    "OMNI_ImageFilter":     OMNI_ImageFilter,
    "OMNI_TextImageFilter": OMNI_TextImageFilter,
    "OMNI_MaskImageFilter": OMNI_MaskImageFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OMNI_ImageFilter":     "OMNI Image Filter",
    "OMNI_TextImageFilter": "OMNI Text/Image Filter",
    "OMNI_MaskImageFilter": "OMNI Mask Filter",
}