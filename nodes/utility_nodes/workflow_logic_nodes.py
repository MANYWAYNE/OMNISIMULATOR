import sys as _sys, os as _os
_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.realpath(__file__)), '..', '..'))
if _ROOT not in _sys.path: _sys.path.insert(0, _ROOT)
del _sys, _os

import threading
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import importlib.util as _ilu

_LANCZOS = getattr(Image, "LANCZOS", None) or Image.Resampling.LANCZOS

try:
    _current_dir = os.path.dirname(os.path.realpath(__file__))
    _gn_path = os.path.join(_current_dir, 'gemini_native.py')
    
    if not os.path.exists(_gn_path):
        _gn_path = os.path.join(_ROOT, 'nodes', 'api_nodes', 'gemini_native.py')

    if os.path.exists(_gn_path):
        _gn_spec = _ilu.spec_from_file_location('omnisimulator_gemini_native', _gn_path)
        _gn_mod  = _ilu.module_from_spec(_gn_spec)
        _gn_spec.loader.exec_module(_gn_mod)
        OmniSimulator_GeminiNative = _gn_mod.OmniSimulator_GeminiNative
    else:
        raise FileNotFoundError(f"File gemini_native.py not found at expected paths.")
except Exception as _e:
    print(f"[OmniSimulator] workflow_logic_nodes: Gemini import warning: {_e}")
    class OmniSimulator_GeminiNative:
        @classmethod
        def INPUT_TYPES(cls): return {"required": {"api_key": ("STRING", {"default": ""}), "model": (["gemini-2.5-pro"],), "seed": ("INT", {"default": 0}), "temperature": ("FLOAT", {"default": 0.1}), "enable_thinking": ("BOOLEAN", {"default": True}), "safety_level": (["Block None"],), "prompt": ("STRING", {"default": ""})}}
        RETURN_TYPES = ("STRING",)
        FUNCTION = "generate_content"
        CATEGORY = "OmniSimulator/API"
        def generate_content(self, **kwargs): return ("Gemini not configured or module missing.",)

DEFAULT_PROMPT_TEMPLATE = """{llm_base_prompt}

Based on the image and the rules above, generate a description.
---
USER REQUEST: "{user_prompt}"
---
IMPORTANT: You MUST modify your description to perfectly match the USER REQUEST. The user's request has the highest priority and should be treated as a final instruction that overrides the image's content if there is a conflict. Provide only the final, clean description without commentary."""


def _fill_template(template, llm_base_prompt, user_prompt):
    if not isinstance(template, str):
        template = DEFAULT_PROMPT_TEMPLATE
    return (template
            .replace("{llm_base_prompt}", llm_base_prompt or "")
            .replace("{user_prompt}", user_prompt or ""))


def tensor_to_pil(image_tensor):
    return Image.fromarray(np.clip(255. * image_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil_to_tensor(pil_image):
    return torch.from_numpy(np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)

def _load_label_font(size: int):
    candidates = [
        os.path.join(_ROOT, 'fonts', 'BricolageGrotesque.ttf'),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
    ]
    for path in candidates:
        try:
            if os.path.isabs(path) and not os.path.exists(path):
                continue
            return ImageFont.truetype(path, max(size, 1))
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=max(size, 1))
    except TypeError:
        return ImageFont.load_default()


class OmniSimulator_FeatureDescriber:
    @classmethod
    def INPUT_TYPES(cls):
        gemini_inputs = OmniSimulator_GeminiNative.INPUT_TYPES()
        required = gemini_inputs.get("required", {})
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "feature_image": ("IMAGE",),
                "use_llm_description": ("BOOLEAN", {"default": True}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "llm_base_prompt": ("STRING", {"multiline": True, "default": "Describe this feature accurately..."}),
                "prompt_template": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT_TEMPLATE}),
                "api_key": required.get("api_key", ("STRING", {"default": ""})),
                "model": required.get("model", (["gemini-2.5-pro", "gemini-2.5-flash"], {"default": "gemini-2.5-pro"})),
                "seed": required.get("seed", ("INT", {"default": 1111111})),
                "temperature": required.get("temperature", ("FLOAT", {"default": 0.111})),
                "enable_thinking": required.get("enable_thinking", ("BOOLEAN", {"default": True})),
                "safety_level": required.get("safety_level", (["Block None"], {"default": "Block None"})),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    FUNCTION = "describe"
    CATEGORY = "OmniSimulator/Workflow Logic"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs): return True

    def describe(self, **kwargs):
        if not kwargs.get('enabled'): return (None, None)
        final_desc = (kwargs.get('user_prompt') or '').strip()
        if kwargs.get('use_llm_description'):
            prompt = _fill_template(kwargs.get('prompt_template', DEFAULT_PROMPT_TEMPLATE), kwargs.get('llm_base_prompt', ''), final_desc)
            gemini_kwargs = {k: v for k, v in kwargs.items() if k not in ('enabled', 'feature_image', 'use_llm_description', 'user_prompt', 'llm_base_prompt', 'prompt_template')}
            final_desc = OmniSimulator_GeminiNative().generate_content(**gemini_kwargs, prompt=prompt, image_1=kwargs.get('feature_image'))[0].strip()
        return (final_desc, kwargs.get('feature_image'))


class OmniSimulator_UniversalDescriber:
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "enabled": ("BOOLEAN", {"default": True}), "use_llm_description": ("BOOLEAN", {"default": True}), "user_prompt": ("STRING", {"multiline": True}), "llm_base_prompt": ("STRING", {"multiline": True}), "prompt_template": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT_TEMPLATE}), }, "optional": { "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",) } }
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE",)
    FUNCTION = "describe"
    CATEGORY = "OmniSimulator/Workflow Logic"

    def describe(self, **kwargs):
        images = (kwargs.get('image_1'), kwargs.get('image_2'), kwargs.get('image_3'), kwargs.get('image_4'))
        return (str(kwargs.get('user_prompt')), *images)


class OmniSimulator_SwapPromptAssembler:
    @classmethod
    def INPUT_TYPES(cls): return { "required": { "hair_prefix": ("STRING", {}), "outfit_prefix": ("STRING", {}), "separator": ("STRING", {"default": " + "}), }, "optional": { "hair_description": ("STRING", {"forceInput": True}), "outfit_description": ("STRING", {"forceInput": True}), } }
    RETURN_TYPES = ("STRING", "BOOLEAN",); FUNCTION = "assemble"; CATEGORY = "OmniSimulator/Workflow Logic"
    def assemble(self, hair_prefix, outfit_prefix, separator, hair_description=None, outfit_description=None):
        parts = []
        if hair_description and hair_description.strip(): parts.append(f"{hair_prefix.strip()} {hair_description.strip()}")
        if outfit_description and outfit_description.strip(): parts.append(f"{outfit_prefix.strip()} {outfit_description.strip()}")
        return (separator.join(parts), bool(parts))

class OmniSimulator_ParallelFeatureDescriber:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"hair_enabled": ("BOOLEAN", {"default": True}), "outfit_enabled": ("BOOLEAN", {"default": True})}, "optional": {"hair_feature_image": ("IMAGE",), "outfit_feature_image": ("IMAGE",)}}
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "IMAGE",)
    FUNCTION = "describe_parallel"
    CATEGORY = "OmniSimulator/Workflow Logic"
    def describe_parallel(self, **kwargs): return (None, kwargs.get("hair_feature_image"), None, kwargs.get("outfit_feature_image"))

class OmniSimulator_SeeDreamPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls): return { "required": { "hair_prefix_single": ("STRING", {}), "outfit_prefix_single": ("STRING", {}), "hair_prefix_multi_template": ("STRING", {"default": "perfectly change hair of image {} to"}), "outfit_prefix_multi_template": ("STRING", {"default": "perfectly change outfit of image {} to"}), "separator": ("STRING", {"default": " + "}), }, "optional": { "hair_description": ("STRING", {"forceInput": True}), "outfit_description": ("STRING", {"forceInput": True}), "hair_image_ref": ("IMAGE",), "outfit_image_ref": ("IMAGE",), } }
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE",); FUNCTION = "build_prompt"; CATEGORY = "OmniSimulator/Workflow Logic"
    def build_prompt(self, hair_prefix_single, outfit_prefix_single, hair_prefix_multi_template, outfit_prefix_multi_template, separator, hair_description=None, outfit_description=None, hair_image_ref=None, outfit_image_ref=None):
        return (f"{hair_description} + {outfit_description}", hair_image_ref, outfit_image_ref)

class OmniSimulator_PreviewAssembler:
    @classmethod
    def INPUT_TYPES(cls): return { "required": { "main_image": ("IMAGE",), "layout": (["Horizontal", "Vertical"],), "spacing": ("INT", {"default": 10}) }, "optional": { "hair_image_ref": ("IMAGE",), "outfit_image_ref": ("IMAGE",) } }
    RETURN_TYPES = ("IMAGE",); FUNCTION = "assemble_preview"; CATEGORY = "OmniSimulator/Workflow Logic"
    def assemble_preview(self, main_image, layout, spacing, hair_image_ref=None, outfit_image_ref=None): return (main_image,)

class OmniSimulator_TeleportInputAssembler:
    @classmethod
    def INPUT_TYPES(cls): return { "required": { "character_image": ("IMAGE",), "teleport_to_image": ("IMAGE",) } }
    RETURN_TYPES = ("IMAGE",); FUNCTION = "assemble_inputs"; CATEGORY = "OmniSimulator/Workflow Logic"
    def assemble_inputs(self, character_image, teleport_to_image): return (character_image,)

NODE_CLASS_MAPPINGS = {
    "OmniSimulator_FeatureDescriber": OmniSimulator_FeatureDescriber,
    "OmniSimulator_UniversalDescriber": OmniSimulator_UniversalDescriber,
    "OmniSimulator_SwapPromptAssembler": OmniSimulator_SwapPromptAssembler,
    "OmniSimulator_ParallelFeatureDescriber": OmniSimulator_ParallelFeatureDescriber,
    "OmniSimulator_SeeDreamPromptBuilder": OmniSimulator_SeeDreamPromptBuilder,
    "OmniSimulator_PreviewAssembler": OmniSimulator_PreviewAssembler,
    "OmniSimulator_TeleportInputAssembler": OmniSimulator_TeleportInputAssembler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniSimulator_FeatureDescriber": "OmniSimulator Feature Describer (Single)",
    "OmniSimulator_UniversalDescriber": "OmniSimulator Universal Describer",
    "OmniSimulator_SwapPromptAssembler": "OmniSimulator Swap Prompt Assembler",
    "OmniSimulator_ParallelFeatureDescriber": "OmniSimulator Feature Describer (Parallel)",
    "OmniSimulator_SeeDreamPromptBuilder": "OmniSimulator SeeDream Prompt Builder",
    "OmniSimulator_PreviewAssembler": "OmniSimulator Preview Assembler",
    "OmniSimulator_TeleportInputAssembler": "OmniSimulator Teleport Input Assembler",
}