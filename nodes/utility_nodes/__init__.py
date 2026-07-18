# ComfyUI_OmniSimulator/nodes/utility_nodes/__init__.py
import sys
import os
import importlib.util

_DIR  = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.path.normpath(os.path.join(_DIR, '..', '..'))
if _ROOT not in sys.path: 
    sys.path.insert(0, _ROOT)

NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _load(filename, extra_classes=None):
    path = os.path.join(_DIR, filename)
    if not os.path.isfile(path):
        return None
    mod_key = 'omni_util__' + filename.replace('.py', '').replace('/', '_')
    try:
        if mod_key in sys.modules:
            mod = sys.modules[mod_key]
        else:
            spec = importlib.util.spec_from_file_location(mod_key, path)
            mod  = importlib.util.module_from_spec(spec)
            mod.__package__ = mod_key
            sys.modules[mod_key] = mod
            spec.loader.exec_module(mod)
        NODE_CLASS_MAPPINGS.update(getattr(mod, 'NODE_CLASS_MAPPINGS', {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, 'NODE_DISPLAY_NAME_MAPPINGS', {}))
        for cls_name in (extra_classes or []):
            cls = getattr(mod, cls_name, None)
            if cls:
                NODE_CLASS_MAPPINGS[cls_name] = cls
        return mod
    except Exception as e:
        print(f'[OmniSimulator] {filename}: {e}')
        return None

_load('seed_generator.py')
_load('grow_mask_with_blur.py')
_load('feather_mask.py')
_load('image_resolution_clamp.py')
_load('api_provider_selector.py')
_load('realistic_noise.py')
_load('realistic_jpeg.py')
_load('workflow_logic_nodes.py')
_load('branding_node.py')
_load('image_resize_advanced.py')
_load('api_model_selector.py')
_load('json_utils.py')
_load('lut_selector.py')
_load('spectral_engine_node.py')
_load('color_science_node.py')
_load('neural_grain_node.py')
_load('lens_simulation_node.py')
_load('compression_node.py')
_load('fft_match.py')
_load('texture_normalize.py')
_load('metadata_inspector.py')
_load('texture_engine.py')
_load('spectral_normalizer_node.py')
_load('pixel_perturb.py')
_load('blend_colors.py')
_load('camera_simulator.py')
_load('load_image_from_path.py')
_load('line_splitter.py')
_load('image_prompt_iterator.py')
_load('debug_prompt_overlay.py')
_load('prompt_batch_preview.py')
_load('mask_to_crop.py')
_load('multi_compression.py')

_load('list_utility_nodes.py',
      ['Omni_BatchFromImageList', 'Omni_ImageListFromBatch',
       'Omni_PickFromList', 'Omni_StringListFromStrings'])

_load('string_utility_nodes.py',
      ['Omni_SplitByCommas', 'Omni_StringToFloat', 'Omni_StringToInt',
       'Omni_AnyListToString', 'Omni_StringCombine'])

_load('mask_utility_nodes.py',
      ['Omni_MaskedSection', 'Omni_MaskCombine'])

_load('auto_white_balance_node.py',
      ['OmniSimulator_AutoWhiteBalance'])

_load('authenticity_profile_selector.py',
      ['OmniSimulator_AuthenticityProfile_Selector'])

NODE_DISPLAY_NAME_MAPPINGS.update({
    'Omni_BatchFromImageList':        'Omni: Batch From Image List',
    'Omni_ImageListFromBatch':        'Omni: Image List From Batch',
    'Omni_PickFromList':              'Omni: Pick From List',
    'Omni_StringListFromStrings':     'Omni: String List',
    'Omni_SplitByCommas':            'Omni: Split String',
    'Omni_StringToFloat':            'Omni: String To Float',
    'Omni_StringToInt':              'Omni: String To Int',
    'Omni_AnyListToString':          'Omni: List To String',
    'Omni_StringCombine':            'Omni: String Combine',
    'Omni_MaskedSection':            'Omni: Masked Section',
    'Omni_MaskCombine':              'Omni: Mask Combine',
    'OmniSimulator_AutoWhiteBalance':         'Omni: Auto White Balance',
    'OmniSimulator_AuthenticityProfile_Selector': 'Omni: Authenticity Profile',
    'OmniSimulator_MultiCompression': 'Multi Compression',
})

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']