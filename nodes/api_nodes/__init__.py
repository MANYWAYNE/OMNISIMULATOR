import sys
import os
import importlib.util

_DIR = os.path.dirname(os.path.realpath(__file__))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _load_module(module_name, file_name):
    try:
        path = os.path.join(_DIR, file_name)
        if not os.path.exists(path):
            return
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        
        NODE_CLASS_MAPPINGS.update(getattr(mod, 'NODE_CLASS_MAPPINGS', {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, 'NODE_DISPLAY_NAME_MAPPINGS', {}))
    except Exception as e:
        print(f"[OmniSimulator] Error loading module {file_name}: {e}")

_load_module('omni_generative_api_base', 'generative_api_nodes.py')
_load_module('omni_gemini_native', 'gemini_native.py')

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']