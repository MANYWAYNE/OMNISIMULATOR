import sys
import os
import types
import importlib.util
from typing import Optional, Any

_DIR = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.path.normpath(os.path.join(_DIR, '..', '..'))

if _ROOT not in sys.path: 
    sys.path.insert(0, _ROOT)

_PKG = 'omni_interactive_pkg'
if _PKG not in sys.modules:
    _m = types.ModuleType(_PKG)
    _m.__path__ = [_DIR]
    _m.__package__ = _PKG
    sys.modules[_PKG] = _m

def _load(filename: str, subname: Optional[str] = None) -> Optional[Any]:
    path = os.path.join(_DIR, filename)
    if not os.path.isfile(path):
        print(f'[OmniSimulator] interactive_nodes/{filename}: file not found')
        return None
        
    base_name = subname or filename.replace('.py', '')
    key = f'{_PKG}.{base_name}'
    
    if key in sys.modules: 
        return sys.modules[key]
        
    try:
        spec = importlib.util.spec_from_file_location(key, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create spec for {path}")
            
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _PKG
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f'[OmniSimulator] Error loading interactive_nodes/{filename}: {e}')
        sys.modules.pop(key, None)
        return None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_load('image_filter_messaging.py', 'image_filter_messaging')

_TARGET_MODULES = (
    'image_filter.py', 
    'interactive_crop.py', 
    'prompt_filter.py', 
    'batch_image_generator.py'
)

for _f in _TARGET_MODULES:
    _mod = _load(_f)
    if _mod:
        NODE_CLASS_MAPPINGS.update(getattr(_mod, 'NODE_CLASS_MAPPINGS', {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, 'NODE_DISPLAY_NAME_MAPPINGS', {}))

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']