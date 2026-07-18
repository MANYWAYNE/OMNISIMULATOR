import sys
import os
import importlib.util

_DIR  = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.path.normpath(os.path.join(_DIR, '..', '..'))
if _ROOT not in sys.path: 
    sys.path.insert(0, _ROOT)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _fname in ('save_with_metadata.py', 'synthesize_with_metadata.py'):
    _path = os.path.join(_DIR, _fname)
    if not os.path.isfile(_path):
        continue
    try:
        _key  = 'omni_out__' + _fname.replace('.py', '')
        _spec = importlib.util.spec_from_file_location(_key, _path)
        _mod  = importlib.util.module_from_spec(_spec)
        sys.modules[_key] = _mod
        _spec.loader.exec_module(_mod)
        NODE_CLASS_MAPPINGS.update(getattr(_mod, 'NODE_CLASS_MAPPINGS', {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, 'NODE_DISPLAY_NAME_MAPPINGS', {}))
    except Exception as _e:
        print(f'[OmniSimulator] output_nodes/{_fname}: {_e}')