# ComfyUI_OmniSimulator/__init__.py
"""OmniSimulator — ComfyUI Custom Node Package"""
import sys
import os
import types
import importlib.util
import traceback

def _find_here():
    try:
        f = globals().get('__file__') or __file__
        return os.path.dirname(os.path.realpath(f))
    except Exception:
        pass
    for p in sys.path:
        candidate = os.path.join(p, 'ComfyUI_OmniSimulator')
        if os.path.isfile(os.path.join(candidate, 'requirements.txt')):
            return candidate
    try:
        import folder_paths
        base = os.path.dirname(folder_paths.base_path)
        for name in ('custom_nodes', 'ComfyUI/custom_nodes'):
            cn = os.path.join(base, name, 'ComfyUI_OmniSimulator')
            if os.path.isdir(cn):
                return cn
    except Exception:
        pass
    raise RuntimeError("OmniSimulator: Cannot determine install directory.")

_HERE = _find_here()
_PKG  = 'ComfyUI_OmniSimulator'

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

def _reg(name, relpath=''):
    path = os.path.join(_HERE, relpath) if relpath else _HERE
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = [path]
        m.__package__ = name
        m.__file__ = os.path.join(path, '__init__.py')
        sys.modules[name] = m
    else:
        m = sys.modules[name]
        if not getattr(m, '__path__', None):
            m.__path__ = [path]

_reg(_PKG)
_reg(f'{_PKG}.nodes',                       'nodes')
_reg(f'{_PKG}.nodes.advanced_loader',       'nodes/advanced_loader')
_reg(f'{_PKG}.nodes.utility_nodes',         'nodes/utility_nodes')
_reg(f'{_PKG}.nodes.output_nodes',          'nodes/output_nodes')
_reg(f'{_PKG}.nodes.interactive_nodes',     'nodes/interactive_nodes')
_reg(f'{_PKG}.modules',                     'modules')
_reg(f'{_PKG}.modules.detection_bypass',    'modules/detection_bypass')
_reg(f'{_PKG}.modules.detection_bypass.utils', 'modules/detection_bypass/utils')

NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _merge(mod):
    if mod is None: return
    NODE_CLASS_MAPPINGS.update(getattr(mod, 'NODE_CLASS_MAPPINGS', {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, 'NODE_DISPLAY_NAME_MAPPINGS', {}))

def _load_file(rel_init, label):
    path = os.path.join(_HERE, rel_init)
    if not os.path.isfile(path):
        print(f'[OmniSimulator] {label}: {path} not found')
        return None
    key = 'omni_simulator_pkg__' + rel_init.replace('/', '_').replace('.py', '')
    try:
        if key in sys.modules:
            mod = sys.modules[key]
        else:
            spec = importlib.util.spec_from_file_location(key, path,
                       submodule_search_locations=[os.path.dirname(path)])
            mod  = importlib.util.module_from_spec(spec)
            mod.__package__ = key
            sys.modules[key] = mod
            spec.loader.exec_module(mod)
        _merge(mod)
        return mod
    except Exception:
        print(f'[OmniSimulator] {label} failed:')
        traceback.print_exc()
        sys.modules.pop(key, None)
        return None

try:
    _sr_path = os.path.join(_HERE, 'nodes', 'advanced_loader', 'server_routes.py')
    _sr_key  = 'omni_simulator_server_routes'
    _sr_spec = importlib.util.spec_from_file_location(_sr_key, _sr_path)
    _sr_mod  = importlib.util.module_from_spec(_sr_spec)
    sys.modules[_sr_key] = _sr_mod
    _sr_spec.loader.exec_module(_sr_mod)
    _sr_mod.setup_routes()
except Exception:
    print('[OmniSimulator] Server routes failed to setup:')
    traceback.print_exc()

_load_file('nodes/advanced_loader/__init__.py',   'Advanced Loader')
_load_file('nodes/utility_nodes/__init__.py',     'Utility Nodes')
_load_file('nodes/output_nodes/__init__.py',      'Output Nodes')
_load_file('nodes/interactive_nodes/__init__.py', 'Interactive Nodes')

WEB_DIRECTORY = './js'

print(f'[OmniSimulator] {len(NODE_CLASS_MAPPINGS)} nodes loaded from {_HERE}')
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']