import sys
import os
import importlib.util

_DIR = os.path.dirname(os.path.realpath(__file__))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_loader_path = os.path.join(_DIR, 'advanced_image_loader.py')
spec_loader  = importlib.util.spec_from_file_location('omni_advanced_loader', _loader_path)
mod_loader   = importlib.util.module_from_spec(spec_loader)
sys.modules['omni_advanced_loader'] = mod_loader
spec_loader.loader.exec_module(mod_loader)

NODE_CLASS_MAPPINGS.update(getattr(mod_loader, 'NODE_CLASS_MAPPINGS', {}))
NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod_loader, 'NODE_DISPLAY_NAME_MAPPINGS', {}))

try:
    _routes_path = os.path.join(_DIR, 'server_routes.py')
    spec_routes  = importlib.util.spec_from_file_location('omni_server_routes', _routes_path)
    mod_routes   = importlib.util.module_from_spec(spec_routes)
    sys.modules['omni_server_routes'] = mod_routes
    spec_routes.loader.exec_module(mod_routes)
    
    if hasattr(mod_routes, 'setup_routes'):
        mod_routes.setup_routes()
except Exception as e:
    print(f"[OmniSimulator] Critical error loading server routes: {e}")