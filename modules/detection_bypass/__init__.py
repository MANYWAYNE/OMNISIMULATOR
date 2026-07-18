# ComfyUI_OmniSimulator/modules/detection_bypass/__init__.py
import sys, os
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

try:
    from modules.detection_bypass.camera_pipeline import simulate_camera_pipeline
    __all__ = ['simulate_camera_pipeline']
except Exception as _e:
    print(f'[OmniSimulator] detection_bypass/__init__: {_e}')