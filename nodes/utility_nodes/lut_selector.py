import os

# Correctly locate the root of your custom node package
NODE_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
OmniSimulator_ROOT_PATH = os.path.abspath(os.path.join(NODE_FILE_PATH, "..", ".."))

class OmniSimulator_LUT_Selector:
    """
    Dynamically finds and lists all .cube LUT files from the internal
    'modules/detection_bypass/_luts' directory, including subdirectories.
    Outputs the full path to the selected LUT file.
    """
    
    LUTS_DIR = os.path.join(OmniSimulator_ROOT_PATH, "modules", "detection_bypass", "_luts")
    LUT_FILES = []
    if os.path.isdir(LUTS_DIR):
        for root, _, files in os.walk(LUTS_DIR):
            for file in files:
                if file.lower().endswith('.cube'):
                    relative_path = os.path.relpath(os.path.join(root, file), LUTS_DIR)
                    LUT_FILES.append(relative_path.replace('\\', '/')) # Normalize slashes for display
    LUT_FILES.sort()

    @classmethod
    def INPUT_TYPES(cls):
        if not cls.LUT_FILES:
            return {
                "required": {
                    "error": ("STRING", {"default": "ERROR: No .cube files found in modules/detection_bypass/_luts/", "multiline": True})
                }
            }
            
        return {
            "required": {
                "lut_name": (cls.LUT_FILES, {"default": cls.LUT_FILES[0] if cls.LUT_FILES else ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lut_path",)
    FUNCTION = "get_path"
    CATEGORY = "OmniSimulator/Utils"

    def get_path(self, lut_name):
        full_path = os.path.join(self.LUTS_DIR, lut_name)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Selected LUT file could not be found: {full_path}")
            
        print(f"OmniSimulator LUT Selector: Providing path '{full_path}'")
        return (full_path,)
NODE_CLASS_MAPPINGS = {"OmniSimulator_LUT_Selector": OmniSimulator_LUT_Selector}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_LUT_Selector": "OmniSimulator LUT Selector"}