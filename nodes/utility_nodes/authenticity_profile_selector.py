import os

class OmniSimulator_AuthenticityProfile_Selector:
    """
    Dynamically finds and lists all .npz authenticity profiles from the 
    'modules/authenticity_profiles' directory located at the project root.
    """
    
    NODE_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
    
    # (utility_nodes -> nodes -> ComfyUI-OmniSimulator)
    OmniSimulator_ROOT_PATH = os.path.abspath(os.path.join(NODE_FILE_PATH, "..", ".."))
    PROFILES_DIR = os.path.join(OmniSimulator_ROOT_PATH, "modules", "authenticity_profiles")

    @classmethod
    def get_profiles(cls):
        if not os.path.isdir(cls.PROFILES_DIR):
            return []
        
        files = [os.path.splitext(f)[0] for f in os.listdir(cls.PROFILES_DIR) if f.lower().endswith('.npz')]
        return sorted(files)

    @classmethod
    def INPUT_TYPES(cls):
        profile_files = cls.get_profiles()
        
        if not profile_files:
            return {
                "required": {
                    "error": ("STRING", {
                        "default": f"ERROR: No .npz files in {cls.PROFILES_DIR}",
                        "multiline": True
                    })
                }
            }
            
        return {
            "required": {
                "profile_name": (profile_files, {"default": profile_files[0]}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("profile_path",)
    FUNCTION = "get_path"
    CATEGORY = "OmniSimulator/Authenticity"

    def get_path(self, profile_name):
        base_path = os.path.join(self.PROFILES_DIR, profile_name)
        
        if not os.path.exists(f"{base_path}.npz"):
            raise FileNotFoundError(f"Profile file not found: {base_path}.npz")
            
        return (base_path,)