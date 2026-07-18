import json
class OmniSimulator_JsonLoad:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"json_text": ("STRING", {"default": "{}", "multiline": True})}}
    RETURN_TYPES = ("STRING",); FUNCTION = "load"; CATEGORY = "OmniSimulator/Utils"
    def load(self, json_text):
        try:
            data = json.loads(json_text)
            return (json.dumps(data, indent=2),)
        except (ValueError, TypeError): return (json_text,)

class OmniSimulator_JsonGet:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"json_text": ("STRING",{"default":"{}"}), "key": ("STRING",{"default":"key"})}}
    RETURN_TYPES = ("STRING",); FUNCTION = "get"; CATEGORY = "OmniSimulator/Utils"
    def get(self, json_text, key):
        try:
            data = json.loads(json_text)
            return (str(data.get(key, "")),) if isinstance(data, dict) else ("",)
        except (ValueError, TypeError): return ("",)

NODE_CLASS_MAPPINGS = {"OmniSimulator_JsonLoad": OmniSimulator_JsonLoad, "OmniSimulator_JsonGet": OmniSimulator_JsonGet}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_JsonLoad": "📋 OmniSimulator JSON Load", "OmniSimulator_JsonGet": "OmniSimulator JSON Get"}