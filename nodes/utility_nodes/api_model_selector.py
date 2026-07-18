class OmniSimulator_ApiModelSelector:
    MODELS = ["gemini-2.5-pro","gemini-2.5-flash","gemini-2.0-flash","gpt-4o","gpt-4o-mini","claude-opus-4-6","claude-sonnet-4-6"]
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"model": (cls.MODELS, {"default": cls.MODELS[0]})}}
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("model",); FUNCTION = "get"; CATEGORY = "OmniSimulator/API"
    def get(self, model): return (model,)

NODE_CLASS_MAPPINGS = {"OmniSimulator_ApiModelSelector": OmniSimulator_ApiModelSelector}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_ApiModelSelector": "OmniSimulator API Model Selector"}