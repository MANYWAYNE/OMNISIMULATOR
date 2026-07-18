class OmniSimulator_ApiProviderSelector:
    PROVIDERS = ["Gemini", "OpenAI", "Anthropic", "OpenRouter", "Local"]
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"provider": (cls.PROVIDERS, {"default": "Gemini"})}}
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("provider",); FUNCTION = "get"; CATEGORY = "OmniSimulator/API"
    def get(self, provider): return (provider,)

NODE_CLASS_MAPPINGS = {"OmniSimulator_ApiProviderSelector": OmniSimulator_ApiProviderSelector}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_ApiProviderSelector": "OmniSimulator API Provider Selector"}