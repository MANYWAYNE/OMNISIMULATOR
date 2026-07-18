class OmniSimulator_SplitByCommas:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"text": ("STRING", {"default": ""}), "index": ("INT", {"default": 0, "min": 0, "max": 999})}}
    RETURN_TYPES = ("STRING",); FUNCTION = "split"; CATEGORY = "OmniSimulator/Utils"
    def split(self, text, index):
        parts = [p.strip() for p in text.split(",")]
        return (parts[index] if index < len(parts) else "",)

class OmniSimulator_StringToFloat:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"value": ("STRING", {"default": "1.0"})}}
    RETURN_TYPES = ("FLOAT",); FUNCTION = "convert"; CATEGORY = "OmniSimulator/Utils"
    def convert(self, value):
        try: return (float(str(value).strip()),)
        except (ValueError, TypeError): return (0.0,)

class OmniSimulator_StringToInt:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"value": ("STRING", {"default": "1"})}}
    RETURN_TYPES = ("INT",); FUNCTION = "convert"; CATEGORY = "OmniSimulator/Utils"
    def convert(self, value):
        # Parse via float first so "3", "3.0" and "3.7" all work ("3.7" -> 3);
        # the old int("3.7") raised and silently returned 0.
        try: return (int(float(str(value).strip())),)
        except (ValueError, TypeError): return (0,)

class OmniSimulator_AnyListToString:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"separator": ("STRING", {"default": ", "})}, "optional": {"value": ("*",)}}
    RETURN_TYPES = ("STRING",); FUNCTION = "convert"; CATEGORY = "OmniSimulator/Utils"
    def convert(self, separator=", ", value=None):
        if value is None: return ("",)
        if isinstance(value, list): return (separator.join(str(v) for v in value),)
        return (str(value),)

class OmniSimulator_StringCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"a": ("STRING", {"default": ""}), "b": ("STRING", {"default": ""}),
                             "c": ("STRING", {"default": ""}), "separator": ("STRING", {"default": " "})}}
    RETURN_TYPES = ("STRING",); FUNCTION = "combine"; CATEGORY = "OmniSimulator/Utils"
    def combine(self, a="", b="", c="", separator=" "):
        parts = [x for x in [a, b, c] if x and x.strip()]
        return (separator.join(parts),)