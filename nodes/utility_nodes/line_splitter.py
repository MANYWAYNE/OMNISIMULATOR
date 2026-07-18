class OmniSimulator_LineSplitter:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"text": ("STRING", {"multiline": True, "default": ""}), "index": ("INT", {"default": 0, "min": 0, "max": 9999})}}
    RETURN_TYPES = ("STRING", "INT"); RETURN_NAMES = ("line", "count"); FUNCTION = "split"; CATEGORY = "OmniSimulator/Utils"
    def split(self, text, index):
        lines = [l for l in text.strip().splitlines() if l.strip()]
        return (lines[index % max(len(lines),1)] if lines else "", len(lines))

NODE_CLASS_MAPPINGS = {"OmniSimulator_LineSplitter": OmniSimulator_LineSplitter}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_LineSplitter": "OmniSimulator Line Splitter"}