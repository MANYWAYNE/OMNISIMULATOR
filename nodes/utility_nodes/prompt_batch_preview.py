class OmniSimulator_PromptBatchPreview:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"prompts": ("STRING", {"multiline": True, "default": ""})}}
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("preview",); FUNCTION = "preview"; CATEGORY = "OmniSimulator/Utils"; OUTPUT_NODE = True
    def preview(self, prompts):
        lines = [l.strip() for l in prompts.strip().splitlines() if l.strip()]
        preview = f"Total prompts: {len(lines)}\n" + "\n".join(f"[{i+1}] {l[:80]}" for i,l in enumerate(lines[:20]))
        return {"ui": {"text": [preview]}, "result": (preview,)}

NODE_CLASS_MAPPINGS = {"OmniSimulator_PromptBatchPreview": OmniSimulator_PromptBatchPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_PromptBatchPreview": "OmniSimulator Prompt Batch Preview"}