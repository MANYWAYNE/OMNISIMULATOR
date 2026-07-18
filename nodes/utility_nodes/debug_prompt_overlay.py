import torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
class OmniSimulator_DebugPromptOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "text": ("STRING", {"multiline": True, "default": "Debug"}),
                             "font_size": ("INT", {"default": 24, "min": 8, "max": 128}),
                             "opacity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0})}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "draw"; CATEGORY = "OmniSimulator/Utils"
    def draw(self, image, text, font_size, opacity):
        results = []
        for i in range(image.shape[0]):
            arr = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr).convert("RGBA")
            overlay = Image.new("RGBA", pil.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            try: font = ImageFont.load_default(size=font_size)
            except: font = ImageFont.load_default()
            draw.rectangle([(0,0),(pil.width,font_size*len(text.splitlines())+20)], fill=(0,0,0,int(180*opacity)))
            draw.text((10,10), text, fill=(255,255,0,255), font=font)
            result = Image.alpha_composite(pil, overlay).convert("RGB")
            results.append(torch.from_numpy(np.array(result).astype(np.float32)/255.0).unsqueeze(0))
        return (torch.cat(results, 0),)

NODE_CLASS_MAPPINGS = {"OmniSimulator_DebugPromptOverlay": OmniSimulator_DebugPromptOverlay}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_DebugPromptOverlay": "OmniSimulator Debug Prompt Overlay"}