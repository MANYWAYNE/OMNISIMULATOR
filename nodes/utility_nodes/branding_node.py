import torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
class OmniSimulator_BrandingOverlay:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"image": ("IMAGE",), "text": ("STRING", {"default": "OmniSimulator"}),
        "position": (["bottom-right","bottom-left","top-right","top-left","center"],), "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0})}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "apply"; CATEGORY = "OmniSimulator/Utils"
    def apply(self, image, text, position, opacity):
        results = []
        for i in range(image.shape[0]):
            arr = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr).convert("RGBA")
            overlay = Image.new("RGBA", pil.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            W, H = pil.size
            try: font = ImageFont.load_default(size=max(12, W//40))
            except: font = ImageFont.load_default()
            bbox = draw.textbbox((0,0), text, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            pad = 10
            pos_map = {"bottom-right":(W-tw-pad,H-th-pad),"bottom-left":(pad,H-th-pad),
                       "top-right":(W-tw-pad,pad),"top-left":(pad,pad),"center":((W-tw)//2,(H-th)//2)}
            x, y = pos_map[position]
            draw.rectangle([x-2,y-2,x+tw+2,y+th+2], fill=(0,0,0,int(128*opacity)))
            draw.text((x,y), text, fill=(255,255,255,int(255*opacity)), font=font)
            final = Image.alpha_composite(pil, overlay).convert("RGB")
            results.append(torch.from_numpy(np.array(final).astype(np.float32)/255.0).unsqueeze(0))
        return (torch.cat(results, dim=0),)

NODE_CLASS_MAPPINGS = {"OmniSimulator_BrandingOverlay": OmniSimulator_BrandingOverlay}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_BrandingOverlay": "OmniSimulator Branding Overlay"}