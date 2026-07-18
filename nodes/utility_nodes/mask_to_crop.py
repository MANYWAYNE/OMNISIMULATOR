import torch
class OmniSimulator_MaskToCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",), "padding": ("INT", {"default": 32, "min": 0, "max": 512})}}
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "x", "y", "width", "height")
    FUNCTION = "crop"; CATEGORY = "OmniSimulator/Masks"
    def crop(self, image, mask, padding):
        m = mask[0] if mask.ndim == 3 else mask
        rows = torch.any(m > 0.5, dim=1); cols = torch.any(m > 0.5, dim=0)
        if not rows.any(): return (image, mask, 0, 0, image.shape[2], image.shape[1])
        y1, y2 = rows.nonzero()[0].item(), rows.nonzero()[-1].item()+1
        x1, x2 = cols.nonzero()[0].item(), cols.nonzero()[-1].item()+1
        H, W = m.shape
        y1 = max(0, y1-padding); y2 = min(H, y2+padding)
        x1 = max(0, x1-padding); x2 = min(W, x2+padding)
        return (image[:, y1:y2, x1:x2], mask[:, y1:y2, x1:x2] if mask.ndim==3 else mask[y1:y2, x1:x2].unsqueeze(0), x1, y1, x2-x1, y2-y1)

NODE_CLASS_MAPPINGS = {"OmniSimulator_MaskToCrop": OmniSimulator_MaskToCrop}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_MaskToCrop": "OmniSimulator Mask To Crop"}