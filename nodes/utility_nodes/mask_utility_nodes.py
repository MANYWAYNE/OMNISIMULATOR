import torch


def _norm_mask(mask):
    """Return a mask as [B,H,W] float regardless of whether it arrived as
    [H,W], [B,H,W] or [B,1,H,W]."""
    m = mask
    if m.ndim == 2:            # [H,W]
        m = m.unsqueeze(0)
    if m.ndim == 4:            # [B,1,H,W]
        m = m.squeeze(1)
    return m.float()


class OmniSimulator_MaskedSection:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"image": ("IMAGE",), "mask": ("MASK",)}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "apply"; CATEGORY = "OmniSimulator/Masks"

    def apply(self, image, mask):
        m = _norm_mask(mask)
        # Broadcast a single mask over an image batch (or vice-versa) instead of
        # letting expand_as() raise when the batch dimensions differ.
        if m.shape[0] != image.shape[0]:
            if m.shape[0] == 1:
                m = m.expand(image.shape[0], -1, -1)
            elif image.shape[0] == 1:
                image = image.expand(m.shape[0], -1, -1, -1)
        return (image * m.unsqueeze(-1),)


class OmniSimulator_MaskCombine:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"mask_a": ("MASK",), "mask_b": ("MASK",), "mode": (["add", "subtract", "multiply"],)}}
    RETURN_TYPES = ("MASK",); FUNCTION = "combine"; CATEGORY = "OmniSimulator/Masks"

    def combine(self, mask_a, mask_b, mode):
        a, b = _norm_mask(mask_a), _norm_mask(mask_b)
        if mode == "add":      return (torch.clamp(a + b, 0, 1),)
        if mode == "subtract": return (torch.clamp(a - b, 0, 1),)
        return (a * b,)