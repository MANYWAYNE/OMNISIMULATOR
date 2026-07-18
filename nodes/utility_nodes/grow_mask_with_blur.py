import torch
import torch.nn.functional as F
import math


def _box_blur_separable(m, ks):
    """Separable (two 1-D passes) box blur — identical result to a dense ks×ks
    box kernel but O(ks) per pixel instead of O(ks²). `m` must be [B,1,H,W]."""
    pad = ks // 2
    kx = torch.ones(1, 1, 1, ks, dtype=m.dtype, device=m.device) / ks
    ky = torch.ones(1, 1, ks, 1, dtype=m.dtype, device=m.device) / ks
    m = F.conv2d(m, kx, padding=(0, pad))
    m = F.conv2d(m, ky, padding=(pad, 0))
    return m


class OmniSimulator_GrowMaskWithBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",), "expand": ("INT", {"default": 6, "min": -999, "max": 999}),
                             "blur": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1})}}
    RETURN_TYPES = ("MASK",); FUNCTION = "apply"; CATEGORY = "OmniSimulator/Masks"

    def apply(self, mask, expand, blur):
        m = mask
        if m.ndim == 2:                     # [H,W] -> [1,H,W]
            m = m.unsqueeze(0)
        m = m.float()
        expand = int(expand)

        if expand != 0:
            k = abs(expand) * 2 + 1
            pad = abs(expand)
            x = m.unsqueeze(1)              # [B,1,H,W]
            if expand > 0:                  # dilate
                x = F.max_pool2d(x, k, stride=1, padding=pad)
            else:                           # erode == dilate of the inverse
                x = -F.max_pool2d(-x, k, stride=1, padding=pad)
            m = x.squeeze(1)

        if blur > 0:
            ks = int(2 * math.ceil(3 * blur) + 1)
            if ks % 2 == 0:
                ks += 1
            m = _box_blur_separable(m.unsqueeze(1), ks).squeeze(1)

        return (torch.clamp(m, 0, 1),)


NODE_CLASS_MAPPINGS = {"OmniSimulator_GrowMaskWithBlur": OmniSimulator_GrowMaskWithBlur}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_GrowMaskWithBlur": "OmniSimulator Grow Mask With Blur"}