import torch
import torch.nn.functional as F


def _box_blur_separable(m, ks):
    """Uniform (box) blur via two 1-D passes.

    Mathematically identical to convolving with a dense ks×ks box kernel
    normalised by ks², but costs O(ks) per pixel instead of O(ks²). That
    difference is what keeps large radii usable: a dense kernel at the max
    feather (ks ≈ 2000) is a ~4M-tap convolution per pixel and will hang or
    run the GPU/CPU out of memory. `m` must be [B,1,H,W] float.
    """
    pad = ks // 2
    kx = torch.ones(1, 1, 1, ks, dtype=m.dtype, device=m.device) / ks
    ky = torch.ones(1, 1, ks, 1, dtype=m.dtype, device=m.device) / ks
    m = F.conv2d(m, kx, padding=(0, pad))
    m = F.conv2d(m, ky, padding=(pad, 0))
    return m


class OmniSimulator_FeatherMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",), "feather": ("INT", {"default": 10, "min": 0, "max": 999})}}
    RETURN_TYPES = ("MASK",); FUNCTION = "apply"; CATEGORY = "OmniSimulator/Masks"

    def apply(self, mask, feather):
        feather = int(feather)
        if feather <= 0:
            return (mask,)
        # Normalise any incoming shape to [B,1,H,W]. ComfyUI usually hands us a
        # [B,H,W] mask, but a bare [H,W] (or [B,1,H,W]) mask must not crash —
        # the previous `mask.unsqueeze(1)` silently mangled a 2-D mask.
        m = mask
        if m.ndim == 2:            # [H,W]
            m = m.unsqueeze(0)
        if m.ndim == 3:            # [B,H,W]
            m = m.unsqueeze(1)
        m = m.float()
        ks = feather * 2 + 1
        m = _box_blur_separable(m, ks)
        return (torch.clamp(m.squeeze(1), 0, 1),)


NODE_CLASS_MAPPINGS = {"OmniSimulator_FeatherMask": OmniSimulator_FeatherMask}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_FeatherMask": "OmniSimulator Feather Mask"}