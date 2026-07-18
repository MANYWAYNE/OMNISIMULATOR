import torch
import torch.nn as nn
import torch.nn.functional as F

_MODEL_CACHE = {}   # per-device cache so we only build once


class OmniSimulator_BuiltinPerceptualLoss(nn.Module):
    """
    Zero-dependency perceptual-distance proxy.

    Compares two images across a small average-pool pyramid using:
      - Sobel gradient-magnitude maps  (captures edges / structure)
      - Local-contrast (windowed std)  (captures fine texture / grain)
    Both are classic hand-crafted correlates of "perceived" difference and,
    unlike a raw pixel L2, are far less sensitive to imperceptible global
    brightness/color shifts — the same property that makes LPIPS useful here.
    """
    def __init__(self, levels: int = 3):
        super().__init__()
        self.levels = levels
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2).contiguous()
        self.register_buffer("_kx", kx, persistent=False)
        self.register_buffer("_ky", ky, persistent=False)

    def _sobel(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        kx = self._kx.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        ky = self._ky.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=c)
        gy = F.conv2d(x, ky, padding=1, groups=c)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    @staticmethod
    def _local_contrast(x: torch.Tensor, k: int = 5) -> torch.Tensor:
        pad = k // 2
        mu  = F.avg_pool2d(x,     k, stride=1, padding=pad, count_include_pad=False)
        mu2 = F.avg_pool2d(x * x, k, stride=1, padding=pad, count_include_pad=False)
        var = (mu2 - mu * mu).clamp(min=0)
        return torch.sqrt(var + 1e-8)

    def _pyramid(self, x: torch.Tensor):
        levels = [x]
        cur = x
        for _ in range(self.levels - 1):
            if min(cur.shape[-2:]) < 4:
                break
            cur = F.avg_pool2d(cur, kernel_size=2, stride=2, ceil_mode=True)
            levels.append(cur)
        return levels

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pyr_a, pyr_b = self._pyramid(a), self._pyramid(b)
        total = torch.zeros((), device=a.device, dtype=a.dtype)
        for i, (pa, pb) in enumerate(zip(pyr_a, pyr_b)):
            w = 1.0 / (2 ** i)
            edge_diff = F.l1_loss(self._sobel(pa), self._sobel(pb))
            tex_diff  = F.l1_loss(self._local_contrast(pa), self._local_contrast(pb))
            total = total + w * (edge_diff + tex_diff)
        return total / len(pyr_a)


def build_perceptual_model(device):
    """
    Returns (model, backend_name). `model` is an nn.Module already moved to
    `device`, in eval mode, with gradients disabled on its own parameters —
    safe to call directly as `model(img_a, img_b)` exactly like lpips.LPIPS.
    Cached per-device so repeated node calls don't rebuild it every time.
    """
    key = str(device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    try:
        import lpips
        model = lpips.LPIPS(net='alex').to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        backend = "lpips"
        print("[OmniSimulator] Perceptual loss backend: lpips (AlexNet) — best quality.")
    except Exception as e:
        model = OmniSimulator_BuiltinPerceptualLoss().to(device).eval()
        backend = "builtin"
        print(f"[OmniSimulator] Perceptual loss backend: built-in dependency-free proxy "
              f"(lpips unavailable: {e}). Optional upgrade: pip install lpips")

    _MODEL_CACHE[key] = (model, backend)
    return model, backend