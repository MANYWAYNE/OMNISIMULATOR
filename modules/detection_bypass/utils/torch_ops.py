import torch
import torch.nn.functional as F


def gaussian_blur2d(x: torch.Tensor, kernel_size, sigma) -> torch.Tensor:
    """
    Dependency-free replacement for kornia.filters.gaussian_blur2d.
    Accepts the same call shape used throughout this codebase:
        gaussian_blur2d(tensor, (k, k), (sigma, sigma))
    kernel_size / sigma may be given as a tuple/list or a plain number;
    only the first element is used (every call site here uses square kernels).
    x: (B, C, H, W)
    """
    k = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else kernel_size
    s = sigma[0] if isinstance(sigma, (tuple, list)) else sigma
    k = int(k)
    if k < 1 or s <= 0:
        return x
    if k % 2 == 0:
        k += 1
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * s * s + 1e-8))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    c = x.shape[1]
    kernel = kernel_2d.expand(c, 1, k, k).contiguous()
    return F.conv2d(x, kernel, padding=k // 2, groups=c)


def rgb_to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """
    Dependency-free replacement for kornia.color.rgb_to_grayscale.
    x: (B, 3, H, W) -> (B, 1, H, W), standard ITU-R BT.601 luma weights.
    """
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b