"""
Adaptive spatial filtering for UnMarker.
Ported from ai-watermark/modules/attack/unmark/cw.py:Filter

This implements learnable, edge-aware filtering that applies perturbations
more strongly in textured regions (where they're invisible) and weakly in
smooth regions (where they'd be noticeable).

Key concept: Joint bilateral filtering with learnable kernels.
"""

import numpy as np
import torch
try:
    import kornia
    _KORNIA_AVAILABLE = True
except ImportError:
    kornia = None
    _KORNIA_AVAILABLE = False
    print("[OmniSimulator] ⚠️  adaptive_filter: kornia not installed. AdaptiveFilter will raise if instantiated. Run: pip install kornia")


class AdaptiveFilter(torch.nn.Module):
    """
    Learnable adaptive filter that applies spatially-varying perturbations.

    This is the secret sauce for imperceptibility. Instead of uniform noise,
    it learns WHERE to apply perturbations based on image content.

    Args:
        kernels: List of (height, width) tuples for filter sizes, e.g., [(7,7), (15,15)]
        shape: Expected input shape (B, C, H, W)
        box: Spatial box size for partitioning, e.g., (1, 1) = no partitioning
        sigma_color: Bilateral filter color sensitivity (0 = spatial only)
        norm: L-p norm for distance computation
        pad_mode: Padding mode for convolutions ("reflect" recommended)
        filter_mode: If True, applies filter at every pixel (slow but accurate)
        loss_factor: Weight for filter regularization loss
        loss_norm: Norm for filter loss computation
    """

    def __init__(
        self,
        kernels,
        shape,
        box=(1, 1),
        sigma_color=0.1,
        norm=1,
        pad_mode="reflect",
        filter_mode=False,
        loss_factor=1,
        loss_norm=2,
    ):
        if not _KORNIA_AVAILABLE:
            raise ImportError("AdaptiveFilter requires kornia. Run: pip install kornia")
        super().__init__()
        self.norm, self.sigma_color, self.pad_mode, self.box, self.filter_mode = (
            norm,
            sigma_color,
            pad_mode,
            box,
            filter_mode,
        )

        # Learnable filter kernels (these are optimized during attack)
        self.kernels = torch.nn.ParameterList(
            [torch.nn.Parameter(self.__get_init_w(kernel, shape)) for kernel in kernels]
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.loss_factor = loss_factor
        self.loss_norm = loss_norm

    def pad_w(self, w):
        """Pad filter kernel to full size."""
        return torch.nn.functional.pad(
            w, (0, w.shape[-1] - 1, 0, w.shape[-2] - 1), "reflect"
        )

    def __get_init_w(self, kernel, shape):
        """Initialize learnable filter weights with Gaussian."""
        repeats, _, h, w = shape
        box = self.box if self.box is not None else kernel
        boxes = [int(np.ceil(h / box[0])), int(np.ceil(w / box[1]))]
        num_boxes = boxes[0] * boxes[1]

        # Initialize with Gaussian kernel
        w = (
            kornia.filters.get_gaussian_kernel2d(kernel, torch.tensor([[0.2, 0.2]]))
            .unsqueeze(0)
            .repeat(repeats, num_boxes, 1, 1)
        )

        # Store only upper-left quadrant (will be mirrored)
        # Log-space for numerical stability
        return (
            w[..., : int(kernel[0] // 2) + 1, : int(kernel[1] // 2) + 1]
            .clamp(1e-5, 0.999999)
            .log()
        )

    def get_dist(self, x, kernel, guidance=None, norm=None):
        """
        Compute spatial distance between pixels and their neighborhoods.
        This is used for bilateral filtering.
        """
        norm = self.norm if norm is None else norm
        unf_inp = self.extract_patches(x, kernel)
        guidance = guidance if guidance is not None else x
        guidance = torch.nn.functional.pad(
            guidance,
            self._box_pad(guidance, kernel),
            mode=self.pad_mode,
        )

        return torch.pow(
            torch.norm(
                unf_inp
                - guidance.view(guidance.shape[0], guidance.shape[1], -1)
                .transpose(1, 2)
                .view(
                    guidance.shape[0],
                    unf_inp.shape[1],
                    unf_inp.shape[2],
                    guidance.shape[1],
                    1,
                ),
                p=norm,
                dim=-2,
                keepdim=True,
            ),
            2,
        )

    def __get_color_kernel(self, guidance, kernel):
        """
        Compute bilateral filter's range kernel (color similarity).
        Pixels with similar colors get more weight.
        """
        if self.sigma_color <= 0:
            return 1  # No color weighting, pure spatial filter

        dist = self.get_dist(guidance.double(), kernel).float()
        ret = (
            (-0.5 / (self.sigma_color**2) * dist)
            .exp()
            .view(guidance.shape[0], dist.shape[1], dist.shape[2], -1, 1)
        )
        return torch.nan_to_num(ret, nan=0.0)

    def _box_pad(self, x, kernel):
        """Compute padding needed for box-based processing."""
        box = self.box if self.box is not None else kernel
        col = (
            box[1] - (x.shape[-1] - (x.shape[-1] // box[1]) * box[1]) % box[1]
        ) % box[1]
        row = (
            box[0] - (x.shape[-2] - (x.shape[-2] // box[0]) * box[0]) % box[0]
        ) % box[0]
        return [0, col, 0, row]

    def _kernel_pad(self, kernel):
        """Compute padding for kernel convolution."""
        return [
            (kernel[1] - 1) // 2,
            (kernel[1] - 1) - (kernel[1] - 1) // 2,
            (kernel[0] - 1) // 2,
            (kernel[0] - 1) - (kernel[0] - 1) // 2,
        ]

    def _median_pad(self, x, kernel, stride):
        """Compute padding for median computation."""
        ph = (
            (kernel[0] - stride[0])
            if x.shape[-2] % stride[0] == 0
            else (kernel[0] - (x.shape[-2] % stride[0]))
        )
        pw = (
            (kernel[1] - stride[1])
            if x.shape[-1] % stride[1] == 0
            else (kernel[1] - (x.shape[-1] % stride[1]))
        )
        return (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)

    def _compute_median(self, x, kernel):
        """Compute local median for regularization."""
        stride = kernel if not self.filter_mode else (1, 1)
        x_p = torch.nn.functional.pad(
            x, self._median_pad(x, kernel, stride), mode="reflect"
        )
        x_unf = x_p.unfold(2, kernel[0], stride[0]).unfold(3, kernel[1], stride[1])
        median = x_unf.contiguous().view(x_unf.size()[:4] + (-1,)).median(dim=-1)[0]
        return (
            median.unsqueeze(-2)
            .unsqueeze(-1)
            .repeat(
                1,
                1,
                1,
                x_p.shape[-2] // median.shape[-2],
                1,
                x_p.shape[-1] // median.shape[-1],
            )
            .flatten(2, 3)
            .flatten(-2)[..., : x.shape[-2], : x.shape[-1]]
        )

    def extract_patches(self, x, kernel):
        """
        Extract overlapping patches for filter application.
        This is the core of the spatial filtering.
        """
        box = self.box if self.box is not None else kernel
        kern = (box[0] + (kernel[0] - 1), box[1] + (kernel[1] - 1))
        pad = [
            b + k for b, k in zip(self._box_pad(x, kernel), self._kernel_pad(kernel))
        ]
        inp_unf = (
            torch.nn.functional.pad(x, pad, mode=self.pad_mode)
            .unfold(2, kern[0], box[0])
            .unfold(3, kern[1], box[1])
            .permute(0, 2, 3, 1, 4, 5)
            .flatten(-2)
            .reshape(-1, x.shape[1], kern[0], kern[1])
        )

        return (
            inp_unf.unfold(2, kernel[0], 1)
            .unfold(3, kernel[1], 1)
            .permute(0, 2, 3, 1, 4, 5)
            .flatten(-2)
            .reshape(
                x.shape[0],
                inp_unf.shape[0] // x.shape[0],
                -1,
                inp_unf.shape[1],
                kernel[0] * kernel[1],
            )
        )

    def __compute_filter_loss(self, x, kernel, norm=2):
        """
        Regularization loss to prevent over-smoothing.
        Penalizes deviation from local median.
        """
        return self.get_dist(
            x, kernel, guidance=self._compute_median(x, kernel), norm=norm
        ).view(x.shape[0], -1).sum(-1, keepdims=True) / torch.prod(
            torch.tensor(x.shape[1:])
        )

    def __apply_filter(self, x, w, guidance=None):
        """
        Apply the learned adaptive filter to the input.

        This is where the magic happens: the filter applies different
        weights to different spatial locations based on local image content.
        """
        w = self.pad_w(w)  # Expand to full kernel
        kernel = (w.shape[-2], w.shape[-1])
        box = self.box if self.box is not None else kernel
        inp_unf = self.extract_patches(x, kernel)

        boxes = [
            int(np.ceil(x.shape[-2] / box[0])),
            int(np.ceil(x.shape[-1] / box[1])),
        ]

        # Compute bilateral weights (spatial + color)
        color_kernel = self.__get_color_kernel(guidance, kernel)

        # Combine learned spatial weights with color weights
        w = (
            self.softmax(w.view(w.shape[0], w.shape[1], -1))
            .unsqueeze(-2)
            .unsqueeze(-1)
            .repeat(1, 1, inp_unf.shape[2], 1, 1)
            * color_kernel
        ).view(w.shape[0], w.shape[1], inp_unf.shape[2], -1, 1)

        # Apply weighted combination
        out = inp_unf.matmul(w).transpose(2, 3).squeeze(-1) / w.squeeze(-1).sum(
            -1
        ).unsqueeze(2)

        out = (
            out.view(-1, inp_unf.shape[-2], inp_unf.shape[-3])
            .reshape(x.shape[0], -1, x.shape[1] * box[0] * box[1])
            .transpose(2, 1)
        )

        return torch.nn.functional.fold(
            out,
            (boxes[0] * box[0], boxes[1] * box[1]),
            box,
            stride=box,
        )[..., : x.shape[-2], : x.shape[-1]]

    def compute_loss(self, x):
        """Compute regularization loss for all filter kernels."""
        kernels = [(f.shape[-2] * 2 - 1, f.shape[-1] * 2 - 1) for f in self.kernels]
        if len(kernels) == 0 or self.loss_factor == 0:
            return 0

        return torch.concat(
            [
                self.__compute_filter_loss(
                    x,
                    k,
                    norm=self.loss_norm,
                )
                for k in kernels
            ],
            dim=-1,
        ).sum(-1)

    def forward(self, x, guidance):
        """
        Apply all learned filters sequentially.

        Args:
            x: Input tensor to filter
            guidance: Guidance image for bilateral weighting (usually original image)

        Returns:
            Filtered tensor
        """
        for filt in self.kernels:
            x = self.__apply_filter(x, filt, guidance=guidance)
        return x.float()