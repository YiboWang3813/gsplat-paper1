import torch

from . import _C


def _check_inputs(
    pixels: torch.Tensor,
    means2d: torch.Tensor,
    inv2d: torch.Tensor,
    weights: torch.Tensor,
    colors: torch.Tensor,
) -> None:
    if not pixels.is_cuda:
        raise ValueError("pixels must be CUDA tensors")
    tensors = [means2d, inv2d, weights, colors]
    if any(not t.is_cuda for t in tensors):
        raise ValueError("all additive_splat inputs must be CUDA tensors")
    if pixels.dtype != torch.float32:
        raise ValueError("additive_splat currently supports float32 only")
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("additive_splat currently supports float32 only")
    if pixels.dim() != 2 or pixels.shape[-1] != 2:
        raise ValueError(f"pixels must have shape (P, 2), got {tuple(pixels.shape)}")
    if means2d.dim() != 2 or means2d.shape[-1] != 2:
        raise ValueError(f"means2d must have shape (N, 2), got {tuple(means2d.shape)}")
    if inv2d.shape != (means2d.shape[0], 2, 2):
        raise ValueError(f"inv2d must have shape (N, 2, 2), got {tuple(inv2d.shape)}")
    if weights.shape != (means2d.shape[0], 1):
        raise ValueError(f"weights must have shape (N, 1), got {tuple(weights.shape)}")
    if colors.shape != (means2d.shape[0], 3):
        raise ValueError(f"colors must have shape (N, 3), got {tuple(colors.shape)}")


def _check_bounded_inputs(
    image_height: int,
    image_width: int,
    half_width: float,
    half_height: float,
    means2d: torch.Tensor,
    inv2d: torch.Tensor,
    weights: torch.Tensor,
    colors: torch.Tensor,
    radius_sigma: float,
) -> None:
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    if half_width <= 0.0 or half_height <= 0.0:
        raise ValueError("physical half extents must be positive")
    if radius_sigma <= 0.0:
        raise ValueError("radius_sigma must be positive")
    tensors = [means2d, inv2d, weights, colors]
    if any(not t.is_cuda for t in tensors):
        raise ValueError("all additive_splat_bounded inputs must be CUDA tensors")
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("additive_splat_bounded currently supports float32 only")
    if means2d.dim() != 2 or means2d.shape[-1] != 2:
        raise ValueError(f"means2d must have shape (N, 2), got {tuple(means2d.shape)}")
    if inv2d.shape != (means2d.shape[0], 2, 2):
        raise ValueError(f"inv2d must have shape (N, 2, 2), got {tuple(inv2d.shape)}")
    if weights.shape != (means2d.shape[0], 1):
        raise ValueError(f"weights must have shape (N, 1), got {tuple(weights.shape)}")
    if colors.shape != (means2d.shape[0], 3):
        raise ValueError(f"colors must have shape (N, 3), got {tuple(colors.shape)}")


class _AdditiveSplat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pixels, means2d, inv2d, weights, colors):
        _check_inputs(pixels, means2d, inv2d, weights, colors)
        pixels_c = pixels.contiguous()
        means_c = means2d.contiguous()
        inv_c = inv2d.contiguous()
        weights_c = weights.contiguous()
        colors_c = colors.contiguous()
        image, alpha = _C.forward(pixels_c, means_c, inv_c, weights_c, colors_c)
        ctx.save_for_backward(pixels_c, means_c, inv_c, weights_c, colors_c)
        return image, alpha

    @staticmethod
    def backward(ctx, grad_image, grad_alpha):
        pixels, means2d, inv2d, weights, colors = ctx.saved_tensors
        grad_image = grad_image.contiguous()
        grad_alpha = grad_alpha.contiguous()
        grad_means, grad_inv, grad_weights, grad_colors = _C.backward(
            pixels,
            means2d,
            inv2d,
            weights,
            colors,
            grad_image,
            grad_alpha,
        )
        return None, grad_means, grad_inv, grad_weights, grad_colors


class _AdditiveSplatBounded(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        image_height,
        image_width,
        half_width,
        half_height,
        means2d,
        inv2d,
        weights,
        colors,
        radius_sigma,
    ):
        _check_bounded_inputs(
            image_height,
            image_width,
            half_width,
            half_height,
            means2d,
            inv2d,
            weights,
            colors,
            radius_sigma,
        )
        means_c = means2d.contiguous()
        inv_c = inv2d.contiguous()
        weights_c = weights.contiguous()
        colors_c = colors.contiguous()
        image, alpha = _C.bounded_forward(
            int(image_height),
            int(image_width),
            float(half_width),
            float(half_height),
            means_c,
            inv_c,
            weights_c,
            colors_c,
            float(radius_sigma),
        )
        ctx.image_height = int(image_height)
        ctx.image_width = int(image_width)
        ctx.half_width = float(half_width)
        ctx.half_height = float(half_height)
        ctx.radius_sigma = float(radius_sigma)
        ctx.save_for_backward(means_c, inv_c, weights_c, colors_c)
        return image, alpha

    @staticmethod
    def backward(ctx, grad_image, grad_alpha):
        means2d, inv2d, weights, colors = ctx.saved_tensors
        grad_image = grad_image.contiguous()
        grad_alpha = grad_alpha.contiguous()
        grad_means, grad_inv, grad_weights, grad_colors = _C.bounded_backward(
            ctx.image_height,
            ctx.image_width,
            ctx.half_width,
            ctx.half_height,
            means2d,
            inv2d,
            weights,
            colors,
            grad_image,
            grad_alpha,
            ctx.radius_sigma,
        )
        return None, None, None, None, grad_means, grad_inv, grad_weights, grad_colors, None


def additive_splat(
    pixels: torch.Tensor,
    means2d: torch.Tensor,
    inv2d: torch.Tensor,
    weights: torch.Tensor,
    colors: torch.Tensor,
):
    """Additively splat 2D Gaussians onto flattened pixels.

    Args:
        pixels: (P, 2) pixel coordinates in physical slice units.
        means2d: (N, 2) Gaussian centers in the same coordinates.
        inv2d: (N, 2, 2) inverse 2D covariances.
        weights: (N, 1) finite-thickness attenuated scalar weights.
        colors: (N, 3) RGB/DC colors.

    Returns:
        image: (P, 3), alpha: (P, 1)
    """
    return _AdditiveSplat.apply(pixels, means2d, inv2d, weights, colors)


def additive_splat_bounded(
    image_height: int,
    image_width: int,
    half_width: float,
    half_height: float,
    means2d: torch.Tensor,
    inv2d: torch.Tensor,
    weights: torch.Tensor,
    colors: torch.Tensor,
    radius_sigma: float = 3.0,
):
    """Additively splat 2D Gaussians onto an image using bounded footprints."""
    return _AdditiveSplatBounded.apply(
        int(image_height),
        int(image_width),
        float(half_width),
        float(half_height),
        means2d,
        inv2d,
        weights,
        colors,
        float(radius_sigma),
    )
