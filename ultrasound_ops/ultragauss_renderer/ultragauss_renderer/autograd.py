from typing import Optional

import torch
import torch.nn.functional as F

from . import _C
from .types import UltraGaussRenderConfig


def ensure_tensor(t: Optional[torch.Tensor], dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if not t.is_cuda:
        t = t.cuda()
    if t.dtype != dtype:
        t = t.to(dtype=dtype)
    return t.contiguous()


def pack_vec3(t: torch.Tensor) -> torch.Tensor:
    if t.shape[-1] == 3:
        t = F.pad(t, (0, 1), mode="constant", value=0.0)
    return t.contiguous()


class _RenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        config: UltraGaussRenderConfig,
        gauss_means_world: torch.Tensor,
        gauss_precision_tril: torch.Tensor,
        gauss_opacities: torch.Tensor,
        gauss_intensities: torch.Tensor,
        pixel_positions_world: torch.Tensor,
        probe_origin_world: torch.Tensor,
        axis_lateral_world: torch.Tensor,
        axis_axial_world: torch.Tensor,
        axis_elevational_world: torch.Tensor,
    ):
        gauss_means_world = ensure_tensor(gauss_means_world)
        gauss_precision_tril = ensure_tensor(gauss_precision_tril)
        gauss_opacities = ensure_tensor(gauss_opacities)
        gauss_intensities = ensure_tensor(gauss_intensities)
        pixel_positions_world = ensure_tensor(pixel_positions_world)
        probe_origin_world = ensure_tensor(probe_origin_world)
        axis_lateral_world = ensure_tensor(axis_lateral_world)
        axis_axial_world = ensure_tensor(axis_axial_world)
        axis_elevational_world = ensure_tensor(axis_elevational_world)

        means_packed = pack_vec3(gauss_means_world)
        pixels_packed = pack_vec3(pixel_positions_world)

        counts, offsets, indices = _C.MapperForwardCUDA(
            pixels_packed,
            means_packed,
            gauss_precision_tril,
            probe_origin_world,
            axis_lateral_world,
            axis_axial_world,
            axis_elevational_world,
            int(config.H),
            int(config.W),
            float(config.lateral_spacing_mm_per_pix),
            float(config.axial_spacing_mm_per_pix),
            float(config.axial_near_mm),
            float(config.lateral_col_offset),
            float(config.chi2_threshold),
            bool(config.mapper_sort_by_depth),
            float(config.diag_beta),
        )

        packed_attr = torch.stack([gauss_opacities, gauss_intensities], dim=1).contiguous()
        image, alpha, depth, sigma_inv_cache = _C.ShaderForwardCUDA(
            packed_attr,
            means_packed,
            gauss_precision_tril,
            pixels_packed,
            probe_origin_world,
            axis_elevational_world,
            offsets,
            counts,
            indices,
            float(config.fixed_bg_alpha),
            float(config.fixed_bg_intensity),
            float(config.chi2_threshold),
            float(config.diag_beta),
        )

        ctx.save_for_backward(
            packed_attr,
            means_packed,
            gauss_precision_tril,
            pixels_packed,
            probe_origin_world,
            axis_elevational_world,
            offsets,
            counts,
            indices,
            sigma_inv_cache,
            alpha,
        )
        ctx.config = config
        return image, alpha, depth, counts, offsets, indices

    @staticmethod
    def backward(ctx, d_image, d_alpha, d_depth, d_counts, d_offsets, d_indices):
        del d_counts, d_offsets, d_indices
        d_image = d_image.contiguous()
        d_alpha = d_alpha.contiguous()
        d_depth = d_depth.contiguous()
        (
            packed_attr,
            means_packed,
            gauss_precision_tril,
            pixels_packed,
            probe_origin_world,
            axis_elevational_world,
            offsets,
            counts,
            indices,
            sigma_inv_cache,
            alpha,
        ) = ctx.saved_tensors
        d_means, d_precision, d_attr = _C.ShaderBackwardCUDA(
            d_image,
            d_alpha,
            d_depth,
            packed_attr,
            means_packed,
            gauss_precision_tril,
            pixels_packed,
            probe_origin_world,
            axis_elevational_world,
            offsets,
            counts,
            indices,
            sigma_inv_cache,
            alpha,
            float(ctx.config.fixed_bg_alpha),
            float(ctx.config.fixed_bg_intensity),
            float(ctx.config.chi2_threshold),
            float(ctx.config.diag_beta),
        )
        d_opacity = d_attr[:, 0]
        d_intensity = d_attr[:, 1]
        return (
            None,
            d_means[:, :3],
            d_precision,
            d_opacity,
            d_intensity,
            None,
            None,
            None,
            None,
            None,
        )
