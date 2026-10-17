from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import _C

from .types import UltragsRenderConfig


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


def mode_to_id(mode: str) -> int:
    m = str(mode).lower()
    if m == "alpha":
        return 0
    if m == "additive_sum":
        return 1
    if m == "additive_norm":
        return 2
    raise ValueError(f"Unsupported compositing_mode: {mode}")


class _RenderFunction(torch.autograd.Function):
    @staticmethod
    def _pack_gauss_attributes(
        gauss_means: torch.Tensor,
        gauss_scales: torch.Tensor,
        gauss_intensities: torch.Tensor,
        gauss_opacities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if gauss_intensities.dim() == 1:
            gauss_intensities = gauss_intensities.unsqueeze(1)
        if gauss_opacities.dim() == 1:
            gauss_opacities = gauss_opacities.unsqueeze(1)
        gauss_means_intensities = torch.cat([gauss_means, gauss_intensities], dim=1).contiguous()
        gauss_scales_opacities = torch.cat([gauss_scales, gauss_opacities], dim=1).contiguous()
        return gauss_means_intensities, gauss_scales_opacities

    @staticmethod
    def forward(
        ctx,
        config: UltragsRenderConfig,
        gauss_means: torch.Tensor,
        gauss_scales: torch.Tensor,
        gauss_quats: torch.Tensor,
        gauss_opacities: torch.Tensor,
        gauss_intensities: torch.Tensor,
        pixel_positions_world: torch.Tensor,
        probe_origin_world: torch.Tensor,
        axis_lateral_world: torch.Tensor,
        axis_axial_world: torch.Tensor,
        axis_elevational_world: torch.Tensor,
        return_intermediates: bool,
    ):
        gauss_means = ensure_tensor(gauss_means)
        gauss_scales = ensure_tensor(gauss_scales)
        gauss_quats = ensure_tensor(gauss_quats)
        gauss_opacities = ensure_tensor(gauss_opacities)
        gauss_intensities = ensure_tensor(gauss_intensities)
        pixel_positions_world = ensure_tensor(pixel_positions_world)
        probe_origin_world = ensure_tensor(probe_origin_world)
        axis_lateral_world = ensure_tensor(axis_lateral_world)
        axis_axial_world = ensure_tensor(axis_axial_world)
        axis_elevational_world = ensure_tensor(axis_elevational_world)

        pixel_positions_packed = pack_vec3(pixel_positions_world)
        gauss_means_packed = pack_vec3(gauss_means)
        gauss_scales_packed = pack_vec3(gauss_scales)
        gauss_means_intensities, gauss_scales_opacities = _RenderFunction._pack_gauss_attributes(
            gauss_means, gauss_scales, gauss_intensities, gauss_opacities
        )

        pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices = _C.MapperForwardCUDA(
            pixel_positions_packed,
            gauss_means_packed,
            gauss_quats,
            gauss_scales_packed,
            probe_origin_world,
            axis_elevational_world,
            axis_lateral_world,
            axis_axial_world,
            int(config.H),
            int(config.W),
            float(config.lateral_spacing_mm_per_pix),
            float(config.axial_spacing_mm_per_pix),
            float(config.slab_half_thickness_mm),
            float(config.axial_near_mm),
            float(config.k_sigma),
            float(config.lateral_col_offset),
            bool(config.mapper_sort_by_depth),
        )

        num_gauss = gauss_means.shape[0]
        if return_intermediates:
            gauss_intermediates = torch.empty((num_gauss, 3), dtype=torch.float32, device=gauss_means.device)
        else:
            gauss_intermediates = torch.empty(0, device=gauss_means.device)

        (
            pixel_colors,
            pixel_depths,
            pixel_accum,
            pixel_T,
            pixel_contrib,
            gauss_Rs,
            gauss_sigma_invs,
        ) = _C.ShaderForwardCUDA(
            int(config.W),
            float(config.fixed_bg_color),
            mode_to_id(config.compositing_mode),
            gauss_means_intensities,
            gauss_scales_opacities,
            gauss_quats,
            pixel_positions_packed,
            probe_origin_world,
            axis_elevational_world,
            pixel_gauss_offsets,
            pixel_gauss_counts,
            pixel_gauss_indices,
            gauss_intermediates,
        )

        ctx.config = config
        ctx.save_for_backward(
            gauss_means_intensities,
            gauss_scales_opacities,
            gauss_quats,
            pixel_positions_packed,
            probe_origin_world,
            axis_elevational_world,
            pixel_gauss_offsets,
            pixel_gauss_counts,
            pixel_gauss_indices,
            gauss_Rs,
            gauss_sigma_invs,
            pixel_accum,
            pixel_T,
        )

        if return_intermediates:
            pixel_intermediates = torch.stack([pixel_accum, pixel_T, pixel_contrib], dim=1)
        else:
            pixel_intermediates = torch.empty(0, device=pixel_positions_world.device)

        return (
            pixel_colors,
            pixel_depths,
            pixel_intermediates,
            gauss_intermediates,
            pixel_gauss_counts,
            pixel_gauss_offsets,
            pixel_gauss_indices,
        )

    @staticmethod
    def backward(
        ctx,
        dL_dPixelColors,
        dL_dPixelDepths,
        dL_dPixelIntermediates,
        dL_dGaussIntermediates,
        dL_dPixelGaussCounts,
        dL_dPixelGaussOffsets,
        dL_dPixelGaussIndices,
    ):
        del dL_dPixelIntermediates
        del dL_dGaussIntermediates
        del dL_dPixelGaussCounts
        del dL_dPixelGaussOffsets
        del dL_dPixelGaussIndices

        dL_dPixelColors = dL_dPixelColors.contiguous()
        dL_dPixelDepths = dL_dPixelDepths.contiguous()

        config = ctx.config
        (
            gauss_means_intensities,
            gauss_scales_opacities,
            gauss_quats,
            pixel_positions_packed,
            probe_origin_world,
            axis_elevational_world,
            pixel_gauss_offsets,
            pixel_gauss_counts,
            pixel_gauss_indices,
            gauss_Rs,
            gauss_sigma_invs,
            pixel_accum,
            pixel_T,
        ) = ctx.saved_tensors

        (
            dL_dGaussMeansIntensities,
            dL_dGaussQuats,
            dL_dGaussScalesOpacities,
        ) = _C.ShaderBackwardCUDA(
            int(config.W),
            float(config.fixed_bg_color),
            mode_to_id(config.compositing_mode),
            gauss_means_intensities,
            gauss_scales_opacities,
            gauss_quats,
            pixel_positions_packed,
            probe_origin_world,
            axis_elevational_world,
            pixel_gauss_offsets,
            pixel_gauss_counts,
            pixel_gauss_indices,
            gauss_Rs,
            gauss_sigma_invs,
            pixel_accum,
            pixel_T,
            dL_dPixelColors,
            dL_dPixelDepths,
        )

        dL_dGaussMeans = dL_dGaussMeansIntensities[:, :3]
        dL_dGaussIntensities = dL_dGaussMeansIntensities[:, 3:4].squeeze(1)
        dL_dGaussScales = dL_dGaussScalesOpacities[:, :3]
        dL_dGaussOpacities = dL_dGaussScalesOpacities[:, 3:4].squeeze(1)

        return (
            None,
            dL_dGaussMeans,
            dL_dGaussScales,
            dL_dGaussQuats,
            dL_dGaussOpacities,
            dL_dGaussIntensities,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
