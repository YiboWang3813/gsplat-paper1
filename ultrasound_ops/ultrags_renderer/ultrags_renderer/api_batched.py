from typing import Dict, Tuple

import torch
import torch.nn as nn

from .autograd_batched import _BatchedRenderFunction
from .check import should_validate_inputs, validate_batched_render_inputs
from .types import BatchSliceGeometry, UltragsRenderConfig


class UltragsBatchedRenderer(nn.Module):
    def __init__(self, config: UltragsRenderConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        gauss_means: torch.Tensor,
        gauss_scales: torch.Tensor,
        gauss_quats: torch.Tensor,
        gauss_opacities: torch.Tensor,
        gauss_intensities: torch.Tensor,
        pixel_positions_world: torch.Tensor,
        geometry: BatchSliceGeometry,
        active_gauss_mask_batched: torch.Tensor = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if should_validate_inputs(self.config):
            validate_batched_render_inputs(
                config=self.config,
                gauss_means=gauss_means,
                gauss_scales=gauss_scales,
                gauss_quats=gauss_quats,
                gauss_opacities=gauss_opacities,
                gauss_intensities=gauss_intensities,
                pixel_positions_world=pixel_positions_world,
                geometry=geometry,
            )

        (
            pixel_colors,
            pixel_depths,
            pixel_intermediates,
            gauss_intermediates,
            pixel_gauss_counts,
            pixel_gauss_offsets,
            pixel_gauss_indices,
            pair_batch_offsets,
        ) = _BatchedRenderFunction.apply(
            self.config,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pixel_positions_world,
            geometry.origin_world,
            geometry.axis_lateral_world,
            geometry.axis_axial_world,
            geometry.axis_elevational_world,
            active_gauss_mask_batched,
            return_intermediates,
        )

        B = pixel_positions_world.shape[0]
        H, W = self.config.H, self.config.W
        color_image = pixel_colors.view(B, H, W)
        depth_image = pixel_depths.view(B, H, W)

        if return_intermediates:
            P = H * W
            intermediates = {
                "pixel_accum": pixel_intermediates[:, 0].view(B, H, W),
                "pixel_T": pixel_intermediates[:, 1].view(B, H, W),
                "pixel_contrib": pixel_intermediates[:, 2].view(B, H, W),
                "gauss_contribution": gauss_intermediates[:, 0],
                "gauss_occlusion": gauss_intermediates[:, 1],
                "gauss_participation": gauss_intermediates[:, 2],
                "pixel_gauss_counts": pixel_gauss_counts.view(B, P),
                "pixel_gauss_offsets": pixel_gauss_offsets.view(B, P),
                "pixel_gauss_indices": pixel_gauss_indices,
                "pair_batch_offsets": pair_batch_offsets,
                "num_pixel_gauss_pairs": torch.tensor(
                    int(pixel_gauss_indices.numel()),
                    dtype=torch.int64,
                    device=pixel_gauss_indices.device,
                ),
            }
        else:
            intermediates = {}

        return color_image, depth_image, intermediates
