from typing import Dict, Tuple

import torch
import torch.nn as nn

from .autograd import _RenderFunction
from .check import should_validate_inputs, validate_render_inputs
from .types import SliceGeometry, UltragsRenderConfig


class UltragsRenderer(nn.Module):
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
        geometry: SliceGeometry,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if should_validate_inputs(self.config):
            validate_render_inputs(
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
        ) = _RenderFunction.apply(
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
            return_intermediates,
        )

        color_image = pixel_colors.view(self.config.H, self.config.W)
        depth_image = pixel_depths.view(self.config.H, self.config.W)

        if return_intermediates:
            intermediates = {
                "pixel_accum": pixel_intermediates[:, 0],
                "pixel_T": pixel_intermediates[:, 1],
                "pixel_contrib": pixel_intermediates[:, 2],
                "gauss_contribution": gauss_intermediates[:, 0],
                "gauss_occlusion": gauss_intermediates[:, 1],
                "gauss_participation": gauss_intermediates[:, 2],
                "pixel_gauss_counts": pixel_gauss_counts,
                "pixel_gauss_offsets": pixel_gauss_offsets,
                "pixel_gauss_indices": pixel_gauss_indices,
                "num_pixel_gauss_pairs": torch.tensor(
                    int(pixel_gauss_indices.numel()),
                    dtype=torch.int64,
                    device=pixel_gauss_indices.device,
                ),
            }
        else:
            intermediates = {}

        return color_image, depth_image, intermediates
