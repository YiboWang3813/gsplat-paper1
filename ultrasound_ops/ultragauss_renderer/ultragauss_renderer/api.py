from typing import Dict, Tuple

import torch
import torch.nn as nn

from .autograd import _RenderFunction
from .types import SliceGeometry, UltraGaussRenderConfig


class UltraGaussRenderer(nn.Module):
    def __init__(self, config: UltraGaussRenderConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        gauss_means_world: torch.Tensor,
        gauss_precision_tril: torch.Tensor,
        gauss_opacities: torch.Tensor,
        gauss_intensities: torch.Tensor,
        pixel_positions_world: torch.Tensor,
        geometry: SliceGeometry,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        image, alpha, depth, pixel_counts, pixel_offsets, pixel_indices = _RenderFunction.apply(
            self.config,
            gauss_means_world,
            gauss_precision_tril,
            gauss_opacities,
            gauss_intensities,
            pixel_positions_world,
            geometry.origin_world,
            geometry.axis_lateral_world,
            geometry.axis_axial_world,
            geometry.axis_elevational_world,
        )
        if self.config.enable_shadow:
            delta = self.config.shadow_delta if self.config.shadow_delta > 0.0 else (1.0 / float(self.config.H))
            alpha_img = alpha.view(self.config.H, self.config.W)
            image = image.view(self.config.H, self.config.W)
            T = torch.exp(-torch.cumsum(alpha_img * delta, dim=0))
            image = (image * T).view(-1)

        image = image.view(self.config.H, self.config.W)
        alpha = alpha.view(self.config.H, self.config.W)
        depth = depth.view(self.config.H, self.config.W)
        if return_intermediates:
            intermediates = {
                "pixel_gauss_counts": pixel_counts,
                "pixel_gauss_offsets": pixel_offsets,
                "pixel_gauss_indices": pixel_indices,
            }
        else:
            intermediates = {}
        return image, alpha, depth, intermediates
