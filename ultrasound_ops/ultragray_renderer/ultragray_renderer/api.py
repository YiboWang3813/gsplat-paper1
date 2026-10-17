import torch
import torch.nn as nn

from .autograd import _UltragrayRenderFunction
from .types import UltragrayRenderConfig, ProbeGeometry


class UltragrayRenderer(nn.Module):
    def __init__(self, config: UltragrayRenderConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        attenuations: torch.Tensor,
        sh_coeffs: torch.Tensor,
        pixel_positions: torch.Tensor,  # (W, H, 3)
        ray_origins: torch.Tensor,      # (W, 3)
        ray_dirs: torch.Tensor,         # (W, 3)
        geometry: ProbeGeometry,
    ) -> torch.Tensor:
        return _UltragrayRenderFunction.apply(
            self.config,
            means,
            scales,
            quats,
            attenuations,
            sh_coeffs,
            pixel_positions,
            ray_origins,
            ray_dirs,
            geometry.origin_world,
            geometry.axis_lateral_world,
            geometry.axis_elevational_world,
            geometry.axis_axial_world,
        )
