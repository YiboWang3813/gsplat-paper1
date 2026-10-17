import torch
import torch.nn as nn

from .autograd import render_projected_slice
from .check import should_validate_inputs, validate_render_inputs
from .types import OursRenderConfig


class OursRenderer(nn.Module):
    """World-mm projected-slice renderer (Gaussians + probe→world 4×4 pose in millimetres)."""

    def __init__(self, config: OursRenderConfig):
        super().__init__()
        self.config = config
        self._slice_grid_geometry = None

    def forward(
        self,
        gauss_means: torch.Tensor,
        gauss_scales: torch.Tensor,
        gauss_quats: torch.Tensor,
        gauss_opacities: torch.Tensor,
        gauss_intensities: torch.Tensor,
        pose_probe_to_world: torch.Tensor,
        gauss_attenuations: torch.Tensor = None,
        collect_path_stats: bool = False,
    ):
        if should_validate_inputs(self.config):
            validate_render_inputs(
                config=self.config,
                gauss_means=gauss_means,
                gauss_scales=gauss_scales,
                gauss_quats=gauss_quats,
                gauss_opacities=gauss_opacities,
                gauss_intensities=gauss_intensities,
                pose_probe_to_world=pose_probe_to_world,
            )

        grid = getattr(self, "_slice_grid_geometry", None)
        if grid is None:
            raise RuntimeError(
                "OursRenderer requires slice_grid_geometry; call set_slice_grid_geometry() from render driver."
            )

        return render_projected_slice(
            config=self.config,
            gauss_means=gauss_means,
            gauss_scales=gauss_scales,
            gauss_quats=gauss_quats,
            gauss_opacities=gauss_opacities,
            gauss_intensities=gauss_intensities,
            gauss_attenuations=gauss_attenuations,
            pose_probe_to_world=pose_probe_to_world,
            slice_grid_geometry=grid,
            collect_path_stats=collect_path_stats,
        )

    def set_slice_grid_geometry(self, slice_grid_geometry: torch.Tensor) -> None:
        self._slice_grid_geometry = slice_grid_geometry
