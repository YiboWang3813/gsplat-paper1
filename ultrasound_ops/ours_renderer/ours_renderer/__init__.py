from .api import OursRenderer
from .check import should_validate_inputs, validate_render_inputs
from .grid_geometry import build_slice_grid_pack, shader_weight_mode_id
from .diagnostics import contribution_residual_stats, path_contribution_stats
from .ray_attenuation import render_ray_tile_attenuation
from .types import OursRenderConfig
from .us_focus_reference import render_us_focus_reference, us_focus_gaussian_footprint_reference

# Bump when CUDA convention changes (e.g. quaternion layout); training code keys renderer cache on this.
QUAT_CONVENTION_TAG = "wxyz_v1"

__all__ = [
    "OursRenderer",
    "OursRenderConfig",
    "QUAT_CONVENTION_TAG",
    "build_slice_grid_pack",
    "contribution_residual_stats",
    "path_contribution_stats",
    "shader_weight_mode_id",
    "render_us_focus_reference",
    "us_focus_gaussian_footprint_reference",
    "render_ray_tile_attenuation",
    "should_validate_inputs",
    "validate_render_inputs",
]
