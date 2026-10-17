import torch

from .autograd import _us_focus_params_vector
from .grid_geometry import shader_weight_mode_id
from .types import OursRenderConfig

try:
    from . import _C  # type: ignore[attr-defined]
except Exception:
    _C = None


def render_ray_tile_attenuation(
    config: OursRenderConfig,
    gauss_means: torch.Tensor,
    gauss_scales: torch.Tensor,
    gauss_quats: torch.Tensor,
    gauss_attenuations: torch.Tensor,
    pose_probe_to_world: torch.Tensor,
    slice_grid_geometry: torch.Tensor,
) -> torch.Tensor:
    """Return a global ray attenuation modulation map.

    The local response branch remains the normal projected-slice renderer. This helper
    computes only the bounded global modifier. The integrated training path uses the
    shared-mapper CUDA forward instead of this standalone helper.
    """
    if not config.ray_attenuation_enabled:
        return torch.ones((config.H, config.W), device=gauss_means.device, dtype=gauss_means.dtype)
    if _C is None or not hasattr(_C, "RayTileAttenuationCUDA"):
        raise RuntimeError("render_ray_tile_attenuation requires ours_renderer._C.RayTileAttenuationCUDA.")
    if config.mapper_kind != "tile":
        raise ValueError("ray attenuation Stage A reuses the tile mapper and requires mapper_kind='tile'.")

    device = gauss_means.device
    us_focus_params = _us_focus_params_vector(config, device)
    modulation = _C.RayTileAttenuationCUDA(
        gauss_means.float().contiguous(),
        gauss_scales.float().contiguous(),
        gauss_quats.float().contiguous(),
        gauss_attenuations.float().contiguous(),
        pose_probe_to_world.float().contiguous(),
        slice_grid_geometry.to(device=device, dtype=torch.float32).contiguous(),
        us_focus_params,
        int(config.H),
        int(config.W),
        float(config.lateral_spacing_mm_per_pix),
        float(config.axial_spacing_mm_per_pix),
        float(config.axial_near_mm),
        int(config.tile_size_px),
        float(config.elev_corridor_half_mm),
        float(config.slab_half_thickness_mm),
        float(config.k_sigma),
        int(shader_weight_mode_id(config.shader_weight_mode)),
        float(config.ray_attenuation_beta),
        float(config.ray_attenuation_strength),
        float(config.ray_attenuation_min_transmission),
        float(config.ray_attenuation_weight_min),
        float(config.ray_attenuation_axial_margin_mm),
    )
    return modulation.to(dtype=gauss_means.dtype)
