import torch
import torch.nn.functional as F

from . import _C

from dataclasses import dataclass
from typing import List, Optional

def ensure_tensor(t: torch.Tensor, dtype=torch.float32):
    if t is None: return None
    if not t.is_cuda: t = t.cuda()
    if t.dtype != dtype: t = t.to(dtype=dtype)
    return t.contiguous()

def pack_vec3(t: torch.Tensor):
    if t.shape[-1] == 3:
        t = F.pad(t, (0, 1), mode='constant', value=0.0)
    return t.contiguous()

@dataclass
class SliceGeometry:
    """Slice pose/axes in world coordinates.

    Coordinate convention (probe frame):
      x=lateral, y=axial, z=elevational

    All axis vectors must be unit vectors in world space.
    """
    origin_world: torch.Tensor              # (3,)
    axis_lateral_world: torch.Tensor        # (3,)
    axis_axial_world: torch.Tensor          # (3,)
    axis_elevational_world: torch.Tensor    # (3,)

@dataclass
class EllipsoidMapperConfig:
    """Mapper configuration (dataset/slice meta derived)."""
    H: int
    W: int
    lateral_spacing_mm_per_pix: float
    axial_spacing_mm_per_pix: float
    slab_half_thickness_mm: float
    axial_near_mm: float
    k_sigma: float
    sort_by_depth: bool = True
    lateral_col_offset: Optional[float] = None

    def __post_init__(self):
        if self.lateral_col_offset is None:
            self.lateral_col_offset = float(self.W - 1) * 0.5

class EllipsoidMapper: 
    def __init__(self):
        pass 

    @staticmethod
    def map(
        # 1) Structured params (required)
        config: EllipsoidMapperConfig,
        geometry: SliceGeometry,
        # 2) Read-only inputs
        pixel_positions_world: torch.Tensor, # (P, 3) world (mm), row-major flatten
        gauss_means_world: torch.Tensor,     # (G, 3) world (mm)
        gauss_scales: torch.Tensor,          # (G, 3) (mm)
        gauss_quats: torch.Tensor,           # (G, 4)
    ) -> List[torch.Tensor]:
        """Compute sparse pixel->gauss candidates via tile culling + in-plane (lat/ax) test.

        Coordinate conventions:
          - Inputs `pixel_positions_world`/`gauss_means_world` are in WORLD coordinates (mm).
          - `geometry.*_world` are WORLD unit vectors.
          - Probe frame axis order is ALWAYS: x=lateral, y=axial, z=elevational.

        Sorting convention within each tile:
          - smaller axial (shallower, encountered earlier) comes first.
        """

        # 1) ensure tensors (cuda, dtype, contiguous)
        pixel_positions_world = ensure_tensor(pixel_positions_world)
        gauss_means_world = ensure_tensor(gauss_means_world)
        gauss_scales = ensure_tensor(gauss_scales)
        gauss_quats = ensure_tensor(gauss_quats)

        slice_origin_world = ensure_tensor(geometry.origin_world)
        axis_lateral_world = ensure_tensor(geometry.axis_lateral_world)
        axis_axial_world = ensure_tensor(geometry.axis_axial_world)
        axis_elevational_world = ensure_tensor(geometry.axis_elevational_world)

        # 2) pack vec3 -> vec4
        pixel_positions_packed = pack_vec3(pixel_positions_world)
        gauss_means_packed = pack_vec3(gauss_means_world)
        gauss_scales_packed = pack_vec3(gauss_scales)

        # 3) launch extension
        (pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices) = _C.launch_ellipsoid_mapper(
            pixel_positions_packed,
            gauss_means_packed,
            gauss_quats,
            gauss_scales_packed,
            slice_origin_world,
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
            bool(config.sort_by_depth),
        )

        return [pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices]
