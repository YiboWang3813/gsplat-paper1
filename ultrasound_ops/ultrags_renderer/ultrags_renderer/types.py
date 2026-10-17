from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class SliceGeometry:
    """Slice pose and probe axes in world coordinates."""

    origin_world: torch.Tensor
    axis_lateral_world: torch.Tensor
    axis_axial_world: torch.Tensor
    axis_elevational_world: torch.Tensor


@dataclass
class BatchSliceGeometry:
    """B slices: each field is (B, 3) in world coordinates."""

    origin_world: torch.Tensor
    axis_lateral_world: torch.Tensor
    axis_axial_world: torch.Tensor
    axis_elevational_world: torch.Tensor


@dataclass
class UltragsRenderConfig:
    """Unified config for mapper + shader pipeline."""

    H: int
    W: int
    lateral_spacing_mm_per_pix: float
    axial_spacing_mm_per_pix: float
    slab_half_thickness_mm: float
    axial_near_mm: float
    k_sigma: float
    mapper_sort_by_depth: bool = True
    fixed_bg_color: float = 0.0
    compositing_mode: str = "alpha"
    enable_input_checks: bool = False
    lateral_col_offset: Optional[float] = None

    def __post_init__(self) -> None:
        if self.H <= 0 or self.W <= 0:
            raise ValueError(f"H/W must be positive, got H={self.H}, W={self.W}.")
        if self.lateral_spacing_mm_per_pix <= 0.0:
            raise ValueError("lateral_spacing_mm_per_pix must be > 0.")
        if self.axial_spacing_mm_per_pix <= 0.0:
            raise ValueError("axial_spacing_mm_per_pix must be > 0.")
        if self.slab_half_thickness_mm < 0.0:
            raise ValueError("slab_half_thickness_mm must be >= 0.")
        if self.k_sigma <= 0.0:
            raise ValueError("k_sigma must be > 0.")
        if self.lateral_col_offset is None:
            self.lateral_col_offset = float(self.W - 1) * 0.5
