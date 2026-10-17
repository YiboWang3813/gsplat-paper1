from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class SliceGeometry:
    origin_world: torch.Tensor
    axis_lateral_world: torch.Tensor
    axis_axial_world: torch.Tensor
    axis_elevational_world: torch.Tensor


@dataclass
class UltraGaussRenderConfig:
    H: int
    W: int
    lateral_spacing_mm_per_pix: float
    axial_spacing_mm_per_pix: float
    axial_near_mm: float
    chi2_threshold: float = 7.815
    mapper_sort_by_depth: bool = True
    fixed_bg_alpha: float = 0.0
    fixed_bg_intensity: float = 0.0
    diag_beta: float = 1e-4
    enable_shadow: bool = False
    shadow_delta: float = 0.0
    lateral_col_offset: Optional[float] = None

    def __post_init__(self) -> None:
        if self.H <= 0 or self.W <= 0:
            raise ValueError("H/W must be positive.")
        if self.lateral_spacing_mm_per_pix <= 0.0 or self.axial_spacing_mm_per_pix <= 0.0:
            raise ValueError("pixel spacing must be > 0.")
        if self.chi2_threshold <= 0.0:
            raise ValueError("chi2_threshold must be > 0.")
        if self.diag_beta <= 0.0:
            raise ValueError("diag_beta must be > 0.")
        if self.lateral_col_offset is None:
            self.lateral_col_offset = float(self.W - 1) * 0.5
