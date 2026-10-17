from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ProbeGeometry:
    origin_world: torch.Tensor
    axis_lateral_world: torch.Tensor
    axis_elevational_world: torch.Tensor
    axis_axial_world: torch.Tensor


@dataclass
class UltragrayRenderConfig:
    image_width: int
    image_height: int
    width_mm: float
    near_mm: float
    elev_cutoff_margin_mm: float = 0.0
    k_sigma: float = 3.0
    ops_max_offset_mm: float = 2.0
    quadrature_window_radius_mm: float = 2.0
    local_transmission_window_only: bool = False
    local_transmission_max_window_mm: float = 17.0
    elev_beam_width_mm: float = 2.0
    lateral_col_offset: Optional[float] = None

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width/image_height must be positive.")
        if self.width_mm <= 0.0:
            raise ValueError("width_mm must be > 0.")
        if self.k_sigma <= 0.0:
            raise ValueError("k_sigma must be > 0.")
        if self.ops_max_offset_mm < 0.0:
            raise ValueError("ops_max_offset_mm must be >= 0.")
        if self.quadrature_window_radius_mm < 0.0:
            raise ValueError("quadrature_window_radius_mm must be >= 0.")
        if self.local_transmission_max_window_mm <= 0.0:
            raise ValueError("local_transmission_max_window_mm must be > 0.")
        if self.lateral_col_offset is None:
            self.lateral_col_offset = float(self.image_width - 1) * 0.5
