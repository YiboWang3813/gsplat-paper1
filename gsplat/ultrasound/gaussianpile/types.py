from dataclasses import dataclass
from typing import Optional


@dataclass
class GaussianPlieRenderConfig:
    H: int
    W: int
    lateral_spacing_mm_per_pix: float
    axial_spacing_mm_per_pix: float
    axial_near_mm: float
    backend: str = "ours_focus"
    tile_size_px: int = 16
    radius_sigma: float = 3.0
    sigma_z_mm: float = 1.0
    prefilter_elevational_radius_mm: float = 0.0
    det_eps: float = 1.0e-6
    cov_eps: float = 1.0e-6
    opacity_cull_threshold: float = 1.0e-6
    render_chunk_size: int = 256
    enable_input_checks: bool = False
    lateral_col_offset: Optional[float] = None

    def __post_init__(self) -> None:
        if self.H <= 0 or self.W <= 0:
            raise ValueError(f"H/W must be positive, got H={self.H}, W={self.W}.")
        if self.lateral_spacing_mm_per_pix <= 0.0:
            raise ValueError("lateral_spacing_mm_per_pix must be > 0.")
        if self.axial_spacing_mm_per_pix <= 0.0:
            raise ValueError("axial_spacing_mm_per_pix must be > 0.")
        if self.lateral_col_offset is None:
            self.lateral_col_offset = float(self.W - 1) * 0.5
        if self.backend not in ("ours_focus", "ultragauss_lowmem"):
            raise ValueError("backend must be 'ours_focus' or 'ultragauss_lowmem'.")
        if self.tile_size_px <= 0:
            raise ValueError("tile_size_px must be > 0.")
        if self.radius_sigma <= 0.0:
            raise ValueError("radius_sigma must be > 0.")
        if self.sigma_z_mm <= 0.0:
            raise ValueError("sigma_z_mm must be > 0.")
        if self.prefilter_elevational_radius_mm < 0.0:
            raise ValueError("prefilter_elevational_radius_mm must be >= 0.")
        if self.det_eps <= 0.0 or self.cov_eps <= 0.0:
            raise ValueError("det_eps and cov_eps must be > 0.")
        if self.opacity_cull_threshold < 0.0:
            raise ValueError("opacity_cull_threshold must be >= 0.")
        if self.render_chunk_size <= 0:
            raise ValueError("render_chunk_size must be > 0.")
