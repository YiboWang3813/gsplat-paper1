from dataclasses import dataclass


@dataclass
class OursRenderConfig:
    """Projected-slice renderer: **world millimetres** for pose, plane sampling, and Gaussian centers/scales.

    ``pose_probe_to_world`` maps homogeneous **probe-local mm** to **world mm** (same convention as ``SliceMeta.pose``).

    Elevational hard culling uses corridor half-width
    ``elev_corridor_half_mm = num_slices_elevational * delta_elevational_mm / 2`` (typically built from dataset +
    model YAML). Mapper adds ellipsoidal ``k_sigma`` radius along elev (see plan §2.0).

    Shader ``probe3d``: probe-frame 3D Mahalanobis at the slice plane.

    Shader ``plane_cond2d``: strict slice cross-section on ``z=0``:
    ``alpha = clamp(opacity * exp(-½(q_cond + mu_z²/S_zz)), 0, 1)`` with Schur ``q_cond`` on lat–ax.
    Opacity should stay in ``(0, 1]``; renderer clamps final alpha to ``[0, 1]``.

    ``pose_probe_to_world`` stores rotation **columns** as probe axes (normalized in CUDA). Not a minimal
    SO(3) parameterization; if pose is optimized, use external SO(3)/SE(3) retraction.

    Gaussian quaternions use repository layout **(w, x, y, z)** per component, matching ``ultrags_renderer`` /
    ``scene`` ``quats_init``.
    """

    H: int
    W: int
    lateral_spacing_mm_per_pix: float
    axial_spacing_mm_per_pix: float
    axial_near_mm: float
    tile_size_px: int = 16
    elev_corridor_half_mm: float = 1.0
    slab_half_thickness_mm: float = 0.0
    k_sigma: float = 3.0
    mapper_kind: str = "tile"  # "tile" | "pixel"
    num_pixels_lateral: int = 5
    num_pixels_axial: int = 5
    mapper_sort_by_v: bool = True
    compositing_mode: str = "additive"  # "additive" | "alpha"
    fixed_bg_color: float = 0.0
    # probe3d: slice-plane 3D Mahalanobis at elev=0; plane_cond2d: Schur conditional 2D on slice grid pack;
    # us_focus: ultrasound focus-weighted additive accumulation (tile/additive path only in the staged rollout).
    shader_weight_mode: str = "probe3d"  # "probe3d" | "plane_cond2d" | "us_focus"
    local_conditioning_mode: str = "adaptive"  # none | fixed | adaptive
    us_focus_sigma_ele_0: float = 1.0
    us_focus_sigma_ele_1: float = 0.0
    us_focus_sigma_ele_min: float = 0.5
    us_focus_sigma_ele_max: float = 8.0
    us_focus_sigma_lat_0: float = 1.0
    us_focus_sigma_lat_1: float = 0.0
    us_focus_sigma_lat_min: float = 0.5
    us_focus_sigma_lat_max: float = 8.0
    us_focus_det_gamma: float = 0.25
    us_focus_det_eps: float = 1e-6
    us_focus_weight_min: float = 1e-6
    us_focus_use_lateral_focus: bool = False
    us_focus_inflate_lat_mm: float = 0.0
    us_focus_inflate_ax_mm: float = 0.0
    ray_attenuation_enabled: bool = False
    ray_attenuation_source: str = "opacity"  # "opacity" | "attenuation"
    ray_attenuation_beta: float = 0.0
    ray_attenuation_strength: float = 0.0
    ray_attenuation_min_transmission: float = 0.75
    ray_attenuation_weight_min: float = 1e-4
    ray_attenuation_axial_margin_mm: float = 0.0
    ray_attenuation_quadrature_window_mm: float = 2.0
    ray_attenuation_quadrature_points: int = 3
    ray_attenuation_mapper_kind: str = "tile_prefix"  # "tile_prefix" | "ray_csr"
    ray_attenuation_formula: str = "integral"  # "integral" | "slice_depth"
    ray_attenuation_candidate_k_sigma: float = -1.0
    render_chunk_size: int = 256
    enable_input_checks: bool = False

    def __post_init__(self) -> None:
        if self.H <= 0 or self.W <= 0:
            raise ValueError(f"H/W must be positive, got H={self.H}, W={self.W}.")
        if self.lateral_spacing_mm_per_pix <= 0.0:
            raise ValueError("lateral_spacing_mm_per_pix must be > 0.")
        if self.axial_spacing_mm_per_pix <= 0.0:
            raise ValueError("axial_spacing_mm_per_pix must be > 0.")
        if self.render_chunk_size <= 0:
            raise ValueError("render_chunk_size must be > 0.")
        if self.tile_size_px <= 0:
            raise ValueError("tile_size_px must be > 0.")
        if self.elev_corridor_half_mm <= 0.0:
            raise ValueError("elev_corridor_half_mm must be > 0.")
        if self.slab_half_thickness_mm < 0.0:
            raise ValueError("slab_half_thickness_mm must be >= 0.")
        if self.k_sigma <= 0.0:
            raise ValueError("k_sigma must be > 0.")
        if self.mapper_kind not in ("tile", "pixel"):
            raise ValueError("mapper_kind must be 'tile' or 'pixel'.")
        if self.mapper_kind == "pixel":
            if self.num_pixels_lateral <= 0 or self.num_pixels_axial <= 0:
                raise ValueError("num_pixels_lateral / num_pixels_axial must be > 0 for pixel mapper.")
        if self.compositing_mode not in ("additive", "alpha"):
            raise ValueError("compositing_mode must be 'additive' or 'alpha'.")
        if self.compositing_mode == "alpha" and not self.mapper_sort_by_v:
            raise ValueError("Alpha compositing requires mapper_sort_by_v=True.")
        if self.local_conditioning_mode not in ("none", "fixed", "adaptive"):
            raise ValueError("local_conditioning_mode must be none, fixed, or adaptive")
        if self.local_conditioning_mode != "adaptive" and (
            self.shader_weight_mode != "us_focus" or self.us_focus_use_lateral_focus
        ):
            raise ValueError("Local ablations require us_focus with lateral focus disabled")
        if self.shader_weight_mode not in ("probe3d", "plane_cond2d", "us_focus"):
            raise ValueError("shader_weight_mode must be 'probe3d', 'plane_cond2d', or 'us_focus'.")
        if self.shader_weight_mode == "us_focus":
            if self.mapper_kind != "tile":
                raise ValueError("us_focus currently supports mapper_kind='tile' only.")
            if self.compositing_mode != "additive":
                raise ValueError("us_focus currently supports compositing_mode='additive' only.")
        if self.ray_attenuation_enabled and self.mapper_kind != "tile":
            raise ValueError("ray attenuation Stage A requires mapper_kind='tile'.")
        if self.ray_attenuation_source not in ("opacity", "attenuation"):
            raise ValueError("ray_attenuation_source must be 'opacity' or 'attenuation'.")
        us_focus_positive = {
            "us_focus_sigma_ele_0": self.us_focus_sigma_ele_0,
            "us_focus_sigma_ele_min": self.us_focus_sigma_ele_min,
            "us_focus_sigma_ele_max": self.us_focus_sigma_ele_max,
            "us_focus_sigma_lat_0": self.us_focus_sigma_lat_0,
            "us_focus_sigma_lat_min": self.us_focus_sigma_lat_min,
            "us_focus_sigma_lat_max": self.us_focus_sigma_lat_max,
            "us_focus_det_eps": self.us_focus_det_eps,
            "us_focus_weight_min": self.us_focus_weight_min,
        }
        for name, value in us_focus_positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0.")
        if self.us_focus_sigma_ele_max < self.us_focus_sigma_ele_min:
            raise ValueError("us_focus_sigma_ele_max must be >= us_focus_sigma_ele_min.")
        if self.us_focus_sigma_lat_max < self.us_focus_sigma_lat_min:
            raise ValueError("us_focus_sigma_lat_max must be >= us_focus_sigma_lat_min.")
        if self.us_focus_det_gamma < 0.0:
            raise ValueError("us_focus_det_gamma must be >= 0.")
        if self.us_focus_inflate_lat_mm < 0.0:
            raise ValueError("us_focus_inflate_lat_mm must be >= 0.")
        if self.us_focus_inflate_ax_mm < 0.0:
            raise ValueError("us_focus_inflate_ax_mm must be >= 0.")
        if self.us_focus_sigma_ele_1 < 0.0:
            raise ValueError("us_focus_sigma_ele_1 must be >= 0.")
        if self.us_focus_sigma_lat_1 < 0.0:
            raise ValueError("us_focus_sigma_lat_1 must be >= 0.")
        if not 0.0 <= self.ray_attenuation_beta <= 1.0:
            raise ValueError("ray_attenuation_beta must be in [0, 1].")
        if self.ray_attenuation_strength < 0.0:
            raise ValueError("ray_attenuation_strength must be >= 0.")
        if not 0.0 < self.ray_attenuation_min_transmission <= 1.0:
            raise ValueError("ray_attenuation_min_transmission must be in (0, 1].")
        if self.ray_attenuation_weight_min < 0.0:
            raise ValueError("ray_attenuation_weight_min must be >= 0.")
        if self.ray_attenuation_axial_margin_mm < 0.0:
            raise ValueError("ray_attenuation_axial_margin_mm must be >= 0.")
        if self.ray_attenuation_quadrature_window_mm < 0.0:
            raise ValueError("ray_attenuation_quadrature_window_mm must be >= 0.")
        if int(self.ray_attenuation_quadrature_points) not in (1, 2, 3):
            raise ValueError("ray_attenuation_quadrature_points must be 1, 2, or 3.")
        if self.ray_attenuation_mapper_kind not in ("tile_prefix", "ray_csr"):
            raise ValueError("ray_attenuation_mapper_kind must be 'tile_prefix' or 'ray_csr'.")
        if self.ray_attenuation_formula not in ("integral", "slice_depth"):
            raise ValueError("ray_attenuation_formula must be 'integral' or 'slice_depth'.")
        if self.ray_attenuation_candidate_k_sigma != -1.0 and self.ray_attenuation_candidate_k_sigma <= 0.0:
            raise ValueError("ray_attenuation_candidate_k_sigma must be -1 or > 0.")
