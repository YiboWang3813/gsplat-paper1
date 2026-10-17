import torch

from .types import GaussianPlieRenderConfig

try:
    from . import _C  # type: ignore[attr-defined]
except Exception:
    _C = None

_META_LEN = 16
_US_FOCUS_PARAMS_LEN = 14

SHADER_WEIGHT_US_FOCUS = 2


def _meta_vector(config: GaussianPlieRenderConfig, device: torch.device) -> torch.Tensor:
    sigma_z = float(config.sigma_z_mm)
    v = torch.zeros((_META_LEN,), dtype=torch.float32, device=device)
    v[0] = float(config.H)
    v[1] = float(config.W)
    v[2] = float(config.lateral_spacing_mm_per_pix)
    v[3] = float(config.axial_spacing_mm_per_pix)
    v[4] = float(config.axial_near_mm)
    v[5] = float(config.tile_size_px)
    v[6] = max(float(config.radius_sigma) * sigma_z, sigma_z * 0.5)
    v[7] = 0.0
    v[8] = float(config.radius_sigma)
    v[9] = 0.0
    v[10] = 0.0
    v[11] = 5.0
    v[12] = 5.0
    v[13] = 0.0
    v[14] = 0.0
    v[15] = float(SHADER_WEIGHT_US_FOCUS)
    return v


def _us_focus_params_vector(config: GaussianPlieRenderConfig, device: torch.device) -> torch.Tensor:
    sigma_z = float(config.sigma_z_mm)
    v = torch.zeros((_US_FOCUS_PARAMS_LEN,), dtype=torch.float32, device=device)
    v[0] = sigma_z
    v[1] = 0.0
    v[2] = sigma_z
    v[3] = sigma_z
    v[4] = 1.0
    v[5] = 0.0
    v[6] = 0.5
    v[7] = 8.0
    v[8] = 0.5
    v[9] = float(config.det_eps)
    v[10] = max(float(config.opacity_cull_threshold), 1.0e-12)
    v[11] = 0.0
    v[12] = 0.0
    v[13] = 0.0
    return v


class _GaussianPileProjectedSliceCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gauss_means: torch.Tensor,
        gauss_scales: torch.Tensor,
        gauss_quats: torch.Tensor,
        gauss_opacities: torch.Tensor,
        gauss_intensities: torch.Tensor,
        pose_probe_to_world: torch.Tensor,
        slice_grid_geometry: torch.Tensor,
        meta: torch.Tensor,
        us_focus_params: torch.Tensor,
    ):
        pose_f = pose_probe_to_world.float().contiguous()
        ctx.pose_dtype = pose_probe_to_world.dtype
        grid_f = slice_grid_geometry.float().contiguous()
        us_focus_params_f = us_focus_params.float().contiguous()

        H = int(meta[0].item())
        W = int(meta[1].item())
        lateral_mm = float(meta[2].item())
        axial_mm = float(meta[3].item())
        axial_near_mm = float(meta[4].item())
        tile_size_px = int(meta[5].item())
        elev_corridor_half_mm = float(meta[6].item())
        slab_half_thickness_mm = float(meta[7].item())
        k_sigma = float(meta[8].item())
        sort_by_v = bool(float(meta[9].item()) != 0.0)
        mapper_kind = int(meta[10].item())
        num_pixels_lateral = int(meta[11].item())
        num_pixels_axial = int(meta[12].item())
        compositing_mode = int(meta[13].item())
        fixed_bg_color = float(meta[14].item())
        shader_weight_mode = int(meta[15].item())

        colors_hw, gauss_rects, csr0, csr1, csr2 = _C.ProjectedSliceForwardCUDA(
            gauss_means.float().contiguous(),
            gauss_scales.float().contiguous(),
            gauss_quats.float().contiguous(),
            gauss_opacities.float().contiguous(),
            gauss_intensities.float().contiguous(),
            pose_f,
            grid_f,
            us_focus_params_f,
            H,
            W,
            lateral_mm,
            axial_mm,
            axial_near_mm,
            tile_size_px,
            elev_corridor_half_mm,
            slab_half_thickness_mm,
            k_sigma,
            mapper_kind,
            num_pixels_lateral,
            num_pixels_axial,
            sort_by_v,
            compositing_mode,
            fixed_bg_color,
            shader_weight_mode,
        )

        ctx.save_for_backward(
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose_f,
            grid_f,
            gauss_rects,
            csr0,
            csr1,
            csr2,
            meta,
            us_focus_params_f,
        )
        ctx.gauss_rect_is_pixels = 0
        return colors_hw

    @staticmethod
    def backward(ctx, grad_colors_hw: torch.Tensor):
        (
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose_f,
            grid_f,
            gauss_rects,
            csr0,
            csr1,
            csr2,
            meta,
            us_focus_params,
        ) = ctx.saved_tensors

        H = int(meta[0].item())
        W = int(meta[1].item())
        lateral_mm = float(meta[2].item())
        axial_mm = float(meta[3].item())
        axial_near_mm = float(meta[4].item())
        tile_size_px = int(meta[5].item())
        compositing_mode = int(meta[13].item())
        fixed_bg_color = float(meta[14].item())
        mapper_kind = int(meta[10].item())
        k_sigma = float(meta[8].item())
        shader_weight_mode = int(meta[15].item())

        if grad_colors_hw is None:
            grad_colors_hw = torch.zeros((H, W), device=gauss_means.device, dtype=torch.float32)
        gc = grad_colors_hw.float().reshape(-1).contiguous()

        d_means, d_scales, d_quats, d_opacity, d_intensity, d_pose = _C.ProjectedSliceBackwardCUDA(
            gauss_means.float().contiguous(),
            gauss_scales.float().contiguous(),
            gauss_quats.float().contiguous(),
            gauss_opacities.float().contiguous(),
            gauss_intensities.float().contiguous(),
            pose_f,
            grid_f,
            us_focus_params,
            gauss_rects,
            int(ctx.gauss_rect_is_pixels),
            H,
            W,
            tile_size_px,
            lateral_mm,
            axial_mm,
            axial_near_mm,
            k_sigma,
            compositing_mode,
            fixed_bg_color,
            mapper_kind,
            csr0,
            csr1,
            csr2,
            gc,
            shader_weight_mode,
            bool(ctx.needs_input_grad[5]),
        )

        d_grid = torch.zeros_like(grid_f)
        d_meta = torch.zeros_like(meta)
        d_us_focus_params = torch.zeros_like(us_focus_params)
        return (
            d_means.to(gauss_means.dtype),
            d_scales.to(gauss_scales.dtype),
            d_quats.to(gauss_quats.dtype),
            d_opacity.to(gauss_opacities.dtype),
            d_intensity.to(gauss_intensities.dtype),
            d_pose.to(getattr(ctx, "pose_dtype", torch.float32)),
            d_grid,
            d_meta,
            d_us_focus_params,
        )


def render_gaussianpile_projected_slice(
    config: GaussianPlieRenderConfig,
    gauss_means: torch.Tensor,
    gauss_scales: torch.Tensor,
    gauss_quats: torch.Tensor,
    gauss_alphas: torch.Tensor,
    pose_probe_to_world: torch.Tensor,
    slice_grid_geometry: torch.Tensor,
) -> torch.Tensor:
    if (
        _C is not None
        and gauss_means.is_cuda
        and hasattr(_C, "ProjectedSliceForwardCUDA")
        and hasattr(_C, "ProjectedSliceBackwardCUDA")
    ):
        device = gauss_means.device
        fixed_intensities = torch.ones_like(gauss_alphas)
        return _GaussianPileProjectedSliceCudaFunction.apply(
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_alphas,
            fixed_intensities,
            pose_probe_to_world,
            slice_grid_geometry.to(device=device, dtype=torch.float32),
            _meta_vector(config, device),
            _us_focus_params_vector(config, device),
        )
    raise RuntimeError("GaussianPile CUDA extension is not built; run ops/gaussianplie_renderer/rebuild.sh.")
