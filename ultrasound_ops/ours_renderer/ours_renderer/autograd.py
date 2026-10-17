import torch

from .grid_geometry import shader_weight_mode_id
from .types import OursRenderConfig

try:
    from . import _C  # type: ignore[attr-defined]
except Exception:
    _C = None

_META_LEN = 16
_US_FOCUS_PARAMS_LEN = 15
_RAY_PARAMS_LEN = 11


def _mapper_kind_id(config: OursRenderConfig) -> int:
    return 0 if config.mapper_kind == "tile" else 1


def _compositing_id(config: OursRenderConfig) -> int:
    return 0 if config.compositing_mode == "additive" else 1


def _meta_vector(config: OursRenderConfig, device: torch.device) -> torch.Tensor:
    v = torch.zeros((_META_LEN,), dtype=torch.float32, device=device)
    v[0] = float(config.H)
    v[1] = float(config.W)
    v[2] = float(config.lateral_spacing_mm_per_pix)
    v[3] = float(config.axial_spacing_mm_per_pix)
    v[4] = float(config.axial_near_mm)
    v[5] = float(config.tile_size_px)
    v[6] = float(config.elev_corridor_half_mm)
    v[7] = float(config.slab_half_thickness_mm)
    v[8] = float(config.k_sigma)
    v[9] = 1.0 if config.mapper_sort_by_v else 0.0
    v[10] = float(_mapper_kind_id(config))
    v[11] = float(config.num_pixels_lateral)
    v[12] = float(config.num_pixels_axial)
    v[13] = float(_compositing_id(config))
    v[14] = float(config.fixed_bg_color)
    v[15] = float(shader_weight_mode_id(config.shader_weight_mode))
    return v


def _us_focus_params_vector(config: OursRenderConfig, device: torch.device) -> torch.Tensor:
    v = torch.zeros((_US_FOCUS_PARAMS_LEN,), dtype=torch.float32, device=device)
    v[0] = float(config.us_focus_sigma_ele_0)
    v[1] = 0.0 if config.local_conditioning_mode == "fixed" else float(config.us_focus_sigma_ele_1)
    v[2] = float(config.us_focus_sigma_ele_min)
    v[3] = float(config.us_focus_sigma_ele_max)
    v[4] = float(config.us_focus_sigma_lat_0)
    v[5] = float(config.us_focus_sigma_lat_1)
    v[6] = float(config.us_focus_sigma_lat_min)
    v[7] = float(config.us_focus_sigma_lat_max)
    v[8] = float(config.us_focus_det_gamma)
    v[9] = float(config.us_focus_det_eps)
    v[10] = float(config.us_focus_weight_min)
    v[11] = 1.0 if config.us_focus_use_lateral_focus else 0.0
    v[12] = float(config.us_focus_inflate_lat_mm)
    v[13] = float(config.us_focus_inflate_ax_mm)
    v[14] = 1.0 if config.local_conditioning_mode == "none" else 0.0
    return v


def _ray_params_vector(config: OursRenderConfig, device: torch.device) -> torch.Tensor:
    v = torch.zeros((_RAY_PARAMS_LEN,), dtype=torch.float32, device=device)
    v[0] = 1.0 if config.ray_attenuation_enabled else 0.0
    v[1] = float(config.ray_attenuation_beta)
    v[2] = float(config.ray_attenuation_strength)
    v[3] = float(config.ray_attenuation_min_transmission)
    v[4] = float(config.ray_attenuation_weight_min)
    v[5] = float(config.ray_attenuation_axial_margin_mm)
    v[6] = 1.0 if config.ray_attenuation_mapper_kind == "ray_csr" else 0.0
    v[7] = float(config.ray_attenuation_quadrature_window_mm)
    v[8] = float(config.ray_attenuation_quadrature_points)
    v[9] = 1.0 if config.ray_attenuation_formula == "slice_depth" else 0.0
    v[10] = float(config.ray_attenuation_candidate_k_sigma)
    return v


class _ProjectedSliceCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gauss_means: torch.Tensor,
        gauss_scales: torch.Tensor,
        gauss_quats: torch.Tensor,
        gauss_opacities: torch.Tensor,
        gauss_intensities: torch.Tensor,
        gauss_attenuations: torch.Tensor,
        pose_probe_to_world: torch.Tensor,
        slice_grid_geometry: torch.Tensor,
        meta: torch.Tensor,
        us_focus_params: torch.Tensor,
        ray_params: torch.Tensor,
        collect_path_stats: bool,
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
        ray_enabled = bool(float(ray_params[0].item()) != 0.0)

        if ray_enabled:
            if mapper_kind != 0:
                raise ValueError("ray attenuation Stage A requires tile mapper.")
            (
                colors_hw,
                modulation_hw,
                gauss_rects,
                csr0,
                csr1,
                csr2,
                ray_start,
                ray_end,
                ray_indices,
                ray_depths,
            ) = _C.ProjectedSliceForwardRayTileCUDA(
                gauss_means.float().contiguous(),
                gauss_scales.float().contiguous(),
                gauss_quats.float().contiguous(),
                gauss_opacities.float().contiguous(),
                gauss_intensities.float().contiguous(),
                gauss_attenuations.float().contiguous(),
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
                sort_by_v,
                compositing_mode,
                fixed_bg_color,
                shader_weight_mode,
                float(ray_params[1].item()),
                float(ray_params[2].item()),
                float(ray_params[3].item()),
                float(ray_params[4].item()),
                float(ray_params[5].item()),
                float(ray_params[7].item()),
                int(ray_params[8].item()),
                int(ray_params[6].item()),
                int(ray_params[9].item()),
                float(ray_params[10].item()),
            )
        else:
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
            modulation_hw = torch.ones((H, W), device=colors_hw.device, dtype=torch.float32)
            ray_start = torch.empty((0,), device=colors_hw.device, dtype=torch.int32)
            ray_end = torch.empty((0,), device=colors_hw.device, dtype=torch.int32)
            ray_indices = torch.empty((0,), device=colors_hw.device, dtype=torch.int32)
            ray_depths = torch.empty((0,), device=colors_hw.device, dtype=torch.float32)

        if bool(collect_path_stats):
            if not ray_enabled or int(ray_params[6].item()) != 1:
                raise ValueError("collect_path_stats requires enabled ray_csr attenuation.")
            if not hasattr(_C, "RayCsrPathContributionStatsFromCsrCUDA"):
                raise RuntimeError("ours_renderer._C lacks reusable-CSR path statistics support.")
            path_contribution, path_support = _C.RayCsrPathContributionStatsFromCsrCUDA(
                gauss_means.float().contiguous(),
                gauss_scales.float().contiguous(),
                gauss_quats.float().contiguous(),
                gauss_attenuations.float().contiguous(),
                pose_f,
                grid_f,
                ray_start,
                ray_end,
                ray_indices,
                ray_depths,
                H,
                W,
                lateral_mm,
                axial_mm,
                axial_near_mm,
                float(ray_params[2].item()),
                float(ray_params[4].item()),
                float(ray_params[5].item()),
                float(ray_params[7].item()),
                int(ray_params[8].item()),
                int(ray_params[9].item()),
            )
        else:
            path_contribution = torch.empty((0,), device=colors_hw.device, dtype=torch.float32)
            path_support = torch.empty((0,), device=colors_hw.device, dtype=torch.float32)

        gauss_rect_is_pixels = 1 if mapper_kind != 0 else 0
        ctx.save_for_backward(
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            gauss_attenuations,
            pose_f,
            grid_f,
            gauss_rects,
            csr0,
            csr1,
            csr2,
            ray_start,
            ray_end,
            ray_indices,
            ray_depths,
            meta,
            us_focus_params_f,
            modulation_hw,
            colors_hw,
            ray_params.float().contiguous(),
        )
        ctx.gauss_rect_is_pixels = gauss_rect_is_pixels
        ctx.ray_enabled = ray_enabled
        ctx.mark_non_differentiable(path_contribution, path_support)
        return colors_hw, path_contribution, path_support

    @staticmethod
    def backward(ctx, grad_colors_hw: torch.Tensor, _grad_path_contribution, _grad_path_support):
        (
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            gauss_attenuations,
            pose_f,
            grid_f,
            gauss_rects,
            csr0,
            csr1,
            csr2,
            ray_start,
            ray_end,
            ray_indices,
            ray_depths,
            meta,
            us_focus_params,
            modulation_hw,
            colors_hw,
            ray_params,
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

        gc = (grad_colors_hw.float() * modulation_hw.float()).reshape(-1).contiguous()

        (
            d_means,
            d_scales,
            d_quats,
            d_opacity,
            d_intensity,
            d_pose,
        ) = _C.ProjectedSliceBackwardCUDA(
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
            bool(ctx.needs_input_grad[6]),
        )

        if bool(getattr(ctx, "ray_enabled", False)) and bool(ctx.needs_input_grad[5]):
            local_colors_hw = colors_hw.float() / modulation_hw.float().clamp_min(1e-8)
            if int(ray_params[6].item()) == 1:
                d_attenuation, d_means_ray, d_scales_ray, d_quats_ray = _C.RayCsrAttenuationBackwardCUDA(
                    gauss_means.float().contiguous(),
                    gauss_scales.float().contiguous(),
                    gauss_quats.float().contiguous(),
                    gauss_attenuations.float().contiguous(),
                    pose_f,
                    grid_f,
                    ray_start,
                    ray_end,
                    ray_indices,
                    ray_depths,
                    local_colors_hw.contiguous(),
                    modulation_hw.float().contiguous(),
                    grad_colors_hw.float().contiguous(),
                    H,
                    W,
                    lateral_mm,
                    axial_mm,
                    axial_near_mm,
                    tile_size_px,
                    float(meta[6].item()),
                    float(meta[7].item()),
                    k_sigma,
                    float(ray_params[1].item()),
                    float(ray_params[2].item()),
                    float(ray_params[3].item()),
                    float(ray_params[4].item()),
                    float(ray_params[5].item()),
                    float(ray_params[7].item()),
                    int(ray_params[8].item()),
                    int(ray_params[9].item()),
                )
                d_means = d_means + d_means_ray.to(d_means.dtype)
                d_scales = d_scales + d_scales_ray.to(d_scales.dtype)
                d_quats = d_quats + d_quats_ray.to(d_quats.dtype)
            else:
                d_attenuation = _C.RayTileAttenuationBackwardFromTileCsrCUDA(
                    gauss_means.float().contiguous(),
                    gauss_scales.float().contiguous(),
                    gauss_quats.float().contiguous(),
                    gauss_attenuations.float().contiguous(),
                    pose_f,
                    grid_f,
                    csr0,
                    csr1,
                    csr2,
                    local_colors_hw.contiguous(),
                    modulation_hw.float().contiguous(),
                    grad_colors_hw.float().contiguous(),
                    H,
                    W,
                    axial_mm,
                    axial_near_mm,
                    tile_size_px,
                    k_sigma,
                    float(ray_params[1].item()),
                    float(ray_params[2].item()),
                    float(ray_params[3].item()),
                    float(ray_params[4].item()),
                    float(ray_params[5].item()),
                )
        else:
            d_attenuation = torch.zeros_like(gauss_attenuations)

        d_meta = torch.zeros_like(meta)
        d_us_focus_params = torch.zeros_like(us_focus_params)
        d_ray_params = torch.zeros((_RAY_PARAMS_LEN,), dtype=torch.float32, device=gauss_means.device)
        d_grid = torch.zeros_like(grid_f)
        d_pose = d_pose.to(getattr(ctx, "pose_dtype", torch.float32))
        d_means = d_means.to(gauss_means.dtype)
        d_scales = d_scales.to(gauss_scales.dtype)
        d_quats = d_quats.to(gauss_quats.dtype)
        d_opacity = d_opacity.to(gauss_opacities.dtype)
        d_intensity = d_intensity.to(gauss_intensities.dtype)
        d_attenuation = d_attenuation.to(gauss_attenuations.dtype)
        return (
            d_means,
            d_scales,
            d_quats,
            d_opacity,
            d_intensity,
            d_attenuation,
            d_pose,
            d_grid,
            d_meta,
            d_us_focus_params,
            d_ray_params,
            None,
        )


def _render_projected_slice_cuda(
    config: OursRenderConfig,
    gauss_means: torch.Tensor,
    gauss_scales: torch.Tensor,
    gauss_quats: torch.Tensor,
    gauss_opacities: torch.Tensor,
    gauss_intensities: torch.Tensor,
    gauss_attenuations: torch.Tensor,
    pose_probe_to_world: torch.Tensor,
    slice_grid_geometry: torch.Tensor,
    collect_path_stats: bool = False,
):
    device = gauss_means.device
    meta = _meta_vector(config, device)
    us_focus_params = _us_focus_params_vector(config, device)
    ray_params = _ray_params_vector(config, device)
    grid = slice_grid_geometry.to(device=device, dtype=torch.float32)

    colors, path_contribution, path_support = _ProjectedSliceCudaFunction.apply(
        gauss_means,
        gauss_scales,
        gauss_quats,
        gauss_opacities,
        gauss_intensities,
        gauss_attenuations,
        pose_probe_to_world,
        grid,
        meta,
        us_focus_params,
        ray_params,
        bool(collect_path_stats),
    )
    if collect_path_stats:
        return colors, path_contribution, path_support
    return colors


def render_projected_slice(
    config: OursRenderConfig,
    gauss_means: torch.Tensor,
    gauss_scales: torch.Tensor,
    gauss_quats: torch.Tensor,
    gauss_opacities: torch.Tensor,
    gauss_intensities: torch.Tensor,
    pose_probe_to_world: torch.Tensor,
    slice_grid_geometry: torch.Tensor,
    gauss_attenuations: torch.Tensor = None,
    collect_path_stats: bool = False,
):
    if (
        _C is not None
        and gauss_means.is_cuda
        and hasattr(_C, "ProjectedSliceForwardCUDA")
            and hasattr(_C, "ProjectedSliceBackwardCUDA")
            and (
                not config.ray_attenuation_enabled
                or (
                    hasattr(_C, "ProjectedSliceForwardRayTileCUDA")
                    and hasattr(_C, "RayTileAttenuationBackwardFromTileCsrCUDA")
                    and hasattr(_C, "RayCsrAttenuationBackwardCUDA")
                )
            )
    ):
        if gauss_attenuations is None:
            gauss_attenuations = gauss_opacities.detach()
        return _render_projected_slice_cuda(
            config,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            gauss_attenuations,
            pose_probe_to_world,
            slice_grid_geometry,
            collect_path_stats=collect_path_stats,
        )

    raise RuntimeError(
        "render_projected_slice requires CUDA + built ours_renderer._C. "
        "CPU reference for checks: tests/ops/ours_renderer/validate_same_effect_cpu_gpu.py"
    )
