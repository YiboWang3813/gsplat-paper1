import math
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F

from . import _C
from .types import UltragrayRenderConfig


_PROFILE_STEP = 0


def _profile_every() -> int:
    try:
        return int(os.environ.get("ULTRAGRAY_PROFILE_EVERY", "0"))
    except ValueError:
        return 0


def _profile_warmup() -> int:
    try:
        return int(os.environ.get("ULTRAGRAY_PROFILE_WARMUP", "0"))
    except ValueError:
        return 0


def _should_profile(step: int) -> bool:
    every = _profile_every()
    if every <= 0 or step <= _profile_warmup():
        return False
    return step % every == 0


def _timed_cuda_call(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    gpu_ms = start.elapsed_time(end)
    return result, wall_ms, gpu_ms


def _ray_stats(ray_counts: torch.Tensor, ray_indices: torch.Tensor) -> str:
    counts = ray_counts.to(dtype=torch.float32)
    active = int((ray_counts > 0).sum().item())
    total_hits = int(ray_indices.numel())
    mean_all = float(counts.mean().item()) if counts.numel() else 0.0
    max_count = int(ray_counts.max().item()) if counts.numel() else 0
    if active:
        active_counts = counts[ray_counts > 0]
        mean_active = float(active_counts.mean().item())
        p50 = float(torch.quantile(active_counts, 0.50).item())
        p90 = float(torch.quantile(active_counts, 0.90).item())
        p99 = float(torch.quantile(active_counts, 0.99).item())
    else:
        mean_active = p50 = p90 = p99 = 0.0
    return (
        f"total_hits={total_hits} active_rays={active}/{ray_counts.numel()} "
        f"hits_mean_all={mean_all:.1f} hits_mean_active={mean_active:.1f} "
        f"hits_p50={p50:.1f} hits_p90={p90:.1f} hits_p99={p99:.1f} "
        f"hits_max={max_count}"
    )


def ensure_tensor(t: Optional[torch.Tensor], dtype: torch.dtype = torch.float32):
    if t is None:
        return None
    if not t.is_cuda:
        t = t.cuda()
    if t.dtype != dtype:
        t = t.to(dtype=dtype)
    return t.contiguous()


def pack_vec3(t: torch.Tensor):
    if t.shape[-1] == 3:
        t = F.pad(t, (0, 1), mode="constant", value=0.0)
    return t.contiguous()


class _UltragrayRenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        config: UltragrayRenderConfig,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        attenuations: torch.Tensor,
        sh_coeffs: torch.Tensor,
        pixel_positions: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
        probe_origin: torch.Tensor,
        axis_lateral: torch.Tensor,
        axis_elevational: torch.Tensor,
        axis_axial: torch.Tensor,
    ):
        global _PROFILE_STEP
        _PROFILE_STEP += 1
        profile_step = _PROFILE_STEP
        profile_enabled = _should_profile(profile_step)

        means = ensure_tensor(means)
        scales = ensure_tensor(scales)
        quats = ensure_tensor(quats)
        attenuations = ensure_tensor(attenuations)
        sh_coeffs = ensure_tensor(sh_coeffs)
        pixel_positions = ensure_tensor(pixel_positions)
        ray_origins = ensure_tensor(ray_origins)
        ray_dirs = ensure_tensor(ray_dirs)
        probe_origin = ensure_tensor(probe_origin)
        axis_lateral = ensure_tensor(axis_lateral)
        axis_elevational = ensure_tensor(axis_elevational)
        axis_axial = ensure_tensor(axis_axial)

        means_packed = pack_vec3(means)
        scales_packed = pack_vec3(scales)

        num_rays = ray_origins.shape[0]
        u = torch.rand(num_rays, device=means.device, dtype=torch.float32)
        perturbations_norm = torch.asin(2.0 * u - 1.0) / (math.pi / 2.0)
        ray_perturbations = (perturbations_norm * config.ops_max_offset_mm).contiguous()

        def run_mapper():
            return _C.MapperForwardCUDA(
                means_packed,
                scales_packed,
                probe_origin,
                axis_lateral,
                axis_elevational,
                axis_axial,
                int(config.image_width),
                float(config.width_mm),
                float(config.near_mm),
                float(config.lateral_col_offset),
                float(config.elev_cutoff_margin_mm),
                float(config.k_sigma),
            )

        if profile_enabled:
            (ray_counts, ray_offsets, ray_indices, sorted_depths), mapper_wall_ms, mapper_gpu_ms = _timed_cuda_call(run_mapper)
        else:
            ray_counts, ray_offsets, ray_indices, sorted_depths = run_mapper()
            mapper_wall_ms = mapper_gpu_ms = None

        def run_shader_forward():
            return _C.ShaderForwardCUDA(
                means_packed,
                scales_packed,
                quats,
                attenuations,
                sh_coeffs,
                ray_counts,
                ray_offsets,
                ray_indices,
                sorted_depths,
                pixel_positions,
                ray_origins,
                ray_dirs,
                axis_elevational,
                ray_perturbations,
                float(config.k_sigma),
                float(config.quadrature_window_radius_mm),
                bool(config.local_transmission_window_only),
                float(config.local_transmission_max_window_mm),
            )

        if profile_enabled:
            image, shader_fwd_wall_ms, shader_fwd_gpu_ms = _timed_cuda_call(run_shader_forward)
            print(
                "[ULTRAGRAY_PROFILE] "
                f"step={profile_step} phase=forward "
                f"mapper_wall_ms={mapper_wall_ms:.3f} mapper_gpu_ms={mapper_gpu_ms:.3f} "
                f"shader_fwd_wall_ms={shader_fwd_wall_ms:.3f} shader_fwd_gpu_ms={shader_fwd_gpu_ms:.3f} "
                f"{_ray_stats(ray_counts, ray_indices)}",
                flush=True,
            )
        else:
            image = run_shader_forward()

        ctx.quadrature_window_radius_mm = float(config.quadrature_window_radius_mm)
        ctx.k_sigma = float(config.k_sigma)
        ctx.local_transmission_window_only = bool(config.local_transmission_window_only)
        ctx.local_transmission_max_window_mm = float(config.local_transmission_max_window_mm)
        ctx.profile_enabled = profile_enabled
        ctx.profile_step = profile_step
        ctx.save_for_backward(
            means_packed,
            scales_packed,
            quats,
            attenuations,
            sh_coeffs,
            ray_counts,
            ray_offsets,
            ray_indices,
            sorted_depths,
            pixel_positions,
            ray_origins,
            ray_dirs,
            axis_elevational,
            ray_perturbations,
        )
        return image

    @staticmethod
    def backward(ctx, dL_dImage):
        (
            means_packed,
            scales_packed,
            quats,
            attenuations,
            sh_coeffs,
            ray_counts,
            ray_offsets,
            ray_indices,
            sorted_depths,
            pixel_positions,
            ray_origins,
            ray_dirs,
            axis_elevational,
            ray_perturbations,
        ) = ctx.saved_tensors

        dL_dImage = dL_dImage.contiguous()

        def run_shader_backward():
            return _C.ShaderBackwardCUDA(
                dL_dImage,
                means_packed,
                scales_packed,
                quats,
                attenuations,
                sh_coeffs,
                ray_counts,
                ray_offsets,
                ray_indices,
                sorted_depths,
                pixel_positions,
                ray_origins,
                ray_dirs,
                axis_elevational,
                ray_perturbations,
                ctx.k_sigma,
                ctx.quadrature_window_radius_mm,
                ctx.local_transmission_window_only,
                ctx.local_transmission_max_window_mm,
            )

        if ctx.profile_enabled:
            backward_result, shader_bwd_wall_ms, shader_bwd_gpu_ms = _timed_cuda_call(run_shader_backward)
            print(
                "[ULTRAGRAY_PROFILE] "
                f"step={ctx.profile_step} phase=backward "
                f"shader_bwd_wall_ms={shader_bwd_wall_ms:.3f} shader_bwd_gpu_ms={shader_bwd_gpu_ms:.3f}",
                flush=True,
            )
            dL_dMeans_packed, dL_dScales_packed, dL_dQuats, dL_dAttens, dL_dSH = backward_result
        else:
            dL_dMeans_packed, dL_dScales_packed, dL_dQuats, dL_dAttens, dL_dSH = run_shader_backward()

        dL_dMeans = dL_dMeans_packed[:, :3]
        dL_dScales = dL_dScales_packed[:, :3]

        return (
            None,
            dL_dMeans,
            dL_dScales,
            dL_dQuats,
            dL_dAttens,
            dL_dSH,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
