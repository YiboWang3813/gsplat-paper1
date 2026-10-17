from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .types import OursRenderConfig

US_FOCUS_PARAMS_LEN = 15
US_FOCUS_SIGMA_ELE_0 = 0
US_FOCUS_SIGMA_ELE_1 = 1
US_FOCUS_SIGMA_ELE_MIN = 2
US_FOCUS_SIGMA_ELE_MAX = 3
US_FOCUS_SIGMA_LAT_0 = 4
US_FOCUS_SIGMA_LAT_1 = 5
US_FOCUS_SIGMA_LAT_MIN = 6
US_FOCUS_SIGMA_LAT_MAX = 7
US_FOCUS_DET_GAMMA = 8
US_FOCUS_DET_EPS = 9
US_FOCUS_WEIGHT_MIN = 10
US_FOCUS_USE_LATERAL_FOCUS = 11
US_FOCUS_INFLATE_LAT_MM = 12
US_FOCUS_INFLATE_AX_MM = 13
US_FOCUS_BYPASS = 14

_PROBE_COV_JITTER = 1e-6


def us_focus_params_from_config(
    config: OursRenderConfig,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    params = torch.zeros((US_FOCUS_PARAMS_LEN,), device=device, dtype=dtype)
    params[US_FOCUS_SIGMA_ELE_0] = float(config.us_focus_sigma_ele_0)
    params[US_FOCUS_SIGMA_ELE_1] = 0.0 if config.local_conditioning_mode == "fixed" else float(config.us_focus_sigma_ele_1)
    params[US_FOCUS_SIGMA_ELE_MIN] = float(config.us_focus_sigma_ele_min)
    params[US_FOCUS_SIGMA_ELE_MAX] = float(config.us_focus_sigma_ele_max)
    params[US_FOCUS_SIGMA_LAT_0] = float(config.us_focus_sigma_lat_0)
    params[US_FOCUS_SIGMA_LAT_1] = float(config.us_focus_sigma_lat_1)
    params[US_FOCUS_SIGMA_LAT_MIN] = float(config.us_focus_sigma_lat_min)
    params[US_FOCUS_SIGMA_LAT_MAX] = float(config.us_focus_sigma_lat_max)
    params[US_FOCUS_DET_GAMMA] = float(config.us_focus_det_gamma)
    params[US_FOCUS_DET_EPS] = float(config.us_focus_det_eps)
    params[US_FOCUS_WEIGHT_MIN] = float(config.us_focus_weight_min)
    params[US_FOCUS_USE_LATERAL_FOCUS] = 1.0 if config.us_focus_use_lateral_focus else 0.0
    params[US_FOCUS_INFLATE_LAT_MM] = float(config.us_focus_inflate_lat_mm)
    params[US_FOCUS_INFLATE_AX_MM] = float(config.us_focus_inflate_ax_mm)
    params[US_FOCUS_BYPASS] = float(config.local_conditioning_mode == "none")
    return params


def _quat_to_rotmat_wxyz(quats: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quats, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _pose_axes_probe_rows(pose_probe_to_world: torch.Tensor) -> torch.Tensor:
    rot = pose_probe_to_world[:3, :3]
    axis_lat = F.normalize(rot[:, 0], dim=0, eps=1e-8)
    axis_ax = F.normalize(rot[:, 1], dim=0, eps=1e-8)
    axis_ele = F.normalize(rot[:, 2], dim=0, eps=1e-8)
    return torch.stack([axis_lat, axis_ax, axis_ele], dim=0)


def _pixel_grid_lat_ax(config: OursRenderConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    xs = (torch.arange(config.W, device=device, dtype=dtype) - (config.W - 1) * 0.5) * float(
        config.lateral_spacing_mm_per_pix
    )
    ys = torch.arange(config.H, device=device, dtype=dtype) * float(config.axial_spacing_mm_per_pix) + float(
        config.axial_near_mm
    )
    ax, lat = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([lat, ax], dim=-1).reshape(-1, 2)


def us_focus_gaussian_footprint_reference(
    config: OursRenderConfig,
    gauss_means: torch.Tensor,
    gauss_scales: torch.Tensor,
    gauss_quats: torch.Tensor,
    pose_probe_to_world: torch.Tensor,
    us_focus_params: Optional[torch.Tensor] = None,
) -> dict:
    """Return per-Gaussian us_focus footprint terms in probe-local mm.

    This mirrors the intended CUDA device path but stays fully differentiable in PyTorch.
    It is intentionally small-batch oriented and used for correctness tests, not training.
    """
    dtype = gauss_means.dtype
    device = gauss_means.device
    params = (
        us_focus_params_from_config(config, device=device, dtype=dtype)
        if us_focus_params is None
        else us_focus_params.to(device=device, dtype=dtype)
    )

    axes = _pose_axes_probe_rows(pose_probe_to_world.to(device=device, dtype=dtype))
    origin = pose_probe_to_world[:3, 3].to(device=device, dtype=dtype)
    mu_c = (gauss_means - origin.unsqueeze(0)) @ axes.t()

    rot_g = _quat_to_rotmat_wxyz(gauss_quats)
    s2 = torch.diag_embed(gauss_scales.clamp(min=1e-6).pow(2))
    sigma_w = rot_g @ s2 @ rot_g.transpose(-1, -2)
    sigma_c = axes.unsqueeze(0) @ sigma_w @ axes.t().unsqueeze(0)
    eye3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    sigma_c = sigma_c + eye3 * _PROBE_COV_JITTER

    lambda_c = torch.linalg.inv(sigma_c)
    depth = (mu_c[:, 1] - float(config.axial_near_mm)).clamp(min=0.0)

    sigma_ele = params[US_FOCUS_SIGMA_ELE_0] + params[US_FOCUS_SIGMA_ELE_1] * depth
    sigma_ele = sigma_ele.clamp(
        min=float(params[US_FOCUS_SIGMA_ELE_MIN].item()),
        max=float(params[US_FOCUS_SIGMA_ELE_MAX].item()),
    )
    sigma_lat = params[US_FOCUS_SIGMA_LAT_0] + params[US_FOCUS_SIGMA_LAT_1] * depth
    sigma_lat = sigma_lat.clamp(
        min=float(params[US_FOCUS_SIGMA_LAT_MIN].item()),
        max=float(params[US_FOCUS_SIGMA_LAT_MAX].item()),
    )

    lambda_f = torch.zeros_like(lambda_c)
    if bool(params[US_FOCUS_USE_LATERAL_FOCUS].item() != 0.0):
        lambda_f[:, 0, 0] = 1.0 / sigma_lat.pow(2)
    lambda_f[:, 2, 2] = 1.0 / sigma_ele.pow(2)

    lambda_e = lambda_c + lambda_f
    sigma_e = torch.linalg.inv(lambda_e)
    mu_e = torch.bmm(sigma_e, torch.bmm(lambda_c, mu_c.unsqueeze(-1))).squeeze(-1)
    bypass = params.numel() > US_FOCUS_BYPASS and bool(params[US_FOCUS_BYPASS].item())
    if bypass:
        lambda_e, sigma_e, mu_e = lambda_c, sigma_c, mu_c

    sigma2 = sigma_e[:, :2, :2].clone()
    sigma2[:, 0, 0] = sigma2[:, 0, 0] + params[US_FOCUS_INFLATE_LAT_MM].clamp(min=0.0).pow(2)
    sigma2[:, 1, 1] = sigma2[:, 1, 1] + params[US_FOCUS_INFLATE_AX_MM].clamp(min=0.0).pow(2)
    mu2 = mu_e[:, :2]
    det2 = sigma2[:, 0, 0] * sigma2[:, 1, 1] - sigma2[:, 0, 1] * sigma2[:, 1, 0]
    inv2 = torch.linalg.inv(sigma2)

    det_eps = params[US_FOCUS_DET_EPS]
    det_gamma = params[US_FOCUS_DET_GAMMA]
    slice_denom = sigma_c[:, 2, 2] + sigma_ele.pow(2) + det_eps
    slice_response = torch.exp(-0.5 * mu_c[:, 2].pow(2) / slice_denom)
    if bypass:
        slice_response = torch.ones_like(slice_response)
    norm_factor = (det2 + det_eps).pow(-det_gamma)

    return {
        "mu_c": mu_c,
        "sigma_c": sigma_c,
        "lambda_c": lambda_c,
        "lambda_e": lambda_e,
        "sigma_e": sigma_e,
        "mu2": mu2,
        "sigma2": sigma2,
        "inv2": inv2,
        "det2": det2,
        "sigma_ele": sigma_ele,
        "sigma_lat": sigma_lat,
        "slice_response": slice_response,
        "norm_factor": norm_factor,
    }


def render_us_focus_reference(
    config: OursRenderConfig,
    gauss_means: torch.Tensor,
    gauss_scales: torch.Tensor,
    gauss_quats: torch.Tensor,
    gauss_opacities: torch.Tensor,
    gauss_intensities: torch.Tensor,
    pose_probe_to_world: torch.Tensor,
    us_focus_params: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if config.shader_weight_mode != "us_focus":
        raise ValueError("render_us_focus_reference expects shader_weight_mode='us_focus'.")
    if config.compositing_mode != "additive":
        raise ValueError("render_us_focus_reference implements us_focus additive rendering only.")

    dtype = gauss_means.dtype
    device = gauss_means.device
    params = (
        us_focus_params_from_config(config, device=device, dtype=dtype)
        if us_focus_params is None
        else us_focus_params.to(device=device, dtype=dtype)
    )
    terms = us_focus_gaussian_footprint_reference(
        config,
        gauss_means,
        gauss_scales,
        gauss_quats,
        pose_probe_to_world,
        params,
    )
    pixels = _pixel_grid_lat_ax(config, device, dtype)

    delta = pixels.unsqueeze(0) - terms["mu2"].unsqueeze(1)
    qd = torch.bmm(delta, terms["inv2"])
    quad = (qd * delta).sum(dim=-1)
    phi = torch.exp(-0.5 * quad)

    det_valid = terms["det2"] > params[US_FOCUS_DET_EPS]
    base_weight = (
        gauss_opacities.reshape(-1).to(device=device, dtype=dtype)
        * terms["slice_response"]
        * terms["norm_factor"]
    )
    weights = base_weight.unsqueeze(1) * phi
    finite = torch.isfinite(weights) & torch.isfinite(quad) & (quad >= 0.0)
    weights = torch.where(
        finite & det_valid.unsqueeze(1) & (weights >= params[US_FOCUS_WEIGHT_MIN]),
        weights,
        torch.zeros_like(weights),
    )

    intensities = gauss_intensities.reshape(-1).to(device=device, dtype=dtype)
    flat = (weights * intensities.unsqueeze(1)).sum(dim=0) + float(config.fixed_bg_color)
    return flat.reshape(config.H, config.W)
