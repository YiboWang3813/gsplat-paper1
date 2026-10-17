from typing import Tuple

import torch
import torch.nn.functional as F


def quat_to_matrix_wxyz(quats: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quats, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rot = torch.empty((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
    rot[..., 0, 0] = ww + xx - yy - zz
    rot[..., 0, 1] = 2.0 * (xy - wz)
    rot[..., 0, 2] = 2.0 * (xz + wy)
    rot[..., 1, 0] = 2.0 * (xy + wz)
    rot[..., 1, 1] = ww - xx + yy - zz
    rot[..., 1, 2] = 2.0 * (yz - wx)
    rot[..., 2, 0] = 2.0 * (xz - wy)
    rot[..., 2, 1] = 2.0 * (yz + wx)
    rot[..., 2, 2] = ww - xx - yy + zz
    return rot


def build_covariance(scales: torch.Tensor, quats: torch.Tensor, cov_eps: float = 1.0e-6) -> torch.Tensor:
    scales = scales.clamp_min(float(cov_eps))
    rot = quat_to_matrix_wxyz(quats)
    cov = rot @ torch.diag_embed(scales * scales) @ rot.transpose(-1, -2)
    eye = torch.eye(3, device=scales.device, dtype=scales.dtype).expand(cov.shape[0], 3, 3)
    return cov + eye * float(cov_eps)


def pose_probe_to_world_to_world_to_camera(pose_probe_to_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rot_pw = pose_probe_to_world[:3, :3]
    trans_pw = pose_probe_to_world[:3, 3]
    rot_wc = rot_pw.transpose(0, 1)
    trans_wc = -(rot_wc @ trans_pw)
    return rot_wc, trans_wc


def world_to_camera(
    means: torch.Tensor,
    covariances: torch.Tensor,
    pose_probe_to_world: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rot_wc, trans_wc = pose_probe_to_world_to_world_to_camera(pose_probe_to_world)
    means_c = means @ rot_wc.transpose(0, 1) + trans_wc
    covs_c = rot_wc.unsqueeze(0) @ covariances @ rot_wc.transpose(0, 1).unsqueeze(0)
    return means_c, covs_c


def focus_project(
    means_c: torch.Tensor,
    covs_c: torch.Tensor,
    opacities: torch.Tensor,
    sigma_z: float,
    det_eps: float = 1.0e-6,
    cov_eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = means_c.shape[0]
    eye = torch.eye(3, device=means_c.device, dtype=means_c.dtype).expand(n, 3, 3)
    covs_c = covs_c + eye * float(cov_eps)
    covs_c_inv = torch.linalg.inv(covs_c)
    axial_precision = torch.zeros_like(covs_c_inv)
    axial_precision[:, 2, 2] = 1.0 / (float(sigma_z) * float(sigma_z))
    covs_e_inv = covs_c_inv + axial_precision
    covs_e = torch.linalg.inv(covs_e_inv)
    means_e = torch.bmm(covs_e, torch.bmm(covs_c_inv, means_c.unsqueeze(-1))).squeeze(-1)

    q_c = torch.sum(means_c.unsqueeze(1) * torch.bmm(covs_c_inv, means_c.unsqueeze(-1)).transpose(1, 2), dim=(1, 2))
    q_e = torch.sum(means_e.unsqueeze(1) * torch.bmm(covs_e_inv, means_e.unsqueeze(-1)).transpose(1, 2), dim=(1, 2))
    opacity_r = torch.exp(-0.5 * (q_c - q_e))

    means_2d = means_e[:, :2]
    covs_2d = covs_e[:, :2, :2]
    det = (
        covs_2d[:, 0, 0] * covs_2d[:, 1, 1]
        - covs_2d[:, 0, 1] * covs_2d[:, 1, 0]
    ).clamp_min(float(det_eps))
    alpha_tilde = opacities.reshape(-1) * opacity_r / torch.sqrt(det)
    return means_2d, covs_2d, alpha_tilde


def render_reference(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    pose_probe_to_world: torch.Tensor,
    xy_grid: torch.Tensor,
    sigma_z: float,
    det_eps: float = 1.0e-6,
    cov_eps: float = 1.0e-6,
) -> torch.Tensor:
    covs = build_covariance(scales, quats, cov_eps=cov_eps)
    means_c, covs_c = world_to_camera(means, covs, pose_probe_to_world)
    means_2d, covs_2d, alpha_tilde = focus_project(
        means_c,
        covs_c,
        opacities,
        sigma_z=sigma_z,
        det_eps=det_eps,
        cov_eps=cov_eps,
    )
    covs_2d = covs_2d + torch.eye(2, device=means.device, dtype=means.dtype).expand(covs_2d.shape[0], 2, 2) * cov_eps
    inv_2d = torch.linalg.inv(covs_2d)
    flat = xy_grid.reshape(-1, 2)
    out = torch.zeros((flat.shape[0],), device=means.device, dtype=means.dtype)
    for i in range(means.shape[0]):
        d = flat - means_2d[i]
        q = torch.sum(d * (d @ inv_2d[i].transpose(0, 1)), dim=-1)
        out = out + alpha_tilde[i] * torch.exp(-0.5 * q)
    return out.reshape(xy_grid.shape[:-1])

