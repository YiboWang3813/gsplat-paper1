import torch
import torch.nn as nn
import torch.nn.functional as F

from .autograd import render_gaussianpile_projected_slice
from .check import should_validate_inputs, validate_render_inputs
from .types import GaussianPlieRenderConfig


class GaussianPlieRenderer(nn.Module):
    """GaussianPile renderer facade.

    This facade exposes the GaussianPile primitive as (mean, scale, quat, alpha).
    The CUDA extension still receives an internal all-ones intensity tensor for ABI compatibility.
    """

    def __init__(self, config: GaussianPlieRenderConfig):
        super().__init__()
        self.config = config
        self._slice_grid_geometry = None
        self._delegate = None

    def _ensure_delegate(self):
        if self._delegate is not None:
            return self._delegate
        if self.config.backend == "ultragauss_lowmem":
            from ultragauss_renderer import UltraGaussRenderConfig, UltraGaussRenderer

            ultra_cfg = UltraGaussRenderConfig(
                H=int(self.config.H),
                W=int(self.config.W),
                lateral_spacing_mm_per_pix=float(self.config.lateral_spacing_mm_per_pix),
                axial_spacing_mm_per_pix=float(self.config.axial_spacing_mm_per_pix),
                axial_near_mm=float(self.config.axial_near_mm),
                lateral_col_offset=float(self.config.lateral_col_offset),
                chi2_threshold=float(self.config.radius_sigma) ** 2,
                mapper_sort_by_depth=False,
                fixed_bg_alpha=1.0e-6,
                fixed_bg_intensity=0.0,
                diag_beta=float(self.config.cov_eps),
                enable_shadow=False,
                shadow_delta=0.0,
            )
            self._delegate = UltraGaussRenderer(ultra_cfg)
            return self._delegate

        return None

    @staticmethod
    def _quat_wxyz_to_matrix(quats: torch.Tensor) -> torch.Tensor:
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

    def _precision_tril_from_scales_quats(self, scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
        scales = scales.clamp_min(float(self.config.cov_eps))
        rot = self._quat_wxyz_to_matrix(quats)
        precision_diag = 1.0 / (scales * scales)
        precision = rot @ torch.diag_embed(precision_diag) @ rot.transpose(1, 2)
        chol = torch.linalg.cholesky(precision)
        beta = float(self.config.cov_eps)
        diag_raw = (torch.diagonal(chol, dim1=1, dim2=2) - beta).clamp_min(1e-8).sqrt()
        return torch.stack(
            [
                diag_raw[:, 0],
                chol[:, 1, 0],
                chol[:, 2, 0],
                diag_raw[:, 1],
                chol[:, 2, 1],
                diag_raw[:, 2],
            ],
            dim=1,
        )

    def forward(
        self,
        gauss_means: torch.Tensor,
        gauss_scales: torch.Tensor,
        gauss_quats: torch.Tensor,
        gauss_alphas: torch.Tensor,
        pose_probe_to_world: torch.Tensor,
        pixel_positions_world: torch.Tensor = None,
        origin_world: torch.Tensor = None,
        axis_lateral_world: torch.Tensor = None,
        axis_axial_world: torch.Tensor = None,
        axis_elevational_world: torch.Tensor = None,
    ) -> torch.Tensor:
        if should_validate_inputs(self.config):
            validate_render_inputs(
                self.config,
                gauss_means,
                gauss_scales,
                gauss_quats,
                gauss_alphas,
                pose_probe_to_world,
            )
        delegate = self._ensure_delegate()
        if self.config.backend == "ultragauss_lowmem":
            if (
                pixel_positions_world is None
                or origin_world is None
                or axis_lateral_world is None
                or axis_axial_world is None
                or axis_elevational_world is None
            ):
                raise RuntimeError("ultragauss_lowmem backend requires pixel positions and slice geometry axes.")
            from ultragauss_renderer import SliceGeometry

            precision_tril = self._precision_tril_from_scales_quats(gauss_scales, gauss_quats)
            pred, _alpha, _depth, _intermediates = delegate(
                gauss_means_world=gauss_means,
                gauss_precision_tril=precision_tril,
                gauss_opacities=gauss_alphas,
                gauss_intensities=torch.ones_like(gauss_alphas),
                pixel_positions_world=pixel_positions_world,
                geometry=SliceGeometry(
                    origin_world=origin_world,
                    axis_lateral_world=axis_lateral_world,
                    axis_axial_world=axis_axial_world,
                    axis_elevational_world=axis_elevational_world,
                ),
                return_intermediates=False,
            )
            return pred

        if self._slice_grid_geometry is None:
            raise RuntimeError("GaussianPlieRenderer requires slice_grid_geometry; call set_slice_grid_geometry().")
        return render_gaussianpile_projected_slice(
            config=self.config,
            gauss_means=gauss_means,
            gauss_scales=gauss_scales,
            gauss_quats=gauss_quats,
            gauss_alphas=gauss_alphas,
            pose_probe_to_world=pose_probe_to_world,
            slice_grid_geometry=self._slice_grid_geometry,
        )

    def set_slice_grid_geometry(self, slice_grid_geometry: torch.Tensor) -> None:
        self._slice_grid_geometry = slice_grid_geometry
        if self._delegate is not None and hasattr(self._delegate, "set_slice_grid_geometry"):
            self._delegate.set_slice_grid_geometry(slice_grid_geometry)
