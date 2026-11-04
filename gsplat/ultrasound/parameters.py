"""Activation views and optimizer surgery over native Gaussian ParameterDicts."""

import torch
import torch.nn.functional as F

from gsplat.strategy.ops import _update_param_with_optimizer


class GaussianParameters:
    def __init__(self, params, config, gradient=None, step=0):
        self.gaussians, self.model_cfg, self.mean_gradient, self.step = (
            params,
            config,
            gradient,
            step,
        )

    @property
    def device(self):
        return self.gaussians["means"].device

    @property
    def num_points(self):
        return len(self.gaussians["means"])

    def activated_param(self, name, indices=None):
        x = self.gaussians[name]
        if indices is not None:
            x = x[indices]
        constraints = getattr(self.model_cfg.gauss_container, "constraints", None)
        if name == "quats":
            return F.normalize(x, dim=-1, eps=1e-8)
        if name in ("opacities", "intensities", "attenuations"):
            maximum = (
                getattr(constraints, "opacity_max", 1.0) if name == "opacities" else 1.0
            )
            return x.sigmoid() * (maximum if maximum > 0 else 1.0)
        if name in ("scales", "scales_2d", "scales_3d"):
            if getattr(constraints, "scale_activation", "exp") == "bounded_sigmoid":
                return (
                    constraints.scale_min_mm
                    + (constraints.scale_max_mm - constraints.scale_min_mm)
                    * x.sigmoid()
                )
            x = x.exp()
            if name == "scales_3d":
                low = getattr(constraints, "min_scale_3d", 0.0)
                high = getattr(constraints, "max_scale_3d", 0.0)
                if low > 0:
                    x = x.clamp_min(low)
                if high > 0:
                    x = x.clamp_max(high)
            return x
        return x

    def inverse(self, name, value):
        c = getattr(self.model_cfg.gauss_container, "constraints", None)
        if name in ("opacities", "intensities", "attenuations"):
            maximum = getattr(c, "opacity_max", 1.0) if name == "opacities" else 1.0
            return torch.logit(
                (value / (maximum if maximum > 0 else 1.0)).clamp(
                    torch.finfo(value.dtype).eps, 1 - torch.finfo(value.dtype).eps
                )
            )
        if name in ("scales", "scales_2d", "scales_3d"):
            if getattr(c, "scale_activation", "exp") == "bounded_sigmoid":
                return torch.logit(
                    (
                        (value - c.scale_min_mm) / (c.scale_max_mm - c.scale_min_mm)
                    ).clamp(
                        torch.finfo(value.dtype).eps, 1 - torch.finfo(value.dtype).eps
                    )
                )
            return value.clamp_min(1e-8).log()
        return value

    def __getattr__(self, name):
        aliases = {
            "alphas": "opacities",
            "precision_tril": "precision_tril",
            "fov_x_offsets": "fov_x",
            "fov_y_offsets": "fov_y",
        }
        if name == "active_sh_degree":
            return min(
                self.model_cfg.render.sh_degree,
                self.model_cfg.render.active_sh_degree_start
                + self.step // max(1, self.model_cfg.render.sh_degree_step_iter),
            )
        if name == "sh_coeffs":
            return torch.cat(
                [self.gaussians["sh_dc"], self.gaussians["sh_rest"]], dim=1
            )
        return self.activated_param(aliases.get(name, name))

    def dc_colors_for(self, indices):
        return (self.gaussians["sh_dc"][indices, 0] * 0.28209479177387814 + 0.5).clamp(
            0, 1
        )


class ParameterOperations:
    def __init__(self, params, optimizers):
        self.params, self.optimizers = params, optimizers

    def concat(self, children):
        n = len(children["means"])
        _update_param_with_optimizer(
            lambda name, x: torch.nn.Parameter(torch.cat([x, children[name]])),
            lambda name, x: torch.cat(
                [x, torch.zeros((n,) + x.shape[1:], device=x.device, dtype=x.dtype)]
            ),
            self.params,
            self.optimizers,
        )

    def prune(self, mask):
        _update_param_with_optimizer(
            lambda name, x: torch.nn.Parameter(x[~mask]),
            lambda name, x: x[~mask],
            self.params,
            self.optimizers,
        )

    def replace(self, name, value):
        _update_param_with_optimizer(
            lambda key, x: torch.nn.Parameter(value),
            lambda key, x: torch.zeros_like(x),
            self.params,
            self.optimizers,
            names=[name],
        )
