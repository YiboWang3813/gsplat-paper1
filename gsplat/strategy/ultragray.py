from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from gsplat.strategy.baseline import NativeBaselineStrategy
from gsplat.ultrasound.parameters import GaussianParameters, ParameterOperations


class UltraGrayStrategy(NativeBaselineStrategy):
    @property
    def model_cfg(self) -> Any:
        return self.config

    def __init__(self, cfg: Any):
        self.config = cfg
        self.ctrl_cfg = self.model_cfg.offline.density_control
        self.xyz_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.last_event_stats: Dict[str, float] = {}

    def scalar_stats(self, iteration: int) -> Dict[str, float]:
        if int(self.last_event_stats.get("iteration", -1)) != int(iteration):
            return {}
        return {
            f"train/dc/{k}": v
            for k, v in self.last_event_stats.items()
            if k != "iteration"
        }

    def should_accumulate_grad_stats(self, iteration: int) -> bool:
        t = self.ctrl_cfg.timing
        return bool(self.ctrl_cfg.enable.densify) and int(
            t.start_iter
        ) <= iteration <= int(t.end_iter)

    def should_densify_prune(self, iteration: int) -> bool:
        if not (bool(self.ctrl_cfg.enable.densify) or bool(self.ctrl_cfg.enable.prune)):
            return False
        t = self.ctrl_cfg.timing
        if iteration < int(t.start_iter) or iteration > int(t.end_iter):
            return False
        return iteration % int(t.freq) == 0

    def update_grad_stats(self, container: GaussianParameters):
        if container.mean_gradient is None:
            return
        n = int(container.num_points)
        if self.xyz_grad_accum.shape[0] != n:
            self.xyz_grad_accum = torch.zeros((n, 1), device=container.device)
            self.denom = torch.zeros((n, 1), device=container.device)
        grad_norm = torch.norm(container.mean_gradient, dim=-1, keepdim=True)
        self.xyz_grad_accum += grad_norm
        self.denom += 1

    @torch.no_grad()
    def prune_and_densify(
        self, iteration: int, solver: ParameterOperations, container: GaussianParameters
    ):
        if container.num_points == 0:
            return
        th = self.ctrl_cfg.thresholds

        n_before = int(container.num_points)
        if self.xyz_grad_accum.shape[0] == n_before:
            grads = self.xyz_grad_accum / self.denom.clamp(min=1.0)
            grads[grads.isnan()] = 0.0
            grads = grads.squeeze(-1)
        else:
            grads = torch.zeros((n_before,), device=container.device)
        max_grad = float(grads.max().item()) if grads.numel() > 0 else 0.0
        mean_grad = float(grads.mean().item()) if grads.numel() > 0 else 0.0

        max_scale = torch.max(container.scales, dim=1).values
        high_grad = (
            (grads > float(th.densify_grad))
            if bool(self.ctrl_cfg.enable.densify)
            else torch.zeros_like(grads, dtype=torch.bool)
        )
        split_mask = high_grad & (max_scale > float(th.split_scale_mm))
        clone_mask = high_grad & (~split_mask)

        new_tensors, n_clone, n_split = self._generate_new_points(
            container, clone_mask, split_mask
        )
        if n_clone + n_split > 0:
            solver.concat(new_tensors)

        # Paper prune rule: scale out-of-range.
        n_after = int(container.num_points)
        prune_mask = torch.zeros((n_after,), device=container.device, dtype=torch.bool)
        if bool(self.ctrl_cfg.enable.prune):
            scales_new = container.scales
            min_scale_new = torch.min(scales_new, dim=1).values
            max_scale_new = torch.max(scales_new, dim=1).values
            prune_mask = (min_scale_new < float(th.min_scale_mm)) | (
                max_scale_new > float(th.max_scale_mm)
            )

        # Optional legacy hard cap. Production comparisons disable this and rely
        # on method thresholds for topology changes.
        n_max_cfg = getattr(self.ctrl_cfg, "n_max", None)
        try:
            n_max = int(n_max_cfg)
        except (TypeError, ValueError):
            n_max = 0
        if bool(self.ctrl_cfg.enable.prune) and n_max > 0 and n_after > n_max:
            overflow = n_after - n_max
            g = torch.zeros((n_after,), device=container.device)
            n_copy = min(int(grads.numel()), n_after)
            g[:n_copy] = grads[:n_copy]
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            _, idx_small = torch.topk(g, k=overflow, largest=False)
            prune_mask[idx_small] = True
        else:
            overflow = 0

        n_prune = int(prune_mask.sum().item())
        if prune_mask.any():
            solver.prune(prune_mask)

        n_final = int(container.num_points)
        self.last_event_stats = {
            "iteration": float(iteration),
            "num_gaussians_before": float(n_before),
            "num_gaussians_after": float(n_final),
            "num_gaussians_delta": float(n_final - n_before),
            "max_grad": float(max_grad),
            "mean_grad": float(mean_grad),
            "densify_grad_threshold": float(th.densify_grad),
            "clone": float(n_clone),
            "split": float(n_split),
            "prune": float(n_prune),
            "prune_nmax": float(overflow),
        }
        self.xyz_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)

    def _generate_new_points(
        self,
        container: GaussianParameters,
        clone_mask: torch.Tensor,
        split_mask: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], int, int]:
        n_clone = int(clone_mask.sum().item())
        n_split = int(split_mask.sum().item())
        if n_clone == 0 and n_split == 0:
            return {}, 0, 0

        param_dict = dict(container.gaussians.items())
        new_tensors_list = {name: [] for name in param_dict.keys()}

        if n_clone > 0:
            for name, tensor in param_dict.items():
                new_tensors_list[name].append(tensor[clone_mask])

        if n_split > 0:
            for name, tensor in param_dict.items():
                repeats = [2] + [1] * (tensor.dim() - 1)
                t_split = tensor[split_mask].repeat(*repeats)
                if name == "scales":
                    t_split -= torch.log(torch.tensor(2.0, device=container.device))
                new_tensors_list[name].append(t_split)

            split_scales_log = new_tensors_list["scales"][-1]
            split_scales = torch.exp(split_scales_log)
            split_means = new_tensors_list["means"][-1]
            split_means.add_(torch.randn_like(split_means) * split_scales * 0.1)

        final_tensors = {
            name: torch.cat(t_list, dim=0) for name, t_list in new_tensors_list.items()
        }
        return final_tensors, n_clone, n_split
