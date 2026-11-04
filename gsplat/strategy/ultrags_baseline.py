from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from gsplat.strategy.baseline import NativeBaselineStrategy


class UltraGSStrategy(NativeBaselineStrategy):
    def __init__(self, cfg: Any):
        self.config = cfg
        self.ctrl_cfg = cfg.offline.density_control
        self.xyz_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.max_radii2d = torch.empty(0)
        self.scene_radius = 1.0
        self.last_event_stats: Dict[str, float] = {}

    def scalar_stats(self, iteration: int) -> Dict[str, float]:
        if int(self.last_event_stats.get("iteration", -1)) != int(iteration):
            return {}
        return {
            f"train/dc/{k}": v
            for k, v in self.last_event_stats.items()
            if k != "iteration"
        }

    def set_scene_radius(self, scene_radius: float):
        self.scene_radius = max(float(scene_radius), 1.0e-6)

    def should_accumulate_grad_stats(self, iteration: int) -> bool:
        return bool(self.ctrl_cfg.enable.densify) and self._in_window(iteration)

    def should_densify_prune(self, iteration: int) -> bool:
        if not (bool(self.ctrl_cfg.enable.densify) or bool(self.ctrl_cfg.enable.prune)):
            return False
        if not self._in_window(iteration):
            return False
        interval = self._densification_interval()
        return iteration % interval == 0

    def should_reset_opacity(self, iteration: int) -> bool:
        if not bool(self.ctrl_cfg.enable.opacity_reset):
            return False
        if not self._in_window(iteration):
            return False
        interval = max(1, int(self.ctrl_cfg.timing.opacity_reset_interval))
        return iteration % interval == 0

    def _in_window(self, iteration: int) -> bool:
        timing = self.ctrl_cfg.timing
        return int(timing.start_iter) <= int(iteration) <= int(timing.end_iter)

    def _densification_interval(self) -> int:
        timing = self.ctrl_cfg.timing
        freq = int(getattr(timing, "freq", 0))
        if freq > 0:
            return freq
        return max(1, int(timing.densification_interval))

    def update_grad_stats(self, container, render_info=None):
        if render_info is None:
            return
        viewspace = render_info.get("viewspace_points")
        visible = render_info.get("visibility_filter")
        radii = render_info.get("radii")
        if viewspace is None or viewspace.grad is None or visible is None:
            return

        n = int(container.num_points)
        if self.xyz_grad_accum.shape[0] != n:
            self.xyz_grad_accum = torch.zeros((n, 1), device=container.device)
            self.denom = torch.zeros((n, 1), device=container.device)
            self.max_radii2d = torch.zeros((n,), device=container.device)

        visible = visible.detach().bool()
        if visible.shape[0] != n:
            return
        grad = viewspace.grad.detach()
        self.xyz_grad_accum[visible] += torch.norm(grad[visible], dim=-1, keepdim=True)
        self.denom[visible] += 1
        if radii is not None and radii.shape[0] == n:
            self.max_radii2d[visible] = torch.maximum(
                self.max_radii2d[visible], radii.detach()[visible].float()
            )

    @torch.no_grad()
    def prune_and_densify(self, iteration, solver, container):
        if container.num_points == 0:
            return

        n_before = int(container.num_points)
        grads = self._current_grads(container)
        max_grad = float(grads.max().item()) if grads.numel() > 0 else 0.0
        mean_grad = float(grads.mean().item()) if grads.numel() > 0 else 0.0
        max_scale = torch.max(container.scales_2d, dim=1).values
        dense_threshold = float(self.ctrl_cfg.thresholds.percent_dense) * float(
            self.scene_radius
        )

        high_grad = grads > float(self.ctrl_cfg.thresholds.densify_grad)
        if bool(self.ctrl_cfg.enable.densify):
            split_mask = high_grad & (max_scale > dense_threshold)
            clone_mask = high_grad & (~split_mask)
        else:
            clone_mask = torch.zeros(
                (n_before,), device=container.device, dtype=torch.bool
            )
            split_mask = torch.zeros(
                (n_before,), device=container.device, dtype=torch.bool
            )

        new_tensors, _, _ = self._generate_new_points(container, clone_mask, split_mask)
        if new_tensors:
            solver.concat(new_tensors)

        prune_mask = torch.zeros(
            (int(container.num_points),), device=container.device, dtype=torch.bool
        )
        if bool(self.ctrl_cfg.enable.densify) and bool(split_mask.any().item()):
            prune_mask[:n_before] = split_mask

        if bool(self.ctrl_cfg.enable.prune):
            prune_mask = prune_mask | (
                container.opacities.squeeze(-1)
                < float(self.ctrl_cfg.thresholds.opacity_cull)
            )
            min_scale = float(getattr(self.ctrl_cfg.thresholds, "min_scale_2d", 0.0))
            max_scale = float(getattr(self.ctrl_cfg.thresholds, "max_scale_2d", 0.0))
            scales = container.scales_2d
            if min_scale > 0.0:
                prune_mask = prune_mask | (torch.min(scales, dim=1).values < min_scale)
            if max_scale > 0.0:
                prune_mask = prune_mask | (torch.max(scales, dim=1).values > max_scale)

        n_max_cfg = getattr(self.ctrl_cfg.thresholds, "n_max", None)
        try:
            n_max = int(n_max_cfg)
        except (TypeError, ValueError):
            n_max = 0
        if n_max > 0:
            kept_after_prune = int(container.num_points) - int(prune_mask.sum().item())
            overflow = max(0, kept_after_prune - n_max)
        else:
            overflow = 0
        if overflow > 0:
            scores = container.opacities.squeeze(-1).detach().clone()
            scores[prune_mask] = float("inf")
            _, idx_small = torch.topk(scores, k=overflow, largest=False)
            prune_mask[idx_small] = True

        if prune_mask.any():
            if bool(prune_mask.all().item()):
                keep_idx = int(torch.argmax(container.opacities.squeeze(-1)).item())
                prune_mask[keep_idx] = False
            n_prune = int(prune_mask.sum().item())
            solver.prune(prune_mask)
        else:
            n_prune = 0

        n_final = int(container.num_points)
        self.last_event_stats = {
            "iteration": float(iteration),
            "num_gaussians_before": float(n_before),
            "num_gaussians_after": float(n_final),
            "num_gaussians_delta": float(n_final - n_before),
            "max_grad": float(max_grad),
            "mean_grad": float(mean_grad),
            "densify_grad_threshold": float(self.ctrl_cfg.thresholds.densify_grad),
            "clone": float(int(clone_mask.sum().item())),
            "split": float(int(split_mask.sum().item())),
            "prune": float(n_prune),
            "prune_nmax": float(overflow),
        }
        self._reset_stats()

    @torch.no_grad()
    def reset_opacity(self, solver, container):
        if container.num_points == 0:
            return
        current = container.opacities
        reset_value = torch.minimum(current, torch.full_like(current, 0.01))
        reset = container.inverse("opacities", reset_value)
        solver.replace("opacities", reset)

    def _current_grads(self, container) -> torch.Tensor:
        n = int(container.num_points)
        if self.xyz_grad_accum.shape[0] != n:
            return torch.zeros((n,), device=container.device)
        grads = self.xyz_grad_accum / self.denom.clamp(min=1.0)
        grads = torch.nan_to_num(grads.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
        return grads

    def _reset_stats(self):
        self.xyz_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.max_radii2d = torch.empty(0)

    def _generate_new_points(
        self,
        container,
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
                if name == "scales_2d":
                    t_split -= torch.log(
                        torch.tensor(1.6, device=container.device, dtype=t_split.dtype)
                    )
                new_tensors_list[name].append(t_split)

            split_means = new_tensors_list["means"][-1]
            split_scales = torch.exp(new_tensors_list["scales_2d"][-1])
            xy_noise = (
                torch.randn(
                    (split_means.shape[0], 2),
                    device=container.device,
                    dtype=split_means.dtype,
                )
                * split_scales
            )
            split_means[:, :2].add_(xy_noise)

        final_tensors = {
            name: torch.cat(t_list, dim=0) for name, t_list in new_tensors_list.items()
        }
        return final_tensors, n_clone, n_split
