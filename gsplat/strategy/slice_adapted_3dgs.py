from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from gsplat.strategy.baseline import NativeBaselineStrategy
from gsplat.ultrasound.parameters import GaussianParameters, ParameterOperations


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(q, dim=-1, eps=1.0e-8)
    w, x, y, z = q.unbind(-1)
    two = 2.0
    return torch.stack(
        [
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


class Slice3DGSStrategy(NativeBaselineStrategy):
    def __init__(self, cfg: Any):
        self.config = cfg
        self.ctrl_cfg = cfg.offline.density_control
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
        if not bool(self.ctrl_cfg.enable.densify):
            return False
        return self._in_window(iteration)

    def should_densify_prune(self, iteration: int) -> bool:
        if not (bool(self.ctrl_cfg.enable.densify) or bool(self.ctrl_cfg.enable.prune)):
            return False
        if not self._in_window(iteration):
            return False
        return int(iteration) % max(1, int(self.ctrl_cfg.timing.freq)) == 0

    def should_reset_opacity(self, iteration: int) -> bool:
        if not bool(self.ctrl_cfg.enable.opacity_reset):
            return False
        if not self._in_window(iteration):
            return False
        interval = max(1, int(self.ctrl_cfg.timing.opacity_reset_interval))
        return int(iteration) % interval == 0

    def _in_window(self, iteration: int) -> bool:
        timing = self.ctrl_cfg.timing
        return int(timing.start_iter) <= int(iteration) <= int(timing.end_iter)

    def update_grad_stats(self, container: GaussianParameters):
        if container.mean_gradient is None:
            return
        n = int(container.num_points)
        if self.xyz_grad_accum.shape[0] != n:
            self.xyz_grad_accum = torch.zeros((n, 1), device=container.device)
            self.denom = torch.zeros((n, 1), device=container.device)
        grad_norm = torch.norm(container.mean_gradient.detach(), dim=-1, keepdim=True)
        self.xyz_grad_accum += grad_norm
        self.denom += 1

    @torch.no_grad()
    def prune_and_densify(
        self,
        iteration: int,
        solver: ParameterOperations,
        container: GaussianParameters,
    ):
        if container.num_points == 0:
            return

        n_before = int(container.num_points)
        thresholds = self.ctrl_cfg.thresholds
        grads = self._current_grads(container)
        max_grad = float(grads.max().item()) if grads.numel() > 0 else 0.0
        mean_grad = float(grads.mean().item()) if grads.numel() > 0 else 0.0
        max_scale = torch.max(container.scales_3d, dim=1).values
        high_grad = grads > float(thresholds.densify_grad)

        if bool(self.ctrl_cfg.enable.densify):
            split_mask = high_grad & (max_scale > float(thresholds.split_scale_mm))
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
            scales = container.scales_3d
            prune_mask = prune_mask | (
                container.opacities.squeeze(-1) < float(thresholds.prune_opacity)
            )
            prune_mask = prune_mask | (
                torch.min(scales, dim=1).values < float(thresholds.min_scale_mm)
            )
            prune_mask = prune_mask | (
                torch.max(scales, dim=1).values > float(thresholds.max_scale_mm)
            )

        n_max = int(self.ctrl_cfg.n_max)
        if n_max > 0:
            kept_after_prune = int(container.num_points) - int(prune_mask.sum().item())
            overflow = max(0, kept_after_prune - n_max)
            if overflow > 0:
                scores = container.opacities.squeeze(-1).detach().clone()
                scores[prune_mask] = float("inf")
                _, idx_small = torch.topk(scores, k=overflow, largest=False)
                prune_mask[idx_small] = True
        else:
            overflow = 0

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
            "densify_grad_threshold": float(thresholds.densify_grad),
            "clone": float(int(clone_mask.sum().item())),
            "split": float(int(split_mask.sum().item())),
            "prune": float(n_prune),
            "prune_nmax": float(overflow),
        }
        self._reset_stats()

    @torch.no_grad()
    def reset_opacity(self, solver: ParameterOperations, container: GaussianParameters):
        if container.num_points == 0:
            return
        current = container.opacities
        reset_value = torch.minimum(current, torch.full_like(current, 0.01))
        reset = container.inverse("opacities", reset_value)
        solver.replace("opacities", reset)

    def _current_grads(self, container: GaussianParameters) -> torch.Tensor:
        n = int(container.num_points)
        if self.xyz_grad_accum.shape[0] != n:
            return torch.zeros((n,), device=container.device)
        grads = self.xyz_grad_accum / self.denom.clamp(min=1.0)
        return torch.nan_to_num(grads.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)

    def _reset_stats(self):
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
                if name == "scales_3d":
                    t_split = t_split - torch.log(
                        torch.tensor(1.6, device=container.device, dtype=t_split.dtype)
                    )
                new_tensors_list[name].append(t_split)

            split_means = new_tensors_list["means"][-1]
            source_scales = container.scales_3d[split_mask].repeat(2, 1)
            source_quats = container.quats[split_mask].repeat(2, 1)
            local_noise = torch.randn_like(split_means) * source_scales
            rotations = _quat_to_matrix(source_quats)
            split_means.add_(rotations.matmul(local_noise.unsqueeze(-1)).squeeze(-1))

        final_tensors = {
            name: torch.cat(t_list, dim=0) for name, t_list in new_tensors_list.items()
        }
        return final_tensors, n_clone, n_split
