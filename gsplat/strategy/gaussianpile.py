from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from gsplat.strategy.baseline import NativeBaselineStrategy
from gsplat.ultrasound.parameters import GaussianParameters, ParameterOperations


class GaussianPileStrategy(NativeBaselineStrategy):
    @property
    def model_cfg(self) -> Any:
        return self.config

    def __init__(self, cfg: Any):
        self.config = cfg
        self.thresh = self.model_cfg.offline.density_control.thresholds
        self.switches = self.model_cfg.offline.density_control.enable
        self.xyz_grad_accum = torch.empty(0)
        self.focus_response_accum = torch.empty(0)
        self.tile_radius_accum = torch.empty(0)
        self.prune_observation_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.last_event_stats: Dict[str, float] = {}
        print(
            "[GaussianPileDensityController] "
            f"Init Densify={self.switches.densify}, Prune={self.switches.prune}"
        )

    def scalar_stats(self, iteration: int) -> Dict[str, float]:
        if int(self.last_event_stats.get("iteration", -1)) != int(iteration):
            return {}
        return {
            f"train/dc/{k}": v
            for k, v in self.last_event_stats.items()
            if k != "iteration"
        }

    def should_accumulate_grad_stats(self, iteration: int) -> bool:
        if not (self.switches.densify or self.switches.prune):
            return False
        t = self.model_cfg.offline.density_control.timing
        if iteration < int(t.start_iter) or iteration > int(t.end_iter):
            return False
        interval = max(1, int(getattr(t, "grad_accum_interval", 1)))
        return iteration % interval == 0 or self.should_densify_prune(iteration)

    def should_densify_prune(self, iteration: int) -> bool:
        if not (self.switches.densify or self.switches.prune):
            return False
        t = self.model_cfg.offline.density_control.timing
        if iteration < int(t.start_iter) or iteration > int(t.end_iter):
            return False
        return iteration % int(t.freq) == 0

    def update_grad_stats(
        self, container: GaussianParameters, dataset_meta=None, slice_meta=None
    ):
        if container.mean_gradient is None:
            return
        n = container.num_points
        if self.xyz_grad_accum.shape[0] != n:
            self.xyz_grad_accum = torch.zeros((n, 1), device=container.device)
            self.focus_response_accum = torch.zeros((n, 1), device=container.device)
            self.tile_radius_accum = torch.zeros((n, 1), device=container.device)
            self.prune_observation_accum = torch.zeros((n, 1), device=container.device)
            self.denom = torch.zeros((n, 1), device=container.device)
        focus_response, tile_radius = self._focus_stats(
            container, dataset_meta, slice_meta
        )
        self.xyz_grad_accum += (
            torch.norm(container.mean_gradient, dim=-1, keepdim=True) * focus_response
        )
        self.focus_response_accum = torch.maximum(
            self.focus_response_accum, focus_response
        )
        self.tile_radius_accum = torch.maximum(self.tile_radius_accum, tile_radius)
        self.prune_observation_accum += self._prune_observation_mask(
            focus_response, tile_radius
        ).float()
        self.denom += 1

    def _focus_stats(
        self, container: GaussianParameters, dataset_meta=None, slice_meta=None
    ):
        n = int(container.num_points)
        if dataset_meta is None or slice_meta is None:
            ones = torch.ones((n, 1), device=container.device)
            return ones, ones

        with torch.no_grad():
            means = container.means
            dtype = means.dtype
            axis_elev = slice_meta.vec_elevational.to(device=means.device, dtype=dtype)
            origin = slice_meta.probe_origin.to(device=means.device, dtype=dtype)
            signed_dist = means.matmul(axis_elev) - origin.dot(axis_elev)

            configured_sigma_z = self.model_cfg.slice_render.sigma_z_mm
            if configured_sigma_z is not None:
                sigma_z = float(configured_sigma_z)
            else:
                ratio = getattr(self.model_cfg.slice_render, "sigma_z_ratio", None)
                if ratio is not None:
                    sigma_z = float(ratio) * float(dataset_meta.delta_elevational_mm)
                else:
                    sigma_z = float(dataset_meta.delta_elevational_mm)
            sigma_z = max(sigma_z, 1.0e-6)
            focus_response = torch.exp(
                -0.5 * (signed_dist / sigma_z).square()
            ).unsqueeze(-1)

            min_spacing = max(
                min(
                    float(dataset_meta.delta_lateral_mm),
                    float(dataset_meta.delta_axial_mm),
                ),
                1.0e-6,
            )
            radius_sigma = float(self.model_cfg.slice_render.radius_sigma)
            tile_size = max(int(self.model_cfg.slice_render.tile_size_px), 1)
            screen_radius_px = (
                radius_sigma * torch.max(container.scales, dim=1).values / min_spacing
            )
            tile_radius = (
                (screen_radius_px / float(tile_size)).unsqueeze(-1).clamp_min(0.0)
            )
            return focus_response, tile_radius

    def prune_and_densify(
        self,
        iteration: int,
        solver: ParameterOperations,
        container: GaussianParameters,
    ):
        if not self.switches.densify and not self.switches.prune:
            return

        n_before = int(container.num_points)
        max_grad = 0.0
        mean_grad = 0.0
        stats = {
            "clone": 0,
            "split": 0,
            "prune_weak": 0,
            "prune_unobserved": 0,
            "prune_limited": 0,
            "prune_guarded": 0,
            "prune_nmax": 0,
        }
        n_pts = int(container.num_points)
        grads = None
        if self.switches.densify:
            grads = self.xyz_grad_accum / self.denom.clamp(min=1)
            grads[grads.isnan()] = 0.0
            grads = grads.squeeze(-1)
            if grads.numel() > 0:
                max_grad = float(grads.max().item())
                mean_grad = float(grads.mean().item())
            print(
                f"[GaussianPile DC Check] Iter {iteration} | "
                f"Max Grad: {max_grad:.2e} | "
                f"Mean Grad: {mean_grad:.2e} | "
                f"Thresh: {float(self.thresh.densify_grad):.2e}"
            )

            if n_pts < int(self.model_cfg.offline.density_control.n_max):
                high_grad = grads > float(self.thresh.densify_grad)
                max_scale = torch.max(container.scales, dim=1).values
                if self.tile_radius_accum.shape[0] == n_pts:
                    focus_tile_radius = self.tile_radius_accum.squeeze(-1)
                    visible = self.focus_response_accum.squeeze(-1) > float(
                        self.thresh.prune_opacity
                    )
                else:
                    focus_tile_radius = torch.zeros_like(max_scale)
                    visible = torch.ones_like(max_scale, dtype=torch.bool)
                split_mask = (
                    high_grad
                    & visible
                    & (
                        (focus_tile_radius > 1.0)
                        | (max_scale > float(self.thresh.split_scale))
                    )
                )
                clone_mask = (
                    high_grad
                    & visible
                    & ~split_mask
                    & (max_scale <= float(self.thresh.split_scale))
                    & (max_scale > float(self.thresh.min_scale_mm))
                )
                new_tensors, n_clone, n_split = self._generate_new_points(
                    container, clone_mask, split_mask
                )
                if n_clone > 0 or n_split > 0:
                    n_room = max(
                        0, int(self.model_cfg.offline.density_control.n_max) - n_pts
                    )
                    n_new = (
                        int(next(iter(new_tensors.values())).shape[0])
                        if new_tensors
                        else 0
                    )
                    if n_room < n_new:
                        new_tensors = {
                            k: v[:n_room].contiguous() for k, v in new_tensors.items()
                        }
                    if new_tensors and next(iter(new_tensors.values())).numel() > 0:
                        solver.concat(new_tensors)
                        stats["clone"] = min(n_clone, n_room)
                        stats["split"] = max(0, min(n_new, n_room) - stats["clone"])
                        self.xyz_grad_accum = torch.empty(0)
                        self.focus_response_accum = torch.empty(0)
                        self.tile_radius_accum = torch.empty(0)
                        self.prune_observation_accum = torch.empty(0)
                        self.denom = torch.empty(0)

        if self.switches.prune:
            n_pts = int(container.num_points)
            alphas = container.alphas.squeeze()
            opacity_scores = self._prune_opacity_scores(alphas, n_pts)
            weak_mask = opacity_scores < float(self.thresh.prune_opacity)
            observed_enough = self._prune_observed_enough(n_pts, alphas.device)
            stats["prune_unobserved"] = int((weak_mask & ~observed_enough).sum().item())
            prune_mask = weak_mask & observed_enough
            prune_mask, n_limited = self._apply_prune_fraction_limit(prune_mask, alphas)
            prune_mask, n_guarded = self._apply_min_keep_guard(prune_mask, alphas)
            stats["prune_limited"] = n_limited
            stats["prune_guarded"] = n_guarded
            stats["prune_weak"] = int(prune_mask.sum().item())
            if prune_mask.any():
                solver.prune(prune_mask)
                self.xyz_grad_accum = torch.empty(0)
                self.focus_response_accum = torch.empty(0)
                self.tile_radius_accum = torch.empty(0)
                self.prune_observation_accum = torch.empty(0)
                self.denom = torch.empty(0)

        if sum(stats.values()) > 0:
            print(
                f"[GaussianPile DC] Iter {iteration} | Split: {stats['split']} "
                f"Clone: {stats['clone']} | Prune: {stats['prune_weak']} "
                f"UnobservedWeak: {stats['prune_unobserved']} | "
                f"Limited: {stats['prune_limited']} | Guarded: {stats['prune_guarded']}"
            )
        n_after = int(container.num_points)
        self.last_event_stats = {
            "iteration": float(iteration),
            "num_gaussians_before": float(n_before),
            "num_gaussians_after": float(n_after),
            "num_gaussians_delta": float(n_after - n_before),
            "max_grad": float(max_grad),
            "mean_grad": float(mean_grad),
            "densify_grad_threshold": float(self.thresh.densify_grad),
            "clone": float(stats["clone"]),
            "split": float(stats["split"]),
            "prune_weak": float(stats["prune_weak"]),
            "prune_unobserved": float(stats["prune_unobserved"]),
            "prune_limited": float(stats["prune_limited"]),
            "prune_guarded": float(stats["prune_guarded"]),
            "prune_nmax": float(stats["prune_nmax"]),
        }

    def _prune_observation_mask(
        self, focus_response: torch.Tensor, tile_radius: torch.Tensor
    ) -> torch.Tensor:
        tile_size = max(float(self.model_cfg.slice_render.tile_size_px), 1.0)
        screen_radius_px = tile_radius.clamp_min(0.0) * tile_size
        support = torch.sqrt(screen_radius_px.clamp_min(1.0))
        signal = focus_response.clamp_min(0.0) * support
        threshold = float(getattr(self.thresh, "prune_observation_score", 0.25))
        return signal >= threshold

    def _prune_observed_enough(self, n_pts: int, device: torch.device) -> torch.Tensor:
        min_observations = int(getattr(self.thresh, "min_prune_observations", 0))
        if min_observations <= 0:
            return torch.ones((n_pts,), dtype=torch.bool, device=device)
        if self.prune_observation_accum.shape[0] != n_pts:
            return torch.ones((n_pts,), dtype=torch.bool, device=device)
        observed = self.prune_observation_accum.squeeze(-1)
        return observed >= float(min_observations)

    def _prune_opacity_scores(self, alphas: torch.Tensor, n_pts: int) -> torch.Tensor:
        if (
            self.focus_response_accum.shape[0] != n_pts
            or self.tile_radius_accum.shape[0] != n_pts
        ):
            return alphas

        focus_response = self.focus_response_accum.squeeze(-1).clamp_min(0.0)
        tile_radius = self.tile_radius_accum.squeeze(-1).clamp_min(0.0)
        tile_size = max(float(self.model_cfg.slice_render.tile_size_px), 1.0)
        screen_radius_px = tile_radius * tile_size
        support = torch.sqrt(screen_radius_px.clamp_min(1.0))
        return alphas * focus_response * support

    def _apply_prune_fraction_limit(
        self, prune_mask: torch.Tensor, alphas: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        max_fraction = float(getattr(self.thresh, "max_prune_fraction_per_step", 0.0))
        if max_fraction <= 0.0 or not bool(prune_mask.any()):
            return prune_mask, 0

        n_current = int(alphas.numel())
        max_prune = int(round(n_current * max_fraction))
        max_prune = max(1, min(max_prune, n_current))
        n_requested = int(prune_mask.sum().item())
        if n_requested <= max_prune:
            return prune_mask, 0

        limited = n_requested - max_prune
        candidate_idx = torch.nonzero(prune_mask, as_tuple=False).squeeze(1)
        candidate_alpha = alphas[candidate_idx]
        prune_in_candidates = torch.topk(
            candidate_alpha, k=max_prune, largest=False
        ).indices
        limited_mask = torch.zeros_like(prune_mask, dtype=torch.bool)
        limited_mask[candidate_idx[prune_in_candidates]] = True
        return limited_mask, limited

    def _apply_min_keep_guard(
        self, prune_mask: torch.Tensor, alphas: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        min_keep_ratio = float(getattr(self.thresh, "min_keep_ratio", 0.0))
        if min_keep_ratio <= 0.0 or not bool(prune_mask.any()):
            return prune_mask, 0

        init_count = max(1, int(self.model_cfg.gauss_init.num_points))
        min_keep = int(round(init_count * min_keep_ratio))
        n_current = int(alphas.numel())
        max_prune = max(0, n_current - min_keep)
        n_requested = int(prune_mask.sum().item())
        if n_requested <= max_prune:
            return prune_mask, 0

        guarded = n_requested - max_prune
        if max_prune <= 0:
            return torch.zeros_like(prune_mask, dtype=torch.bool), guarded

        candidate_idx = torch.nonzero(prune_mask, as_tuple=False).squeeze(1)
        candidate_alpha = alphas[candidate_idx]
        keep_in_candidates = torch.topk(
            candidate_alpha, k=max_prune, largest=False
        ).indices
        limited_mask = torch.zeros_like(prune_mask, dtype=torch.bool)
        limited_mask[candidate_idx[keep_in_candidates]] = True
        return limited_mask, guarded

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
                    split_scale_values = container.scales[split_mask].repeat(2, 1) * 0.5
                    t_split = container.inverse("scales", split_scale_values)
                new_tensors_list[name].append(t_split)
            split_scales = container.scales[split_mask].repeat(2, 1)
            perturb = (
                torch.randn_like(new_tensors_list["means"][-1]) * split_scales * 0.1
            )
            new_tensors_list["means"][-1].add_(perturb)

        return (
            {
                name: torch.cat(parts, dim=0).contiguous()
                for name, parts in new_tensors_list.items()
            },
            n_clone,
            n_split,
        )
