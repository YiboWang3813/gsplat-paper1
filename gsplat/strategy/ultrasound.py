"""Ultrasound clone/split/prune strategy using gsplat parameter/optimizer APIs."""

from dataclasses import dataclass
from typing import Literal

import torch

from .base import Strategy
from .ops import _update_param_with_optimizer


@dataclass
class UltrasoundStrategy(Strategy):
    enabled: bool = False
    refine_start: int = 1000
    refine_stop: int = 8000
    refine_every: int = 1000
    densify: bool = True
    pruning: Literal["none", "opacity", "local_aware", "path_aware"] = "opacity"
    densification: Literal["gradient", "residual_guided"] = "gradient"
    gradient_threshold: float = 1.5e-5
    split_scale_mm: float = 0.6
    prune_echo_alpha: float = 0.015
    max_scale_mm: float = 0.0
    score_quantile: float = 0.98
    min_score: float = 1e-6
    max_fraction: float = 0.05
    low_contribution_quantile: float = 0.1
    path_contribution_quantile: float = 0.8
    path_threshold_mode: Literal["quantile", "fixed"] = "quantile"
    path_threshold: float = 0.0
    high_residual_quantile: float = 0.9

    def initialize_state(self):
        if (
            self.refine_every <= 0
            or self.refine_start < 1
            or self.refine_stop < self.refine_start
        ):
            raise ValueError(
                "Invalid ultrasound refinement schedule (physical iterations start at 1)."
            )
        for value in (
            self.score_quantile,
            self.low_contribution_quantile,
            self.path_contribution_quantile,
            self.high_residual_quantile,
        ):
            if not 0 <= value <= 1:
                raise ValueError("Strategy quantiles must be within [0,1].")
        return {"accumulators": {}, "frames": 0, "last_event": {}}

    def check_sanity(self, params, optimizers):
        super().check_sanity(params, optimizers)
        required = {"means", "scales", "quats", "echo_alphas", "attenuations"}
        if set(params) != required:
            raise ValueError(f"Expected ultrasound parameters {required}.")
        for name, parameter in params.items():
            owned = optimizers[name].param_groups[0]["params"]
            if len(owned) != 1 or owned[0] is not parameter:
                raise ValueError(
                    f"Optimizer {name} does not own the current parameter."
                )

    def in_window(self, step):
        return self.enabled and self.refine_start <= step <= self.refine_stop

    def is_event(self, step):
        return self.in_window(step) and step % self.refine_every == 0

    def collect_local(self, step):
        return self.is_event(step) and (
            self.densification == "residual_guided"
            or self.pruning in ("local_aware", "path_aware")
        )

    def collect_path(self, step):
        return (
            self.in_window(step)
            and self.pruning == "path_aware"
            and step <= self.refine_stop // self.refine_every * self.refine_every
        )

    @torch.no_grad()
    def step_post_backward(self, params, optimizers, state, step, info, **kwargs):
        """Called after the optimizer update, with captured unscaled backward stats."""
        if not self.in_window(step):
            return
        accum = state["accumulators"]

        def add(name, value):
            value = value.detach().flatten()
            if name not in accum:
                accum[name] = torch.zeros_like(value)
            if accum[name].shape != value.shape:
                raise RuntimeError(
                    "Gaussian topology changed inside a refinement interval."
                )
            accum[name] += value

        grad = info.get("means_grad")
        if grad is None:
            raise RuntimeError(
                "Density control requires captured unscaled mean gradients."
            )
        add("gradient", grad.norm(dim=-1))
        state["frames"] += 1
        if "local_stats" in info:
            for name, value in zip(
                ("contribution", "residual", "hits", "support"), info["local_stats"]
            ):
                add(name, value)
        if "path_contribution" in info:
            add("path", info["path_contribution"])
        if not self.is_event(step):
            return
        n = len(params["means"])
        scales, echo = info["activated_scales"], info["activated_echo"]
        high = accum["gradient"] / state["frames"] > self.gradient_threshold
        if self.densification == "residual_guided" and "contribution" in accum:
            c = accum["contribution"]
            score = torch.nan_to_num(
                accum["residual"] / c.clamp_min(1e-8) * torch.log1p(c),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            positive = score[score > 0]
            if positive.numel():
                threshold = torch.quantile(positive, self.score_quantile).clamp_min(
                    self.min_score
                )
                if 0 < self.max_fraction < 1:
                    k = max(1, int(n * self.max_fraction))
                    if positive.numel() > k:
                        threshold = torch.maximum(
                            threshold, torch.topk(score, k).values[-1]
                        )
                high = score >= threshold
        maximum = scales.max(dim=-1).values
        clone = (
            high & (maximum <= self.split_scale_mm) & (maximum > 0.01) & self.densify
        )
        split = high & (maximum > self.split_scale_mm) & self.densify
        children = {}
        for name, value in params.items():
            repeats = (2,) + (1,) * (value.ndim - 1)
            split_values = value[split].repeat(repeats).clone()
            # Retain the frozen algorithm's raw-scale shift. Changing this to a
            # physical inverse-activation rule would be a separate method change.
            if name == "scales":
                split_values -= torch.log(torch.tensor(2.0, device=value.device))
            children[name] = torch.cat([value[clone], split_values])
        split_count = int(split.sum()) * 2
        if split_count:
            children["means"][-split_count:] += (
                torch.randn_like(children["means"][-split_count:])
                * children["scales"][-split_count:].exp()
                * 0.1
            )
        child_count = len(children["means"])
        if child_count:
            _update_param_with_optimizer(
                lambda name, value: torch.nn.Parameter(
                    torch.cat([value, children[name]])
                ),
                lambda key, value: torch.cat(
                    [
                        value,
                        torch.zeros(
                            (child_count,) + value.shape[1:],
                            device=value.device,
                            dtype=value.dtype,
                        ),
                    ]
                ),
                params,
                optimizers,
            )
        remove = torch.zeros(n + child_count, device=echo.device, dtype=torch.bool)
        if self.pruning != "none":
            # Children have identical echo parameters to parents but no local/path
            # history. Local/path policies protect them during their birth event.
            remove = torch.cat(
                [
                    echo < self.prune_echo_alpha,
                    info["echo_max"] * children["echo_alphas"].sigmoid()
                    < self.prune_echo_alpha,
                ]
            )
            if self.pruning in ("local_aware", "path_aware"):
                if "contribution" not in accum:
                    raise RuntimeError(
                        "Local-aware pruning requires contribution statistics."
                    )
                contribution = torch.cat(
                    [
                        accum["contribution"],
                        torch.full((child_count,), float("inf"), device=echo.device),
                    ]
                )
                remove &= contribution <= torch.quantile(
                    contribution, self.low_contribution_quantile
                )
                if self.pruning == "path_aware":
                    if "path" not in accum:
                        # Missing history must never silently remove global-path Gaussians.
                        remove.zero_()
                    else:
                        raw = accum["path"]
                        path = torch.nan_to_num(
                            raw, nan=0.0, posinf=0.0, neginf=0.0
                        ).clamp_min(0)
                        threshold = (
                            torch.quantile(path, self.path_contribution_quantile)
                            if self.path_threshold_mode == "quantile"
                            else self.path_threshold
                        )
                        protect = (path >= threshold) | ~torch.isfinite(raw)
                        remove &= ~torch.cat(
                            [
                                protect,
                                torch.ones(
                                    child_count, device=echo.device, dtype=torch.bool
                                ),
                            ]
                        )
            if self.max_scale_mm > 0:
                child_scales = (
                    info["scale_min"]
                    + (info["scale_max"] - info["scale_min"])
                    * children["scales"].sigmoid()
                )
                remove |= (
                    torch.cat([maximum, child_scales.max(dim=-1).values])
                    > self.max_scale_mm
                )
        if remove.all():
            raise RuntimeError(
                "Density policy would prune every Gaussian; refusing an invalid empty model."
            )
        if remove.any():
            _update_param_with_optimizer(
                lambda name, value: torch.nn.Parameter(value[~remove]),
                lambda key, value: value[~remove],
                params,
                optimizers,
            )
        state["last_event"] = {
            "step": step,
            "before": n,
            "after": len(params["means"]),
            "cloned": int(clone.sum()),
            "split_parents": int(split.sum()),
            "pruned": int(remove.sum()),
        }
        state["accumulators"], state["frames"] = {}, 0
