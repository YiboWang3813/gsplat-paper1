"""Native lifecycle for method-specific ultrasound adaptation strategies."""

from dataclasses import is_dataclass

from .base import Strategy


class NativeBaselineStrategy(Strategy):
    def initialize_state(self):
        return {
            key: value for key, value in vars(self).items() if not is_dataclass(value)
        }

    def collect_local(self, step):
        return False

    def collect_path(self, step):
        return False

    def step_post_backward(self, params, optimizers, state, step, info, **kwargs):
        from gsplat.ultrasound.parameters import GaussianParameters, ParameterOperations

        for key, value in state.items():
            setattr(self, key, value)
        current = GaussianParameters(params, self.config, info.get("means_grad"), step)
        before = GaussianParameters(
            info.get("before_parameters", params),
            self.config,
            info.get("means_grad"),
            step,
        )
        operations = ParameterOperations(params, optimizers)
        method = self.config.name
        if self.should_accumulate_grad_stats(step):
            if method == "gaussianpile":
                self.update_grad_stats(before, info["geometry"], info["slice"])
            elif method == "ultrags_baseline":
                self.update_grad_stats(before, info.get("render_info"))
            else:
                self.update_grad_stats(before)
        if self.should_densify_prune(step):
            self.prune_and_densify(step, operations, current)
        if hasattr(self, "should_reset_opacity") and self.should_reset_opacity(step):
            self.reset_opacity(operations, current)
        state.clear()
        state.update(self.initialize_state())
