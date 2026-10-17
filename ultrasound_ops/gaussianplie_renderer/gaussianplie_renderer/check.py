import torch

from .types import GaussianPlieRenderConfig


def should_validate_inputs(config: GaussianPlieRenderConfig) -> bool:
    return bool(config.enable_input_checks)


def validate_render_inputs(
    config: GaussianPlieRenderConfig,
    gauss_means: torch.Tensor,
    gauss_scales: torch.Tensor,
    gauss_quats: torch.Tensor,
    gauss_opacities: torch.Tensor,
    pose_probe_to_world: torch.Tensor,
) -> None:
    del config
    n = gauss_means.shape[0]
    if gauss_means.ndim != 2 or gauss_means.shape != (n, 3):
        raise ValueError(f"gauss_means must be (N, 3), got {tuple(gauss_means.shape)}.")
    if gauss_scales.shape != (n, 3):
        raise ValueError(f"gauss_scales must be (N, 3), got {tuple(gauss_scales.shape)}.")
    if gauss_quats.shape != (n, 4):
        raise ValueError(f"gauss_quats must be (N, 4), got {tuple(gauss_quats.shape)}.")
    if gauss_opacities.shape not in ((n,), (n, 1)):
        raise ValueError(f"gauss_opacities must be (N,) or (N, 1), got {tuple(gauss_opacities.shape)}.")
    if pose_probe_to_world.shape != (4, 4):
        raise ValueError(f"pose_probe_to_world must be (4, 4), got {tuple(pose_probe_to_world.shape)}.")

