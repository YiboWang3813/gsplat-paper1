import torch

from .types import OursRenderConfig


def should_validate_inputs(config: OursRenderConfig) -> bool:
    return bool(getattr(config, "enable_input_checks", False))


def validate_render_inputs(
    config: OursRenderConfig,
    gauss_means: torch.Tensor,
    gauss_scales: torch.Tensor,
    gauss_quats: torch.Tensor,
    gauss_opacities: torch.Tensor,
    gauss_intensities: torch.Tensor,
    pose_probe_to_world: torch.Tensor,
) -> None:
    if gauss_means.ndim != 2 or gauss_means.shape[1] != 3:
        raise ValueError(f"gauss_means must have shape (G, 3), got {tuple(gauss_means.shape)}.")
    if gauss_scales.ndim != 2 or gauss_scales.shape[1] != 3:
        raise ValueError(f"gauss_scales must have shape (G, 3), got {tuple(gauss_scales.shape)}.")
    if gauss_quats.ndim != 2 or gauss_quats.shape[1] != 4:
        raise ValueError(f"gauss_quats must have shape (G, 4), got {tuple(gauss_quats.shape)}.")

    num_gauss = gauss_means.shape[0]
    if gauss_scales.shape[0] != num_gauss or gauss_quats.shape[0] != num_gauss:
        raise ValueError("gauss_means/gauss_scales/gauss_quats must share the same leading dimension G.")
    if gauss_opacities.numel() != num_gauss:
        raise ValueError(f"gauss_opacities must contain G elements, got {gauss_opacities.numel()} vs G={num_gauss}.")
    if gauss_intensities.numel() != num_gauss:
        raise ValueError(
            f"gauss_intensities must contain G elements, got {gauss_intensities.numel()} vs G={num_gauss}."
        )

    if pose_probe_to_world.shape != (4, 4):
        raise ValueError(f"pose_probe_to_world must be (4, 4), got {tuple(pose_probe_to_world.shape)}.")
    if not torch.isfinite(pose_probe_to_world).all():
        raise ValueError("pose_probe_to_world contains NaN/Inf.")
