import torch

from .types import BatchSliceGeometry, SliceGeometry, UltragsRenderConfig


def should_validate_inputs(config: UltragsRenderConfig) -> bool:
    return bool(getattr(config, "enable_input_checks", False))


def validate_render_inputs(
    config: UltragsRenderConfig,
    gauss_means: torch.Tensor,
    gauss_scales: torch.Tensor,
    gauss_quats: torch.Tensor,
    gauss_opacities: torch.Tensor,
    gauss_intensities: torch.Tensor,
    pixel_positions_world: torch.Tensor,
    geometry: SliceGeometry,
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

    if pixel_positions_world.ndim != 2 or pixel_positions_world.shape[1] != 3:
        raise ValueError(
            f"pixel_positions_world must have shape (P, 3), got {tuple(pixel_positions_world.shape)}."
        )
    if pixel_positions_world.shape[0] != config.H * config.W:
        raise ValueError(
            f"pixel_positions_world first dim must equal H*W={config.H * config.W}, "
            f"got P={pixel_positions_world.shape[0]}."
        )

    probe_origin_world = geometry.origin_world
    axis_lateral_world = geometry.axis_lateral_world
    axis_axial_world = geometry.axis_axial_world
    axis_elevational_world = geometry.axis_elevational_world

    for name, axis in (
        ("probe_origin_world", probe_origin_world),
        ("axis_lateral_world", axis_lateral_world),
        ("axis_axial_world", axis_axial_world),
        ("axis_elevational_world", axis_elevational_world),
    ):
        if axis.ndim != 1 or axis.shape[0] != 3:
            raise ValueError(f"{name} must have shape (3,), got {tuple(axis.shape)}.")
        if not torch.isfinite(axis).all():
            raise ValueError(f"{name} contains NaN/Inf.")

    for axis_name, axis in (
        ("axis_lateral_world", axis_lateral_world),
        ("axis_axial_world", axis_axial_world),
        ("axis_elevational_world", axis_elevational_world),
    ):
        if torch.linalg.norm(axis).item() <= 1e-8:
            raise ValueError(f"{axis_name} has near-zero norm.")


def validate_batched_render_inputs(
    config: UltragsRenderConfig,
    gauss_means: torch.Tensor,
    gauss_scales: torch.Tensor,
    gauss_quats: torch.Tensor,
    gauss_opacities: torch.Tensor,
    gauss_intensities: torch.Tensor,
    pixel_positions_world: torch.Tensor,
    geometry: BatchSliceGeometry,
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

    if pixel_positions_world.ndim != 3 or pixel_positions_world.shape[2] != 3:
        raise ValueError(
            f"pixel_positions_world must have shape (B, P, 3), got {tuple(pixel_positions_world.shape)}."
        )
    B = pixel_positions_world.shape[0]
    P = pixel_positions_world.shape[1]
    if P != config.H * config.W:
        raise ValueError(
            f"pixel_positions_world P must equal H*W={config.H * config.W}, got P={P}."
        )

    probe_origin_world = geometry.origin_world
    axis_lateral_world = geometry.axis_lateral_world
    axis_axial_world = geometry.axis_axial_world
    axis_elevational_world = geometry.axis_elevational_world

    for name, axis in (
        ("probe_origin_world", probe_origin_world),
        ("axis_lateral_world", axis_lateral_world),
        ("axis_axial_world", axis_axial_world),
        ("axis_elevational_world", axis_elevational_world),
    ):
        if axis.ndim != 2 or axis.shape[0] != B or axis.shape[1] != 3:
            raise ValueError(f"{name} must have shape ({B}, 3), got {tuple(axis.shape)}.")
        if not torch.isfinite(axis).all():
            raise ValueError(f"{name} contains NaN/Inf.")

    for axis_name, axis in (
        ("axis_lateral_world", axis_lateral_world),
        ("axis_axial_world", axis_axial_world),
        ("axis_elevational_world", axis_elevational_world),
    ):
        norms = torch.linalg.norm(axis, dim=-1)
        if (norms <= 1e-8).any():
            raise ValueError(f"{axis_name} has near-zero norm on some batch rows.")
