"""Execute the native ultrasound kernels, comparing against independent Torch math."""

from types import SimpleNamespace

import pytest
import torch

from gsplat.ultrasound import (
    UltrasoundRenderConfig,
    build_slice_grid_pack,
    rasterization,
)
from gsplat.ultrasound.us_focus_reference import render_us_focus_reference


def inputs(ray=False):
    device = "cuda"
    config = UltrasoundRenderConfig(
        H=16,
        W=16,
        lateral_spacing_mm_per_pix=0.5,
        axial_spacing_mm_per_pix=0.5,
        axial_near_mm=0.0,
        tile_size_px=8,
        elev_corridor_half_mm=2.0,
        k_sigma=8.0,
        shader_weight_mode="us_focus",
        compositing_mode="additive",
        mapper_sort_by_v=False,
        us_focus_sigma_ele_0=1.2,
        us_focus_weight_min=1e-8,
        ray_attenuation_enabled=ray,
        ray_attenuation_source="attenuation",
        ray_attenuation_mapper_kind="ray_csr",
        ray_attenuation_formula="slice_depth",
        ray_attenuation_beta=0.6,
        ray_attenuation_strength=0.3,
    )
    means = torch.tensor(
        [[0.0, 4.0, 0.0], [1.0, 5.0, 0.8], [-1.2, 3.5, 2.5]],
        device=device,
        requires_grad=True,
    )
    scales = torch.tensor(
        [[0.8, 0.9, 0.7], [0.7, 0.8, 0.9], [1.1, 0.6, 0.8]],
        device=device,
        requires_grad=True,
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, device=device, requires_grad=True)
    opacity = torch.tensor([0.6, 0.4, 0.2], device=device, requires_grad=True)
    intensity = torch.tensor([0.75, 0.5, 0.9], device=device, requires_grad=True)
    pose = torch.eye(4, device=device, requires_grad=True)
    meta = SimpleNamespace(
        H=16,
        W=16,
        delta_lateral_mm=0.5,
        delta_axial_mm=0.5,
        near_mm=0.0,
        center_lateral=True,
    )
    return (
        config,
        [means, scales, quats, opacity, intensity, pose],
        build_slice_grid_pack(meta, device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution required")
def test_local_forward_and_gradients_match_torch_reference():
    config, tensors, grid = inputs()
    reference = [t.detach().clone().requires_grad_(True) for t in tensors]
    image = rasterization(config, *tensors, grid)
    expected = render_us_focus_reference(config, *reference)
    torch.testing.assert_close(image, expected, atol=3e-4, rtol=4e-3)
    torch.manual_seed(3)
    upstream = torch.randn_like(image) * 0.1
    (image * upstream).sum().backward()
    (expected * upstream).sum().backward()
    for index in (0, 3, 4):
        torch.testing.assert_close(
            tensors[index].grad, reference[index].grad, atol=3e-4, rtol=4e-3
        )
    for tensor in tensors:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution required")
def test_global_attenuation_and_path_statistics_execute():
    config, tensors, grid = inputs(ray=True)
    attenuation = torch.tensor([0.1, 0.2, 0.3], device="cuda", requires_grad=True)
    image, contribution, support = rasterization(
        config, *tensors, grid, attenuation, True
    )
    image.sum().backward()
    assert contribution.shape == support.shape == (3,)
    assert torch.isfinite(image).all() and torch.isfinite(contribution).all()
    assert attenuation.grad is not None and torch.isfinite(attenuation.grad).all()
    assert attenuation.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution required")
def test_pose_translation_gradient_finite_difference():
    config, tensors, grid = inputs()
    image = rasterization(config, *tensors, grid)
    image.sum().backward()
    analytical = tensors[-1].grad[0, 3].item()
    values = []
    for sign in (-1, 1):
        pose = tensors[-1].detach().clone()
        pose[0, 3] += sign * 1e-3
        values.append(rasterization(config, *tensors[:-1], pose, grid).sum().item())
    numerical = (values[1] - values[0]) / 2e-3
    assert analytical == pytest.approx(numerical, abs=0.01, rel=0.02)
