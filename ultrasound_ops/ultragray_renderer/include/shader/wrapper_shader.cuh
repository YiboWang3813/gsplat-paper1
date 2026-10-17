#pragma once

#include <torch/extension.h>

torch::Tensor ShaderForwardCUDA(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& attenuations,
    const torch::Tensor& sh_coeffs,
    const torch::Tensor& ray_counts,
    const torch::Tensor& ray_offsets,
    const torch::Tensor& ray_indices,
    const torch::Tensor& sorted_depths,
    const torch::Tensor& pixel_positions,
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_dirs,
    const torch::Tensor& axis_elevational,
    const torch::Tensor& ray_perturbations,
    float k_sigma,
    float quadrature_window_radius_mm,
    bool local_transmission_window_only,
    float local_transmission_max_window_mm
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ShaderBackwardCUDA(
    const torch::Tensor& dL_dImage,
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& attenuations,
    const torch::Tensor& sh_coeffs,
    const torch::Tensor& ray_counts,
    const torch::Tensor& ray_offsets,
    const torch::Tensor& ray_indices,
    const torch::Tensor& sorted_depths,
    const torch::Tensor& pixel_positions,
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_dirs,
    const torch::Tensor& axis_elevational,
    const torch::Tensor& ray_perturbations,
    float k_sigma,
    float quadrature_window_radius_mm,
    bool local_transmission_window_only,
    float local_transmission_max_window_mm
);
