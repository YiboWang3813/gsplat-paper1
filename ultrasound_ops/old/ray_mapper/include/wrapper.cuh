#pragma once 
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> launch_ray_mapper(
    // Geometry Tensors
    const torch::Tensor& means,           // (N, 4)
    const torch::Tensor& scales,          // (N, 4)
    const torch::Tensor& perturbations,   // (N,) Python 预计算的随机扰动
    const torch::Tensor& origin,          // (3,)
    const torch::Tensor& axis_lat,        // (3,)
    const torch::Tensor& axis_ele,        // (3,)
    const torch::Tensor& axis_ax,         // (3,)
    // Scalar Params
    const int image_width,
    const float width_mm,
    const float near_mm,
    const float elev_cutoff_mm,
    const float k_sigma
);