#pragma once
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MapperForwardCUDA(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& origin,
    const torch::Tensor& axis_lat,
    const torch::Tensor& axis_ele,
    const torch::Tensor& axis_ax,
    int image_width,
    float width_mm,
    float near_mm,
    float lateral_col_offset,
    float elev_cutoff_mm,
    float k_sigma
);
