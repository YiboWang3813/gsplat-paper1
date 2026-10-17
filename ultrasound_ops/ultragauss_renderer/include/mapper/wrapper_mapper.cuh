#pragma once
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MapperForwardCUDA(
    const torch::Tensor& p_pos_packed,
    const torch::Tensor& means_packed,
    const torch::Tensor& precision_tril,
    const torch::Tensor& probe_org,
    const torch::Tensor& vec_lat,
    const torch::Tensor& vec_ax,
    const torch::Tensor& vec_ele,
    const int image_height,
    const int image_width,
    const float delta_lat,
    const float delta_ax,
    const float near_mm,
    const float lateral_col_offset,
    const float chi2_threshold,
    const bool sort_by_depth,
    const float diag_beta
);
