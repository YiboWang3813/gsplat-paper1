#pragma once
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MapperForwardCUDA(
    const torch::Tensor& p_pos_packed,
    const torch::Tensor& means_packed,
    const torch::Tensor& quats,
    const torch::Tensor& scales_packed,
    const torch::Tensor& probe_org,
    const torch::Tensor& vec_ele,
    const torch::Tensor& vec_lat,
    const torch::Tensor& vec_ax,
    const int image_height,
    const int image_width,
    const float delta_lat,
    const float delta_ax,
    const float slab_half_thickness_mm,
    const float near_mm,
    const float k_sigma,
    const float lateral_col_offset,
    const bool sort_by_depth
);

/// Batched mapper: p_pos_packed [B, H*W, 4], probe_org / vec_* [B, 3].
/// Returns pixel_gauss_* over flattened pixels (B*H*W), and pair_batch_offsets [B+1] int64.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MapperForwardCUDABatched(
    const torch::Tensor& p_pos_packed,
    const torch::Tensor& means_packed,
    const torch::Tensor& quats,
    const torch::Tensor& scales_packed,
    const torch::Tensor& probe_org,
    const torch::Tensor& vec_ele,
    const torch::Tensor& vec_lat,
    const torch::Tensor& vec_ax,
    const int image_height,
    const int image_width,
    const float delta_lat,
    const float delta_ax,
    const float slab_half_thickness_mm,
    const float near_mm,
    const float k_sigma,
    const float lateral_col_offset,
    const bool sort_by_depth,
    const torch::Tensor& active_gauss_mask_batched,
    const bool return_pair_batch_offsets
);
