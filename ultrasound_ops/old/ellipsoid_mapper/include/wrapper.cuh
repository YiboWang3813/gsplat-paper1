#pragma once
#include <torch/extension.h>

// ============================================================================
// EllipsoidMapper (CUDA)
//
// Returns the sparse pixel->gauss mapping in the same format used across
// other ops (e.g. shaders):
//   - pixel_gauss_counts  : (P,) int32
//   - pixel_gauss_offsets : (P,) int32 (exclusive prefix sum of counts)
//   - pixel_gauss_indices : (TotalPairs,) int32 (concatenated gauss ids)
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launch_ellipsoid_mapper(
    // 几何输入
    const torch::Tensor& p_pos_packed, 
    const torch::Tensor& means_packed,     
    const torch::Tensor& quats,
    const torch::Tensor& scales_packed,
    // 探头参数
    const torch::Tensor& probe_org,    
    const torch::Tensor& vec_ele,    
    const torch::Tensor& vec_lat,    
    const torch::Tensor& vec_ax,      
    // 配置
    const int   image_height,
    const int   image_width,
    const float delta_lat,       
    const float delta_ax,
    const float slab_half_thickness_mm,
    const float near_mm,
    const float k_sigma,
    const float lateral_col_offset,
    const bool  sort_by_depth
);
