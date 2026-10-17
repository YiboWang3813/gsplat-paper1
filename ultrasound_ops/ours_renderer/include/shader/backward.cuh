#pragma once
#include <cuda_runtime.h>

namespace ours_backward_cu {

void RenderOursSliceBackwardCUDA_Impl(
    int n_pixels,
    int n_gauss,
    int H,
    int W,
    int tile_size_px,
    int gauss_rect_is_pixels,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float k_sigma,
    int compositing_mode,
    float fixed_bg_color,
    int mapper_kind,
    const int* tile_range_start,
    const int* tile_range_end,
    const int* sorted_gauss_ids,
    const int* pixel_gauss_offsets,
    const int* pixel_gauss_counts,
    const int* pixel_gauss_indices,
    float* workspace,
    int workspace_floats,
    const int* gauss_rects,
    const float* gauss_means,
    const float* gauss_scales,
    const float* gauss_quats,
    const float* gauss_opacities,
    const float* gauss_intensities,
    const float* pose_probe_to_world,
    const float* grad_pixel_colors,
    float* grad_gauss_means,
    float* grad_gauss_scales,
    float* grad_gauss_quats,
    float* grad_gauss_opacities,
    float* grad_gauss_intensities,
    float* grad_pose_probe_to_world,
    cudaStream_t stream);

}  // namespace ours_backward_cu
