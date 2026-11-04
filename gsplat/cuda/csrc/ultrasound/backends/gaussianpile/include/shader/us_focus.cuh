#pragma once
#include <cuda_runtime.h>

namespace us_focus_backward_cu {

void RenderOursSliceBackwardUsFocus_Impl(
    int n_gauss,
    int H,
    int W,
    int Tw,
    int Th,
    int tile_size_px,
    float k_sigma,
    const int* gauss_rects,
    const int* tile_range_start,
    const int* tile_range_end,
    const int* sorted_gauss_ids,
    const float* gauss_means,
    const float* gauss_scales,
    const float* gauss_quats,
    const float* gauss_opacities,
    const float* gauss_intensities,
    const float* pose_probe_to_world,
    const float* slice_grid_pack,
    const float* us_focus_params,
    const float* grad_pixel_colors,
    float* workspace,
    int workspace_numel,
    bool compute_pose_grad,
    float* grad_gauss_means,
    float* grad_gauss_scales,
    float* grad_gauss_quats,
    float* grad_gauss_opacities,
    float* grad_gauss_intensities,
    float* grad_pose_probe_to_world,
    cudaStream_t stream);

}  // namespace us_focus_backward_cu
