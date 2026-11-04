#pragma once
#include <cuda_runtime.h>

#include "shader/plane_cond2d_device.cuh"
#include "shader/shared.cuh"
#include "shader/slice_grid_geometry.cuh"

/** Workspace stride per Gaussian for plane_cond2d alpha backward (matches OURS_ALPHA_WSP_STRIDE). */
#define PLANE_COND2D_WSP_STRIDE 14
/** Packed workspace layout (floats per Gaussian). */
#define PLANE_COND2D_WSP_G_MU2_LAT 0
#define PLANE_COND2D_WSP_G_MU2_AX 1
#define PLANE_COND2D_WSP_G_MU_Z 2
#define PLANE_COND2D_WSP_G_I00 3
#define PLANE_COND2D_WSP_G_I01 4
#define PLANE_COND2D_WSP_G_SZZ 5
#define PLANE_COND2D_WSP_G_I11 6
#define PLANE_COND2D_WSP_G_OPACITY 12
#define PLANE_COND2D_WSP_G_INTENSITY 13

__device__ void ours_plane_cond2d_bwd_wire(
    int col,
    int row,
    const float* __restrict__ slice_grid_pack,
    float mu_lat_2d,
    float mu_ax_2d,
    float s2d_inv00,
    float s2d_inv01,
    float s2d_inv11,
    float mu_z,
    float s_zz,
    float o,
    float Ir,
    float dL_dalpha,
    float dL_dI,
    float* __restrict__ wrow);

void RenderOursSliceBackwardPlaneCond2D_Impl(
    int n_pixels,
    int n_gauss,
    int H,
    int W,
    int tile_size_px,
    int gauss_rect_is_pixels,
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
    const float* slice_grid_pack,
    const float* grad_pixel_colors,
    float* grad_gauss_means,
    float* grad_gauss_scales,
    float* grad_gauss_quats,
    float* grad_gauss_opacities,
    float* grad_gauss_intensities,
    float* grad_pose_probe_to_world,
    cudaStream_t stream);
