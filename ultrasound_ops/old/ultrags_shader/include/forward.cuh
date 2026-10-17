#pragma once
#include <cuda_runtime.h>

// 参数顺序与 wrapper 一致：标量 -> 只读高斯 -> 只读像素/探头 -> 只读映射 -> 可写中间/输出
void RasterizeGaussiansForwardCUDA_Impl(
    const int n_pixels,
    const int n_gauss,
    const float fixed_bg_color,
    const int compositing_mode,
    const float4* gauss_means_intensities,   // (G, 4) .xyz=mean, .w=intensity
    const float4* gauss_scales_opacities,    // (G, 4) .xyz=scale, .w=opacity
    const float4* gauss_quats,               // (G, 4)
    const float4* pixel_positions_packed,    // (P, 4) .xyz=pos, .w=pad
    const float* probe_origin,                // (3,)
    const float* probe_vec_elevational,       // (3,)
    const int* pixel_gauss_offsets,           // (P,)
    const int* pixel_gauss_counts,            // (P,)
    const int* pixel_gauss_indices,           // (Total_Interactions,)
    float* gauss_Rs,                         // (G, 9)
    float* gauss_sigma_invs,                 // (G, 9)
    float* pixel_accum,
    float* pixel_T,
    float* pixel_contrib,
    float* pixel_depths,
    float* pixel_colors,
    float* gauss_intermediates,               // (G, 3) optional, may be nullptr
    const bool fill_gauss_intermediates,
    cudaStream_t stream
);
