#pragma once
#include <cuda_runtime.h>

// 参数顺序与 wrapper / __init__.py 一致；内部仍需要 dL_dGaussSigmaInvs 等，仅不返回给 Python
void RasterizeGaussiansBackwardCUDA_Impl(
    const int n_pixels,
    const int n_gauss,
    const float fixed_bg_color,
    const int compositing_mode,
    const float4* gauss_means_intensities,
    const float4* gauss_scales_opacities,
    const float4* gauss_quats,
    const float4* pixel_positions_packed,
    const float* probe_origin,
    const float* probe_vec_elevational,
    const int* pixel_gauss_offsets,
    const int* pixel_gauss_counts,
    const int* pixel_gauss_indices,
    const float* gauss_Rs,
    const float* gauss_sigma_invs,
    const float* pixel_accum,
    const float* pixel_T,
    const float* dL_dPixelColors,
    const float* dL_dPixelDepths,
    float* dL_dGaussSigmaInvs,
    float* dL_dPixelAccum,
    float* dL_dPixelT,
    float4* dL_dGaussMeansIntensities,
    float4* dL_dGaussQuats,
    float4* dL_dGaussScalesOpacities,
    cudaStream_t stream
);
