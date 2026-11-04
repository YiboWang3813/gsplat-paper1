#pragma once
#include <cuda_runtime.h>

void RasterizeGaussiansBackwardCUDA_Impl(
    const int n_pixels,
    const int n_gauss,
    const float* dL_dImage,
    const float* dL_dAlpha,
    const float* dL_dDepth,
    const float2* gauss_opacity_intensity,
    const float4* gauss_means,
    const float* gauss_precision_tril,
    const float4* pixel_positions,
    const float* probe_origin,
    const float* probe_vec_elevational,
    const int* pixel_gauss_offsets,
    const int* pixel_gauss_counts,
    const int* pixel_gauss_indices,
    const float* sigma_inv_cache,
    const float* pixel_alpha,
    const float bg_alpha,
    const float bg_intensity,
    const float chi2_threshold,
    const float diag_beta,
    float4* dL_dMeans,
    float* dL_dPrecisionTril,
    float2* dL_dOpacityIntensity,
    cudaStream_t stream
);
