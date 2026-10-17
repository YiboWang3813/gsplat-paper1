#pragma once
#include <cuda_runtime.h>

void RasterizeGaussiansForwardCUDA_Impl(
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
    float* gauss_Rs,
    float* gauss_sigma_invs,
    float* pixel_accum,
    float* pixel_T,
    float* pixel_contrib,
    float* pixel_depths,
    float* pixel_colors,
    float* gauss_intermediates,
    const bool fill_gauss_intermediates,
    cudaStream_t stream
);

/// probe_origin / probe_vec_elevational: row-major [batch_size, 3].
/// n_pixels_total == batch_size * pixels_per_image; pixel_buffers indexed by gp.
void RasterizeGaussiansForwardCUDA_Impl_Batched(
    const int batch_size,
    const int pixels_per_image,
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
    float* gauss_Rs,
    float* gauss_sigma_invs,
    float* pixel_accum,
    float* pixel_T,
    float* pixel_contrib,
    float* pixel_depths,
    float* pixel_colors,
    float* gauss_intermediates,
    const bool fill_gauss_intermediates,
    cudaStream_t stream
);
