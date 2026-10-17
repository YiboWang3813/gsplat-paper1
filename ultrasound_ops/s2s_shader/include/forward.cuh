#pragma once
#include <cuda_runtime.h>

// 仅做几何+强度渲染，不做 alpha compositing / depth
void RasterizeGaussiansForwardCUDA_Impl(
    const int n_pixels,
    const int n_gauss,
    const float* gauss_means,         // (G,3)
    const float* gauss_scales,        // (G,3)
    const float* gauss_quats,         // (G,4)
    const float* gauss_opacities,     // (G,)
    const float* gauss_intensities,   // (G,)
    const float* pixel_positions,     // (P,3)
    const int*   pixel_gauss_offsets, // (P,)
    const int*   pixel_gauss_counts,  // (P,)
    const int*   pixel_gauss_indices, // (Total_Interactions,)
    float*       gauss_Rs,            // (G,9)
    float*       gauss_sigma_invs,    // (G,9)
    float*       pixel_intensities,   // (P,)
    cudaStream_t stream
);