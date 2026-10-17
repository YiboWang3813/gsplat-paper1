#pragma once
#include <cuda_runtime.h>

// 反向实现：从像素强度梯度回传到 Mean / Scale / Quat / Opacity / Intensity
void RasterizeGaussiansBackwardCUDA_Impl(
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
    const float* gauss_Rs,            // (G,9)
    const float* gauss_sigma_invs,    // (G,9)
    const float* dL_dPixelColors,     // (P,)
    float*       dL_dGaussMeans,      // (G,3)
    float*       dL_dGaussScales,     // (G,3)
    float*       dL_dGaussQuats,      // (G,4)
    float*       dL_dGaussOpacities,  // (G,)
    float*       dL_dGaussIntensities,// (G,)
    float*       dL_dGaussSigmaInvs,  // (G,9) 中间缓冲
    cudaStream_t stream
);