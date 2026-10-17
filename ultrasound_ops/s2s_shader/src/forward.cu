#include <math_constants.h>

#include "math_utils.cuh"
#include "covariance.cuh"
#include "shared.cuh"
#include "forward.cuh"

// ------------------------------------------------------------------------
// Kernel: 高斯预处理 — 计算旋转矩阵 R 和协方差逆 SigmaInv
// ------------------------------------------------------------------------
__global__ void kernel_gauss_preprocess_forward(
    const int n_gauss,
    const float* __restrict__ gauss_quats,   // (G,4)
    const float* __restrict__ gauss_scales,  // (G,3)
    float*       __restrict__ gauss_Rs,      // (G,9)
    float*       __restrict__ gauss_sigma_invs // (G,9)
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) return;

    // 读 quat
    const float* quat_ptr = gauss_quats + g * 4;
    float4 gauss_quat = make_float4(quat_ptr[0], quat_ptr[1], quat_ptr[2], quat_ptr[3]);

    float gauss_R[9];
    compute_rot_from_quat(gauss_quat, gauss_R);
    mat3_copy(gauss_R, gauss_Rs + 9 * g);

    // 读 scale
    const float* scale_ptr = gauss_scales + g * 3;
    float3 gauss_scale = make_float3(scale_ptr[0], scale_ptr[1], scale_ptr[2]);

    float gauss_sigma_inv[9];
    compute_cov3d_from_scale_rot(gauss_scale, gauss_R, gauss_sigma_inv);
    mat3_copy(gauss_sigma_inv, gauss_sigma_invs + 9 * g);
}

// ------------------------------------------------------------------------
// Kernel: 像素强度累积（无 alpha compositing，仅线性叠加）
// ------------------------------------------------------------------------
__global__ void kernel_pixel_intensity_accum_forward(
    const int n_pixels,
    const float* __restrict__ pixel_positions,     // (P,3)
    const int*   __restrict__ pixel_gauss_offsets, // (P,)
    const int*   __restrict__ pixel_gauss_counts,  // (P,)
    const int*   __restrict__ pixel_gauss_indices, // (Total_Interactions,)
    const float* __restrict__ gauss_means,         // (G,3)
    const float* __restrict__ gauss_sigma_invs,    // (G,9)
    const float* __restrict__ gauss_opacities,     // (G,)
    const float* __restrict__ gauss_intensities,   // (G,)
    float*       __restrict__ pixel_intensities    // (P,)
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;

    const float px = pixel_positions[p * 3 + 0];
    const float py = pixel_positions[p * 3 + 1];
    const float pz = pixel_positions[p * 3 + 2];
    float3 pixel_pos = make_float3(px, py, pz);

    const int start = pixel_gauss_offsets[p];
    const int count = pixel_gauss_counts[p];

    float accum = 0.0f;

    for (int k = 0; k < count; ++k) {
        const int idx = start + k;
        const int g   = pixel_gauss_indices[idx];

        const float* mean_ptr = gauss_means + g * 3;
        float3 gauss_mean = make_float3(mean_ptr[0], mean_ptr[1], mean_ptr[2]);

        const float* sigma_inv = gauss_sigma_invs + g * 9;
        const float opacity    = gauss_opacities[g];
        const float intensity  = gauss_intensities[g];

        float gaussian_weight = compute_gaussian_weight(pixel_pos, gauss_mean, sigma_inv);
        float alpha = opacity * gaussian_weight;

        accum += alpha * intensity;
    }

    pixel_intensities[p] = accum;
}

// ------------------------------------------------------------------------
// Host Entry
// ------------------------------------------------------------------------
void RasterizeGaussiansForwardCUDA_Impl(
    const int n_pixels,
    const int n_gauss,
    const float* gauss_means,
    const float* gauss_scales,
    const float* gauss_quats,
    const float* gauss_opacities,
    const float* gauss_intensities,
    const float* pixel_positions,
    const int*   pixel_gauss_offsets,
    const int*   pixel_gauss_counts,
    const int*   pixel_gauss_indices,
    float*       gauss_Rs,
    float*       gauss_sigma_invs,
    float*       pixel_intensities,
    cudaStream_t stream
) {
    const int blockSize = 256;

    // 1. 高斯预处理
    int grid_pre = (n_gauss + blockSize - 1) / blockSize;
    kernel_gauss_preprocess_forward<<<grid_pre, blockSize, 0, stream>>>(
        n_gauss, gauss_quats, gauss_scales,
        gauss_Rs, gauss_sigma_invs
    );

    // 2. 像素强度累积
    int grid_pix = (n_pixels + blockSize - 1) / blockSize;
    kernel_pixel_intensity_accum_forward<<<grid_pix, blockSize, 0, stream>>>(
        n_pixels,
        pixel_positions,
        pixel_gauss_offsets,
        pixel_gauss_counts,
        pixel_gauss_indices,
        gauss_means,
        gauss_sigma_invs,
        gauss_opacities,
        gauss_intensities,
        pixel_intensities
    );
}