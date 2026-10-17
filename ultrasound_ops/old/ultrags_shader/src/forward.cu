#include <math_constants.h>

#include "math_utils.cuh"
#include "covariance.cuh"

#include "shared.cuh"
#include "forward.cuh"

#define DEBUG_CHANNELS 3
#define COMPOSITING_ALPHA 0
#define COMPOSITING_ADDITIVE_SUM 1
#define COMPOSITING_ADDITIVE_NORM 2

// ------------------------------------------------------------------------
// Kernel: 高斯预处理 — 计算旋转矩阵 R 和协方差逆 SigmaInv
// ------------------------------------------------------------------------
__global__ void kernel_gauss_preprocess_forward(
    const int n_gauss,
    const float4* __restrict__ gauss_quats,
    const float4* __restrict__ gauss_scales_opacities,
    float* __restrict__ gauss_Rs,
    float* __restrict__ gauss_sigma_invs
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) return;

    float4 gauss_quat = gauss_quats[g];
    float gauss_R[9];
    compute_rot_from_quat(gauss_quat, gauss_R);
    mat3_copy(gauss_R, gauss_Rs + 9 * g);

    float4 gauss_scale_opacity = gauss_scales_opacities[g];
    float3 gauss_scale = make_float3(gauss_scale_opacity.x, gauss_scale_opacity.y, gauss_scale_opacity.z);

    float gauss_sigma_inv[9];
    compute_cov3d_from_scale_rot(gauss_scale, gauss_R, gauss_sigma_inv);
    mat3_copy(gauss_sigma_inv, gauss_sigma_invs + 9 * g);
}

// ------------------------------------------------------------------------
// Kernel: 像素 Alpha 累积 (Result-First Alpha Blending)
// ------------------------------------------------------------------------
__global__ void kernel_pixel_alpha_accum_forward(
    const int n_pixels,
    const float4* __restrict__ pixel_positions_packed,
    const int* __restrict__ pixel_gauss_offsets,
    const int* __restrict__ pixel_gauss_counts,
    const int* __restrict__ pixel_gauss_indices,
    const float4* __restrict__ gauss_means_intensities,
    const float4* __restrict__ gauss_scales_opacities,
    const float* __restrict__ gauss_sigma_invs,
    const float* __restrict__ probe_origin,
    const float* __restrict__ probe_vec_elevational,
    float* __restrict__ pixel_accum,
    float* __restrict__ pixel_T,
    float* __restrict__ pixel_contrib,
    float* __restrict__ pixel_depths,
    float* __restrict__ gauss_intermediates,
    const bool fill_gauss_intermediates,
    const int compositing_mode
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;

    int pixel_gauss_start = pixel_gauss_offsets[p];
    int pixel_gauss_count = pixel_gauss_counts[p];

    float4 pixel_pos_packed = pixel_positions_packed[p];
    float3 pixel_pos = make_float3(pixel_pos_packed.x, pixel_pos_packed.y, pixel_pos_packed.z);

    __shared__ float s_probe_origin[3];
    __shared__ float s_probe_vec_elevational[3];
    if (threadIdx.x == 0) {
        vec3_copy(probe_origin, s_probe_origin);
        vec3_copy(probe_vec_elevational, s_probe_vec_elevational);
    }
    __syncthreads();
    float3 probe_origin_val = make_float3(s_probe_origin[0], s_probe_origin[1], s_probe_origin[2]);
    float3 probe_vec_elevational_val = make_float3(s_probe_vec_elevational[0], s_probe_vec_elevational[1], s_probe_vec_elevational[2]);

    float transmittance = 1.0f;
    float color_accum = 0.0f;
    float alpha_accum = 0.0f;
    float depth_accum = 0.0f;
    float additive_num = 0.0f;
    float additive_den = 0.0f;

    for (int k = 0; k < pixel_gauss_count; ++k) {
        int g = pixel_gauss_indices[pixel_gauss_start + k];

        float4 gauss_mean_intensity = gauss_means_intensities[g];
        float3 gauss_mean = make_float3(gauss_mean_intensity.x, gauss_mean_intensity.y, gauss_mean_intensity.z);
        float intensity = gauss_mean_intensity.w;

        float4 gauss_scale_opacity = gauss_scales_opacities[g];
        float opacity = gauss_scale_opacity.w;

        const float* gauss_sigma_inv = gauss_sigma_invs + 9 * g;

        float gaussian_weight = compute_gaussian_weight(pixel_pos, gauss_mean, gauss_sigma_inv);
        float alpha = opacity * gaussian_weight;

        if (alpha < 1.0f / 255.0f) continue;

        float weight = 0.0f;
        float contribution = 0.0f;
        if (compositing_mode == COMPOSITING_ALPHA) {
            weight = alpha * transmittance;
            contribution = intensity * weight;
        } else {
            weight = alpha;
            contribution = intensity * weight;
        }

        float3 gauss_to_probe_offset = gauss_mean - probe_origin_val;
        float depth_elevational = dot_prod(gauss_to_probe_offset, probe_vec_elevational_val);
        depth_accum += weight * depth_elevational;

        if (fill_gauss_intermediates && gauss_intermediates != nullptr) {
            atomicAdd(&gauss_intermediates[g * 3 + 0], contribution);
            atomicAdd(&gauss_intermediates[g * 3 + 1], weight);
            atomicAdd(&gauss_intermediates[g * 3 + 2], 1.0f);
        }

        if (compositing_mode == COMPOSITING_ADDITIVE_NORM) {
            additive_num += contribution;
            additive_den += weight;
        } else {
            color_accum += contribution;
        }
        alpha_accum += alpha;

        float transmittance_next = transmittance * (1.0f - alpha);
        if (transmittance_next < 0.0001f) {
            transmittance = 0.0f;
            if (compositing_mode == COMPOSITING_ALPHA) break;
        } else {
            transmittance = transmittance_next;
        }
    }

    if (compositing_mode == COMPOSITING_ADDITIVE_NORM) {
        color_accum = additive_num / fmaxf(additive_den, 1e-8f);
    }
    pixel_accum[p] = color_accum;
    pixel_T[p] = (compositing_mode == COMPOSITING_ALPHA) ? transmittance : 0.0f;
    pixel_contrib[p] = alpha_accum;
    pixel_depths[p] = depth_accum;
}

// ------------------------------------------------------------------------
// Kernel: 像素颜色与背景合成
// ------------------------------------------------------------------------
__global__ void kernel_pixel_compose_color_forward(
    const int n_pixels,
    const float* __restrict__ pixel_accum,
    const float* __restrict__ pixel_T,
    const float fixed_bg_color,
    float* __restrict__ pixel_colors
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;

    float composed_color = pixel_accum[p] + pixel_T[p] * fixed_bg_color;
    pixel_colors[p] = fmaxf(0.0f, fminf(1.0f, composed_color));
}

// ------------------------------------------------------------------------
// Wrapper Implementation
// ------------------------------------------------------------------------
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
) {
    int blockSize = 256;

    int grid_pre = (n_gauss + blockSize - 1) / blockSize;
    kernel_gauss_preprocess_forward<<<grid_pre, blockSize, 0, stream>>>(
        n_gauss, gauss_quats, gauss_scales_opacities,
        gauss_Rs, gauss_sigma_invs
    );

    if (fill_gauss_intermediates && gauss_intermediates != nullptr) {
        cudaMemsetAsync(gauss_intermediates, 0, n_gauss * 3 * sizeof(float), stream);
    }

    int grid_accum = (n_pixels + blockSize - 1) / blockSize;
    kernel_pixel_alpha_accum_forward<<<grid_accum, blockSize, 0, stream>>>(
        n_pixels, pixel_positions_packed, pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
        gauss_means_intensities, gauss_scales_opacities, gauss_sigma_invs,
        probe_origin, probe_vec_elevational,
        pixel_accum, pixel_T, pixel_contrib, pixel_depths,
        gauss_intermediates, fill_gauss_intermediates, compositing_mode
    );

    kernel_pixel_compose_color_forward<<<grid_accum, blockSize, 0, stream>>>(
        n_pixels, pixel_accum, pixel_T, fixed_bg_color, pixel_colors
    );
}