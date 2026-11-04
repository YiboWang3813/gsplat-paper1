#include "math_utils.cuh"

#include "shader/shared.cuh"
#include "shader/backward.cuh"

__global__ void kernel_precompute_sigma_inv(
    int n_gauss,
    const float* __restrict__ gauss_precision_tril,
    float diag_beta,
    float* __restrict__ sigma_inv_cache
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) return;
    build_sigma_inv_from_tril(gauss_precision_tril + g * 6, diag_beta, sigma_inv_cache + g * 9);
}

__global__ void kernel_pixel_ultragauss_backward(
    const int n_pixels,
    const int n_gauss,
    const float* __restrict__ dL_dImage,
    const float* __restrict__ dL_dAlpha,
    const float* __restrict__ dL_dDepth,
    const float2* __restrict__ gauss_attr,
    const float4* __restrict__ gauss_means,
    const float* __restrict__ gauss_precision_tril,
    const float4* __restrict__ pixel_positions,
    const float* __restrict__ probe_origin,
    const float* __restrict__ probe_vec_elevational,
    const int* __restrict__ pixel_gauss_offsets,
    const int* __restrict__ pixel_gauss_counts,
    const int* __restrict__ pixel_gauss_indices,
    const float* __restrict__ sigma_inv_cache,
    const float* __restrict__ pixel_alpha,
    const float bg_alpha,
    const float bg_intensity,
    const float chi2_threshold,
    const float diag_beta,
    float4* __restrict__ dL_dMeans,
    float* __restrict__ dL_dPrecisionTril,
    float2* __restrict__ dL_dAttr
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    float3 pixel_pos = make_float3(pixel_positions[p].x, pixel_positions[p].y, pixel_positions[p].z);
    float3 origin = make_float3(probe_origin[0], probe_origin[1], probe_origin[2]);
    float3 elev = make_float3(probe_vec_elevational[0], probe_vec_elevational[1], probe_vec_elevational[2]);
    int start = pixel_gauss_offsets[p];
    int count = pixel_gauss_counts[p];
    float sum_alpha = pixel_alpha[p] - bg_alpha;
    float sum_alpha_c = 0.0f;
    float sum_depth_weighted = 0.0f;
    for (int k = 0; k < count; ++k) {
        int g = pixel_gauss_indices[start + k];
        if (g < 0 || g >= n_gauss) continue;
        const float* sigma_inv = sigma_inv_cache + g * 9;
        float3 mean = make_float3(gauss_means[g].x, gauss_means[g].y, gauss_means[g].z);
        float q = compute_gaussian_quadratic(pixel_pos, mean, sigma_inv);
        if (q > chi2_threshold) continue;
        float w = __expf(-0.5f * q);
        float ah = gauss_attr[g].x * w;
        sum_alpha_c += ah * gauss_attr[g].y;
        sum_depth_weighted += ah * dot_prod(mean - origin, elev);
    }
    float alpha_total = sum_alpha + bg_alpha;
    float inv_alpha = 1.0f / fmaxf(alpha_total, 1e-8f);
    float num = sum_alpha_c + bg_alpha * bg_intensity;
    float dL_dNum = dL_dImage[p] * inv_alpha;
    float dL_dAlphaTotal = dL_dAlpha[p] - dL_dImage[p] * num * inv_alpha * inv_alpha;
    float inv_sum_alpha = 1.0f / fmaxf(sum_alpha, 1e-8f);
    float dL_dDepthWeighted = dL_dDepth[p] * inv_sum_alpha;
    float dL_dSumAlphaFromDepth = -dL_dDepth[p] * sum_depth_weighted * inv_sum_alpha * inv_sum_alpha;

    for (int k = 0; k < count; ++k) {
        int g = pixel_gauss_indices[start + k];
        if (g < 0 || g >= n_gauss) continue;
        const float* sigma_inv = sigma_inv_cache + g * 9;
        float3 mean = make_float3(gauss_means[g].x, gauss_means[g].y, gauss_means[g].z);
        float opacity = gauss_attr[g].x;
        float intensity = gauss_attr[g].y;
        float q = compute_gaussian_quadratic(pixel_pos, mean, sigma_inv);
        if (q > chi2_threshold) continue;
        float w = __expf(-0.5f * q);
        float ah = opacity * w;

        float dL_dAh = dL_dNum * intensity + dL_dAlphaTotal + dL_dSumAlphaFromDepth;
        float depth_term = dot_prod(mean - origin, elev);
        dL_dAh += dL_dDepthWeighted * depth_term;

        atomicAdd(&dL_dAttr[g].y, dL_dNum * ah);
        atomicAdd(&dL_dAttr[g].x, dL_dAh * w);

        float dL_dw = dL_dAh * opacity;
        float dx = pixel_pos.x - mean.x;
        float dy = pixel_pos.y - mean.y;
        float dz = pixel_pos.z - mean.z;
        float dL_dQ = dL_dw * w * (-0.5f);
        float v0 = sigma_inv[0] * dx + sigma_inv[1] * dy + sigma_inv[2] * dz;
        float v1 = sigma_inv[3] * dx + sigma_inv[4] * dy + sigma_inv[5] * dz;
        float v2 = sigma_inv[6] * dx + sigma_inv[7] * dy + sigma_inv[8] * dz;
        atomicAdd(&dL_dMeans[g].x, dL_dQ * (-2.0f * v0) + dL_dDepthWeighted * ah * elev.x);
        atomicAdd(&dL_dMeans[g].y, dL_dQ * (-2.0f * v1) + dL_dDepthWeighted * ah * elev.y);
        atomicAdd(&dL_dMeans[g].z, dL_dQ * (-2.0f * v2) + dL_dDepthWeighted * ah * elev.z);

        float dS[9];
        dS[0] = dL_dQ * dx * dx; dS[1] = dL_dQ * dx * dy; dS[2] = dL_dQ * dx * dz;
        dS[3] = dL_dQ * dy * dx; dS[4] = dL_dQ * dy * dy; dS[5] = dL_dQ * dy * dz;
        dS[6] = dL_dQ * dz * dx; dS[7] = dL_dQ * dz * dy; dS[8] = dL_dQ * dz * dz;

        const float* tri = gauss_precision_tril + g * 6;
        float l11 = tri[0] * tri[0] + diag_beta;
        float l12 = tri[1];
        float l13 = tri[2];
        float l22 = tri[3] * tri[3] + diag_beta;
        float l23 = tri[4];
        float l33 = tri[5] * tri[5] + diag_beta;

        float dl11 = dS[0] * (2.0f * l11) + (dS[1] + dS[3]) * l12 + (dS[2] + dS[6]) * l13;
        float dl12 = (dS[1] + dS[3]) * l11 + dS[4] * (2.0f * l12) + (dS[5] + dS[7]) * l13;
        float dl13 = (dS[2] + dS[6]) * l11 + (dS[5] + dS[7]) * l12 + dS[8] * (2.0f * l13);
        float dl22 = dS[4] * (2.0f * l22) + (dS[5] + dS[7]) * l23;
        float dl23 = (dS[5] + dS[7]) * l22 + dS[8] * (2.0f * l23);
        float dl33 = dS[8] * (2.0f * l33);

        atomicAdd(dL_dPrecisionTril + g * 6 + 0, dl11 * (2.0f * tri[0]));
        atomicAdd(dL_dPrecisionTril + g * 6 + 1, dl12);
        atomicAdd(dL_dPrecisionTril + g * 6 + 2, dl13);
        atomicAdd(dL_dPrecisionTril + g * 6 + 3, dl22 * (2.0f * tri[3]));
        atomicAdd(dL_dPrecisionTril + g * 6 + 4, dl23);
        atomicAdd(dL_dPrecisionTril + g * 6 + 5, dl33 * (2.0f * tri[5]));
    }
}

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
) {
    (void)n_gauss;
    int block = 256;
    int grid = (n_pixels + block - 1) / block;
    kernel_pixel_ultragauss_backward<<<grid, block, 0, stream>>>(
        n_pixels,
        n_gauss,
        dL_dImage,
        dL_dAlpha,
        dL_dDepth,
        gauss_opacity_intensity,
        gauss_means,
        gauss_precision_tril,
        pixel_positions,
        probe_origin,
        probe_vec_elevational,
        pixel_gauss_offsets,
        pixel_gauss_counts,
        pixel_gauss_indices,
        sigma_inv_cache,
        pixel_alpha,
        bg_alpha,
        bg_intensity,
        chi2_threshold,
        diag_beta,
        dL_dMeans,
        dL_dPrecisionTril,
        dL_dOpacityIntensity
    );
}
