#include "math_utils.cuh"

#include "shader/shared.cuh"
#include "shader/forward.cuh"

__global__ void kernel_precompute_sigma_inv_forward(
    int n_gauss,
    const float* __restrict__ gauss_precision_tril,
    float diag_beta,
    float* __restrict__ sigma_inv_cache
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) return;
    build_sigma_inv_from_tril(gauss_precision_tril + g * 6, diag_beta, sigma_inv_cache + g * 9);
}

__global__ void kernel_pixel_ultragauss_forward(
    const int n_pixels,
    const int n_gauss,
    const float2* __restrict__ gauss_attr,
    const float4* __restrict__ gauss_means,
    const float* __restrict__ gauss_precision_tril,
    const float4* __restrict__ pixel_positions,
    const float* __restrict__ probe_origin,
    const float* __restrict__ probe_vec_elevational,
    const int* __restrict__ pixel_gauss_offsets,
    const int* __restrict__ pixel_gauss_counts,
    const int* __restrict__ pixel_gauss_indices,
    const float bg_alpha,
    const float bg_intensity,
    const float chi2_threshold,
    const float diag_beta,
    float* __restrict__ pixel_image,
    float* __restrict__ pixel_alpha,
    float* __restrict__ pixel_depth,
    const float* __restrict__ sigma_inv_cache
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;

    float3 pixel_pos = make_float3(pixel_positions[p].x, pixel_positions[p].y, pixel_positions[p].z);
    float3 origin = make_float3(probe_origin[0], probe_origin[1], probe_origin[2]);
    float3 elev = make_float3(probe_vec_elevational[0], probe_vec_elevational[1], probe_vec_elevational[2]);

    int start = pixel_gauss_offsets[p];
    int count = pixel_gauss_counts[p];
    float sum_alpha = 0.0f;
    float sum_alpha_c = 0.0f;
    float sum_depth_weighted = 0.0f;
    for (int k = 0; k < count; ++k) {
        int g = pixel_gauss_indices[start + k];
        if (g < 0 || g >= n_gauss) continue;
        const float* tri = gauss_precision_tril + g * 6;
        float sigma_inv_local[9];
        const float* sigma_inv = sigma_inv_cache ? (sigma_inv_cache + g * 9) : sigma_inv_local;
        if (!sigma_inv_cache) {
            build_sigma_inv_from_tril(tri, diag_beta, sigma_inv_local);
        }
        float3 mean = make_float3(gauss_means[g].x, gauss_means[g].y, gauss_means[g].z);
        float q = compute_gaussian_quadratic(pixel_pos, mean, sigma_inv);
        if (q > chi2_threshold) continue;
        float w = __expf(-0.5f * q);
        float alpha_hat = gauss_attr[g].x * w;
        if (alpha_hat <= 1e-8f) continue;
        sum_alpha += alpha_hat;
        sum_alpha_c += alpha_hat * gauss_attr[g].y;
        sum_depth_weighted += alpha_hat * dot_prod(mean - origin, elev);
    }
    float alpha_total = sum_alpha + bg_alpha;
    float intensity_num = sum_alpha_c + bg_alpha * bg_intensity;
    float image = intensity_num / fmaxf(alpha_total, 1e-8f);
    pixel_alpha[p] = alpha_total;
    pixel_image[p] = fmaxf(0.0f, fminf(1.0f, image));
    pixel_depth[p] = sum_depth_weighted / fmaxf(sum_alpha, 1e-8f);

}

void RasterizeGaussiansForwardCUDA_Impl(
    const int n_pixels,
    const int n_gauss,
    const float2* gauss_opacity_intensity,
    const float4* gauss_means,
    const float* gauss_precision_tril,
    const float4* pixel_positions,
    const float* probe_origin,
    const float* probe_vec_elevational,
    const int* pixel_gauss_offsets,
    const int* pixel_gauss_counts,
    const int* pixel_gauss_indices,
    const float bg_alpha,
    const float bg_intensity,
    const float chi2_threshold,
    const float diag_beta,
    float* pixel_image,
    float* pixel_alpha,
    float* pixel_depth,
    float* sigma_inv_cache,
    cudaStream_t stream
) {
    int block = 256;
    int grid_g = (n_gauss + block - 1) / block;
    kernel_precompute_sigma_inv_forward<<<grid_g, block, 0, stream>>>(
        n_gauss, gauss_precision_tril, diag_beta, sigma_inv_cache);
    int grid = (n_pixels + block - 1) / block;
    kernel_pixel_ultragauss_forward<<<grid, block, 0, stream>>>(
        n_pixels,
        n_gauss,
        gauss_opacity_intensity,
        gauss_means,
        gauss_precision_tril,
        pixel_positions,
        probe_origin,
        probe_vec_elevational,
        pixel_gauss_offsets,
        pixel_gauss_counts,
        pixel_gauss_indices,
        bg_alpha,
        bg_intensity,
        chi2_threshold,
        diag_beta,
        pixel_image,
        pixel_alpha,
        pixel_depth,
        sigma_inv_cache
    );
}
