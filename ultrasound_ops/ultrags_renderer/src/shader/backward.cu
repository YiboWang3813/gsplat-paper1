#include <math_constants.h>

#include "math_utils.cuh"
#include "covariance.cuh"

#include "shader/shared.cuh"
#include "shader/backward.cuh"

#define MAX_G 256
#define COMPOSITING_ALPHA 0
#define COMPOSITING_ADDITIVE_SUM 1
#define COMPOSITING_ADDITIVE_NORM 2

__global__ void kernel_gauss_preprocess_backward(
    const int n_gauss,
    const float4* __restrict__ gauss_quats,
    const float4* __restrict__ gauss_scales_opacities,
    const float* __restrict__ gauss_Rs,
    const float* __restrict__ dL_dGaussSigmaInvs,
    float4* __restrict__ dL_dGaussQuats,
    float4* __restrict__ dL_dGaussScalesOpacities
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) return;

    float4 gauss_quat = gauss_quats[g];
    float4 gauss_scale_opacity = gauss_scales_opacities[g];
    float3 gauss_scale = make_float3(gauss_scale_opacity.x, gauss_scale_opacity.y, gauss_scale_opacity.z);

    const float* gauss_R_ptr = gauss_Rs + 9 * g;
    float gauss_R[9];
    mat3_copy(gauss_R_ptr, gauss_R);

    const float* dL_dGaussSigmaInv = dL_dGaussSigmaInvs + 9 * g;

    float dL_dR[9];
    mat3_zero(dL_dR);
    float3 dL_dScale = {0.0f, 0.0f, 0.0f};

    transform_grads_sigma_to_R_S(gauss_scale, gauss_R, dL_dGaussSigmaInv, dL_dR, dL_dScale);

    float4 dL_dGaussQuat;
    transform_grads_R_to_Quat(gauss_quat, dL_dR, dL_dGaussQuat);

    dL_dGaussQuats[g] = dL_dGaussQuat;

    float* dL_dGaussScalesOpacity_ptr = (float*)dL_dGaussScalesOpacities + 4 * g;
    atomicAdd(dL_dGaussScalesOpacity_ptr + 0, dL_dScale.x);
    atomicAdd(dL_dGaussScalesOpacity_ptr + 1, dL_dScale.y);
    atomicAdd(dL_dGaussScalesOpacity_ptr + 2, dL_dScale.z);
}

__global__ void kernel_pixel_alpha_accum_backward(
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
    const float* __restrict__ dL_dPixelAccum,
    const float* __restrict__ dL_dPixelT,
    const float* __restrict__ dL_dPixelDepths,
    float4* __restrict__ dL_dGaussMeansIntensities,
    float4* __restrict__ dL_dGaussScalesOpacities,
    float* __restrict__ dL_dGaussSigmaInvs,
    const int compositing_mode
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;

    int pixel_gauss_start = pixel_gauss_offsets[p];
    int pixel_gauss_count = pixel_gauss_counts[p];
    if (pixel_gauss_count == 0) return;

    float3 probe_origin_vec = make_float3(probe_origin[0], probe_origin[1], probe_origin[2]);
    float3 vec_elevational = make_float3(probe_vec_elevational[0], probe_vec_elevational[1], probe_vec_elevational[2]);

    float dL_dPixelAccum_final = dL_dPixelAccum[p];
    float dL_dPixelT_final = dL_dPixelT[p];
    float dL_dPixelDepth_final = dL_dPixelDepths[p];

    int num_gauss_capped = min(pixel_gauss_count, MAX_G);

    float cache_alpha[MAX_G];
    float cache_color[MAX_G];
    float cache_transmittance[MAX_G];
    int cache_gauss_idx[MAX_G];

    float4 pixel_pos_packed = pixel_positions_packed[p];
    float3 pixel_pos = make_float3(pixel_pos_packed.x, pixel_pos_packed.y, pixel_pos_packed.z);

    float transmittance = 1.0f;
    int accepted_cnt = 0;
    for (int k = 0; k < num_gauss_capped; ++k) {
        int g = pixel_gauss_indices[pixel_gauss_start + k];

        float4 gauss_mean_intensity = gauss_means_intensities[g];
        float3 gauss_mean = make_float3(gauss_mean_intensity.x, gauss_mean_intensity.y, gauss_mean_intensity.z);
        float color = gauss_mean_intensity.w;

        float4 gauss_scale_opacity = gauss_scales_opacities[g];
        float opacity = gauss_scale_opacity.w;
        const float* gauss_sigma_inv = gauss_sigma_invs + 9 * g;

        float gaussian_weight = compute_gaussian_weight(pixel_pos, gauss_mean, gauss_sigma_inv);
        float alpha = opacity * gaussian_weight;
        if (alpha < 1.0f / 255.0f) continue;

        cache_gauss_idx[accepted_cnt] = g;
        cache_alpha[accepted_cnt] = alpha;
        cache_color[accepted_cnt] = color;
        cache_transmittance[accepted_cnt] = transmittance;
        accepted_cnt++;

        transmittance *= (1.0f - alpha);
        if (compositing_mode == COMPOSITING_ALPHA && transmittance < 0.0001f) break;
    }

    if (compositing_mode == COMPOSITING_ADDITIVE_SUM || compositing_mode == COMPOSITING_ADDITIVE_NORM) {
        float num = 0.0f, den = 0.0f;
        for (int k = 0; k < accepted_cnt; ++k) {
            float alpha = cache_alpha[k];
            float color = cache_color[k];
            num += color * alpha;
            den += alpha;
        }
        for (int k = accepted_cnt - 1; k >= 0; --k) {
            int g = cache_gauss_idx[k];
            float alpha = cache_alpha[k];
            float color = cache_color[k];
            float4 gauss_mean_intensity = gauss_means_intensities[g];
            float3 gauss_mean = make_float3(gauss_mean_intensity.x, gauss_mean_intensity.y, gauss_mean_intensity.z);
            const float* gauss_sigma_inv = gauss_sigma_invs + 9 * g;
            float gaussian_weight = compute_gaussian_weight(pixel_pos, gauss_mean, gauss_sigma_inv);
            float opacity = gauss_scales_opacities[g].w;
            float3 gauss_to_probe_offset = make_float3(gauss_mean.x - probe_origin_vec.x, gauss_mean.y - probe_origin_vec.y, gauss_mean.z - probe_origin_vec.z);
            float depth_elevational = gauss_to_probe_offset.x * vec_elevational.x + gauss_to_probe_offset.y * vec_elevational.y + gauss_to_probe_offset.z * vec_elevational.z;

            float dL_dI = 0.0f;
            float dL_dAlpha = 0.0f;
            if (compositing_mode == COMPOSITING_ADDITIVE_SUM) {
                dL_dI = dL_dPixelAccum_final * alpha;
                dL_dAlpha = dL_dPixelAccum_final * color + dL_dPixelDepth_final * depth_elevational;
            } else {
                float den_safe = fmaxf(den, 1e-8f);
                dL_dI = dL_dPixelAccum_final * (alpha / den_safe);
                dL_dAlpha = dL_dPixelAccum_final * ((color * den_safe - num) / (den_safe * den_safe))
                    + dL_dPixelDepth_final * depth_elevational;
            }

            float dL_dOpacity = dL_dAlpha * gaussian_weight;
            float dL_dGaussian_weight = dL_dAlpha * opacity;

            atomicAdd(&dL_dGaussMeansIntensities[g].w, dL_dI);
            atomicAdd(&dL_dGaussScalesOpacities[g].w, dL_dOpacity);

            float dL_dDepth_elevational = dL_dPixelDepth_final * alpha;
            atomicAdd(&dL_dGaussMeansIntensities[g].x, dL_dDepth_elevational * vec_elevational.x);
            atomicAdd(&dL_dGaussMeansIntensities[g].y, dL_dDepth_elevational * vec_elevational.y);
            atomicAdd(&dL_dGaussMeansIntensities[g].z, dL_dDepth_elevational * vec_elevational.z);

            float sigma_inv_grad_scale = dL_dGaussian_weight * gaussian_weight * (-0.5f);
            float dx = pixel_pos_packed.x - gauss_mean.x;
            float dy = pixel_pos_packed.y - gauss_mean.y;
            float dz = pixel_pos_packed.z - gauss_mean.z;
            float* dL_dGaussSigmaInv_ptr = (float*)dL_dGaussSigmaInvs + 9 * g;
            atomicAdd(dL_dGaussSigmaInv_ptr + 0, sigma_inv_grad_scale * dx * dx);
            atomicAdd(dL_dGaussSigmaInv_ptr + 1, sigma_inv_grad_scale * dx * dy);
            atomicAdd(dL_dGaussSigmaInv_ptr + 2, sigma_inv_grad_scale * dx * dz);
            atomicAdd(dL_dGaussSigmaInv_ptr + 3, sigma_inv_grad_scale * dy * dx);
            atomicAdd(dL_dGaussSigmaInv_ptr + 4, sigma_inv_grad_scale * dy * dy);
            atomicAdd(dL_dGaussSigmaInv_ptr + 5, sigma_inv_grad_scale * dy * dz);
            atomicAdd(dL_dGaussSigmaInv_ptr + 6, sigma_inv_grad_scale * dz * dx);
            atomicAdd(dL_dGaussSigmaInv_ptr + 7, sigma_inv_grad_scale * dz * dy);
            atomicAdd(dL_dGaussSigmaInv_ptr + 8, sigma_inv_grad_scale * dz * dz);

            float dG_dMu_x = gauss_sigma_inv[0] * dx + gauss_sigma_inv[1] * dy + gauss_sigma_inv[2] * dz;
            float dG_dMu_y = gauss_sigma_inv[3] * dx + gauss_sigma_inv[4] * dy + gauss_sigma_inv[5] * dz;
            float dG_dMu_z = gauss_sigma_inv[6] * dx + gauss_sigma_inv[7] * dy + gauss_sigma_inv[8] * dz;
            float dL_dMu_factor = dL_dGaussian_weight * gaussian_weight;
            atomicAdd(&dL_dGaussMeansIntensities[g].x, dL_dMu_factor * dG_dMu_x);
            atomicAdd(&dL_dGaussMeansIntensities[g].y, dL_dMu_factor * dG_dMu_y);
            atomicAdd(&dL_dGaussMeansIntensities[g].z, dL_dMu_factor * dG_dMu_z);
        }
        return;
    }

    float dL_dT_running = dL_dPixelT_final;
    for (int k = accepted_cnt - 1; k >= 0; --k) {
        int g = cache_gauss_idx[k];
        float alpha = cache_alpha[k];
        float color = cache_color[k];
        float transmittance_before = cache_transmittance[k];

        float4 gauss_mean_intensity = gauss_means_intensities[g];
        float3 gauss_mean = make_float3(gauss_mean_intensity.x, gauss_mean_intensity.y, gauss_mean_intensity.z);
        const float* gauss_sigma_inv = gauss_sigma_invs + 9 * g;
        float gaussian_weight = compute_gaussian_weight(pixel_pos, gauss_mean, gauss_sigma_inv);
        float opacity = gauss_scales_opacities[g].w;

        float3 gauss_to_probe_offset = make_float3(gauss_mean.x - probe_origin_vec.x, gauss_mean.y - probe_origin_vec.y, gauss_mean.z - probe_origin_vec.z);
        float depth_elevational = gauss_to_probe_offset.x * vec_elevational.x + gauss_to_probe_offset.y * vec_elevational.y + gauss_to_probe_offset.z * vec_elevational.z;

        float dL_dColor = dL_dPixelAccum_final * alpha * transmittance_before;
        float dL_dDepth_elevational = dL_dPixelDepth_final * alpha * transmittance_before;

        float dL_dAlpha_from_color = dL_dPixelAccum_final * color * transmittance_before;
        float dL_dAlpha_from_depth = dL_dPixelDepth_final * depth_elevational * transmittance_before;
        float dL_dAlpha_from_T_next = dL_dT_running * (-transmittance_before);
        float dL_dAlpha = dL_dAlpha_from_color + dL_dAlpha_from_depth + dL_dAlpha_from_T_next;

        float dL_dT_from_contrib = dL_dPixelAccum_final * color * alpha;
        float dL_dT_from_depth = dL_dPixelDepth_final * depth_elevational * alpha;
        float dL_dT_from_next = dL_dT_running * (1.0f - alpha);
        dL_dT_running = dL_dT_from_contrib + dL_dT_from_depth + dL_dT_from_next;

        float dL_dOpacity = dL_dAlpha * gaussian_weight;
        float dL_dGaussian_weight = dL_dAlpha * opacity;

        atomicAdd(&dL_dGaussMeansIntensities[g].w, dL_dColor);
        atomicAdd(&dL_dGaussScalesOpacities[g].w, dL_dOpacity);
        atomicAdd(&dL_dGaussMeansIntensities[g].x, dL_dDepth_elevational * vec_elevational.x);
        atomicAdd(&dL_dGaussMeansIntensities[g].y, dL_dDepth_elevational * vec_elevational.y);
        atomicAdd(&dL_dGaussMeansIntensities[g].z, dL_dDepth_elevational * vec_elevational.z);

        float sigma_inv_grad_scale = dL_dGaussian_weight * gaussian_weight * (-0.5f);
        float dx = pixel_pos_packed.x - gauss_mean.x;
        float dy = pixel_pos_packed.y - gauss_mean.y;
        float dz = pixel_pos_packed.z - gauss_mean.z;

        float* dL_dGaussSigmaInv_ptr = (float*)dL_dGaussSigmaInvs + 9 * g;
        atomicAdd(dL_dGaussSigmaInv_ptr + 0, sigma_inv_grad_scale * dx * dx);
        atomicAdd(dL_dGaussSigmaInv_ptr + 1, sigma_inv_grad_scale * dx * dy);
        atomicAdd(dL_dGaussSigmaInv_ptr + 2, sigma_inv_grad_scale * dx * dz);
        atomicAdd(dL_dGaussSigmaInv_ptr + 3, sigma_inv_grad_scale * dy * dx);
        atomicAdd(dL_dGaussSigmaInv_ptr + 4, sigma_inv_grad_scale * dy * dy);
        atomicAdd(dL_dGaussSigmaInv_ptr + 5, sigma_inv_grad_scale * dy * dz);
        atomicAdd(dL_dGaussSigmaInv_ptr + 6, sigma_inv_grad_scale * dz * dx);
        atomicAdd(dL_dGaussSigmaInv_ptr + 7, sigma_inv_grad_scale * dz * dy);
        atomicAdd(dL_dGaussSigmaInv_ptr + 8, sigma_inv_grad_scale * dz * dz);

        float dG_dMu_x = gauss_sigma_inv[0] * dx + gauss_sigma_inv[1] * dy + gauss_sigma_inv[2] * dz;
        float dG_dMu_y = gauss_sigma_inv[3] * dx + gauss_sigma_inv[4] * dy + gauss_sigma_inv[5] * dz;
        float dG_dMu_z = gauss_sigma_inv[6] * dx + gauss_sigma_inv[7] * dy + gauss_sigma_inv[8] * dz;
        float dL_dMu_factor = dL_dGaussian_weight * gaussian_weight;

        atomicAdd(&dL_dGaussMeansIntensities[g].x, dL_dMu_factor * dG_dMu_x);
        atomicAdd(&dL_dGaussMeansIntensities[g].y, dL_dMu_factor * dG_dMu_y);
        atomicAdd(&dL_dGaussMeansIntensities[g].z, dL_dMu_factor * dG_dMu_z);
    }
}

__global__ void kernel_pixel_alpha_accum_backward_batched(
    const int n_pixels_total,
    const int pixels_per_image,
    const int batch_size,
    const float4* __restrict__ pixel_positions_packed,
    const int* __restrict__ pixel_gauss_offsets,
    const int* __restrict__ pixel_gauss_counts,
    const int* __restrict__ pixel_gauss_indices,
    const float4* __restrict__ gauss_means_intensities,
    const float4* __restrict__ gauss_scales_opacities,
    const float* __restrict__ gauss_sigma_invs,
    const float* __restrict__ probe_origin,
    const float* __restrict__ probe_vec_elevational,
    const float* __restrict__ dL_dPixelAccum,
    const float* __restrict__ dL_dPixelT,
    const float* __restrict__ dL_dPixelDepths,
    float4* __restrict__ dL_dGaussMeansIntensities,
    float4* __restrict__ dL_dGaussScalesOpacities,
    float* __restrict__ dL_dGaussSigmaInvs,
    const int compositing_mode
) {
    int gp = blockIdx.x * blockDim.x + threadIdx.x;
    if (gp >= n_pixels_total) return;

    int b = gp / pixels_per_image;
    (void)batch_size;

    int pixel_gauss_start = pixel_gauss_offsets[gp];
    int pixel_gauss_count = pixel_gauss_counts[gp];
    if (pixel_gauss_count == 0) return;

    const float* org = probe_origin + b * 3;
    const float* pe = probe_vec_elevational + b * 3;
    float3 probe_origin_vec = make_float3(org[0], org[1], org[2]);
    float3 vec_elevational = make_float3(pe[0], pe[1], pe[2]);

    float dL_dPixelAccum_final = dL_dPixelAccum[gp];
    float dL_dPixelT_final = dL_dPixelT[gp];
    float dL_dPixelDepth_final = dL_dPixelDepths[gp];

    int num_gauss_capped = min(pixel_gauss_count, MAX_G);

    float cache_alpha[MAX_G];
    float cache_color[MAX_G];
    float cache_transmittance[MAX_G];
    int cache_gauss_idx[MAX_G];

    float4 pixel_pos_packed = pixel_positions_packed[gp];
    float3 pixel_pos = make_float3(pixel_pos_packed.x, pixel_pos_packed.y, pixel_pos_packed.z);

    float transmittance = 1.0f;
    int accepted_cnt = 0;
    for (int k = 0; k < num_gauss_capped; ++k) {
        int g = pixel_gauss_indices[pixel_gauss_start + k];

        float4 gauss_mean_intensity = gauss_means_intensities[g];
        float3 gauss_mean = make_float3(gauss_mean_intensity.x, gauss_mean_intensity.y, gauss_mean_intensity.z);
        float color = gauss_mean_intensity.w;

        float4 gauss_scale_opacity = gauss_scales_opacities[g];
        float opacity = gauss_scale_opacity.w;
        const float* gauss_sigma_inv = gauss_sigma_invs + 9 * g;

        float gaussian_weight = compute_gaussian_weight(pixel_pos, gauss_mean, gauss_sigma_inv);
        float alpha = opacity * gaussian_weight;
        if (alpha < 1.0f / 255.0f) continue;

        cache_gauss_idx[accepted_cnt] = g;
        cache_alpha[accepted_cnt] = alpha;
        cache_color[accepted_cnt] = color;
        cache_transmittance[accepted_cnt] = transmittance;
        accepted_cnt++;

        transmittance *= (1.0f - alpha);
        if (compositing_mode == COMPOSITING_ALPHA && transmittance < 0.0001f) break;
    }

    if (compositing_mode == COMPOSITING_ADDITIVE_SUM || compositing_mode == COMPOSITING_ADDITIVE_NORM) {
        float num = 0.0f, den = 0.0f;
        for (int k = 0; k < accepted_cnt; ++k) {
            float alpha = cache_alpha[k];
            float color = cache_color[k];
            num += color * alpha;
            den += alpha;
        }
        for (int k = accepted_cnt - 1; k >= 0; --k) {
            int g = cache_gauss_idx[k];
            float alpha = cache_alpha[k];
            float color = cache_color[k];
            float4 gauss_mean_intensity = gauss_means_intensities[g];
            float3 gauss_mean = make_float3(gauss_mean_intensity.x, gauss_mean_intensity.y, gauss_mean_intensity.z);
            const float* gauss_sigma_inv = gauss_sigma_invs + 9 * g;
            float gaussian_weight = compute_gaussian_weight(pixel_pos, gauss_mean, gauss_sigma_inv);
            float opacity = gauss_scales_opacities[g].w;
            float3 gauss_to_probe_offset = make_float3(gauss_mean.x - probe_origin_vec.x, gauss_mean.y - probe_origin_vec.y, gauss_mean.z - probe_origin_vec.z);
            float depth_elevational = gauss_to_probe_offset.x * vec_elevational.x + gauss_to_probe_offset.y * vec_elevational.y + gauss_to_probe_offset.z * vec_elevational.z;

            float dL_dI = 0.0f;
            float dL_dAlpha = 0.0f;
            if (compositing_mode == COMPOSITING_ADDITIVE_SUM) {
                dL_dI = dL_dPixelAccum_final * alpha;
                dL_dAlpha = dL_dPixelAccum_final * color + dL_dPixelDepth_final * depth_elevational;
            } else {
                float den_safe = fmaxf(den, 1e-8f);
                dL_dI = dL_dPixelAccum_final * (alpha / den_safe);
                dL_dAlpha = dL_dPixelAccum_final * ((color * den_safe - num) / (den_safe * den_safe))
                    + dL_dPixelDepth_final * depth_elevational;
            }

            float dL_dOpacity = dL_dAlpha * gaussian_weight;
            float dL_dGaussian_weight = dL_dAlpha * opacity;

            atomicAdd(&dL_dGaussMeansIntensities[g].w, dL_dI);
            atomicAdd(&dL_dGaussScalesOpacities[g].w, dL_dOpacity);

            float dL_dDepth_elevational = dL_dPixelDepth_final * alpha;
            atomicAdd(&dL_dGaussMeansIntensities[g].x, dL_dDepth_elevational * vec_elevational.x);
            atomicAdd(&dL_dGaussMeansIntensities[g].y, dL_dDepth_elevational * vec_elevational.y);
            atomicAdd(&dL_dGaussMeansIntensities[g].z, dL_dDepth_elevational * vec_elevational.z);

            float sigma_inv_grad_scale = dL_dGaussian_weight * gaussian_weight * (-0.5f);
            float dx = pixel_pos_packed.x - gauss_mean.x;
            float dy = pixel_pos_packed.y - gauss_mean.y;
            float dz = pixel_pos_packed.z - gauss_mean.z;
            float* dL_dGaussSigmaInv_ptr = (float*)dL_dGaussSigmaInvs + 9 * g;
            atomicAdd(dL_dGaussSigmaInv_ptr + 0, sigma_inv_grad_scale * dx * dx);
            atomicAdd(dL_dGaussSigmaInv_ptr + 1, sigma_inv_grad_scale * dx * dy);
            atomicAdd(dL_dGaussSigmaInv_ptr + 2, sigma_inv_grad_scale * dx * dz);
            atomicAdd(dL_dGaussSigmaInv_ptr + 3, sigma_inv_grad_scale * dy * dx);
            atomicAdd(dL_dGaussSigmaInv_ptr + 4, sigma_inv_grad_scale * dy * dy);
            atomicAdd(dL_dGaussSigmaInv_ptr + 5, sigma_inv_grad_scale * dy * dz);
            atomicAdd(dL_dGaussSigmaInv_ptr + 6, sigma_inv_grad_scale * dz * dx);
            atomicAdd(dL_dGaussSigmaInv_ptr + 7, sigma_inv_grad_scale * dz * dy);
            atomicAdd(dL_dGaussSigmaInv_ptr + 8, sigma_inv_grad_scale * dz * dz);

            float dG_dMu_x = gauss_sigma_inv[0] * dx + gauss_sigma_inv[1] * dy + gauss_sigma_inv[2] * dz;
            float dG_dMu_y = gauss_sigma_inv[3] * dx + gauss_sigma_inv[4] * dy + gauss_sigma_inv[5] * dz;
            float dG_dMu_z = gauss_sigma_inv[6] * dx + gauss_sigma_inv[7] * dy + gauss_sigma_inv[8] * dz;
            float dL_dMu_factor = dL_dGaussian_weight * gaussian_weight;
            atomicAdd(&dL_dGaussMeansIntensities[g].x, dL_dMu_factor * dG_dMu_x);
            atomicAdd(&dL_dGaussMeansIntensities[g].y, dL_dMu_factor * dG_dMu_y);
            atomicAdd(&dL_dGaussMeansIntensities[g].z, dL_dMu_factor * dG_dMu_z);
        }
        return;
    }

    float dL_dT_running = dL_dPixelT_final;
    for (int k = accepted_cnt - 1; k >= 0; --k) {
        int g = cache_gauss_idx[k];
        float alpha = cache_alpha[k];
        float color = cache_color[k];
        float transmittance_before = cache_transmittance[k];

        float4 gauss_mean_intensity = gauss_means_intensities[g];
        float3 gauss_mean = make_float3(gauss_mean_intensity.x, gauss_mean_intensity.y, gauss_mean_intensity.z);
        const float* gauss_sigma_inv = gauss_sigma_invs + 9 * g;
        float gaussian_weight = compute_gaussian_weight(pixel_pos, gauss_mean, gauss_sigma_inv);
        float opacity = gauss_scales_opacities[g].w;

        float3 gauss_to_probe_offset = make_float3(gauss_mean.x - probe_origin_vec.x, gauss_mean.y - probe_origin_vec.y, gauss_mean.z - probe_origin_vec.z);
        float depth_elevational = gauss_to_probe_offset.x * vec_elevational.x + gauss_to_probe_offset.y * vec_elevational.y + gauss_to_probe_offset.z * vec_elevational.z;

        float dL_dColor = dL_dPixelAccum_final * alpha * transmittance_before;
        float dL_dDepth_elevational = dL_dPixelDepth_final * alpha * transmittance_before;

        float dL_dAlpha_from_color = dL_dPixelAccum_final * color * transmittance_before;
        float dL_dAlpha_from_depth = dL_dPixelDepth_final * depth_elevational * transmittance_before;
        float dL_dAlpha_from_T_next = dL_dT_running * (-transmittance_before);
        float dL_dAlpha = dL_dAlpha_from_color + dL_dAlpha_from_depth + dL_dAlpha_from_T_next;

        float dL_dT_from_contrib = dL_dPixelAccum_final * color * alpha;
        float dL_dT_from_depth = dL_dPixelDepth_final * depth_elevational * alpha;
        float dL_dT_from_next = dL_dT_running * (1.0f - alpha);
        dL_dT_running = dL_dT_from_contrib + dL_dT_from_depth + dL_dT_from_next;

        float dL_dOpacity = dL_dAlpha * gaussian_weight;
        float dL_dGaussian_weight = dL_dAlpha * opacity;

        atomicAdd(&dL_dGaussMeansIntensities[g].w, dL_dColor);
        atomicAdd(&dL_dGaussScalesOpacities[g].w, dL_dOpacity);
        atomicAdd(&dL_dGaussMeansIntensities[g].x, dL_dDepth_elevational * vec_elevational.x);
        atomicAdd(&dL_dGaussMeansIntensities[g].y, dL_dDepth_elevational * vec_elevational.y);
        atomicAdd(&dL_dGaussMeansIntensities[g].z, dL_dDepth_elevational * vec_elevational.z);

        float sigma_inv_grad_scale = dL_dGaussian_weight * gaussian_weight * (-0.5f);
        float dx = pixel_pos_packed.x - gauss_mean.x;
        float dy = pixel_pos_packed.y - gauss_mean.y;
        float dz = pixel_pos_packed.z - gauss_mean.z;

        float* dL_dGaussSigmaInv_ptr = (float*)dL_dGaussSigmaInvs + 9 * g;
        atomicAdd(dL_dGaussSigmaInv_ptr + 0, sigma_inv_grad_scale * dx * dx);
        atomicAdd(dL_dGaussSigmaInv_ptr + 1, sigma_inv_grad_scale * dx * dy);
        atomicAdd(dL_dGaussSigmaInv_ptr + 2, sigma_inv_grad_scale * dx * dz);
        atomicAdd(dL_dGaussSigmaInv_ptr + 3, sigma_inv_grad_scale * dy * dx);
        atomicAdd(dL_dGaussSigmaInv_ptr + 4, sigma_inv_grad_scale * dy * dy);
        atomicAdd(dL_dGaussSigmaInv_ptr + 5, sigma_inv_grad_scale * dy * dz);
        atomicAdd(dL_dGaussSigmaInv_ptr + 6, sigma_inv_grad_scale * dz * dx);
        atomicAdd(dL_dGaussSigmaInv_ptr + 7, sigma_inv_grad_scale * dz * dy);
        atomicAdd(dL_dGaussSigmaInv_ptr + 8, sigma_inv_grad_scale * dz * dz);

        float dG_dMu_x = gauss_sigma_inv[0] * dx + gauss_sigma_inv[1] * dy + gauss_sigma_inv[2] * dz;
        float dG_dMu_y = gauss_sigma_inv[3] * dx + gauss_sigma_inv[4] * dy + gauss_sigma_inv[5] * dz;
        float dG_dMu_z = gauss_sigma_inv[6] * dx + gauss_sigma_inv[7] * dy + gauss_sigma_inv[8] * dz;
        float dL_dMu_factor = dL_dGaussian_weight * gaussian_weight;

        atomicAdd(&dL_dGaussMeansIntensities[g].x, dL_dMu_factor * dG_dMu_x);
        atomicAdd(&dL_dGaussMeansIntensities[g].y, dL_dMu_factor * dG_dMu_y);
        atomicAdd(&dL_dGaussMeansIntensities[g].z, dL_dMu_factor * dG_dMu_z);
    }
}

__global__ void kernel_pixel_compose_color_backward(
    const int n_pixels,
    const float* __restrict__ dL_dPixelColors,
    const float fixed_bg_color,
    const int compositing_mode,
    float* __restrict__ dL_dPixelAccum,
    float* __restrict__ dL_dPixelT
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;

    float dL_dComposedColor = dL_dPixelColors[p];
    dL_dPixelAccum[p] = dL_dComposedColor;
    dL_dPixelT[p] = (compositing_mode == COMPOSITING_ALPHA) ? (dL_dComposedColor * fixed_bg_color) : 0.0f;
}

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
) {
    int blockSize = 256;

    int grid_accum = (n_pixels + blockSize - 1) / blockSize;
    kernel_pixel_compose_color_backward<<<grid_accum, blockSize, 0, stream>>>(
        n_pixels, dL_dPixelColors, fixed_bg_color, compositing_mode,
        dL_dPixelAccum, dL_dPixelT
    );

    kernel_pixel_alpha_accum_backward<<<grid_accum, blockSize, 0, stream>>>(
        n_pixels, pixel_positions_packed, pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
        gauss_means_intensities, gauss_scales_opacities, gauss_sigma_invs,
        probe_origin, probe_vec_elevational,
        dL_dPixelAccum, dL_dPixelT, dL_dPixelDepths,
        dL_dGaussMeansIntensities, dL_dGaussScalesOpacities, dL_dGaussSigmaInvs, compositing_mode
    );

    int grid_pre = (n_gauss + blockSize - 1) / blockSize;
    kernel_gauss_preprocess_backward<<<grid_pre, blockSize, 0, stream>>>(
        n_gauss, gauss_quats, gauss_scales_opacities, gauss_Rs,
        dL_dGaussSigmaInvs, dL_dGaussQuats, dL_dGaussScalesOpacities
    );
}

void RasterizeGaussiansBackwardCUDA_Impl_Batched(
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
) {
    int blockSize = 256;
    const int n_pixels_total = batch_size * pixels_per_image;

    int grid_accum = (n_pixels_total + blockSize - 1) / blockSize;
    kernel_pixel_compose_color_backward<<<grid_accum, blockSize, 0, stream>>>(
        n_pixels_total, dL_dPixelColors, fixed_bg_color, compositing_mode,
        dL_dPixelAccum, dL_dPixelT
    );

    kernel_pixel_alpha_accum_backward_batched<<<grid_accum, blockSize, 0, stream>>>(
        n_pixels_total,
        pixels_per_image,
        batch_size,
        pixel_positions_packed,
        pixel_gauss_offsets,
        pixel_gauss_counts,
        pixel_gauss_indices,
        gauss_means_intensities,
        gauss_scales_opacities,
        gauss_sigma_invs,
        probe_origin,
        probe_vec_elevational,
        dL_dPixelAccum,
        dL_dPixelT,
        dL_dPixelDepths,
        dL_dGaussMeansIntensities,
        dL_dGaussScalesOpacities,
        dL_dGaussSigmaInvs,
        compositing_mode
    );

    int grid_pre = (n_gauss + blockSize - 1) / blockSize;
    kernel_gauss_preprocess_backward<<<grid_pre, blockSize, 0, stream>>>(
        n_gauss, gauss_quats, gauss_scales_opacities, gauss_Rs,
        dL_dGaussSigmaInvs, dL_dGaussQuats, dL_dGaussScalesOpacities
    );
}
