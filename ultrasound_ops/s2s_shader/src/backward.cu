#include <math_constants.h>

#include "math_utils.cuh"
#include "covariance.cuh"
#include "shared.cuh"
#include "backward.cuh"

// 像素核：现场重算 d 和 e，累加到 Mean / Opacity / Intensity / SigmaInv
__global__ void kernel_pixel_intensity_accum_backward(
    const int n_pixels,
    const float* __restrict__ pixel_positions,
    const int*   __restrict__ pixel_gauss_offsets,
    const int*   __restrict__ pixel_gauss_counts,
    const int*   __restrict__ pixel_gauss_indices,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_sigma_invs,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    const float* __restrict__ dL_dPixelColors,
    float*       __restrict__ dL_dGaussMeans,      // (G,3)
    float*       __restrict__ dL_dGaussOpacities,  // (G,)
    float*       __restrict__ dL_dGaussIntensities,// (G,)
    float*       __restrict__ dL_dGaussSigmaInvs   // (G,9)
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;

    float grad_Ip = dL_dPixelColors[p];
    if (grad_Ip == 0.0f) return;

    const float px = pixel_positions[p * 3 + 0];
    const float py = pixel_positions[p * 3 + 1];
    const float pz = pixel_positions[p * 3 + 2];
    float3 pixel_pos = make_float3(px, py, pz);

    const int start = pixel_gauss_offsets[p];
    const int count = pixel_gauss_counts[p];

    for (int k = 0; k < count; ++k) {
        const int idx = start + k;
        const int g   = pixel_gauss_indices[idx];

        const float* mean_ptr = gauss_means + g * 3;
        float3 gauss_mean = make_float3(mean_ptr[0], mean_ptr[1], mean_ptr[2]);

        const float* S = gauss_sigma_invs + g * 9;
        const float opacity   = gauss_opacities[g];
        const float intensity = gauss_intensities[g];

        // d = p - mu
        float dx = px - gauss_mean.x;
        float dy = py - gauss_mean.y;
        float dz = pz - gauss_mean.z;

        // e = -0.5 d^T S d
        float qx = S[0] * dx + S[1] * dy + S[2] * dz;
        float qy = S[3] * dx + S[4] * dy + S[5] * dz;
        float qz = S[6] * dx + S[7] * dy + S[8] * dz;
        float e  = -0.5f * (dx * qx + dy * qy + dz * qz);

        float exp_e = __expf(e);
        float alpha_gp = opacity * exp_e;

        // dL/dI_g
        atomicAdd(dL_dGaussIntensities + g, grad_Ip * alpha_gp);

        // dL/do_g
        atomicAdd(dL_dGaussOpacities + g, grad_Ip * intensity * exp_e);

        // C_gp = dL/de
        float C_gp = grad_Ip * opacity * intensity * exp_e;

        // mean: de/dmu = SigmaInv * d
        float Sd1 = S[0] * dx + S[1] * dy + S[2] * dz;
        float Sd2 = S[3] * dx + S[4] * dy + S[5] * dz;
        float Sd3 = S[6] * dx + S[7] * dy + S[8] * dz;

        float* dMu = dL_dGaussMeans + g * 3;
        atomicAdd(dMu + 0, C_gp * Sd1);
        atomicAdd(dMu + 1, C_gp * Sd2);
        atomicAdd(dMu + 2, C_gp * Sd3);

        // SigmaInv: de/dS_ij = -0.5 d_i d_j
        float coeff = -0.5f * C_gp;
        float* dSg = dL_dGaussSigmaInvs + g * 9;

        atomicAdd(dSg + 0, coeff * (dx * dx));
        atomicAdd(dSg + 1, coeff * (dx * dy));
        atomicAdd(dSg + 2, coeff * (dx * dz));
        atomicAdd(dSg + 3, coeff * (dy * dx));
        atomicAdd(dSg + 4, coeff * (dy * dy));
        atomicAdd(dSg + 5, coeff * (dy * dz));
        atomicAdd(dSg + 6, coeff * (dz * dx));
        atomicAdd(dSg + 7, coeff * (dz * dy));
        atomicAdd(dSg + 8, coeff * (dz * dz));
    }
}

// 几何核：从 dL/dSigmaInv 回传到 scale / quat
__global__ void kernel_gauss_preprocess_backward(
    const int n_gauss,
    const float* __restrict__ gauss_quats,   // (G,4)
    const float* __restrict__ gauss_scales,  // (G,3)
    const float* __restrict__ gauss_Rs,      // (G,9)
    const float* __restrict__ dL_dGaussSigmaInvs,
    float*       __restrict__ dL_dGaussQuats,   // (G,4)
    float*       __restrict__ dL_dGaussScales  // (G,3)
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) return;

    const float* quat_ptr  = gauss_quats + g * 4;
    const float* scale_ptr = gauss_scales + g * 3;
    const float* R_ptr     = gauss_Rs + g * 9;
    const float* dS_ptr    = dL_dGaussSigmaInvs + g * 9;

    float4 gauss_quat = make_float4(quat_ptr[0], quat_ptr[1], quat_ptr[2], quat_ptr[3]);
    float3 gauss_scale = make_float3(scale_ptr[0], scale_ptr[1], scale_ptr[2]);

    float R[9];
    mat3_copy(R_ptr, R);

    float dR[9];
    mat3_zero(dR);
    float3 dScale = make_float3(0.f, 0.f, 0.f);

    transform_grads_sigma_to_R_S(gauss_scale, R, dS_ptr, dR, dScale);

    float4 dQuat4;
    transform_grads_R_to_Quat(gauss_quat, dR, dQuat4);

    float* dq = dL_dGaussQuats + g * 4;
    dq[0] = dQuat4.x;
    dq[1] = dQuat4.y;
    dq[2] = dQuat4.z;
    dq[3] = dQuat4.w;

    float* ds = dL_dGaussScales + g * 3;
    ds[0] += dScale.x;
    ds[1] += dScale.y;
    ds[2] += dScale.z;
}

// Host Entry
void RasterizeGaussiansBackwardCUDA_Impl(
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
    const float* gauss_Rs,
    const float* gauss_sigma_invs,
    const float* dL_dPixelColors,
    float*       dL_dGaussMeans,
    float*       dL_dGaussScales,
    float*       dL_dGaussQuats,
    float*       dL_dGaussOpacities,
    float*       dL_dGaussIntensities,
    float*       dL_dGaussSigmaInvs,
    cudaStream_t stream
) {
    const int blockSize = 256;

    int grid_pix = (n_pixels + blockSize - 1) / blockSize;
    kernel_pixel_intensity_accum_backward<<<grid_pix, blockSize, 0, stream>>>(
        n_pixels,
        pixel_positions,
        pixel_gauss_offsets,
        pixel_gauss_counts,
        pixel_gauss_indices,
        gauss_means,
        gauss_sigma_invs,
        gauss_opacities,
        gauss_intensities,
        dL_dPixelColors,
        dL_dGaussMeans,
        dL_dGaussOpacities,
        dL_dGaussIntensities,
        dL_dGaussSigmaInvs
    );

    int grid_geo = (n_gauss + blockSize - 1) / blockSize;
    kernel_gauss_preprocess_backward<<<grid_geo, blockSize, 0, stream>>>(
        n_gauss,
        gauss_quats,
        gauss_scales,
        gauss_Rs,
        dL_dGaussSigmaInvs,
        dL_dGaussQuats,
        dL_dGaussScales
    );
}