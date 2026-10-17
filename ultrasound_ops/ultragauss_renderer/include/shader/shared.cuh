#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ void build_sigma_inv_from_tril(
    const float* tri,
    const float beta,
    float* sigma_inv
) {
    const float l11 = tri[0] * tri[0] + beta;
    const float l12 = tri[1];
    const float l13 = tri[2];
    const float l22 = tri[3] * tri[3] + beta;
    const float l23 = tri[4];
    const float l33 = tri[5] * tri[5] + beta;

    sigma_inv[0] = l11 * l11;
    sigma_inv[1] = l11 * l12;
    sigma_inv[2] = l11 * l13;
    sigma_inv[3] = sigma_inv[1];
    sigma_inv[4] = l12 * l12 + l22 * l22;
    sigma_inv[5] = l12 * l13 + l22 * l23;
    sigma_inv[6] = sigma_inv[2];
    sigma_inv[7] = sigma_inv[5];
    sigma_inv[8] = l13 * l13 + l23 * l23 + l33 * l33;
}

__device__ __forceinline__ float compute_gaussian_weight(
    const float3& p,
    const float3& mean,
    const float* sigma_inv
) {
    float dx = p.x - mean.x;
    float dy = p.y - mean.y;
    float dz = p.z - mean.z;
    float q = (sigma_inv[0] * dx + sigma_inv[1] * dy + sigma_inv[2] * dz) * dx
            + (sigma_inv[3] * dx + sigma_inv[4] * dy + sigma_inv[5] * dz) * dy
            + (sigma_inv[6] * dx + sigma_inv[7] * dy + sigma_inv[8] * dz) * dz;
    return __expf(-0.5f * q);
}

__device__ __forceinline__ float compute_gaussian_quadratic(
    const float3& p,
    const float3& mean,
    const float* sigma_inv
) {
    float dx = p.x - mean.x;
    float dy = p.y - mean.y;
    float dz = p.z - mean.z;
    return (sigma_inv[0] * dx + sigma_inv[1] * dy + sigma_inv[2] * dz) * dx
         + (sigma_inv[3] * dx + sigma_inv[4] * dy + sigma_inv[5] * dz) * dy
         + (sigma_inv[6] * dx + sigma_inv[7] * dy + sigma_inv[8] * dz) * dz;
}
