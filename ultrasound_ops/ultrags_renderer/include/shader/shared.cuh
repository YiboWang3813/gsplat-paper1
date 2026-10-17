#pragma once

__device__ __forceinline__ float compute_gaussian_weight(
    const float3& p,
    const float3& mean,
    const float* sigma_inv
) {
    float dx = p.x - mean.x;
    float dy = p.y - mean.y;
    float dz = p.z - mean.z;

    float power = -0.5f * (
        (sigma_inv[0] * dx + sigma_inv[1] * dy + sigma_inv[2] * dz) * dx +
        (sigma_inv[3] * dx + sigma_inv[4] * dy + sigma_inv[5] * dz) * dy +
        (sigma_inv[6] * dx + sigma_inv[7] * dy + sigma_inv[8] * dz) * dz
    );
    return __expf(power);
}
