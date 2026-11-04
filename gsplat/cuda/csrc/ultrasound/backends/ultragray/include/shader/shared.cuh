#pragma once

#include "math_utils.cuh"

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

__device__ __forceinline__ float eval_sh_degree1(const float4& sh, const float3& d) {
    // Degree-1 SH compact form: [1, dx, dy, dz].
    return sh.x + sh.y * d.x + sh.z * d.y + sh.w * d.z;
}

// 3-point Gauss-Legendre approximation over segment [0, l].
__device__ __forceinline__ float compute_ray_gaussian_integral(
    const float3& ray_origin,
    const float3& ray_dir,
    const float3& mu,
    const float3& s,
    const float* R,
    float l,
    float quadrature_window_radius_mm
) {
    if (l <= 0.0f) return 0.0f;

    float3 diff = ray_origin - mu;
    float3 o_local;
    o_local.x = R[0] * diff.x + R[3] * diff.y + R[6] * diff.z;
    o_local.y = R[1] * diff.x + R[4] * diff.y + R[7] * diff.z;
    o_local.z = R[2] * diff.x + R[5] * diff.y + R[8] * diff.z;

    float3 d_local;
    d_local.x = R[0] * ray_dir.x + R[3] * ray_dir.y + R[6] * ray_dir.z;
    d_local.y = R[1] * ray_dir.x + R[4] * ray_dir.y + R[7] * ray_dir.z;
    d_local.z = R[2] * ray_dir.x + R[5] * ray_dir.y + R[8] * ray_dir.z;

    float3 o_g = make_float3(o_local.x / s.x, o_local.y / s.y, o_local.z / s.z);
    float3 d_g = make_float3(d_local.x / s.x, d_local.y / s.y, d_local.z / s.z);

    const float xi0 = -0.7745966692f;
    const float xi1 = 0.0f;
    const float xi2 = 0.7745966692f;
    const float w0 = 0.5555555556f;
    const float w1 = 0.8888888889f;
    const float w2 = 0.5555555556f;

    float dg_norm2 = dot_prod(d_g, d_g);
    float t_center = -dot_prod(o_g, d_g) / (dg_norm2 + 1e-8f);
    float t_start = 0.0f;
    float t_end = l;
    if (quadrature_window_radius_mm > 0.0f) {
        t_start = fmaxf(0.0f, t_center - quadrature_window_radius_mm);
        t_end = fminf(l, t_center + quadrature_window_radius_mm);
    }
    if (t_end <= t_start) return 0.0f;

    float half = 0.5f * (t_end - t_start);
    float c = 0.5f * (t_end + t_start);

    float t0 = half * xi0 + c;
    float t1 = half * xi1 + c;
    float t2 = half * xi2 + c;

    float3 x0 = o_g + d_g * t0;
    float3 x1 = o_g + d_g * t1;
    float3 x2 = o_g + d_g * t2;

    float f0 = __expf(-0.5f * dot_prod(x0, x0));
    float f1 = __expf(-0.5f * dot_prod(x1, x1));
    float f2 = __expf(-0.5f * dot_prod(x2, x2));

    return half * (w0 * f0 + w1 * f1 + w2 * f2);
}
