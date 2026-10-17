#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ float ours_dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 ours_normalize3(const float3& v) {
    float n = sqrtf(fmaxf(1e-12f, ours_dot3(v, v)));
    return make_float3(v.x / n, v.y / n, v.z / n);
}

/** Quaternion layout matches repo / ultrags: (w, x, y, z) in q_in.x, .y, .z, .w. */
__device__ __forceinline__ void ours_quat_to_rot(const float4& q_in, float R[9]) {
    float nq = rsqrtf(fmaxf(1e-12f, q_in.x * q_in.x + q_in.y * q_in.y + q_in.z * q_in.z + q_in.w * q_in.w));
    float w = q_in.x * nq;
    float x = q_in.y * nq;
    float y = q_in.z * nq;
    float z = q_in.w * nq;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    R[0] = 1.f - 2.f * (yy + zz);
    R[1] = 2.f * (xy - wz);
    R[2] = 2.f * (xz + wy);
    R[3] = 2.f * (xy + wz);
    R[4] = 1.f - 2.f * (xx + zz);
    R[5] = 2.f * (yz - wx);
    R[6] = 2.f * (xz - wy);
    R[7] = 2.f * (yz + wx);
    R[8] = 1.f - 2.f * (xx + yy);
}

/** Same cutoff as alpha compositing / UltraGS-style negligible splat skip. */
__device__ constexpr float OURS_ALPHA_MIN = 1.f / 255.f;

/** L2 projected radius along probe axis (mm), matching tile/pixel mapper footprint. */
__device__ __forceinline__ void ours_lat_ax_radii_mm(
    float sx,
    float sy,
    float sz,
    const float Rg[9],
    float3 axis_lat,
    float3 axis_ax,
    float k_sigma,
    float* r_lat_mm,
    float* r_ax_mm) {
    float ex0 = sx * (Rg[0] * axis_lat.x + Rg[3] * axis_lat.y + Rg[6] * axis_lat.z);
    float ex1 = sy * (Rg[1] * axis_lat.x + Rg[4] * axis_lat.y + Rg[7] * axis_lat.z);
    float ex2 = sz * (Rg[2] * axis_lat.x + Rg[5] * axis_lat.y + Rg[8] * axis_lat.z);
    float ey0 = sx * (Rg[0] * axis_ax.x + Rg[3] * axis_ax.y + Rg[6] * axis_ax.z);
    float ey1 = sy * (Rg[1] * axis_ax.x + Rg[4] * axis_ax.y + Rg[7] * axis_ax.z);
    float ey2 = sz * (Rg[2] * axis_ax.x + Rg[5] * axis_ax.y + Rg[8] * axis_ax.z);
    *r_lat_mm = sqrtf(fmaxf(1e-12f, ex0 * ex0 + ex1 * ex1 + ex2 * ex2)) * k_sigma;
    *r_ax_mm = sqrtf(fmaxf(1e-12f, ey0 * ey0 + ey1 * ey1 + ey2 * ey2)) * k_sigma;
}

/** Per-pixel lat/ax AABB reject (UltraGS ``check_intersection_optimized`` in probe plane). */
__device__ __forceinline__ bool ours_lat_ax_aabb_reject(
    float x_c,
    float y_c,
    float mu_x,
    float mu_y,
    float r_lat_mm,
    float r_ax_mm) {
    return fabsf(x_c - mu_x) > r_lat_mm || fabsf(y_c - mu_y) > r_ax_mm;
}

/** Quantized v (row) key for radix sort; must be monotonic in v ascending. */
__device__ __forceinline__ uint32_t ours_v_to_depth_key(float v_pix) {
    const float kDepthKeyOffset = 500.0f;
    const float kDepthKeyScale = 1000.0f;
    return (uint32_t)((v_pix + kDepthKeyOffset) * kDepthKeyScale);
}

/** Slice weight α and intensity I at pixel (x,y) for Gaussian g (shared by forward / alpha-backward replay). */
__device__ __forceinline__ void ours_eval_gauss_contrib_at_pixel(
    int x,
    int y,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float k_sigma,
    int g,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    const float* __restrict__ pose,
    float3 axis_lat,
    float3 axis_ax,
    float3 axis_ele,
    float3 origin,
    float* __restrict__ out_alpha,
    float* __restrict__ out_intensity) {
    float x_c = (x - 0.5f * (W - 1)) * lateral_spacing_mm_per_pix;
    float y_c = y * axial_spacing_mm_per_pix + axial_near_mm;
    float z_c = 0.f;

    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;
    float3 mw = make_float3(mptr[0], mptr[1], mptr[2]);
    float3 d = make_float3(mw.x - origin.x, mw.y - origin.y, mw.z - origin.z);
    float mu_x = ours_dot3(d, axis_lat);
    float mu_y = ours_dot3(d, axis_ax);
    float mu_z = ours_dot3(d, axis_ele);

    float Rg[9];
    ours_quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
    float sx = fmaxf(sptr[0], 1e-6f);
    float sy = fmaxf(sptr[1], 1e-6f);
    float sz = fmaxf(sptr[2], 1e-6f);

    float r_lat_mm = 0.f;
    float r_ax_mm = 0.f;
    ours_lat_ax_radii_mm(sx, sy, sz, Rg, axis_lat, axis_ax, k_sigma, &r_lat_mm, &r_ax_mm);
    if (ours_lat_ax_aabb_reject(x_c, y_c, mu_x, mu_y, r_lat_mm, r_ax_mm)) {
        *out_alpha = 0.f;
        *out_intensity = 0.f;
        return;
    }

    float ex0 = sx * (Rg[0] * axis_lat.x + Rg[3] * axis_lat.y + Rg[6] * axis_lat.z);
    float ex1 = sy * (Rg[1] * axis_lat.x + Rg[4] * axis_lat.y + Rg[7] * axis_lat.z);
    float ex2 = sz * (Rg[2] * axis_lat.x + Rg[5] * axis_lat.y + Rg[8] * axis_lat.z);
    float ey0 = sx * (Rg[0] * axis_ax.x + Rg[3] * axis_ax.y + Rg[6] * axis_ax.z);
    float ey1 = sy * (Rg[1] * axis_ax.x + Rg[4] * axis_ax.y + Rg[7] * axis_ax.z);
    float ey2 = sz * (Rg[2] * axis_ax.x + Rg[5] * axis_ax.y + Rg[8] * axis_ax.z);
    float ez0 = sx * (Rg[0] * axis_ele.x + Rg[3] * axis_ele.y + Rg[6] * axis_ele.z);
    float ez1 = sy * (Rg[1] * axis_ele.x + Rg[4] * axis_ele.y + Rg[7] * axis_ele.z);
    float ez2 = sz * (Rg[2] * axis_ele.x + Rg[5] * axis_ele.y + Rg[8] * axis_ele.z);

    float c00 = ex0 * ex0 + ex1 * ex1 + ex2 * ex2;
    float c01 = ex0 * ey0 + ex1 * ey1 + ex2 * ey2;
    float c02 = ex0 * ez0 + ex1 * ez1 + ex2 * ez2;
    float c11 = ey0 * ey0 + ey1 * ey1 + ey2 * ey2;
    float c12 = ey0 * ez0 + ey1 * ez1 + ey2 * ez2;
    float c22 = ez0 * ez0 + ez1 * ez1 + ez2 * ez2;
    c00 += 1e-6f;
    c11 += 1e-6f;
    c22 += 1e-6f;
    float det = c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02) + c02 * (c01 * c12 - c11 * c02);
    det = fmaxf(det, 1e-12f);
    float invdet = 1.f / det;
    float i00 = (c11 * c22 - c12 * c12) * invdet;
    float i01 = (c02 * c12 - c01 * c22) * invdet;
    float i02 = (c01 * c12 - c02 * c11) * invdet;
    float i11 = (c00 * c22 - c02 * c02) * invdet;
    float i12 = (c01 * c02 - c00 * c12) * invdet;
    float i22 = (c00 * c11 - c01 * c01) * invdet;

    float dx = x_c - mu_x;
    float dy = y_c - mu_y;
    float dz = z_c - mu_z;
    float quad = dx * (i00 * dx + i01 * dy + i02 * dz) + dy * (i01 * dx + i11 * dy + i12 * dz)
        + dz * (i02 * dx + i12 * dy + i22 * dz);
    float base = expf(-0.5f * quad);
    float alpha = gauss_opacities[g] * base;
    *out_alpha = alpha;
    *out_intensity = gauss_intensities[g];
}
