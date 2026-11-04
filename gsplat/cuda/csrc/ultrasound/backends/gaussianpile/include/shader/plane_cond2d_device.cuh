#pragma once

#include "shader/shared.cuh"
#include "shader/slice_grid_geometry.cuh"

namespace plane_cond2d_device {

/** Match ``ours_eval_gauss_contrib_at_pixel`` probe-frame covariance diagonal jitter. */
constexpr float kProbeCovJitter = 1e-6f;
constexpr float kDetMin = 1e-12f;
constexpr float kMaxInvDet = 1e8f;
constexpr float kGradClamp = 1e3f;

__device__ __forceinline__ void add_probe_cov_jitter(float Sc[9]) {
    Sc[0] += kProbeCovJitter;
    Sc[4] += kProbeCovJitter;
    Sc[8] += kProbeCovJitter;
}

__device__ __forceinline__ bool invert_sigma2_sym(
    float s2_00,
    float s2_01,
    float s2_11,
    float* inv2_00,
    float* inv2_01,
    float* inv2_11) {
    const float det2 = s2_00 * s2_11 - s2_01 * s2_01;
    if (!(det2 >= kDetMin) || !isfinite(det2)) {
        return false;
    }
    float inv_det = 1.f / det2;
    inv_det = fminf(inv_det, kMaxInvDet);
    *inv2_00 = s2_11 * inv_det;
    *inv2_01 = -s2_01 * inv_det;
    *inv2_11 = s2_00 * inv_det;
    return true;
}

/** True when forward ``invert_sigma2_sym`` saturated ``inv_det`` at ``kMaxInvDet``. */
__device__ __forceinline__ bool invert_sigma2_inv_det_clamped(float s2_00, float s2_01, float s2_11) {
    const float det2 = s2_00 * s2_11 - s2_01 * s2_01;
    if (!(det2 >= kDetMin) || !isfinite(det2)) {
        return false;
    }
    return (1.f / det2) > kMaxInvDet;
}

__device__ __forceinline__ void mat3_sigma_world(
    float sx,
    float sy,
    float sz,
    const float Rg[9],
    float Sigma_w[9]) {
    float M[9];
    M[0] = Rg[0] * sx * sx;
    M[1] = Rg[1] * sy * sy;
    M[2] = Rg[2] * sz * sz;
    M[3] = Rg[3] * sx * sx;
    M[4] = Rg[4] * sy * sy;
    M[5] = Rg[5] * sz * sz;
    M[6] = Rg[6] * sx * sx;
    M[7] = Rg[7] * sy * sy;
    M[8] = Rg[8] * sz * sz;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Sigma_w[i * 3 + j] = M[i * 3 + 0] * Rg[j] + M[i * 3 + 1] * Rg[3 + j] + M[i * 3 + 2] * Rg[6 + j];
        }
    }
}

__device__ __forceinline__ void build_sigma_c_probe(
    const float Sigma_w[9],
    float3 axis_lat,
    float3 axis_ax,
    float3 axis_ele,
    float Sc[9]) {
    float3 c0 = make_float3(
        Sigma_w[0] * axis_lat.x + Sigma_w[1] * axis_lat.y + Sigma_w[2] * axis_lat.z,
        Sigma_w[3] * axis_lat.x + Sigma_w[4] * axis_lat.y + Sigma_w[5] * axis_lat.z,
        Sigma_w[6] * axis_lat.x + Sigma_w[7] * axis_lat.y + Sigma_w[8] * axis_lat.z);
    float3 c1 = make_float3(
        Sigma_w[0] * axis_ax.x + Sigma_w[1] * axis_ax.y + Sigma_w[2] * axis_ax.z,
        Sigma_w[3] * axis_ax.x + Sigma_w[4] * axis_ax.y + Sigma_w[5] * axis_ax.z,
        Sigma_w[6] * axis_ax.x + Sigma_w[7] * axis_ax.y + Sigma_w[8] * axis_ax.z);
    float3 c2 = make_float3(
        Sigma_w[0] * axis_ele.x + Sigma_w[1] * axis_ele.y + Sigma_w[2] * axis_ele.z,
        Sigma_w[3] * axis_ele.x + Sigma_w[4] * axis_ele.y + Sigma_w[5] * axis_ele.z,
        Sigma_w[6] * axis_ele.x + Sigma_w[7] * axis_ele.y + Sigma_w[8] * axis_ele.z);
    Sc[0] = axis_lat.x * c0.x + axis_lat.y * c0.y + axis_lat.z * c0.z;
    Sc[1] = axis_lat.x * c1.x + axis_lat.y * c1.y + axis_lat.z * c1.z;
    Sc[2] = axis_lat.x * c2.x + axis_lat.y * c2.y + axis_lat.z * c2.z;
    Sc[3] = axis_ax.x * c0.x + axis_ax.y * c0.y + axis_ax.z * c0.z;
    Sc[4] = axis_ax.x * c1.x + axis_ax.y * c1.y + axis_ax.z * c1.z;
    Sc[5] = axis_ax.x * c2.x + axis_ax.y * c2.y + axis_ax.z * c2.z;
    Sc[6] = axis_ele.x * c0.x + axis_ele.y * c0.y + axis_ele.z * c0.z;
    Sc[7] = axis_ele.x * c1.x + axis_ele.y * c1.y + axis_ele.z * c1.z;
    Sc[8] = axis_ele.x * c2.x + axis_ele.y * c2.y + axis_ele.z * c2.z;
}

/**
 * Schur complement of probe covariance Sc on elev (index 2), lat-ax block (0,1).
 *   mu_2d = mu_p - Sigma_pz / Sigma_zz * mu_z
 *   Sigma_2d = Sigma_pp - Sigma_pz Sigma_pz^T / Sigma_zz
 */
__device__ __forceinline__ bool schur_conditional_plane(
    float mu_lat,
    float mu_ax,
    float mu_z,
    const float Sc[9],
    float* mu2_lat,
    float* mu2_ax,
    float* s2_00,
    float* s2_01,
    float* s2_11,
    float* inv2_00,
    float* inv2_01,
    float* inv2_11) {
    const float szz = Sc[8];
    const float inv_zz = 1.f / szz;
    const float spz_lat = Sc[2];
    const float spz_ax = Sc[5];
    *mu2_lat = mu_lat - spz_lat * inv_zz * mu_z;
    *mu2_ax = mu_ax - spz_ax * inv_zz * mu_z;
    *s2_00 = Sc[0] - inv_zz * spz_lat * spz_lat;
    *s2_01 = Sc[1] - inv_zz * spz_lat * spz_ax;
    *s2_11 = Sc[4] - inv_zz * spz_ax * spz_ax;
    return invert_sigma2_sym(*s2_00, *s2_01, *s2_11, inv2_00, inv2_01, inv2_11);
}

/** dL/dS = -S^{-1} (dL/dS^{-1}) S^{-1} for symmetric 2x2 S=[[a,b],[b,c]]. Zeros if inv_det clamp active. */
__device__ __forceinline__ void inv2_sym_bwd(
    float a,
    float b,
    float c,
    float gi00,
    float gi01,
    float gi11,
    float* ga,
    float* gb,
    float* gc) {
    const float det2 = a * c - b * b;
    if (!(det2 >= kDetMin) || !isfinite(det2) || invert_sigma2_inv_det_clamped(a, b, c)) {
        *ga = *gb = *gc = 0.f;
        return;
    }
    const float inv_det = 1.f / det2;
    const float i00 = c * inv_det;
    const float i01 = -b * inv_det;
    const float i11 = a * inv_det;
    const float m00 = i00 * (gi00 * i00 + 2.f * gi01 * i01) + i01 * i01 * gi11;
    const float m01 = i00 * (gi00 * i01 + gi01 * i11) + i01 * (gi01 * i00 + gi11 * i01);
    const float m11 = i01 * (2.f * gi00 * i01) + i11 * (gi01 * i01 + gi11 * i11);
    *ga = -m00;
    *gb = -2.f * m01;
    *gc = -m11;
}

/** Conservative axis-aligned half-width (mm) for Schur Sigma_2d at mu2. */
__device__ __forceinline__ void plane_cond2d_schur_aabb_radii_mm(
    float s2_00,
    float s2_01,
    float s2_11,
    float k_sigma,
    float* r_lat_mm,
    float* r_ax_mm) {
    const float cross = fabsf(s2_01);
    *r_lat_mm = k_sigma * sqrtf(fmaxf(s2_00 + cross, 1e-12f));
    *r_ax_mm = k_sigma * sqrtf(fmaxf(s2_11 + cross, 1e-12f));
}

/**
 * Plane in-slice weight. quad = delta^T Sigma_2d^{-1} delta must be >= 0 for PD Sigma_2d.
 * Never use fmax(quad,0): quad<0 would incorrectly yield exp(0)=1 (opaque blobs).
 */
__device__ __forceinline__ float plane_cond2d_plane_weight(float quad_raw) {
    if (!(quad_raw >= 0.f) || !isfinite(quad_raw)) {
        return 0.f;
    }
    return expf(-0.5f * quad_raw);
}

/**
 * Mapper footprint: center at Schur mu2; radius is max(ellipsoid k-sigma, Schur AABB)
 * plus |mu_p - mu2| so tiles cover the shader Schur AABB at mu2.
 */
__device__ __forceinline__ bool plane_cond2d_mapper_footprint_mm(
    const float* __restrict__ mptr,
    const float* __restrict__ sptr,
    const float* __restrict__ qptr,
    float3 axis_lat,
    float3 axis_ax,
    float3 axis_ele,
    float3 origin,
    float k_sigma,
    float* __restrict__ mu2_lat_mm,
    float* __restrict__ mu2_ax_mm,
    float* __restrict__ r_lat_mm,
    float* __restrict__ r_ax_mm) {
    float3 d = make_float3(mptr[0] - origin.x, mptr[1] - origin.y, mptr[2] - origin.z);
    const float mu_lat = axis_lat.x * d.x + axis_lat.y * d.y + axis_lat.z * d.z;
    const float mu_ax = axis_ax.x * d.x + axis_ax.y * d.y + axis_ax.z * d.z;
    const float mu_z = axis_ele.x * d.x + axis_ele.y * d.y + axis_ele.z * d.z;

    float Rg[9];
    ours_quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
    const float sx = fmaxf(sptr[0], 1e-6f);
    const float sy = fmaxf(sptr[1], 1e-6f);
    const float sz = fmaxf(sptr[2], 1e-6f);

    float Sigma_w[9];
    mat3_sigma_world(sx, sy, sz, Rg, Sigma_w);
    float Sc[9];
    build_sigma_c_probe(Sigma_w, axis_lat, axis_ax, axis_ele, Sc);
    add_probe_cov_jitter(Sc);

    float mu2_lat = 0.f;
    float mu2_ax = 0.f;
    float s2_00 = 0.f;
    float s2_01 = 0.f;
    float s2_11 = 0.f;
    float i00 = 0.f;
    float i01 = 0.f;
    float i11 = 0.f;
    if (!schur_conditional_plane(mu_lat, mu_ax, mu_z, Sc, &mu2_lat, &mu2_ax, &s2_00, &s2_01, &s2_11, &i00, &i01, &i11)) {
        return false;
    }

    float r_lat = 0.f;
    float r_ax = 0.f;
    ours_lat_ax_radii_mm(sx, sy, sz, Rg, axis_lat, axis_ax, k_sigma, &r_lat, &r_ax);
    const float pad_lat = fabsf(mu_lat - mu2_lat);
    const float pad_ax = fabsf(mu_ax - mu2_ax);
    float r_lat_schur = 0.f;
    float r_ax_schur = 0.f;
    plane_cond2d_schur_aabb_radii_mm(s2_00, s2_01, s2_11, k_sigma, &r_lat_schur, &r_ax_schur);

    *mu2_lat_mm = mu2_lat;
    *mu2_ax_mm = mu2_ax;
    *r_lat_mm = fmaxf(r_lat + pad_lat, r_lat_schur);
    *r_ax_mm = fmaxf(r_ax + pad_ax, r_ax_schur);
    return true;
}

__device__ __forceinline__ float3 mat3_vec_sym(const float M[9], float3 v) {
    return make_float3(
        M[0] * v.x + M[1] * v.y + M[2] * v.z,
        M[3] * v.x + M[4] * v.y + M[5] * v.z,
        M[6] * v.x + M[7] * v.y + M[8] * v.z);
}

__device__ __forceinline__ float3 rot_col(const float Rg[9], int k) {
    return make_float3(Rg[k], Rg[3 + k], Rg[6 + k]);
}

/** Backprop for Sigma_w = sum_k sk^2 * c_k c_k^T with c_k = column k of Rg (row-major). */
__device__ __forceinline__ void sigma_w_to_scale_rot_grads(
    const float gSigma_w[9],
    const float Rg[9],
    float sx,
    float sy,
    float sz,
    float* g_sx,
    float* g_sy,
    float* g_sz,
    float grad_R[9]) {
    for (int i = 0; i < 9; ++i) {
        grad_R[i] = 0.f;
    }
    const float sk[3] = {sx, sy, sz};
    const float sk2[3] = {sx * sx, sy * sy, sz * sz};
    float* g_sk_out[3] = {g_sx, g_sy, g_sz};
    for (int k = 0; k < 3; ++k) {
        float3 ck = rot_col(Rg, k);
        float3 gck = mat3_vec_sym(gSigma_w, ck);
        const float q = ck.x * gck.x + ck.y * gck.y + ck.z * gck.z;
        *g_sk_out[k] = 2.f * sk[k] * q;
        const float3 gcol = make_float3(
            2.f * sk2[k] * gck.x,
            2.f * sk2[k] * gck.y,
            2.f * sk2[k] * gck.z);
        grad_R[k] += gcol.x;
        grad_R[3 + k] += gcol.y;
        grad_R[6 + k] += gcol.z;
    }
}

}  // namespace plane_cond2d_device

__device__ __forceinline__ void ours_plane_cond2d_eval_at_pixel(
    int col,
    int row,
    float k_sigma,
    int g,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    float3 axis_lat,
    float3 axis_ax,
    float3 axis_ele,
    float3 origin,
    const float* __restrict__ slice_grid_pack,
    float* __restrict__ out_alpha,
    float* __restrict__ out_intensity) {
    *out_alpha = 0.f;
    *out_intensity = 0.f;
    if (sg_probe_type(slice_grid_pack) != SLICE_GRID_PROBE_LINEAR) {
        return;
    }

    float xi_lat = 0.f;
    float xi_ax = 0.f;
    sg_linear_pixel_to_plane_mm(slice_grid_pack, col, row, &xi_lat, &xi_ax);

    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;
    float3 mw = make_float3(mptr[0], mptr[1], mptr[2]);
    float3 d = make_float3(mw.x - origin.x, mw.y - origin.y, mw.z - origin.z);
    const float mu_lat = ours_dot3(d, axis_lat);
    const float mu_ax = ours_dot3(d, axis_ax);
    const float mu_z = ours_dot3(d, axis_ele);

    float Rg[9];
    ours_quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
    const float sx = fmaxf(sptr[0], 1e-6f);
    const float sy = fmaxf(sptr[1], 1e-6f);
    const float sz = fmaxf(sptr[2], 1e-6f);

    float Sigma_w[9];
    plane_cond2d_device::mat3_sigma_world(sx, sy, sz, Rg, Sigma_w);
    float Sc[9];
    plane_cond2d_device::build_sigma_c_probe(Sigma_w, axis_lat, axis_ax, axis_ele, Sc);
    plane_cond2d_device::add_probe_cov_jitter(Sc);

    float mu2_lat = 0.f;
    float mu2_ax = 0.f;
    float s2_00 = 0.f;
    float s2_01 = 0.f;
    float s2_11 = 0.f;
    float i00 = 0.f;
    float i01 = 0.f;
    float i11 = 0.f;
    if (!plane_cond2d_device::schur_conditional_plane(
            mu_lat, mu_ax, mu_z, Sc, &mu2_lat, &mu2_ax, &s2_00, &s2_01, &s2_11, &i00, &i01, &i11)) {
        return;
    }

    float r_lat = 0.f;
    float r_ax = 0.f;
    plane_cond2d_device::plane_cond2d_schur_aabb_radii_mm(s2_00, s2_01, s2_11, k_sigma, &r_lat, &r_ax);
    if (fabsf(xi_lat - mu2_lat) > r_lat || fabsf(xi_ax - mu2_ax) > r_ax) {
        return;
    }

    const float dlat = xi_lat - mu2_lat;
    const float dax = xi_ax - mu2_ax;
    const float quad_raw = dlat * (i00 * dlat + i01 * dax) + dax * (i01 * dlat + i11 * dax);
    const float base_plane = plane_cond2d_device::plane_cond2d_plane_weight(quad_raw);
    const float elev_quad = (mu_z * mu_z) / Sc[8];
    const float elev_atten = expf(-0.5f * elev_quad);
    const float base = base_plane * elev_atten;
    float alpha = gauss_opacities[g] * base;
    alpha = fminf(fmaxf(alpha, 0.f), 1.f);
    *out_alpha = alpha;
    *out_intensity = gauss_intensities[g];
}
