#pragma once

#include "shader/plane_cond2d_device.cuh"
#include "shader/shared.cuh"
#include "shader/slice_grid_geometry.cuh"

namespace us_focus_device {

constexpr int US_FOCUS_PARAMS_LEN = 14;
constexpr int US_FOCUS_SIGMA_ELE_0 = 0;
constexpr int US_FOCUS_SIGMA_ELE_1 = 1;
constexpr int US_FOCUS_SIGMA_ELE_MIN = 2;
constexpr int US_FOCUS_SIGMA_ELE_MAX = 3;
constexpr int US_FOCUS_SIGMA_LAT_0 = 4;
constexpr int US_FOCUS_SIGMA_LAT_1 = 5;
constexpr int US_FOCUS_SIGMA_LAT_MIN = 6;
constexpr int US_FOCUS_SIGMA_LAT_MAX = 7;
constexpr int US_FOCUS_DET_GAMMA = 8;
constexpr int US_FOCUS_DET_EPS = 9;
constexpr int US_FOCUS_WEIGHT_MIN = 10;
constexpr int US_FOCUS_USE_LATERAL_FOCUS = 11;
constexpr int US_FOCUS_INFLATE_LAT_MM = 12;
constexpr int US_FOCUS_INFLATE_AX_MM = 13;

struct UsFocusEvalOut {
    float weight;
    float intensity;
    float mu2_lat;
    float mu2_ax;
    float sig2_00;
    float sig2_01;
    float sig2_11;
    float inv2_00;
    float inv2_01;
    float inv2_11;
    float det2;
    float slice_response;
    float norm_factor;
    float quad;
};

struct UsFocusFootprintOut {
    float mu2_lat;
    float mu2_ax;
    float sig2_00;
    float sig2_01;
    float sig2_11;
    float inv2_00;
    float inv2_01;
    float inv2_11;
    float det2;
    float sigma_ele;
    float sigma_lat;
    float mu_ax;
    float mu_ele;
    float mu_e_lat;
    float mu_e_ax;
    float mu_e_ele;
    float sc22;
    float lc00;
    float lc01;
    float lc02;
    float lc10;
    float lc11;
    float lc12;
    float lc20;
    float lc21;
    float lc22;
    float se00;
    float se01;
    float se02;
    float se10;
    float se11;
    float se12;
    float se20;
    float se21;
    float se22;
    float b00;
    float b01;
    float b02;
    float b10;
    float b11;
    float b12;
    float r_lat_mm;
    float r_ax_mm;
};

__device__ __forceinline__ float clamp_param(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ float us_focus_param_or_default(
    const float* __restrict__ params,
    int idx,
    float default_v) {
    return params == nullptr ? default_v : params[idx];
}

__device__ __forceinline__ float det_norm_factor(float det2_plus_eps, float det_gamma) {
    if (det_gamma == 0.f) {
        return 1.f;
    }
    if (fabsf(det_gamma - 0.25f) < 1e-7f) {
        return 1.f / sqrtf(sqrtf(det2_plus_eps));
    }
    if (fabsf(det_gamma - 0.5f) < 1e-7f) {
        return 1.f / sqrtf(det2_plus_eps);
    }
    if (fabsf(det_gamma - 1.f) < 1e-7f) {
        return 1.f / det2_plus_eps;
    }
    return powf(det2_plus_eps, -det_gamma);
}

__device__ __forceinline__ bool invert3x3_sym(const float M[9], float Minv[9]) {
    const float a00 = M[0];
    const float a01 = M[1];
    const float a02 = M[2];
    const float a10 = M[3];
    const float a11 = M[4];
    const float a12 = M[5];
    const float a20 = M[6];
    const float a21 = M[7];
    const float a22 = M[8];

    const float c00 = a11 * a22 - a12 * a21;
    const float c01 = a02 * a21 - a01 * a22;
    const float c02 = a01 * a12 - a02 * a11;
    const float c10 = a12 * a20 - a10 * a22;
    const float c11 = a00 * a22 - a02 * a20;
    const float c12 = a02 * a10 - a00 * a12;
    const float c20 = a10 * a21 - a11 * a20;
    const float c21 = a01 * a20 - a00 * a21;
    const float c22 = a00 * a11 - a01 * a10;

    const float det = a00 * c00 + a01 * c10 + a02 * c20;
    if (!(det > 1e-12f) || !isfinite(det)) {
        return false;
    }
    const float inv_det = 1.f / det;
    Minv[0] = c00 * inv_det;
    Minv[1] = c01 * inv_det;
    Minv[2] = c02 * inv_det;
    Minv[3] = c10 * inv_det;
    Minv[4] = c11 * inv_det;
    Minv[5] = c12 * inv_det;
    Minv[6] = c20 * inv_det;
    Minv[7] = c21 * inv_det;
    Minv[8] = c22 * inv_det;
    return true;
}

__device__ __forceinline__ void mat3_vec(const float M[9], const float v[3], float out[3]) {
    out[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
    out[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
    out[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
}

__device__ __forceinline__ bool invert2x2_sym(
    float s00,
    float s01,
    float s11,
    float det_eps,
    float* inv00,
    float* inv01,
    float* inv11,
    float* det2) {
    *det2 = s00 * s11 - s01 * s01;
    if (!(*det2 > det_eps) || !isfinite(*det2)) {
        return false;
    }
    const float inv_det = 1.f / *det2;
    *inv00 = s11 * inv_det;
    *inv01 = -s01 * inv_det;
    *inv11 = s00 * inv_det;
    return true;
}

__device__ __forceinline__ bool compute_us_focus_footprint_mm(
    const float* __restrict__ mptr,
    const float* __restrict__ sptr,
    const float* __restrict__ qptr,
    float3 axis_lat,
    float3 axis_ax,
    float3 axis_ele,
    float3 origin,
    float axial_near_mm,
    float k_sigma,
    const float* __restrict__ us_focus_params,
    UsFocusFootprintOut* __restrict__ out) {
    float3 d = make_float3(mptr[0] - origin.x, mptr[1] - origin.y, mptr[2] - origin.z);
    const float mu_lat = ours_dot3(d, axis_lat);
    const float mu_ax = ours_dot3(d, axis_ax);
    const float mu_ele = ours_dot3(d, axis_ele);

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

    float Lc[9];
    if (!invert3x3_sym(Sc, Lc)) {
        return false;
    }

    const float depth = fmaxf(mu_ax - axial_near_mm, 0.f);
    const float sigma_ele_0 = us_focus_param_or_default(us_focus_params, US_FOCUS_SIGMA_ELE_0, 1.f);
    const float sigma_ele_1 = us_focus_param_or_default(us_focus_params, US_FOCUS_SIGMA_ELE_1, 0.f);
    const float sigma_ele_min = us_focus_param_or_default(us_focus_params, US_FOCUS_SIGMA_ELE_MIN, 0.5f);
    const float sigma_ele_max = us_focus_param_or_default(us_focus_params, US_FOCUS_SIGMA_ELE_MAX, 8.f);
    const float sigma_lat_0 = us_focus_param_or_default(us_focus_params, US_FOCUS_SIGMA_LAT_0, 1.f);
    const float sigma_lat_1 = us_focus_param_or_default(us_focus_params, US_FOCUS_SIGMA_LAT_1, 0.f);
    const float sigma_lat_min = us_focus_param_or_default(us_focus_params, US_FOCUS_SIGMA_LAT_MIN, 0.5f);
    const float sigma_lat_max = us_focus_param_or_default(us_focus_params, US_FOCUS_SIGMA_LAT_MAX, 8.f);
    const bool use_lateral_focus =
        us_focus_param_or_default(us_focus_params, US_FOCUS_USE_LATERAL_FOCUS, 0.f) != 0.f;

    const float sigma_ele = clamp_param(sigma_ele_0 + sigma_ele_1 * depth, sigma_ele_min, sigma_ele_max);
    const float sigma_lat = clamp_param(sigma_lat_0 + sigma_lat_1 * depth, sigma_lat_min, sigma_lat_max);

    float Le[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        Le[i] = Lc[i];
    }
    if (use_lateral_focus) {
        Le[0] += 1.f / fmaxf(sigma_lat * sigma_lat, 1e-12f);
    }
    Le[8] += 1.f / fmaxf(sigma_ele * sigma_ele, 1e-12f);

    float Se[9];
    if (!invert3x3_sym(Le, Se)) {
        return false;
    }

    const float mu_c[3] = {mu_lat, mu_ax, mu_ele};
    float tmp[3];
    mat3_vec(Lc, mu_c, tmp);
    float mu_e[3];
    mat3_vec(Se, tmp, mu_e);
    float B[9];
    B[0] = Se[0] * Lc[0] + Se[1] * Lc[3] + Se[2] * Lc[6];
    B[1] = Se[0] * Lc[1] + Se[1] * Lc[4] + Se[2] * Lc[7];
    B[2] = Se[0] * Lc[2] + Se[1] * Lc[5] + Se[2] * Lc[8];
    B[3] = Se[3] * Lc[0] + Se[4] * Lc[3] + Se[5] * Lc[6];
    B[4] = Se[3] * Lc[1] + Se[4] * Lc[4] + Se[5] * Lc[7];
    B[5] = Se[3] * Lc[2] + Se[4] * Lc[5] + Se[5] * Lc[8];

    const float inflate_lat = fmaxf(us_focus_param_or_default(us_focus_params, US_FOCUS_INFLATE_LAT_MM, 0.f), 0.f);
    const float inflate_ax = fmaxf(us_focus_param_or_default(us_focus_params, US_FOCUS_INFLATE_AX_MM, 0.f), 0.f);
    const float s2_00 = Se[0] + inflate_lat * inflate_lat;
    const float s2_01 = 0.5f * (Se[1] + Se[3]);
    const float s2_11 = Se[4] + inflate_ax * inflate_ax;
    const float det_eps = us_focus_param_or_default(us_focus_params, US_FOCUS_DET_EPS, 1e-6f);
    float inv00 = 0.f;
    float inv01 = 0.f;
    float inv11 = 0.f;
    float det2 = 0.f;
    if (!invert2x2_sym(s2_00, s2_01, s2_11, det_eps, &inv00, &inv01, &inv11, &det2)) {
        return false;
    }

    out->mu2_lat = mu_e[0];
    out->mu2_ax = mu_e[1];
    out->sig2_00 = s2_00;
    out->sig2_01 = s2_01;
    out->sig2_11 = s2_11;
    out->inv2_00 = inv00;
    out->inv2_01 = inv01;
    out->inv2_11 = inv11;
    out->det2 = det2;
    out->sigma_ele = sigma_ele;
    out->sigma_lat = sigma_lat;
    out->mu_ax = mu_ax;
    out->mu_ele = mu_ele;
    out->mu_e_lat = mu_e[0];
    out->mu_e_ax = mu_e[1];
    out->mu_e_ele = mu_e[2];
    out->sc22 = Sc[8];
    out->lc00 = Lc[0];
    out->lc01 = Lc[1];
    out->lc02 = Lc[2];
    out->lc10 = Lc[3];
    out->lc11 = Lc[4];
    out->lc12 = Lc[5];
    out->lc20 = Lc[6];
    out->lc21 = Lc[7];
    out->lc22 = Lc[8];
    out->se00 = Se[0];
    out->se01 = Se[1];
    out->se02 = Se[2];
    out->se10 = Se[3];
    out->se11 = Se[4];
    out->se12 = Se[5];
    out->se20 = Se[6];
    out->se21 = Se[7];
    out->se22 = Se[8];
    out->b00 = B[0];
    out->b01 = B[1];
    out->b02 = B[2];
    out->b10 = B[3];
    out->b11 = B[4];
    out->b12 = B[5];

    const float cross = fabsf(s2_01);
    out->r_lat_mm = k_sigma * sqrtf(fmaxf(s2_00 + cross, 1e-12f));
    out->r_ax_mm = k_sigma * sqrtf(fmaxf(s2_11 + cross, 1e-12f));
    return true;
}

__device__ __forceinline__ bool us_focus_eval_at_pixel(
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
    const float* __restrict__ us_focus_params,
    UsFocusEvalOut* __restrict__ out) {
    if (sg_probe_type(slice_grid_pack) != SLICE_GRID_PROBE_LINEAR) {
        return false;
    }

    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;

    UsFocusFootprintOut fp;
    if (!compute_us_focus_footprint_mm(
            mptr,
            sptr,
            qptr,
            axis_lat,
            axis_ax,
            axis_ele,
            origin,
            slice_grid_pack[SG_NEAR_MM],
            k_sigma,
            us_focus_params,
            &fp)) {
        return false;
    }

    float xi_lat = 0.f;
    float xi_ax = 0.f;
    sg_linear_pixel_to_plane_mm(slice_grid_pack, col, row, &xi_lat, &xi_ax);
    if (fabsf(xi_lat - fp.mu2_lat) > fp.r_lat_mm || fabsf(xi_ax - fp.mu2_ax) > fp.r_ax_mm) {
        return false;
    }

    const float dlat = xi_lat - fp.mu2_lat;
    const float dax = xi_ax - fp.mu2_ax;
    const float quad = dlat * (fp.inv2_00 * dlat + fp.inv2_01 * dax)
        + dax * (fp.inv2_01 * dlat + fp.inv2_11 * dax);
    if (!(quad >= 0.f) || !isfinite(quad)) {
        return false;
    }

    const float det_eps = us_focus_param_or_default(us_focus_params, US_FOCUS_DET_EPS, 1e-6f);
    const float det_gamma = us_focus_param_or_default(us_focus_params, US_FOCUS_DET_GAMMA, 0.25f);
    const float weight_min = us_focus_param_or_default(us_focus_params, US_FOCUS_WEIGHT_MIN, 1e-6f);
    const float phi = expf(-0.5f * quad);
    const float denom_ele = fp.sc22 + fp.sigma_ele * fp.sigma_ele + det_eps;
    const float slice_response = expf(-0.5f * fp.mu_ele * fp.mu_ele / fmaxf(denom_ele, 1e-12f));
    const float norm_factor = det_norm_factor(fp.det2 + det_eps, det_gamma);
    const float weight = gauss_opacities[g] * slice_response * norm_factor * phi;
    if (!isfinite(weight) || weight < weight_min) {
        return false;
    }

    out->weight = weight;
    out->intensity = gauss_intensities[g];
    out->mu2_lat = fp.mu2_lat;
    out->mu2_ax = fp.mu2_ax;
    out->sig2_00 = fp.sig2_00;
    out->sig2_01 = fp.sig2_01;
    out->sig2_11 = fp.sig2_11;
    out->inv2_00 = fp.inv2_00;
    out->inv2_01 = fp.inv2_01;
    out->inv2_11 = fp.inv2_11;
    out->det2 = fp.det2;
    out->slice_response = slice_response;
    out->norm_factor = norm_factor;
    out->quad = quad;
    return true;
}

}  // namespace us_focus_device
