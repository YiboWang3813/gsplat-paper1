#include <cuda_runtime.h>
#include <math_constants.h>

#include "shader/us_focus.cuh"
#include "shader/us_focus_device.cuh"

namespace {

using plane_cond2d_device::kGradClamp;

constexpr int US_FOCUS_TILE_SHAPE_BATCH = 64;
constexpr int US_FOCUS_BWD_WSP_STRIDE = 35;
constexpr int US_FOCUS_WSP_G_MU_PROBE = 0;
constexpr int US_FOCUS_WSP_G_SC = 3;
constexpr int US_FOCUS_WSP_G_MU_E_SUM = 12;
constexpr int US_FOCUS_WSP_G_SE = 15;
constexpr int US_FOCUS_WSP_G_OPACITY = 24;
constexpr int US_FOCUS_WSP_G_INTENSITY = 25;
constexpr int US_FOCUS_WSP_G_FP_EVAL = 26;

struct UsFocusTileShapePairCandidate {
    int g;
    float intensity;
    float base_weight_p;
    float base_weight_m;
    float mu2_lat_p;
    float mu2_ax_p;
    float inv2_00_p;
    float inv2_01_p;
    float inv2_11_p;
    float r_lat_mm_p;
    float r_ax_mm_p;
    float mu2_lat_m;
    float mu2_ax_m;
    float inv2_00_m;
    float inv2_01_m;
    float inv2_11_m;
    float r_lat_mm_m;
    float r_ax_mm_m;
};

struct UsFocusEvalFootprintAdjoint {
    float mu2_lat;
    float mu2_ax;
    float inv2_00;
    float inv2_01;
    float inv2_11;
    float det2;
    float mu_ele;
    float sc22;
    float sigma_ele;
};

__device__ __forceinline__ float us_focus_dot3f(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 us_focus_normalize3_backward(
    const float3& raw,
    const float3& n_unit,
    const float3& grad_out) {
    const float len = sqrtf(fmaxf(1e-12f, us_focus_dot3f(raw, raw)));
    const float inv_len = 1.f / len;
    const float dot_ng = us_focus_dot3f(n_unit, grad_out);
    return make_float3(
        (grad_out.x - n_unit.x * dot_ng) * inv_len,
        (grad_out.y - n_unit.y * dot_ng) * inv_len,
        (grad_out.z - n_unit.z * dot_ng) * inv_len);
}

__device__ void us_focus_sc_to_world_grads(
    const float g_Sc[9],
    float3 axis_lat,
    float3 axis_ax,
    float3 axis_ele,
    float gSigma_w[9]) {
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        gSigma_w[i] = 0.f;
    }
    const float3 axes[3] = {axis_lat, axis_ax, axis_ele};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const float gij = g_Sc[i * 3 + j];
            const float3 ai = axes[i];
            const float3 aj = axes[j];
            gSigma_w[0] += gij * ai.x * aj.x;
            gSigma_w[1] += gij * ai.x * aj.y;
            gSigma_w[2] += gij * ai.x * aj.z;
            gSigma_w[3] += gij * ai.y * aj.x;
            gSigma_w[4] += gij * ai.y * aj.y;
            gSigma_w[5] += gij * ai.y * aj.z;
            gSigma_w[6] += gij * ai.z * aj.x;
            gSigma_w[7] += gij * ai.z * aj.y;
            gSigma_w[8] += gij * ai.z * aj.z;
        }
    }
}

__device__ void us_focus_sc_to_axis_grads(
    const float g_Sc[9],
    const float Sigma_w[9],
    float3 axis_lat,
    float3 axis_ax,
    float3 axis_ele,
    float3* g_axis_lat,
    float3* g_axis_ax,
    float3* g_axis_ele) {
    *g_axis_lat = make_float3(0.f, 0.f, 0.f);
    *g_axis_ax = make_float3(0.f, 0.f, 0.f);
    *g_axis_ele = make_float3(0.f, 0.f, 0.f);
    float3 axes[3] = {axis_lat, axis_ax, axis_ele};
    float3* g_axes[3] = {g_axis_lat, g_axis_ax, g_axis_ele};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const float gij = g_Sc[i * 3 + j];
            const float3 Sai = plane_cond2d_device::mat3_vec_sym(Sigma_w, axes[i]);
            const float3 Saj = plane_cond2d_device::mat3_vec_sym(Sigma_w, axes[j]);
            g_axes[i]->x += gij * Saj.x;
            g_axes[i]->y += gij * Saj.y;
            g_axes[i]->z += gij * Saj.z;
            g_axes[j]->x += gij * Sai.x;
            g_axes[j]->y += gij * Sai.y;
            g_axes[j]->z += gij * Sai.z;
        }
    }
}

__device__ __forceinline__ float us_focus_sanitize_grad(float g) {
    if (!isfinite(g)) {
        return 0.f;
    }
    if (fabsf(g) > kGradClamp) {
        return copysignf(kGradClamp, g);
    }
    return g;
}

__device__ __forceinline__ void us_focus_outer_add(const float a[3], const float b[3], float M[9]) {
    M[0] += a[0] * b[0];
    M[1] += a[0] * b[1];
    M[2] += a[0] * b[2];
    M[3] += a[1] * b[0];
    M[4] += a[1] * b[1];
    M[5] += a[1] * b[2];
    M[6] += a[2] * b[0];
    M[7] += a[2] * b[1];
    M[8] += a[2] * b[2];
}

__device__ __forceinline__ void us_focus_atomic_add_if_nonzero(float* ptr, float v) {
    if (v != 0.f) {
        atomicAdd(ptr, v);
    }
}

__device__ __forceinline__ bool us_focus_fill_shape_pair_candidate(
    int g,
    bool is_quat,
    int param_idx,
    float eps,
    float k_sigma,
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
    UsFocusTileShapePairCandidate* __restrict__ out) {
    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;
    float sp[3] = {sptr[0], sptr[1], sptr[2]};
    float sm[3] = {sptr[0], sptr[1], sptr[2]};
    float qp[4] = {qptr[0], qptr[1], qptr[2], qptr[3]};
    float qm[4] = {qptr[0], qptr[1], qptr[2], qptr[3]};
    if (is_quat) {
        qp[param_idx] += eps;
        qm[param_idx] -= eps;
    } else {
        sp[param_idx] += eps;
        sm[param_idx] -= eps;
    }

    us_focus_device::UsFocusFootprintOut fp_p;
    const bool ok_p = us_focus_device::compute_us_focus_footprint_mm(
        mptr,
        sp,
        qp,
        axis_lat,
        axis_ax,
        axis_ele,
        origin,
        slice_grid_pack[SG_NEAR_MM],
        k_sigma,
        us_focus_params,
        &fp_p);
    us_focus_device::UsFocusFootprintOut fp_m;
    const bool ok_m = us_focus_device::compute_us_focus_footprint_mm(
        mptr,
        sm,
        qm,
        axis_lat,
        axis_ax,
        axis_ele,
        origin,
        slice_grid_pack[SG_NEAR_MM],
        k_sigma,
        us_focus_params,
        &fp_m);
    if (!ok_p && !ok_m) {
        out->g = -1;
        return false;
    }

    const float det_eps =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_DET_EPS, 1e-6f);
    const float det_gamma =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_DET_GAMMA, 0.25f);
    const float opacity = gauss_opacities[g];
    const float denom_ele_p = ok_p ? fmaxf(fp_p.sc22 + fp_p.sigma_ele * fp_p.sigma_ele + det_eps, 1e-12f) : 1.f;
    const float slice_response_p = ok_p ? expf(-0.5f * fp_p.mu_ele * fp_p.mu_ele / denom_ele_p) : 0.f;
    const float norm_factor_p = ok_p ? us_focus_device::det_norm_factor(fp_p.det2 + det_eps, det_gamma) : 0.f;
    const float denom_ele_m = ok_m ? fmaxf(fp_m.sc22 + fp_m.sigma_ele * fp_m.sigma_ele + det_eps, 1e-12f) : 1.f;
    const float slice_response_m = ok_m ? expf(-0.5f * fp_m.mu_ele * fp_m.mu_ele / denom_ele_m) : 0.f;
    const float norm_factor_m = ok_m ? us_focus_device::det_norm_factor(fp_m.det2 + det_eps, det_gamma) : 0.f;

    out->g = g;
    out->intensity = gauss_intensities[g];
    out->base_weight_p = ok_p ? opacity * slice_response_p * norm_factor_p : 0.f;
    out->base_weight_m = ok_m ? opacity * slice_response_m * norm_factor_m : 0.f;
    out->mu2_lat_p = ok_p ? fp_p.mu2_lat : 0.f;
    out->mu2_ax_p = ok_p ? fp_p.mu2_ax : 0.f;
    out->inv2_00_p = ok_p ? fp_p.inv2_00 : 0.f;
    out->inv2_01_p = ok_p ? fp_p.inv2_01 : 0.f;
    out->inv2_11_p = ok_p ? fp_p.inv2_11 : 0.f;
    out->r_lat_mm_p = ok_p ? fp_p.r_lat_mm : 0.f;
    out->r_ax_mm_p = ok_p ? fp_p.r_ax_mm : 0.f;
    out->mu2_lat_m = ok_m ? fp_m.mu2_lat : 0.f;
    out->mu2_ax_m = ok_m ? fp_m.mu2_ax : 0.f;
    out->inv2_00_m = ok_m ? fp_m.inv2_00 : 0.f;
    out->inv2_01_m = ok_m ? fp_m.inv2_01 : 0.f;
    out->inv2_11_m = ok_m ? fp_m.inv2_11 : 0.f;
    out->r_lat_mm_m = ok_m ? fp_m.r_lat_mm : 0.f;
    out->r_ax_mm_m = ok_m ? fp_m.r_ax_mm : 0.f;
    return isfinite(out->base_weight_p) || isfinite(out->base_weight_m);
}

__device__ __forceinline__ float us_focus_eval_shape_pair_pixel(
    const UsFocusTileShapePairCandidate& cand,
    float xi_lat,
    float xi_ax,
    float upstream_intensity,
    float weight_min,
    float inv_2eps) {
    if (cand.g < 0) {
        return 0.f;
    }
    float lp = 0.f;
    if (cand.base_weight_p != 0.f &&
        fabsf(xi_lat - cand.mu2_lat_p) <= cand.r_lat_mm_p &&
        fabsf(xi_ax - cand.mu2_ax_p) <= cand.r_ax_mm_p) {
        const float dlat = xi_lat - cand.mu2_lat_p;
        const float dax = xi_ax - cand.mu2_ax_p;
        const float qlat = cand.inv2_00_p * dlat + cand.inv2_01_p * dax;
        const float qax = cand.inv2_01_p * dlat + cand.inv2_11_p * dax;
        const float quad = dlat * qlat + dax * qax;
        if (quad >= 0.f && isfinite(quad)) {
            const float weight = cand.base_weight_p * expf(-0.5f * quad);
            if (isfinite(weight) && weight >= weight_min) {
                lp = upstream_intensity * weight;
            }
        }
    }
    float lm = 0.f;
    if (cand.base_weight_m != 0.f &&
        fabsf(xi_lat - cand.mu2_lat_m) <= cand.r_lat_mm_m &&
        fabsf(xi_ax - cand.mu2_ax_m) <= cand.r_ax_mm_m) {
        const float dlat = xi_lat - cand.mu2_lat_m;
        const float dax = xi_ax - cand.mu2_ax_m;
        const float qlat = cand.inv2_00_m * dlat + cand.inv2_01_m * dax;
        const float qax = cand.inv2_01_m * dlat + cand.inv2_11_m * dax;
        const float quad = dlat * qlat + dax * qax;
        if (quad >= 0.f && isfinite(quad)) {
            const float weight = cand.base_weight_m * expf(-0.5f * quad);
            if (isfinite(weight) && weight >= weight_min) {
                lm = upstream_intensity * weight;
            }
        }
    }
    return (lp - lm) * inv_2eps;
}

__device__ __forceinline__ float us_focus_footprint_eval_adjoint_dot(
    const us_focus_device::UsFocusFootprintOut& fp,
    const UsFocusEvalFootprintAdjoint& g) {
    return g.mu2_lat * fp.mu2_lat +
           g.mu2_ax * fp.mu2_ax +
           g.inv2_00 * fp.inv2_00 +
           g.inv2_01 * fp.inv2_01 +
           g.inv2_11 * fp.inv2_11 +
           g.det2 * fp.det2 +
           g.mu_ele * fp.mu_ele +
           g.sc22 * fp.sc22 +
           g.sigma_ele * fp.sigma_ele;
}

__device__ __forceinline__ float us_focus_eval_adjoint_shape_finite_diff(
    int g,
    bool is_quat,
    int param_idx,
    float eps,
    float k_sigma,
    const UsFocusEvalFootprintAdjoint& g_fp_eval,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    float3 axis_lat,
    float3 axis_ax,
    float3 axis_ele,
    float3 origin,
    const float* __restrict__ slice_grid_pack,
    const float* __restrict__ us_focus_params) {
    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;
    float sp[3] = {sptr[0], sptr[1], sptr[2]};
    float sm[3] = {sptr[0], sptr[1], sptr[2]};
    float qp[4] = {qptr[0], qptr[1], qptr[2], qptr[3]};
    float qm[4] = {qptr[0], qptr[1], qptr[2], qptr[3]};
    if (is_quat) {
        qp[param_idx] += eps;
        qm[param_idx] -= eps;
    } else {
        sp[param_idx] += eps;
        sm[param_idx] -= eps;
    }

    us_focus_device::UsFocusFootprintOut fp_p;
    const bool ok_p = us_focus_device::compute_us_focus_footprint_mm(
        mptr,
        sp,
        qp,
        axis_lat,
        axis_ax,
        axis_ele,
        origin,
        slice_grid_pack[SG_NEAR_MM],
        k_sigma,
        us_focus_params,
        &fp_p);
    us_focus_device::UsFocusFootprintOut fp_m;
    const bool ok_m = us_focus_device::compute_us_focus_footprint_mm(
        mptr,
        sm,
        qm,
        axis_lat,
        axis_ax,
        axis_ele,
        origin,
        slice_grid_pack[SG_NEAR_MM],
        k_sigma,
        us_focus_params,
        &fp_m);
    const float lp = ok_p ? us_focus_footprint_eval_adjoint_dot(fp_p, g_fp_eval) : 0.f;
    const float lm = ok_m ? us_focus_footprint_eval_adjoint_dot(fp_m, g_fp_eval) : 0.f;
    return (lp - lm) / (2.f * eps);
}

__device__ __forceinline__ void us_focus_inv3_bwd(const float Inv[9], const float g_Inv[9], float g_M[9]) {
    float U[9];
    #pragma unroll
    for (int row = 0; row < 3; ++row) {
        #pragma unroll
        for (int col = 0; col < 3; ++col) {
            float acc = 0.f;
            #pragma unroll
            for (int a = 0; a < 3; ++a) {
                #pragma unroll
                for (int b = 0; b < 3; ++b) {
                    acc += Inv[a * 3 + row] * g_Inv[a * 3 + b] * Inv[col * 3 + b];
                }
            }
            U[row * 3 + col] = acc;
        }
    }
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        g_M[i] -= U[i];
    }
}

__device__ __forceinline__ void us_focus_accumulate_pixel_sufficient_backward(
    float xi_lat,
    float xi_ax,
    float upstream,
    const us_focus_device::UsFocusFootprintOut& fp,
    float denom_ele,
    float slice_response,
    float norm_factor,
    float det_eps,
    float det_gamma,
    float weight_min,
    float opacity,
    float intensity,
    float d_sigma_ele_d_ax,
    float g_mu_e_sum[3],
    float g_Se[9],
    float g_mu_probe_direct[3],
    float g_Sc_direct[9],
    float* g_opacity,
    float* g_intensity,
    UsFocusEvalFootprintAdjoint* g_fp_eval) {
    if (fabsf(xi_lat - fp.mu2_lat) > fp.r_lat_mm || fabsf(xi_ax - fp.mu2_ax) > fp.r_ax_mm) {
        return;
    }

    const float dlat = xi_lat - fp.mu2_lat;
    const float dax = xi_ax - fp.mu2_ax;
    const float qlat = fp.inv2_00 * dlat + fp.inv2_01 * dax;
    const float qax = fp.inv2_01 * dlat + fp.inv2_11 * dax;
    const float quad = dlat * qlat + dax * qax;
    if (!(quad >= 0.f) || !isfinite(quad)) {
        return;
    }

    const float phi = expf(-0.5f * quad);
    const float base_no_opacity = slice_response * norm_factor * phi;
    const float weight = opacity * base_no_opacity;
    if (!isfinite(weight) || weight < weight_min) {
        return;
    }

    const float g_weight = upstream * intensity;
    *g_intensity += upstream * weight;
    *g_opacity += g_weight * base_no_opacity;

    const float common = g_weight * weight;
    g_mu_e_sum[0] += common * qlat;
    g_mu_e_sum[1] += common * qax;
    if (g_fp_eval != nullptr) {
        g_fp_eval->mu2_lat += common * qlat;
        g_fp_eval->mu2_ax += common * qax;
        g_fp_eval->inv2_00 += common * (-0.5f * dlat * dlat);
        g_fp_eval->inv2_01 += common * (-dlat * dax);
        g_fp_eval->inv2_11 += common * (-0.5f * dax * dax);
    }

    const float inv_det_norm = 1.f / fmaxf(fp.det2 + det_eps, 1e-12f);
    if (g_fp_eval != nullptr) {
        g_fp_eval->det2 += common * (-det_gamma * inv_det_norm);
    }
    const float g_s2_00 = common * (0.5f * qlat * qlat - det_gamma * fp.sig2_11 * inv_det_norm);
    const float g_s2_01 = common * (qlat * qax + 2.f * det_gamma * fp.sig2_01 * inv_det_norm);
    const float g_s2_11 = common * (0.5f * qax * qax - det_gamma * fp.sig2_00 * inv_det_norm);
    g_Se[0] += g_s2_00;
    g_Se[1] += 0.5f * g_s2_01;
    g_Se[3] += 0.5f * g_s2_01;
    g_Se[4] += g_s2_11;

    const float inv_denom_ele = 1.f / denom_ele;
    g_mu_probe_direct[2] += common * (-fp.mu_ele * inv_denom_ele);
    g_Sc_direct[8] += common * (0.5f * fp.mu_ele * fp.mu_ele * inv_denom_ele * inv_denom_ele);
    g_mu_probe_direct[1] +=
        common * (fp.mu_ele * fp.mu_ele * d_sigma_ele_d_ax * inv_denom_ele * inv_denom_ele);
    if (g_fp_eval != nullptr) {
        const float inv_denom_ele2 = inv_denom_ele * inv_denom_ele;
        g_fp_eval->mu_ele += common * (-fp.mu_ele * inv_denom_ele);
        g_fp_eval->sc22 += common * (0.5f * fp.mu_ele * fp.mu_ele * inv_denom_ele2);
        g_fp_eval->sigma_ele += common * (fp.mu_ele * fp.mu_ele * fp.sigma_ele * inv_denom_ele2);
    }
}

__global__ void kernel_us_focus_additive_backward(
    int n_gauss,
    int H,
    int W,
    int tile_size_px,
    float k_sigma,
    const int4* __restrict__ gauss_rects,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    const float* __restrict__ pose,
    const float* __restrict__ slice_grid_pack,
    const float* __restrict__ us_focus_params,
    const float* __restrict__ grad_pixel_colors,
    const float* __restrict__ workspace,
    bool use_workspace_accum,
    bool compute_pose_grad,
    bool compute_shape_grad,
    float* __restrict__ grad_gauss_means,
    float* __restrict__ grad_gauss_scales,
    float* __restrict__ grad_gauss_quats,
    float* __restrict__ grad_gauss_opacities,
    float* __restrict__ grad_gauss_intensities,
    float* __restrict__ grad_pose_probe_to_world) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) return;
    if (sg_probe_type(slice_grid_pack) != SLICE_GRID_PROBE_LINEAR) return;

    int4 rect = gauss_rects[g];
    if (rect.x >= rect.z || rect.y >= rect.w) return;

    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 raw_ele = make_float3(pose[2], pose[6], pose[10]);
    float3 axis_lat = ours_normalize3(raw_lat);
    float3 axis_ax = ours_normalize3(raw_ax);
    float3 axis_ele = ours_normalize3(raw_ele);
    float3 origin = make_float3(pose[3], pose[7], pose[11]);

    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;
    us_focus_device::UsFocusFootprintOut fp;
    if (!us_focus_device::compute_us_focus_footprint_mm(
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
        return;
    }

    const float det_eps =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_DET_EPS, 1e-6f);
    const float det_gamma =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_DET_GAMMA, 0.25f);
    const float weight_min =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_WEIGHT_MIN, 1e-6f);
    const float opacity = gauss_opacities[g];
    const float intensity = gauss_intensities[g];
    const float denom_ele = fmaxf(fp.sc22 + fp.sigma_ele * fp.sigma_ele + det_eps, 1e-12f);
    const float slice_response = expf(-0.5f * fp.mu_ele * fp.mu_ele / denom_ele);
    const float norm_factor = us_focus_device::det_norm_factor(fp.det2 + det_eps, det_gamma);
    const float sigma_ele_0 =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_SIGMA_ELE_0, 1.f);
    const float sigma_ele_1 =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_SIGMA_ELE_1, 0.f);
    const float sigma_ele_min =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_SIGMA_ELE_MIN, 0.5f);
    const float sigma_ele_max =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_SIGMA_ELE_MAX, 8.f);
    const float sigma_lat_0 =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_SIGMA_LAT_0, 1.f);
    const float sigma_lat_1 =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_SIGMA_LAT_1, 0.f);
    const float sigma_lat_min =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_SIGMA_LAT_MIN, 0.5f);
    const float sigma_lat_max =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_SIGMA_LAT_MAX, 8.f);
    const bool use_lateral_focus =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_USE_LATERAL_FOCUS, 0.f) != 0.f;
    const float depth = fmaxf(fp.mu_ax - slice_grid_pack[SG_NEAR_MM], 0.f);
    const bool depth_active = fp.mu_ax > slice_grid_pack[SG_NEAR_MM];
    const float sigma_ele_raw = sigma_ele_0 + sigma_ele_1 * depth;
    const float sigma_lat_raw = sigma_lat_0 + sigma_lat_1 * depth;
    const float d_sigma_ele_d_ax =
        (depth_active && sigma_ele_raw > sigma_ele_min && sigma_ele_raw < sigma_ele_max) ? sigma_ele_1 : 0.f;
    const float d_sigma_lat_d_ax =
        (use_lateral_focus && depth_active && sigma_lat_raw > sigma_lat_min && sigma_lat_raw < sigma_lat_max)
            ? sigma_lat_1
            : 0.f;
    const float d_lambda_ele_d_ax =
        -2.f * d_sigma_ele_d_ax / fmaxf(fp.sigma_ele * fp.sigma_ele * fp.sigma_ele, 1e-12f);
    const float d_lambda_lat_d_ax =
        use_lateral_focus
            ? -2.f * d_sigma_lat_d_ax / fmaxf(fp.sigma_lat * fp.sigma_lat * fp.sigma_lat, 1e-12f)
            : 0.f;
    float3 dw = make_float3(mptr[0] - origin.x, mptr[1] - origin.y, mptr[2] - origin.z);
    const float mu_c[3] = {ours_dot3(dw, axis_lat), fp.mu_ax, fp.mu_ele};
    const float Lc[9] = {
        fp.lc00, fp.lc01, fp.lc02,
        fp.lc10, fp.lc11, fp.lc12,
        fp.lc20, fp.lc21, fp.lc22,
    };
    const float Se[9] = {
        fp.se00, fp.se01, fp.se02,
        fp.se10, fp.se11, fp.se12,
        fp.se20, fp.se21, fp.se22,
    };
    float y_lc_mu[3];
    us_focus_device::mat3_vec(Lc, mu_c, y_lc_mu);

    const int x0 = max(0, rect.x * tile_size_px);
    const int y0 = max(0, rect.y * tile_size_px);
    const int x1 = min(W, rect.z * tile_size_px);
    const int y1 = min(H, rect.w * tile_size_px);

    float g_mu_probe[3] = {0.f, 0.f, 0.f};
    float g_Sc[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float g_mu_e_sum[3] = {0.f, 0.f, 0.f};
    float g_Se[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float g_opacity = 0.f;
    float g_intensity = 0.f;
    UsFocusEvalFootprintAdjoint g_fp_eval = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const float lat_delta = slice_grid_pack[SG_DELTA_LAT_MM];
    const float ax_delta = slice_grid_pack[SG_DELTA_AX_MM];
    const float lat_col_offset = slice_grid_pack[SG_LAT_COL_OFFSET];
    const float near_mm = slice_grid_pack[SG_NEAR_MM];

    if (use_workspace_accum) {
        if (workspace == nullptr) {
            return;
        }
        const float* wsp = workspace + g * US_FOCUS_BWD_WSP_STRIDE;
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            g_mu_probe[i] = wsp[US_FOCUS_WSP_G_MU_PROBE + i];
            g_mu_e_sum[i] = wsp[US_FOCUS_WSP_G_MU_E_SUM + i];
        }
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            g_Sc[i] = wsp[US_FOCUS_WSP_G_SC + i];
            g_Se[i] = wsp[US_FOCUS_WSP_G_SE + i];
        }
        g_opacity = wsp[US_FOCUS_WSP_G_OPACITY];
        g_intensity = wsp[US_FOCUS_WSP_G_INTENSITY];
        g_fp_eval.mu2_lat = wsp[US_FOCUS_WSP_G_FP_EVAL + 0];
        g_fp_eval.mu2_ax = wsp[US_FOCUS_WSP_G_FP_EVAL + 1];
        g_fp_eval.inv2_00 = wsp[US_FOCUS_WSP_G_FP_EVAL + 2];
        g_fp_eval.inv2_01 = wsp[US_FOCUS_WSP_G_FP_EVAL + 3];
        g_fp_eval.inv2_11 = wsp[US_FOCUS_WSP_G_FP_EVAL + 4];
        g_fp_eval.det2 = wsp[US_FOCUS_WSP_G_FP_EVAL + 5];
        g_fp_eval.mu_ele = wsp[US_FOCUS_WSP_G_FP_EVAL + 6];
        g_fp_eval.sc22 = wsp[US_FOCUS_WSP_G_FP_EVAL + 7];
        g_fp_eval.sigma_ele = wsp[US_FOCUS_WSP_G_FP_EVAL + 8];
    } else {
        for (int y = y0; y < y1; ++y) {
            const float xi_ax = static_cast<float>(y) * ax_delta + near_mm;
            for (int x = x0; x < x1; ++x) {
                const int p = y * W + x;
                const float xi_lat = (static_cast<float>(x) - lat_col_offset) * lat_delta;
                us_focus_accumulate_pixel_sufficient_backward(
                    xi_lat,
                    xi_ax,
                    grad_pixel_colors[p],
                    fp,
                    denom_ele,
                    slice_response,
                    norm_factor,
                    det_eps,
                    det_gamma,
                    weight_min,
                    opacity,
                    intensity,
                    d_sigma_ele_d_ax,
                    g_mu_e_sum,
                    g_Se,
                    g_mu_probe,
                    g_Sc,
                    &g_opacity,
                    &g_intensity,
                    &g_fp_eval);
            }
        }
    }

    float g_y[3] = {
        Se[0] * g_mu_e_sum[0] + Se[3] * g_mu_e_sum[1] + Se[6] * g_mu_e_sum[2],
        Se[1] * g_mu_e_sum[0] + Se[4] * g_mu_e_sum[1] + Se[7] * g_mu_e_sum[2],
        Se[2] * g_mu_e_sum[0] + Se[5] * g_mu_e_sum[1] + Se[8] * g_mu_e_sum[2],
    };
    float g_Lc[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    us_focus_outer_add(g_y, mu_c, g_Lc);
    g_mu_probe[0] += Lc[0] * g_y[0] + Lc[3] * g_y[1] + Lc[6] * g_y[2];
    g_mu_probe[1] += Lc[1] * g_y[0] + Lc[4] * g_y[1] + Lc[7] * g_y[2];
    g_mu_probe[2] += Lc[2] * g_y[0] + Lc[5] * g_y[1] + Lc[8] * g_y[2];

    us_focus_outer_add(g_mu_e_sum, y_lc_mu, g_Se);
    float g_Le[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    us_focus_inv3_bwd(Se, g_Se, g_Le);
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        g_Lc[i] += g_Le[i];
    }
    g_mu_probe[1] += g_Le[0] * d_lambda_lat_d_ax + g_Le[8] * d_lambda_ele_d_ax;
    us_focus_inv3_bwd(Lc, g_Lc, g_Sc);

    grad_gauss_intensities[g] = us_focus_sanitize_grad(g_intensity);
    grad_gauss_opacities[g] = us_focus_sanitize_grad(g_opacity);
    if (compute_shape_grad) {
        float fd_scales[3] = {0.f, 0.f, 0.f};
        float fd_quats[4] = {0.f, 0.f, 0.f, 0.f};
        const float eps_scale = 1e-3f;
        const float eps_quat = 1e-3f;
        for (int k = 0; k < 3; ++k) {
            fd_scales[k] = us_focus_eval_adjoint_shape_finite_diff(
                g,
                false,
                k,
                eps_scale,
                k_sigma,
                g_fp_eval,
                gauss_means,
                gauss_scales,
                gauss_quats,
                axis_lat,
                axis_ax,
                axis_ele,
                origin,
                slice_grid_pack,
                us_focus_params);
        }
        for (int k = 0; k < 4; ++k) {
            fd_quats[k] = us_focus_eval_adjoint_shape_finite_diff(
                g,
                true,
                k,
                eps_quat,
                k_sigma,
                g_fp_eval,
                gauss_means,
                gauss_scales,
                gauss_quats,
                axis_lat,
                axis_ax,
                axis_ele,
                origin,
                slice_grid_pack,
                us_focus_params);
        }
        grad_gauss_scales[g * 3 + 0] = us_focus_sanitize_grad(fd_scales[0]);
        grad_gauss_scales[g * 3 + 1] = us_focus_sanitize_grad(fd_scales[1]);
        grad_gauss_scales[g * 3 + 2] = us_focus_sanitize_grad(fd_scales[2]);
        grad_gauss_quats[g * 4 + 0] = us_focus_sanitize_grad(fd_quats[0]);
        grad_gauss_quats[g * 4 + 1] = us_focus_sanitize_grad(fd_quats[1]);
        grad_gauss_quats[g * 4 + 2] = us_focus_sanitize_grad(fd_quats[2]);
        grad_gauss_quats[g * 4 + 3] = us_focus_sanitize_grad(fd_quats[3]);
    }

    float3 g_delta = make_float3(
        g_mu_probe[0] * axis_lat.x + g_mu_probe[1] * axis_ax.x + g_mu_probe[2] * axis_ele.x,
        g_mu_probe[0] * axis_lat.y + g_mu_probe[1] * axis_ax.y + g_mu_probe[2] * axis_ele.y,
        g_mu_probe[0] * axis_lat.z + g_mu_probe[1] * axis_ax.z + g_mu_probe[2] * axis_ele.z);
    grad_gauss_means[g * 3 + 0] = us_focus_sanitize_grad(g_delta.x);
    grad_gauss_means[g * 3 + 1] = us_focus_sanitize_grad(g_delta.y);
    grad_gauss_means[g * 3 + 2] = us_focus_sanitize_grad(g_delta.z);

    if (!compute_pose_grad) {
        return;
    }

    float Rg[9];
    ours_quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
    const float sx = fmaxf(sptr[0], 1e-6f);
    const float sy = fmaxf(sptr[1], 1e-6f);
    const float sz = fmaxf(sptr[2], 1e-6f);
    float gSigma_w[9];
    us_focus_sc_to_world_grads(g_Sc, axis_lat, axis_ax, axis_ele, gSigma_w);

    atomicAdd(grad_pose_probe_to_world + 3, -g_delta.x);
    atomicAdd(grad_pose_probe_to_world + 7, -g_delta.y);
    atomicAdd(grad_pose_probe_to_world + 11, -g_delta.z);

    float3 g_axis_lat = make_float3(0.f, 0.f, 0.f);
    float3 g_axis_ax = make_float3(0.f, 0.f, 0.f);
    float3 g_axis_ele = make_float3(0.f, 0.f, 0.f);
    float Sigma_w[9];
    plane_cond2d_device::mat3_sigma_world(sx, sy, sz, Rg, Sigma_w);
    us_focus_sc_to_axis_grads(g_Sc, Sigma_w, axis_lat, axis_ax, axis_ele, &g_axis_lat, &g_axis_ax, &g_axis_ele);

    float3 g_lat_mu = make_float3(g_mu_probe[0] * dw.x, g_mu_probe[0] * dw.y, g_mu_probe[0] * dw.z);
    float3 g_ax_mu = make_float3(g_mu_probe[1] * dw.x, g_mu_probe[1] * dw.y, g_mu_probe[1] * dw.z);
    float3 g_ele_mu = make_float3(g_mu_probe[2] * dw.x, g_mu_probe[2] * dw.y, g_mu_probe[2] * dw.z);

    float3 g_raw_lat = us_focus_normalize3_backward(
        raw_lat,
        axis_lat,
        make_float3(g_lat_mu.x + g_axis_lat.x, g_lat_mu.y + g_axis_lat.y, g_lat_mu.z + g_axis_lat.z));
    float3 g_raw_ax = us_focus_normalize3_backward(
        raw_ax,
        axis_ax,
        make_float3(g_ax_mu.x + g_axis_ax.x, g_ax_mu.y + g_axis_ax.y, g_ax_mu.z + g_axis_ax.z));
    float3 g_raw_ele = us_focus_normalize3_backward(
        raw_ele,
        axis_ele,
        make_float3(g_ele_mu.x + g_axis_ele.x, g_ele_mu.y + g_axis_ele.y, g_ele_mu.z + g_axis_ele.z));

    atomicAdd(grad_pose_probe_to_world + 0, g_raw_lat.x);
    atomicAdd(grad_pose_probe_to_world + 4, g_raw_lat.y);
    atomicAdd(grad_pose_probe_to_world + 8, g_raw_lat.z);
    atomicAdd(grad_pose_probe_to_world + 1, g_raw_ax.x);
    atomicAdd(grad_pose_probe_to_world + 5, g_raw_ax.y);
    atomicAdd(grad_pose_probe_to_world + 9, g_raw_ax.z);
    atomicAdd(grad_pose_probe_to_world + 2, g_raw_ele.x);
    atomicAdd(grad_pose_probe_to_world + 6, g_raw_ele.y);
    atomicAdd(grad_pose_probe_to_world + 10, g_raw_ele.z);
}

__global__ void kernel_us_focus_tile2d_sufficient_backward(
    int H,
    int W,
    int Tw,
    int tile_size_px,
    float k_sigma,
    const int* __restrict__ tile_range_start,
    const int* __restrict__ tile_range_end,
    const int* __restrict__ sorted_gauss_ids,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    const float* __restrict__ pose,
    const float* __restrict__ slice_grid_pack,
    const float* __restrict__ us_focus_params,
    const float* __restrict__ grad_pixel_colors,
    float* __restrict__ workspace) {
    const int tx = blockIdx.x;
    const int ty = blockIdx.y;
    const int tid = ty * Tw + tx;
    const int s = tile_range_start[tid];
    const int e = tile_range_end[tid];
    if (s < 0 || e <= s) {
        return;
    }

    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 raw_ele = make_float3(pose[2], pose[6], pose[10]);
    const float3 axis_lat = ours_normalize3(raw_lat);
    const float3 axis_ax = ours_normalize3(raw_ax);
    const float3 axis_ele = ours_normalize3(raw_ele);
    const float3 origin = make_float3(pose[3], pose[7], pose[11]);

    const float det_eps =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_DET_EPS, 1e-6f);
    const float det_gamma =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_DET_GAMMA, 0.25f);
    const float weight_min =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_WEIGHT_MIN, 1e-6f);
    const float sigma_ele_0 =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_SIGMA_ELE_0, 1.f);
    const float sigma_ele_1 =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_SIGMA_ELE_1, 0.f);
    const float sigma_ele_min =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_SIGMA_ELE_MIN, 0.5f);
    const float sigma_ele_max =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_SIGMA_ELE_MAX, 8.f);
    const float lat_delta = slice_grid_pack[SG_DELTA_LAT_MM];
    const float ax_delta = slice_grid_pack[SG_DELTA_AX_MM];
    const float lat_col_offset = slice_grid_pack[SG_LAT_COL_OFFSET];
    const float near_mm = slice_grid_pack[SG_NEAR_MM];

    const int x0 = max(0, tx * tile_size_px);
    const int y0 = max(0, ty * tile_size_px);
    const int x1 = min(W, (tx + 1) * tile_size_px);
    const int y1 = min(H, (ty + 1) * tile_size_px);

    for (int idx = s + threadIdx.x; idx < e; idx += blockDim.x) {
        const int g = sorted_gauss_ids[idx];
        const float* mptr = gauss_means + g * 3;
        const float* sptr = gauss_scales + g * 3;
        const float* qptr = gauss_quats + g * 4;

        us_focus_device::UsFocusFootprintOut fp;
        if (!us_focus_device::compute_us_focus_footprint_mm(
                mptr,
                sptr,
                qptr,
                axis_lat,
                axis_ax,
                axis_ele,
                origin,
                near_mm,
                k_sigma,
                us_focus_params,
                &fp)) {
            continue;
        }

        const int cand_x0 = max(x0, (int)ceilf((fp.mu2_lat - fp.r_lat_mm) / lat_delta + lat_col_offset));
        const int cand_x1 = min(x1, (int)floorf((fp.mu2_lat + fp.r_lat_mm) / lat_delta + lat_col_offset) + 1);
        const int cand_y0 = max(y0, (int)ceilf((fp.mu2_ax - fp.r_ax_mm - near_mm) / ax_delta));
        const int cand_y1 = min(y1, (int)floorf((fp.mu2_ax + fp.r_ax_mm - near_mm) / ax_delta) + 1);
        if (cand_x0 >= cand_x1 || cand_y0 >= cand_y1) {
            continue;
        }

        const float opacity = gauss_opacities[g];
        const float intensity = gauss_intensities[g];
        const float denom_ele = fmaxf(fp.sc22 + fp.sigma_ele * fp.sigma_ele + det_eps, 1e-12f);
        const float slice_response = expf(-0.5f * fp.mu_ele * fp.mu_ele / denom_ele);
        const float norm_factor = us_focus_device::det_norm_factor(fp.det2 + det_eps, det_gamma);
        const float depth = fmaxf(fp.mu_ax - near_mm, 0.f);
        const bool depth_active = fp.mu_ax > near_mm;
        const float sigma_ele_raw = sigma_ele_0 + sigma_ele_1 * depth;
        const float d_sigma_ele_d_ax =
            (depth_active && sigma_ele_raw > sigma_ele_min && sigma_ele_raw < sigma_ele_max) ? sigma_ele_1 : 0.f;

        float g_mu_probe[3] = {0.f, 0.f, 0.f};
        float g_Sc[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float g_mu_e_sum[3] = {0.f, 0.f, 0.f};
        float g_Se[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float g_opacity = 0.f;
        float g_intensity = 0.f;
        UsFocusEvalFootprintAdjoint g_fp_eval = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        for (int y = cand_y0; y < cand_y1; ++y) {
            const float xi_ax = static_cast<float>(y) * ax_delta + near_mm;
            for (int x = cand_x0; x < cand_x1; ++x) {
                const int p = y * W + x;
                const float xi_lat = (static_cast<float>(x) - lat_col_offset) * lat_delta;
                us_focus_accumulate_pixel_sufficient_backward(
                    xi_lat,
                    xi_ax,
                    grad_pixel_colors[p],
                    fp,
                    denom_ele,
                    slice_response,
                    norm_factor,
                    det_eps,
                    det_gamma,
                    weight_min,
                    opacity,
                    intensity,
                    d_sigma_ele_d_ax,
                    g_mu_e_sum,
                    g_Se,
                    g_mu_probe,
                    g_Sc,
                    &g_opacity,
                    &g_intensity,
                    &g_fp_eval);
            }
        }

        float* wsp = workspace + g * US_FOCUS_BWD_WSP_STRIDE;
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_MU_PROBE + i, g_mu_probe[i]);
            us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_MU_E_SUM + i, g_mu_e_sum[i]);
        }
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_SC + i, g_Sc[i]);
            us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_SE + i, g_Se[i]);
        }
        us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_OPACITY, g_opacity);
        us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_INTENSITY, g_intensity);
        us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_FP_EVAL + 0, g_fp_eval.mu2_lat);
        us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_FP_EVAL + 1, g_fp_eval.mu2_ax);
        us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_FP_EVAL + 2, g_fp_eval.inv2_00);
        us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_FP_EVAL + 3, g_fp_eval.inv2_01);
        us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_FP_EVAL + 4, g_fp_eval.inv2_11);
        us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_FP_EVAL + 5, g_fp_eval.det2);
        us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_FP_EVAL + 6, g_fp_eval.mu_ele);
        us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_FP_EVAL + 7, g_fp_eval.sc22);
        us_focus_atomic_add_if_nonzero(wsp + US_FOCUS_WSP_G_FP_EVAL + 8, g_fp_eval.sigma_ele);
    }
}

__global__ void kernel_us_focus_tile2d_shape_fd_backward(
    int H,
    int W,
    int Tw,
    int tile_size_px,
    float k_sigma,
    const int* __restrict__ tile_range_start,
    const int* __restrict__ tile_range_end,
    const int* __restrict__ sorted_gauss_ids,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    const float* __restrict__ pose,
    const float* __restrict__ slice_grid_pack,
    const float* __restrict__ us_focus_params,
    const float* __restrict__ grad_pixel_colors,
    float* __restrict__ grad_gauss_scales,
    float* __restrict__ grad_gauss_quats) {
    __shared__ UsFocusTileShapePairCandidate sh_cands[US_FOCUS_TILE_SHAPE_BATCH];

    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int lane = local_y * blockDim.x + local_x;
    const int tx = blockIdx.x;
    const int ty = blockIdx.y;
    const int x = tx * tile_size_px + local_x;
    const int y = ty * tile_size_px + local_y;
    const bool pixel_valid = (local_x < tile_size_px) && (local_y < tile_size_px) && (x < W) && (y < H);
    const int tid = ty * Tw + tx;

    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 raw_ele = make_float3(pose[2], pose[6], pose[10]);
    const float3 axis_lat = ours_normalize3(raw_lat);
    const float3 axis_ax = ours_normalize3(raw_ax);
    const float3 axis_ele = ours_normalize3(raw_ele);
    const float3 origin = make_float3(pose[3], pose[7], pose[11]);

    float xi_lat = 0.f;
    float xi_ax = 0.f;
    float upstream = 0.f;
    if (pixel_valid) {
        xi_ax = static_cast<float>(y) * slice_grid_pack[SG_DELTA_AX_MM] + slice_grid_pack[SG_NEAR_MM];
        xi_lat = (static_cast<float>(x) - slice_grid_pack[SG_LAT_COL_OFFSET]) * slice_grid_pack[SG_DELTA_LAT_MM];
        upstream = grad_pixel_colors[y * W + x];
    }
    const float weight_min =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_WEIGHT_MIN, 1e-6f);
    const int s = tile_range_start[tid];
    const int e = tile_range_end[tid];
    if (s < 0 || e <= s) {
        return;
    }

    constexpr float eps_scale = 1e-3f;
    constexpr float eps_quat = 1e-3f;
    constexpr float inv_2eps_scale = 1.f / (2.f * eps_scale);
    constexpr float inv_2eps_quat = 1.f / (2.f * eps_quat);

    for (int param = 0; param < 7; ++param) {
        const bool is_quat = param >= 3;
        const int param_idx = is_quat ? (param - 3) : param;
        const float eps = is_quat ? eps_quat : eps_scale;
        const float inv_2eps = is_quat ? inv_2eps_quat : inv_2eps_scale;
        for (int base = s; base < e; base += US_FOCUS_TILE_SHAPE_BATCH) {
            const int batch_count = min(US_FOCUS_TILE_SHAPE_BATCH, e - base);
            if (lane < batch_count) {
                const int g = sorted_gauss_ids[base + lane];
                if (!us_focus_fill_shape_pair_candidate(
                        g,
                        is_quat,
                        param_idx,
                        eps,
                        k_sigma,
                        gauss_means,
                        gauss_scales,
                        gauss_quats,
                        gauss_opacities,
                        gauss_intensities,
                        axis_lat,
                        axis_ax,
                        axis_ele,
                        origin,
                        slice_grid_pack,
                        us_focus_params,
                        &sh_cands[lane])) {
                    sh_cands[lane].g = -1;
                }
            }
            __syncthreads();

            if (pixel_valid && upstream != 0.f) {
                #pragma unroll
                for (int k = 0; k < US_FOCUS_TILE_SHAPE_BATCH; ++k) {
                    if (k >= batch_count) {
                        break;
                    }
                    const UsFocusTileShapePairCandidate cand = sh_cands[k];
                    if (cand.g < 0) {
                        continue;
                    }
                    const float g_param = us_focus_sanitize_grad(us_focus_eval_shape_pair_pixel(
                        cand,
                        xi_lat,
                        xi_ax,
                        upstream * cand.intensity,
                        weight_min,
                        inv_2eps));
                    if (g_param == 0.f) {
                        continue;
                    }
                    if (is_quat) {
                        atomicAdd(grad_gauss_quats + cand.g * 4 + param_idx, g_param);
                    } else {
                        atomicAdd(grad_gauss_scales + cand.g * 3 + param_idx, g_param);
                    }
                }
            }
            __syncthreads();
        }
    }
}

}  // namespace

namespace us_focus_backward_cu {

void RenderOursSliceBackwardUsFocus_Impl(
    int n_gauss,
    int H,
    int W,
    int Tw,
    int Th,
    int tile_size_px,
    float k_sigma,
    const int* gauss_rects,
    const int* tile_range_start,
    const int* tile_range_end,
    const int* sorted_gauss_ids,
    const float* gauss_means,
    const float* gauss_scales,
    const float* gauss_quats,
    const float* gauss_opacities,
    const float* gauss_intensities,
    const float* pose_probe_to_world,
    const float* slice_grid_pack,
    const float* us_focus_params,
    const float* grad_pixel_colors,
    float* workspace,
    int workspace_numel,
    bool compute_pose_grad,
    float* grad_gauss_means,
    float* grad_gauss_scales,
    float* grad_gauss_quats,
    float* grad_gauss_opacities,
    float* grad_gauss_intensities,
    float* grad_pose_probe_to_world,
    cudaStream_t stream) {
    if (n_gauss <= 0) return;

    const bool use_tile_shape_fd = false;
    const bool use_tile_sufficient =
        workspace != nullptr && workspace_numel >= n_gauss * US_FOCUS_BWD_WSP_STRIDE && tile_range_start != nullptr &&
        tile_range_end != nullptr && sorted_gauss_ids != nullptr;
    const int block = 128;
    const int grid = (n_gauss + block - 1) / block;
    if (use_tile_sufficient) {
        const int tile_block = 256;
        const dim3 tile_grid(Tw, Th, 1);
        kernel_us_focus_tile2d_sufficient_backward<<<tile_grid, tile_block, 0, stream>>>(
            H,
            W,
            Tw,
            tile_size_px,
            k_sigma,
            tile_range_start,
            tile_range_end,
            sorted_gauss_ids,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose_probe_to_world,
            slice_grid_pack,
            us_focus_params,
            grad_pixel_colors,
            workspace);
    }
    kernel_us_focus_additive_backward<<<grid, block, 0, stream>>>(
        n_gauss,
        H,
        W,
        tile_size_px,
        k_sigma,
        reinterpret_cast<const int4*>(gauss_rects),
        gauss_means,
        gauss_scales,
        gauss_quats,
        gauss_opacities,
        gauss_intensities,
        pose_probe_to_world,
        slice_grid_pack,
        us_focus_params,
        grad_pixel_colors,
        use_tile_sufficient ? workspace : nullptr,
        use_tile_sufficient,
        compute_pose_grad,
        !use_tile_shape_fd,
        grad_gauss_means,
        grad_gauss_scales,
        grad_gauss_quats,
        grad_gauss_opacities,
        grad_gauss_intensities,
        grad_pose_probe_to_world);
    if (use_tile_shape_fd) {
        const dim3 tile_block(tile_size_px, tile_size_px, 1);
        const dim3 tile_grid(Tw, Th, 1);
        kernel_us_focus_tile2d_shape_fd_backward<<<tile_grid, tile_block, 0, stream>>>(
            H,
            W,
            Tw,
            tile_size_px,
            k_sigma,
            tile_range_start,
            tile_range_end,
            sorted_gauss_ids,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose_probe_to_world,
            slice_grid_pack,
            us_focus_params,
            grad_pixel_colors,
            grad_gauss_scales,
            grad_gauss_quats);
    }
}

}  // namespace us_focus_backward_cu
