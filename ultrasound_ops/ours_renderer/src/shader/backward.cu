#include <cuda_runtime.h>
#include <math_constants.h>

#include "shader/backward.cuh"
#include "shader/shared.cuh"

namespace ours_backward_cu {

constexpr int OURS_ALPHA_WSP_STRIDE = 14;
constexpr int OURS_MAX_ALPHA_LAYERS = 256;

__device__ __forceinline__ float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 normalize3_forward(const float3& v) {
    float n = sqrtf(fmaxf(1e-12f, dot3(v, v)));
    return make_float3(v.x / n, v.y / n, v.z / n);
}

__device__ __forceinline__ float3 normalize3_backward(const float3& raw, const float3& n_unit, const float3& grad_out) {
    float len = sqrtf(fmaxf(1e-12f, dot3(raw, raw)));
    float inv_len = 1.f / len;
    float dot_ng = dot3(n_unit, grad_out);
    return make_float3(
        (grad_out.x - n_unit.x * dot_ng) * inv_len,
        (grad_out.y - n_unit.y * dot_ng) * inv_len,
        (grad_out.z - n_unit.z * dot_ng) * inv_len
    );
}

__device__ __forceinline__ void quat_to_rot(const float4& q_in, float R[9]) {
    ours_quat_to_rot(q_in, R);
}

__device__ __forceinline__ float frob_dot_grad_R(const float grad_R[9], const float Rp[9]) {
    float s = 0.f;
    for (int i = 0; i < 9; ++i) s += grad_R[i] * Rp[i];
    return s;
}

__device__ __forceinline__ void grad_quat_finite_diff(const float4& q_raw, const float grad_R[9], float* grad_q) {
    const float eps = 1e-3f;
    for (int k = 0; k < 4; ++k) {
        float4 qp = q_raw;
        float4 qm = q_raw;
        float* pk = (&qp.x) + k;
        float* mk = (&qm.x) + k;
        *pk += eps;
        *mk -= eps;
        float Rp[9], Rm[9];
        quat_to_rot(qp, Rp);
        quat_to_rot(qm, Rm);
        float fp = frob_dot_grad_R(grad_R, Rp);
        float fm = frob_dot_grad_R(grad_R, Rm);
        grad_q[k] = (fp - fm) / (2.f * eps);
    }
}

__device__ __forceinline__ void mat3_mul(const float A[9], const float B[9], float C[9]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            C[i * 3 + j] = A[i * 3 + 0] * B[0 * 3 + j] + A[i * 3 + 1] * B[1 * 3 + j] + A[i * 3 + 2] * B[2 * 3 + j];
        }
    }
}

__device__ void ours_bwd_fill_gauss_probe_inv(
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
    float* __restrict__ mu_x,
    float* __restrict__ mu_y,
    float* __restrict__ mu_z,
    float* __restrict__ i00,
    float* __restrict__ i01,
    float* __restrict__ i02,
    float* __restrict__ i11,
    float* __restrict__ i12,
    float* __restrict__ i22,
    float* __restrict__ o_out,
    float* __restrict__ I_out) {
    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;
    float ox = pose[3], oy = pose[7], oz = pose[11];
    float dw_x = mptr[0] - ox, dw_y = mptr[1] - oy, dw_z = mptr[2] - oz;
    *mu_x = dw_x * axis_lat.x + dw_y * axis_lat.y + dw_z * axis_lat.z;
    *mu_y = dw_x * axis_ax.x + dw_y * axis_ax.y + dw_z * axis_ax.z;
    *mu_z = dw_x * axis_ele.x + dw_y * axis_ele.y + dw_z * axis_ele.z;

    float Rg[9];
    quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
    float sx = fmaxf(sptr[0], 1e-6f);
    float sy = fmaxf(sptr[1], 1e-6f);
    float sz = fmaxf(sptr[2], 1e-6f);

    float a0_lat = Rg[0] * axis_lat.x + Rg[3] * axis_lat.y + Rg[6] * axis_lat.z;
    float a1_lat = Rg[1] * axis_lat.x + Rg[4] * axis_lat.y + Rg[7] * axis_lat.z;
    float a2_lat = Rg[2] * axis_lat.x + Rg[5] * axis_lat.y + Rg[8] * axis_lat.z;
    float ex0 = sx * a0_lat, ex1 = sy * a1_lat, ex2 = sz * a2_lat;

    float a0_ax = Rg[0] * axis_ax.x + Rg[3] * axis_ax.y + Rg[6] * axis_ax.z;
    float a1_ax = Rg[1] * axis_ax.x + Rg[4] * axis_ax.y + Rg[7] * axis_ax.z;
    float a2_ax = Rg[2] * axis_ax.x + Rg[5] * axis_ax.y + Rg[8] * axis_ax.z;
    float ey0 = sx * a0_ax, ey1 = sy * a1_ax, ey2 = sz * a2_ax;

    float a0_ele = Rg[0] * axis_ele.x + Rg[3] * axis_ele.y + Rg[6] * axis_ele.z;
    float a1_ele = Rg[1] * axis_ele.x + Rg[4] * axis_ele.y + Rg[7] * axis_ele.z;
    float a2_ele = Rg[2] * axis_ele.x + Rg[5] * axis_ele.y + Rg[8] * axis_ele.z;
    float ez0 = sx * a0_ele, ez1 = sy * a1_ele, ez2 = sz * a2_ele;

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
    *i00 = (c11 * c22 - c12 * c12) * invdet;
    *i01 = (c02 * c12 - c01 * c22) * invdet;
    *i02 = (c01 * c12 - c02 * c11) * invdet;
    *i11 = (c00 * c22 - c02 * c02) * invdet;
    *i12 = (c01 * c02 - c00 * c12) * invdet;
    *i22 = (c00 * c11 - c01 * c01) * invdet;
    *o_out = gauss_opacities[g];
    *I_out = gauss_intensities[g];
}

__device__ void ours_alpha_atomic_bwd_wire(
    int px,
    int py,
    int W,
    float lateral_spacing_per_pix,
    float axial_spacing_per_pix,
    float axial_near,
    float mu_x,
    float mu_y,
    float mu_z,
    float i00,
    float i01,
    float i02,
    float i11,
    float i12,
    float i22,
    float o,
    float Ir,
    float dL_dalpha,
    float dL_dI,
    float* __restrict__ wrow) {
    float x_c = (px - 0.5f * (W - 1)) * lateral_spacing_per_pix;
    float y_c = py * axial_spacing_per_pix + axial_near;
    float dx = x_c - mu_x;
    float dy = y_c - mu_y;
    float dz = -mu_z;
    float vx[3] = {dx, dy, dz};
    float sv0 = i00 * vx[0] + i01 * vx[1] + i02 * vx[2];
    float sv1 = i01 * vx[0] + i11 * vx[1] + i12 * vx[2];
    float sv2 = i02 * vx[0] + i12 * vx[1] + i22 * vx[2];
    float quad = dx * sv0 + dy * sv1 + dz * sv2;
    float base = expf(-0.5f * quad);
    float dL_dbase = dL_dalpha * o;
    atomicAdd(wrow + 12, dL_dalpha * base);
    atomicAdd(wrow + 13, dL_dI);
    float dL_dquad = dL_dbase * (-0.5f * base);
    for (int rr = 0; rr < 3; ++rr) {
        for (int cc = 0; cc < 3; ++cc) {
            atomicAdd(wrow + 3 + rr * 3 + cc, dL_dquad * vx[rr] * vx[cc]);
        }
    }
    float grad_v[3];
    grad_v[0] = -dL_dquad * sv0;
    grad_v[1] = -dL_dquad * sv1;
    grad_v[2] = -dL_dquad * sv2;
    atomicAdd(wrow + 0, -grad_v[0]);
    atomicAdd(wrow + 1, -grad_v[1]);
    atomicAdd(wrow + 2, -grad_v[2]);
}

__global__ void kernel_ours_alpha_back_pixel_csr(
    int n_pixels,
    int H,
    int W,
    float lateral_spacing_per_pix,
    float axial_spacing_per_pix,
    float axial_near,
    float k_sigma,
    float fixed_bg_color,
    const int* __restrict__ pixel_gauss_offsets,
    const int* __restrict__ pixel_gauss_counts,
    const int* __restrict__ pixel_gauss_indices,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    const float* __restrict__ pose,
    const float* __restrict__ grad_pixel_colors,
    float* __restrict__ workspace) {
    (void)H;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    float gc = grad_pixel_colors[p];
    if (fabsf(gc) < 1e-24f) return;
    int y = p / W;
    int x = p - y * W;

    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 raw_ele = make_float3(pose[2], pose[6], pose[10]);
    float3 axis_lat = normalize3_forward(raw_lat);
    float3 axis_ax = normalize3_forward(raw_ax);
    float3 axis_ele = normalize3_forward(raw_ele);

    int off = pixel_gauss_offsets[p];
    int cnt = pixel_gauss_counts[p];

    float ca[OURS_MAX_ALPHA_LAYERS];
    float cI[OURS_MAX_ALPHA_LAYERS];
    float cT[OURS_MAX_ALPHA_LAYERS];
    int cg[OURS_MAX_ALPHA_LAYERS];
    int acc = 0;
    float T = 1.f;
    for (int k = 0; k < cnt; ++k) {
        if (acc >= OURS_MAX_ALPHA_LAYERS) break;
        int g = pixel_gauss_indices[off + k];
        float alpha = 0.f, Iv = 0.f;
        float3 o3 = make_float3(pose[3], pose[7], pose[11]);
        ours_eval_gauss_contrib_at_pixel(
            x,
            y,
            W,
            lateral_spacing_per_pix,
            axial_spacing_per_pix,
            axial_near,
            k_sigma,
            g,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose,
            axis_lat,
            axis_ax,
            axis_ele,
            o3,
            &alpha,
            &Iv);
        if (alpha < OURS_ALPHA_MIN) continue;
        ca[acc] = alpha;
        cI[acc] = Iv;
        cT[acc] = T;
        cg[acc] = g;
        ++acc;
        T *= (1.f - alpha);
        if (T < 1e-4f) break;
    }

    float dL_dT = gc * fixed_bg_color;
    for (int kk = acc - 1; kk >= 0; --kk) {
        float alpha = ca[kk];
        float Iv = cI[kk];
        float Tb = cT[kk];
        int g = cg[kk];
        float dL_dI = gc * alpha * Tb;
        float dL_dalpha = gc * Iv * Tb + dL_dT * (-Tb);
        dL_dT = gc * Iv * alpha + dL_dT * (1.f - alpha);

        float mu_x, mu_y, mu_z, i00, i01, i02, i11, i12, i22, o, Ir;
        ours_bwd_fill_gauss_probe_inv(
            g,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose,
            axis_lat,
            axis_ax,
            axis_ele,
            &mu_x,
            &mu_y,
            &mu_z,
            &i00,
            &i01,
            &i02,
            &i11,
            &i12,
            &i22,
            &o,
            &Ir);
        float* wrow = workspace + g * OURS_ALPHA_WSP_STRIDE;
        ours_alpha_atomic_bwd_wire(
            x,
            y,
            W,
            lateral_spacing_per_pix,
            axial_spacing_per_pix,
            axial_near,
            mu_x,
            mu_y,
            mu_z,
            i00,
            i01,
            i02,
            i11,
            i12,
            i22,
            o,
            Ir,
            dL_dalpha,
            dL_dI,
            wrow);
    }
}

__global__ void kernel_ours_alpha_back_tile_csr(
    int n_pixels,
    int H,
    int W,
    int Tw,
    int tile_size_px,
    float lateral_spacing_per_pix,
    float axial_spacing_per_pix,
    float axial_near,
    float k_sigma,
    float fixed_bg_color,
    const int* __restrict__ tile_range_start,
    const int* __restrict__ tile_range_end,
    const int* __restrict__ sorted_gauss_ids,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    const float* __restrict__ pose,
    const float* __restrict__ grad_pixel_colors,
    float* __restrict__ workspace) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    float gc = grad_pixel_colors[p];
    if (fabsf(gc) < 1e-24f) return;
    int y = p / W;
    int x = p - y * W;

    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 raw_ele = make_float3(pose[2], pose[6], pose[10]);
    float3 axis_lat = normalize3_forward(raw_lat);
    float3 axis_ax = normalize3_forward(raw_ax);
    float3 axis_ele = normalize3_forward(raw_ele);
    float3 o3 = make_float3(pose[3], pose[7], pose[11]);

    int ts = tile_size_px;
    int tx = x / ts;
    int ty = y / ts;
    int tid = ty * Tw + tx;
    int s = tile_range_start[tid];
    int e = tile_range_end[tid];
    if (s < 0 || e <= s) return;

    float ca[OURS_MAX_ALPHA_LAYERS];
    float cI[OURS_MAX_ALPHA_LAYERS];
    float cT[OURS_MAX_ALPHA_LAYERS];
    int cg[OURS_MAX_ALPHA_LAYERS];
    int acc = 0;
    float T = 1.f;
    for (int j = s; j < e; ++j) {
        if (acc >= OURS_MAX_ALPHA_LAYERS) break;
        int g = sorted_gauss_ids[j];
        float alpha = 0.f, Iv = 0.f;
        ours_eval_gauss_contrib_at_pixel(
            x,
            y,
            W,
            lateral_spacing_per_pix,
            axial_spacing_per_pix,
            axial_near,
            k_sigma,
            g,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose,
            axis_lat,
            axis_ax,
            axis_ele,
            o3,
            &alpha,
            &Iv);
        if (alpha < OURS_ALPHA_MIN) continue;
        ca[acc] = alpha;
        cI[acc] = Iv;
        cT[acc] = T;
        cg[acc] = g;
        ++acc;
        T *= (1.f - alpha);
        if (T < 1e-4f) break;
    }

    float dL_dT = gc * fixed_bg_color;
    for (int kk = acc - 1; kk >= 0; --kk) {
        float alpha = ca[kk];
        float Iv = cI[kk];
        float Tb = cT[kk];
        int g = cg[kk];
        float dL_dI = gc * alpha * Tb;
        float dL_dalpha = gc * Iv * Tb + dL_dT * (-Tb);
        dL_dT = gc * Iv * alpha + dL_dT * (1.f - alpha);

        float mu_x, mu_y, mu_z, i00, i01, i02, i11, i12, i22, o, Ir;
        ours_bwd_fill_gauss_probe_inv(
            g,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose,
            axis_lat,
            axis_ax,
            axis_ele,
            &mu_x,
            &mu_y,
            &mu_z,
            &i00,
            &i01,
            &i02,
            &i11,
            &i12,
            &i22,
            &o,
            &Ir);
        float* wrow = workspace + g * OURS_ALPHA_WSP_STRIDE;
        ours_alpha_atomic_bwd_wire(
            x,
            y,
            W,
            lateral_spacing_per_pix,
            axial_spacing_per_pix,
            axial_near,
            mu_x,
            mu_y,
            mu_z,
            i00,
            i01,
            i02,
            i11,
            i12,
            i22,
            o,
            Ir,
            dL_dalpha,
            dL_dI,
            wrow);
    }
}

__global__ void kernel_ours_alpha_finalize_gauss(
    int n_gauss,
    const float* __restrict__ workspace,
    const int* __restrict__ gauss_rects,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ pose,
    float* __restrict__ grad_gauss_means,
    float* __restrict__ grad_gauss_scales,
    float* __restrict__ grad_gauss_quats,
    float* __restrict__ grad_gauss_opacities,
    float* __restrict__ grad_gauss_intensities,
    float* __restrict__ grad_pose) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) return;

    int r0 = gauss_rects[g * 4 + 0];
    int r1 = gauss_rects[g * 4 + 1];
    int r2 = gauss_rects[g * 4 + 2];
    int r3 = gauss_rects[g * 4 + 3];
    if (r0 >= r2 || r1 >= r3) return;

    const float* wrow = workspace + g * OURS_ALPHA_WSP_STRIDE;
    float g_opacity = wrow[12];
    float g_intensity = wrow[13];
    float gSinv_acc[9];
    for (int i = 0; i < 9; ++i) gSinv_acc[i] = wrow[3 + i];
    float g_mu_x_acc = wrow[0];
    float g_mu_y_acc = wrow[1];
    float g_mu_z_acc = wrow[2];

    if (fabsf(g_opacity) + fabsf(g_intensity) < 1e-30f) {
        float ssum = 0.f;
        for (int i = 0; i < 9; ++i) ssum += fabsf(gSinv_acc[i]);
        if (fabsf(g_mu_x_acc) + fabsf(g_mu_y_acc) + fabsf(g_mu_z_acc) + ssum < 1e-30f) return;
    }

    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 raw_ele = make_float3(pose[2], pose[6], pose[10]);
    float3 axis_lat = normalize3_forward(raw_lat);
    float3 axis_ax = normalize3_forward(raw_ax);
    float3 axis_ele = normalize3_forward(raw_ele);

    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;
    float mx = mptr[0], my = mptr[1], mz = mptr[2];
    float ox = pose[3], oy = pose[7], oz = pose[11];
    float dw_x = mx - ox, dw_y = my - oy, dw_z = mz - oz;

    float Rg[9];
    quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
    float sx = fmaxf(sptr[0], 1e-6f);
    float sy = fmaxf(sptr[1], 1e-6f);
    float sz = fmaxf(sptr[2], 1e-6f);

    float col0x = Rg[0], col0y = Rg[3], col0z = Rg[6];
    float col1x = Rg[1], col1y = Rg[4], col1z = Rg[7];
    float col2x = Rg[2], col2y = Rg[5], col2z = Rg[8];

    float a0_lat = col0x * axis_lat.x + col0y * axis_lat.y + col0z * axis_lat.z;
    float a1_lat = col1x * axis_lat.x + col1y * axis_lat.y + col1z * axis_lat.z;
    float a2_lat = col2x * axis_lat.x + col2y * axis_lat.y + col2z * axis_lat.z;
    float ex0 = sx * a0_lat, ex1 = sy * a1_lat, ex2 = sz * a2_lat;

    float a0_ax = col0x * axis_ax.x + col0y * axis_ax.y + col0z * axis_ax.z;
    float a1_ax = col1x * axis_ax.x + col1y * axis_ax.y + col1z * axis_ax.z;
    float a2_ax = col2x * axis_ax.x + col2y * axis_ax.y + col2z * axis_ax.z;
    float ey0 = sx * a0_ax, ey1 = sy * a1_ax, ey2 = sz * a2_ax;

    float a0_ele = col0x * axis_ele.x + col0y * axis_ele.y + col0z * axis_ele.z;
    float a1_ele = col1x * axis_ele.x + col1y * axis_ele.y + col1z * axis_ele.z;
    float a2_ele = col2x * axis_ele.x + col2y * axis_ele.y + col2z * axis_ele.z;
    float ez0 = sx * a0_ele, ez1 = sy * a1_ele, ez2 = sz * a2_ele;

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
    float Sinv[9] = {i00, i01, i02, i01, i11, i12, i02, i12, i22};

    grad_gauss_opacities[g] = g_opacity;
    grad_gauss_intensities[g] = g_intensity;

    float Sinv_mat[9];
    for (int i = 0; i < 9; ++i) Sinv_mat[i] = Sinv[i];
    float gSinv_mat[9];
    for (int i = 0; i < 9; ++i) gSinv_mat[i] = gSinv_acc[i];

    float tmp1[9], tmp2[9], gSigma[9];
    mat3_mul(Sinv_mat, gSinv_mat, tmp1);
    mat3_mul(tmp1, Sinv_mat, tmp2);
    for (int i = 0; i < 9; ++i) gSigma[i] = -tmp2[i];

    float G00 = gSigma[0], G01 = gSigma[1], G02 = gSigma[2];
    float G11 = gSigma[4], G12 = gSigma[5];
    float G22 = gSigma[8];

    float g_ex0 = 2.f * G00 * ex0 + 2.f * G01 * ey0 + 2.f * G02 * ez0;
    float g_ex1 = 2.f * G00 * ex1 + 2.f * G01 * ey1 + 2.f * G02 * ez1;
    float g_ex2 = 2.f * G00 * ex2 + 2.f * G01 * ey2 + 2.f * G02 * ez2;
    float g_ey0 = 2.f * G01 * ex0 + 2.f * G11 * ey0 + 2.f * G12 * ez0;
    float g_ey1 = 2.f * G01 * ex1 + 2.f * G11 * ey1 + 2.f * G12 * ez1;
    float g_ey2 = 2.f * G01 * ex2 + 2.f * G11 * ey2 + 2.f * G12 * ez2;
    float g_ez0 = 2.f * G02 * ex0 + 2.f * G12 * ey0 + 2.f * G22 * ez0;
    float g_ez1 = 2.f * G02 * ex1 + 2.f * G12 * ey1 + 2.f * G22 * ez1;
    float g_ez2 = 2.f * G02 * ex2 + 2.f * G12 * ey2 + 2.f * G22 * ez2;

    float g_sx = g_ex0 * a0_lat + g_ey0 * a0_ax + g_ez0 * a0_ele;
    float g_sy = g_ex1 * a1_lat + g_ey1 * a1_ax + g_ez1 * a1_ele;
    float g_sz = g_ex2 * a2_lat + g_ey2 * a2_ax + g_ez2 * a2_ele;

    grad_gauss_scales[g * 3 + 0] = g_sx;
    grad_gauss_scales[g * 3 + 1] = g_sy;
    grad_gauss_scales[g * 3 + 2] = g_sz;

    float3 g_col0 = make_float3(
        g_ex0 * sx * axis_lat.x + g_ey0 * sy * axis_ax.x + g_ez0 * sz * axis_ele.x,
        g_ex0 * sx * axis_lat.y + g_ey0 * sy * axis_ax.y + g_ez0 * sz * axis_ele.y,
        g_ex0 * sx * axis_lat.z + g_ey0 * sy * axis_ax.z + g_ez0 * sz * axis_ele.z);
    float3 g_col1 = make_float3(
        g_ex1 * sx * axis_lat.x + g_ey1 * sy * axis_ax.x + g_ez1 * sz * axis_ele.x,
        g_ex1 * sx * axis_lat.y + g_ey1 * sy * axis_ax.y + g_ez1 * sz * axis_ele.y,
        g_ex1 * sx * axis_lat.z + g_ey1 * sy * axis_ax.z + g_ez1 * sz * axis_ele.z);
    float3 g_col2 = make_float3(
        g_ex2 * sx * axis_lat.x + g_ey2 * sy * axis_ax.x + g_ez2 * sz * axis_ele.x,
        g_ex2 * sx * axis_lat.y + g_ey2 * sy * axis_ax.y + g_ez2 * sz * axis_ele.y,
        g_ex2 * sx * axis_lat.z + g_ey2 * sy * axis_ax.z + g_ez2 * sz * axis_ele.z);

    float grad_R[9];
    grad_R[0] = g_col0.x;
    grad_R[3] = g_col0.y;
    grad_R[6] = g_col0.z;
    grad_R[1] = g_col1.x;
    grad_R[4] = g_col1.y;
    grad_R[7] = g_col1.z;
    grad_R[2] = g_col2.x;
    grad_R[5] = g_col2.y;
    grad_R[8] = g_col2.z;

    float gq_local[4];
    grad_quat_finite_diff(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), grad_R, gq_local);
    grad_gauss_quats[g * 4 + 0] = gq_local[0];
    grad_gauss_quats[g * 4 + 1] = gq_local[1];
    grad_gauss_quats[g * 4 + 2] = gq_local[2];
    grad_gauss_quats[g * 4 + 3] = gq_local[3];

    float g_mu_x = g_mu_x_acc;
    float g_mu_y = g_mu_y_acc;
    float g_mu_z = g_mu_z_acc;
    float3 g_delta = make_float3(
        g_mu_x * axis_lat.x + g_mu_y * axis_ax.x + g_mu_z * axis_ele.x,
        g_mu_x * axis_lat.y + g_mu_y * axis_ax.y + g_mu_z * axis_ele.y,
        g_mu_x * axis_lat.z + g_mu_y * axis_ax.z + g_mu_z * axis_ele.z);

    grad_gauss_means[g * 3 + 0] = g_delta.x;
    grad_gauss_means[g * 3 + 1] = g_delta.y;
    grad_gauss_means[g * 3 + 2] = g_delta.z;

    atomicAdd(grad_pose + 3, -g_delta.x);
    atomicAdd(grad_pose + 7, -g_delta.y);
    atomicAdd(grad_pose + 11, -g_delta.z);

    float3 g_lat_cov = make_float3(
        g_ex0 * sx * col0x + g_ex1 * sy * col1x + g_ex2 * sz * col2x,
        g_ex0 * sx * col0y + g_ex1 * sy * col1y + g_ex2 * sz * col2y,
        g_ex0 * sx * col0z + g_ex1 * sy * col1z + g_ex2 * sz * col2z);
    float3 g_ax_cov = make_float3(
        g_ey0 * sx * col0x + g_ey1 * sy * col1x + g_ey2 * sz * col2x,
        g_ey0 * sx * col0y + g_ey1 * sy * col1y + g_ey2 * sz * col2y,
        g_ey0 * sx * col0z + g_ey1 * sy * col1z + g_ey2 * sz * col2z);
    float3 g_ele_cov = make_float3(
        g_ez0 * sx * col0x + g_ez1 * sy * col1x + g_ez2 * sz * col2x,
        g_ez0 * sx * col0y + g_ez1 * sy * col1y + g_ez2 * sz * col2y,
        g_ez0 * sx * col0z + g_ez1 * sy * col1z + g_ez2 * sz * col2z);

    float3 g_lat_mu = make_float3(g_mu_x * dw_x, g_mu_x * dw_y, g_mu_x * dw_z);
    float3 g_ax_mu = make_float3(g_mu_y * dw_x, g_mu_y * dw_y, g_mu_y * dw_z);
    float3 g_ele_mu = make_float3(g_mu_z * dw_x, g_mu_z * dw_y, g_mu_z * dw_z);

    float3 g_raw_lat = normalize3_backward(raw_lat, axis_lat, make_float3(g_lat_mu.x + g_lat_cov.x, g_lat_mu.y + g_lat_cov.y, g_lat_mu.z + g_lat_cov.z));
    float3 g_raw_ax = normalize3_backward(raw_ax, axis_ax, make_float3(g_ax_mu.x + g_ax_cov.x, g_ax_mu.y + g_ax_cov.y, g_ax_mu.z + g_ax_cov.z));
    float3 g_raw_ele = normalize3_backward(raw_ele, axis_ele, make_float3(g_ele_mu.x + g_ele_cov.x, g_ele_mu.y + g_ele_cov.y, g_ele_mu.z + g_ele_cov.z));

    atomicAdd(grad_pose + 0, g_raw_lat.x);
    atomicAdd(grad_pose + 4, g_raw_lat.y);
    atomicAdd(grad_pose + 8, g_raw_lat.z);
    atomicAdd(grad_pose + 1, g_raw_ax.x);
    atomicAdd(grad_pose + 5, g_raw_ax.y);
    atomicAdd(grad_pose + 9, g_raw_ax.z);
    atomicAdd(grad_pose + 2, g_raw_ele.x);
    atomicAdd(grad_pose + 6, g_raw_ele.y);
    atomicAdd(grad_pose + 10, g_raw_ele.z);
}

__global__ void kernel_projected_slice_backward(
    int n_pixels,
    int n_gauss,
    int H,
    int W,
    int tile_size_px,
    int gauss_rect_is_pixels,
    float lateral_spacing_per_pix,
    float axial_spacing_per_pix,
    float axial_near,
    float k_sigma,
    const int* __restrict__ gauss_rects,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    const float* __restrict__ pose,
    const float* __restrict__ grad_pixel_colors,
    float* __restrict__ grad_gauss_means,
    float* __restrict__ grad_gauss_scales,
    float* __restrict__ grad_gauss_quats,
    float* __restrict__ grad_gauss_opacities,
    float* __restrict__ grad_gauss_intensities,
    float* __restrict__ grad_pose
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) return;

    int r0 = gauss_rects[g * 4 + 0];
    int r1 = gauss_rects[g * 4 + 1];
    int r2 = gauss_rects[g * 4 + 2];
    int r3 = gauss_rects[g * 4 + 3];
    if (r0 >= r2 || r1 >= r3) return;

    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 raw_ele = make_float3(pose[2], pose[6], pose[10]);
    float3 axis_lat = normalize3_forward(raw_lat);
    float3 axis_ax = normalize3_forward(raw_ax);
    float3 axis_ele = normalize3_forward(raw_ele);

    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;

    float mx = mptr[0], my = mptr[1], mz = mptr[2];
    float ox = pose[3], oy = pose[7], oz = pose[11];

    float dw_x = mx - ox;
    float dw_y = my - oy;
    float dw_z = mz - oz;

    float mu_x = dw_x * axis_lat.x + dw_y * axis_lat.y + dw_z * axis_lat.z;
    float mu_y = dw_x * axis_ax.x + dw_y * axis_ax.y + dw_z * axis_ax.z;
    float mu_z = dw_x * axis_ele.x + dw_y * axis_ele.y + dw_z * axis_ele.z;

    float Rg[9];
    quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
    float sx = fmaxf(sptr[0], 1e-6f);
    float sy = fmaxf(sptr[1], 1e-6f);
    float sz = fmaxf(sptr[2], 1e-6f);

    float col0x = Rg[0], col0y = Rg[3], col0z = Rg[6];
    float col1x = Rg[1], col1y = Rg[4], col1z = Rg[7];
    float col2x = Rg[2], col2y = Rg[5], col2z = Rg[8];

    float a0_lat = col0x * axis_lat.x + col0y * axis_lat.y + col0z * axis_lat.z;
    float a1_lat = col1x * axis_lat.x + col1y * axis_lat.y + col1z * axis_lat.z;
    float a2_lat = col2x * axis_lat.x + col2y * axis_lat.y + col2z * axis_lat.z;
    float ex0 = sx * a0_lat;
    float ex1 = sy * a1_lat;
    float ex2 = sz * a2_lat;

    float a0_ax = col0x * axis_ax.x + col0y * axis_ax.y + col0z * axis_ax.z;
    float a1_ax = col1x * axis_ax.x + col1y * axis_ax.y + col1z * axis_ax.z;
    float a2_ax = col2x * axis_ax.x + col2y * axis_ax.y + col2z * axis_ax.z;
    float ey0 = sx * a0_ax;
    float ey1 = sy * a1_ax;
    float ey2 = sz * a2_ax;

    float a0_ele = col0x * axis_ele.x + col0y * axis_ele.y + col0z * axis_ele.z;
    float a1_ele = col1x * axis_ele.x + col1y * axis_ele.y + col1z * axis_ele.z;
    float a2_ele = col2x * axis_ele.x + col2y * axis_ele.y + col2z * axis_ele.z;
    float ez0 = sx * a0_ele;
    float ez1 = sy * a1_ele;
    float ez2 = sz * a2_ele;

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

    float Sinv[9] = {i00, i01, i02, i01, i11, i12, i02, i12, i22};

    float gSinv_acc[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    float g_opacity = 0.f;
    float g_intensity = 0.f;

    float g_mu_x_acc = 0.f;
    float g_mu_y_acc = 0.f;
    float g_mu_z_acc = 0.f;

    float delta_lat = lateral_spacing_per_pix;
    float delta_ax = axial_spacing_per_pix;
    float near_ax = axial_near;

    float o = gauss_opacities[g];
    float I = gauss_intensities[g];

    float r_lat_mm = 0.f;
    float r_ax_mm = 0.f;
    ours_lat_ax_radii_mm(sx, sy, sz, Rg, axis_lat, axis_ax, k_sigma, &r_lat_mm, &r_ax_mm);

    int ts = tile_size_px;
#define OURS_BWD_ONE_PIXEL(pxv, pyv)                                                                 \
    do {                                                                                             \
        int p = (pyv)*W + (pxv);                                                                     \
        float gc = grad_pixel_colors[p];                                                             \
        if (fabsf(gc) >= 1e-24f) {                                                                  \
            float x_c = ((pxv)-0.5f * (W - 1)) * delta_lat;                                         \
            float y_c = (pyv)*delta_ax + near_ax;                                                    \
            if (!ours_lat_ax_aabb_reject(x_c, y_c, mu_x, mu_y, r_lat_mm, r_ax_mm)) {                \
            float dx = x_c - mu_x;                                                                   \
            float dy = y_c - mu_y;                                                                   \
            float dz = -mu_z;                                                                        \
            float vx[3] = {dx, dy, dz};                                                              \
            float sv0 = i00 * vx[0] + i01 * vx[1] + i02 * vx[2];                                     \
            float sv1 = i01 * vx[0] + i11 * vx[1] + i12 * vx[2];                                     \
            float sv2 = i02 * vx[0] + i12 * vx[1] + i22 * vx[2];                                     \
            float quad = dx * sv0 + dy * sv1 + dz * sv2;                                           \
            float base = expf(-0.5f * quad);                                                        \
            float alpha = o * base;                                                                    \
            if (alpha >= OURS_ALPHA_MIN) {                                                         \
            float dL_dalpha = gc * I;                                                                \
            float dL_dbase = dL_dalpha * o;                                                          \
            g_opacity += gc * I * base;                                                              \
            g_intensity += gc * alpha;                                                               \
            float dL_dquad = dL_dbase * (-0.5f * base);                                              \
            for (int rr = 0; rr < 3; ++rr) {                                                         \
                for (int cc = 0; cc < 3; ++cc) {                                                     \
                    gSinv_acc[rr * 3 + cc] += dL_dquad * vx[rr] * vx[cc];                            \
                }                                                                                    \
            }                                                                                        \
            float grad_v[3];                                                                         \
            grad_v[0] = -dL_dquad * sv0;                                                             \
            grad_v[1] = -dL_dquad * sv1;                                                             \
            grad_v[2] = -dL_dquad * sv2;                                                             \
            g_mu_x_acc += -grad_v[0];                                                                \
            g_mu_y_acc += -grad_v[1];                                                                \
            g_mu_z_acc += -grad_v[2];                                                               \
            }                                                                                        \
            }                                                                                        \
        }                                                                                            \
    } while (0)

    if (gauss_rect_is_pixels != 0) {
        for (int py = r1; py < r3; ++py) {
            for (int px = r0; px < r2; ++px) {
                OURS_BWD_ONE_PIXEL(px, py);
            }
        }
    } else {
        for (int tty = r1; tty < r3; ++tty) {
            for (int ttx = r0; ttx < r2; ++ttx) {
                int px0 = ttx * ts;
                int px1 = min(W - 1, (ttx + 1) * ts - 1);
                int py0 = tty * ts;
                int py1 = min(H - 1, (tty + 1) * ts - 1);
                for (int py = py0; py <= py1; ++py) {
                    for (int px = px0; px <= px1; ++px) {
                        OURS_BWD_ONE_PIXEL(px, py);
                    }
                }
            }
        }
    }
#undef OURS_BWD_ONE_PIXEL

    grad_gauss_opacities[g] = g_opacity;
    grad_gauss_intensities[g] = g_intensity;

    float Sinv_mat[9];
    for (int i = 0; i < 9; ++i) Sinv_mat[i] = Sinv[i];
    float gSinv_mat[9];
    for (int i = 0; i < 9; ++i) gSinv_mat[i] = gSinv_acc[i];

    float tmp1[9], tmp2[9], gSigma[9];
    mat3_mul(Sinv_mat, gSinv_mat, tmp1);
    mat3_mul(tmp1, Sinv_mat, tmp2);
    for (int i = 0; i < 9; ++i) gSigma[i] = -tmp2[i];

    float G00 = gSigma[0], G01 = gSigma[1], G02 = gSigma[2];
    float G11 = gSigma[4], G12 = gSigma[5];
    float G22 = gSigma[8];

    float g_ex0 = 2.f * G00 * ex0 + 2.f * G01 * ey0 + 2.f * G02 * ez0;
    float g_ex1 = 2.f * G00 * ex1 + 2.f * G01 * ey1 + 2.f * G02 * ez1;
    float g_ex2 = 2.f * G00 * ex2 + 2.f * G01 * ey2 + 2.f * G02 * ez2;

    float g_ey0 = 2.f * G01 * ex0 + 2.f * G11 * ey0 + 2.f * G12 * ez0;
    float g_ey1 = 2.f * G01 * ex1 + 2.f * G11 * ey1 + 2.f * G12 * ez1;
    float g_ey2 = 2.f * G01 * ex2 + 2.f * G11 * ey2 + 2.f * G12 * ez2;

    float g_ez0 = 2.f * G02 * ex0 + 2.f * G12 * ey0 + 2.f * G22 * ez0;
    float g_ez1 = 2.f * G02 * ex1 + 2.f * G12 * ey1 + 2.f * G22 * ez1;
    float g_ez2 = 2.f * G02 * ex2 + 2.f * G12 * ey2 + 2.f * G22 * ez2;

    float g_sx = g_ex0 * a0_lat + g_ey0 * a0_ax + g_ez0 * a0_ele;
    float g_sy = g_ex1 * a1_lat + g_ey1 * a1_ax + g_ez1 * a1_ele;
    float g_sz = g_ex2 * a2_lat + g_ey2 * a2_ax + g_ez2 * a2_ele;

    grad_gauss_scales[g * 3 + 0] = g_sx;
    grad_gauss_scales[g * 3 + 1] = g_sy;
    grad_gauss_scales[g * 3 + 2] = g_sz;

    float3 g_col0 = make_float3(
        g_ex0 * sx * axis_lat.x + g_ey0 * sy * axis_ax.x + g_ez0 * sz * axis_ele.x,
        g_ex0 * sx * axis_lat.y + g_ey0 * sy * axis_ax.y + g_ez0 * sz * axis_ele.y,
        g_ex0 * sx * axis_lat.z + g_ey0 * sy * axis_ax.z + g_ez0 * sz * axis_ele.z
    );
    float3 g_col1 = make_float3(
        g_ex1 * sx * axis_lat.x + g_ey1 * sy * axis_ax.x + g_ez1 * sz * axis_ele.x,
        g_ex1 * sx * axis_lat.y + g_ey1 * sy * axis_ax.y + g_ez1 * sz * axis_ele.y,
        g_ex1 * sx * axis_lat.z + g_ey1 * sy * axis_ax.z + g_ez1 * sz * axis_ele.z
    );
    float3 g_col2 = make_float3(
        g_ex2 * sx * axis_lat.x + g_ey2 * sy * axis_ax.x + g_ez2 * sz * axis_ele.x,
        g_ex2 * sx * axis_lat.y + g_ey2 * sy * axis_ax.y + g_ez2 * sz * axis_ele.y,
        g_ex2 * sx * axis_lat.z + g_ey2 * sy * axis_ax.z + g_ez2 * sz * axis_ele.z
    );

    float grad_R[9];
    grad_R[0] = g_col0.x;
    grad_R[3] = g_col0.y;
    grad_R[6] = g_col0.z;
    grad_R[1] = g_col1.x;
    grad_R[4] = g_col1.y;
    grad_R[7] = g_col1.z;
    grad_R[2] = g_col2.x;
    grad_R[5] = g_col2.y;
    grad_R[8] = g_col2.z;

    float gq_local[4];
    grad_quat_finite_diff(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), grad_R, gq_local);
    grad_gauss_quats[g * 4 + 0] = gq_local[0];
    grad_gauss_quats[g * 4 + 1] = gq_local[1];
    grad_gauss_quats[g * 4 + 2] = gq_local[2];
    grad_gauss_quats[g * 4 + 3] = gq_local[3];

    float g_mu_x = g_mu_x_acc;
    float g_mu_y = g_mu_y_acc;
    float g_mu_z = g_mu_z_acc;

    float3 g_delta = make_float3(
        g_mu_x * axis_lat.x + g_mu_y * axis_ax.x + g_mu_z * axis_ele.x,
        g_mu_x * axis_lat.y + g_mu_y * axis_ax.y + g_mu_z * axis_ele.y,
        g_mu_x * axis_lat.z + g_mu_y * axis_ax.z + g_mu_z * axis_ele.z
    );

    grad_gauss_means[g * 3 + 0] = g_delta.x;
    grad_gauss_means[g * 3 + 1] = g_delta.y;
    grad_gauss_means[g * 3 + 2] = g_delta.z;

    atomicAdd(grad_pose + 3, -g_delta.x);
    atomicAdd(grad_pose + 7, -g_delta.y);
    atomicAdd(grad_pose + 11, -g_delta.z);

    float3 g_lat_cov = make_float3(
        g_ex0 * sx * col0x + g_ex1 * sy * col1x + g_ex2 * sz * col2x,
        g_ex0 * sx * col0y + g_ex1 * sy * col1y + g_ex2 * sz * col2y,
        g_ex0 * sx * col0z + g_ex1 * sy * col1z + g_ex2 * sz * col2z
    );
    float3 g_ax_cov = make_float3(
        g_ey0 * sx * col0x + g_ey1 * sy * col1x + g_ey2 * sz * col2x,
        g_ey0 * sx * col0y + g_ey1 * sy * col1y + g_ey2 * sz * col2y,
        g_ey0 * sx * col0z + g_ey1 * sy * col1z + g_ey2 * sz * col2z
    );
    float3 g_ele_cov = make_float3(
        g_ez0 * sx * col0x + g_ez1 * sy * col1x + g_ez2 * sz * col2x,
        g_ez0 * sx * col0y + g_ez1 * sy * col1y + g_ez2 * sz * col2y,
        g_ez0 * sx * col0z + g_ez1 * sy * col1z + g_ez2 * sz * col2z
    );

    float3 g_lat_mu = make_float3(g_mu_x * dw_x, g_mu_x * dw_y, g_mu_x * dw_z);
    float3 g_ax_mu = make_float3(g_mu_y * dw_x, g_mu_y * dw_y, g_mu_y * dw_z);
    float3 g_ele_mu = make_float3(g_mu_z * dw_x, g_mu_z * dw_y, g_mu_z * dw_z);

    float3 g_raw_lat = normalize3_backward(raw_lat, axis_lat, make_float3(g_lat_mu.x + g_lat_cov.x, g_lat_mu.y + g_lat_cov.y, g_lat_mu.z + g_lat_cov.z));
    float3 g_raw_ax = normalize3_backward(raw_ax, axis_ax, make_float3(g_ax_mu.x + g_ax_cov.x, g_ax_mu.y + g_ax_cov.y, g_ax_mu.z + g_ax_cov.z));
    float3 g_raw_ele = normalize3_backward(raw_ele, axis_ele, make_float3(g_ele_mu.x + g_ele_cov.x, g_ele_mu.y + g_ele_cov.y, g_ele_mu.z + g_ele_cov.z));

    atomicAdd(grad_pose + 0, g_raw_lat.x);
    atomicAdd(grad_pose + 4, g_raw_lat.y);
    atomicAdd(grad_pose + 8, g_raw_lat.z);
    atomicAdd(grad_pose + 1, g_raw_ax.x);
    atomicAdd(grad_pose + 5, g_raw_ax.y);
    atomicAdd(grad_pose + 9, g_raw_ax.z);
    atomicAdd(grad_pose + 2, g_raw_ele.x);
    atomicAdd(grad_pose + 6, g_raw_ele.y);
    atomicAdd(grad_pose + 10, g_raw_ele.z);
}

void RenderOursSliceBackwardCUDA_Impl(
    int n_pixels,
    int n_gauss,
    int H,
    int W,
    int tile_size_px,
    int gauss_rect_is_pixels,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float k_sigma,
    int compositing_mode,
    float fixed_bg_color,
    int mapper_kind,
    const int* tile_range_start,
    const int* tile_range_end,
    const int* sorted_gauss_ids,
    const int* pixel_gauss_offsets,
    const int* pixel_gauss_counts,
    const int* pixel_gauss_indices,
    float* workspace,
    int workspace_floats,
    const int* gauss_rects,
    const float* gauss_means,
    const float* gauss_scales,
    const float* gauss_quats,
    const float* gauss_opacities,
    const float* gauss_intensities,
    const float* pose_probe_to_world,
    const float* grad_pixel_colors,
    float* grad_gauss_means,
    float* grad_gauss_scales,
    float* grad_gauss_quats,
    float* grad_gauss_opacities,
    float* grad_gauss_intensities,
    float* grad_pose_probe_to_world,
    cudaStream_t stream) {
    const int threads = 128;
    cudaMemsetAsync(grad_gauss_means, 0, sizeof(float) * n_gauss * 3, stream);
    cudaMemsetAsync(grad_gauss_scales, 0, sizeof(float) * n_gauss * 3, stream);
    cudaMemsetAsync(grad_gauss_quats, 0, sizeof(float) * n_gauss * 4, stream);
    cudaMemsetAsync(grad_gauss_opacities, 0, sizeof(float) * n_gauss, stream);
    cudaMemsetAsync(grad_gauss_intensities, 0, sizeof(float) * n_gauss, stream);
    cudaMemsetAsync(grad_pose_probe_to_world, 0, sizeof(float) * 16, stream);

    if (compositing_mode == 0) {
        const int blocks = (n_gauss + threads - 1) / threads;
        kernel_projected_slice_backward<<<blocks, threads, 0, stream>>>(
            n_pixels,
            n_gauss,
            H,
            W,
            tile_size_px,
            gauss_rect_is_pixels,
            lateral_spacing_mm_per_pix,
            axial_spacing_mm_per_pix,
            axial_near_mm,
            k_sigma,
            gauss_rects,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose_probe_to_world,
            grad_pixel_colors,
            grad_gauss_means,
            grad_gauss_scales,
            grad_gauss_quats,
            grad_gauss_opacities,
            grad_gauss_intensities,
            grad_pose_probe_to_world);
        return;
    }

    if (workspace == nullptr || workspace_floats < n_gauss * OURS_ALPHA_WSP_STRIDE) {
        return;
    }
    cudaMemsetAsync(workspace, 0, sizeof(float) * (size_t)n_gauss * OURS_ALPHA_WSP_STRIDE, stream);

    const int pb = 128;
    const int pg = (n_pixels + pb - 1) / pb;
    if (mapper_kind == 0) {
        const int Tw = (W + tile_size_px - 1) / tile_size_px;
        kernel_ours_alpha_back_tile_csr<<<pg, pb, 0, stream>>>(
            n_pixels,
            H,
            W,
            Tw,
            tile_size_px,
            lateral_spacing_mm_per_pix,
            axial_spacing_mm_per_pix,
            axial_near_mm,
            k_sigma,
            fixed_bg_color,
            tile_range_start,
            tile_range_end,
            sorted_gauss_ids,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose_probe_to_world,
            grad_pixel_colors,
            workspace);
    } else {
        kernel_ours_alpha_back_pixel_csr<<<pg, pb, 0, stream>>>(
            n_pixels,
            H,
            W,
            lateral_spacing_mm_per_pix,
            axial_spacing_mm_per_pix,
            axial_near_mm,
            k_sigma,
            fixed_bg_color,
            pixel_gauss_offsets,
            pixel_gauss_counts,
            pixel_gauss_indices,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose_probe_to_world,
            grad_pixel_colors,
            workspace);
    }

    const int gb = 128;
    const int gg = (n_gauss + gb - 1) / gb;
    kernel_ours_alpha_finalize_gauss<<<gg, gb, 0, stream>>>(
        n_gauss,
        workspace,
        gauss_rects,
        gauss_means,
        gauss_scales,
        gauss_quats,
        pose_probe_to_world,
        grad_gauss_means,
        grad_gauss_scales,
        grad_gauss_quats,
        grad_gauss_opacities,
        grad_gauss_intensities,
        grad_pose_probe_to_world);
}

}  // namespace ours_backward_cu
