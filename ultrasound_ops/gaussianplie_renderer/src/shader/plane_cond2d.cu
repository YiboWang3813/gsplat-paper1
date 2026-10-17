#include <cuda_runtime.h>
#include <math_constants.h>

#include "shader/plane_cond2d.cuh"
#include "shader/shared.cuh"

namespace plane_cond2d_cu {

constexpr int OURS_MAX_ALPHA_LAYERS = 256;

using plane_cond2d_device::kGradClamp;

__device__ __forceinline__ float dot3f(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 normalize3_forward(const float3& v) {
    float n = sqrtf(fmaxf(1e-12f, dot3f(v, v)));
    return make_float3(v.x / n, v.y / n, v.z / n);
}

__device__ __forceinline__ float3 normalize3_backward(const float3& raw, const float3& n_unit, const float3& grad_out) {
    float len = sqrtf(fmaxf(1e-12f, dot3f(raw, raw)));
    float inv_len = 1.f / len;
    float dot_ng = dot3f(n_unit, grad_out);
    return make_float3(
        (grad_out.x - n_unit.x * dot_ng) * inv_len,
        (grad_out.y - n_unit.y * dot_ng) * inv_len,
        (grad_out.z - n_unit.z * dot_ng) * inv_len);
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
    Sc[0] = dot3f(axis_lat, c0);
    Sc[1] = dot3f(axis_lat, c1);
    Sc[2] = dot3f(axis_lat, c2);
    Sc[3] = dot3f(axis_ax, c0);
    Sc[4] = dot3f(axis_ax, c1);
    Sc[5] = dot3f(axis_ax, c2);
    Sc[6] = dot3f(axis_ele, c0);
    Sc[7] = dot3f(axis_ele, c1);
    Sc[8] = dot3f(axis_ele, c2);
}

__device__ __forceinline__ void schur_bwd(
    float mu_lat,
    float mu_ax,
    float mu_z,
    const float Sc[9],
    float g_mu2_lat,
    float g_mu2_ax,
    float g_s2_00,
    float g_s2_01,
    float g_s2_11,
    float g_mu_lat[3],
    float g_Sc[9]) {
    for (int i = 0; i < 9; ++i) g_Sc[i] = 0.f;
    g_mu_lat[0] = g_mu_lat[1] = g_mu_lat[2] = 0.f;

    const float inv_zz = 1.f / Sc[8];
    const float spz_lat = Sc[2];
    const float spz_ax = Sc[5];

    g_mu_lat[0] += g_mu2_lat;
    g_mu_lat[1] += g_mu2_ax;
    g_mu_lat[2] += g_mu2_lat * (-spz_lat * inv_zz) + g_mu2_ax * (-spz_ax * inv_zz);

    float g_spz_lat = g_mu2_lat * (-mu_z * inv_zz);
    float g_spz_ax = g_mu2_ax * (-mu_z * inv_zz);
    float g_szz = g_mu2_lat * (spz_lat * mu_z * inv_zz * inv_zz) + g_mu2_ax * (spz_ax * mu_z * inv_zz * inv_zz);

    g_Sc[0] += g_s2_00;
    g_Sc[1] += g_s2_01;
    g_Sc[4] += g_s2_11;

    g_spz_lat += -inv_zz * (2.f * g_s2_00 * spz_lat + g_s2_01 * spz_ax);
    g_spz_ax += -inv_zz * (2.f * g_s2_11 * spz_ax + g_s2_01 * spz_lat);
    g_szz += g_s2_00 * spz_lat * spz_lat * inv_zz * inv_zz + g_s2_11 * spz_ax * spz_ax * inv_zz * inv_zz
        + g_s2_01 * spz_lat * spz_ax * inv_zz * inv_zz;

    g_Sc[2] += g_spz_lat;
    g_Sc[5] += g_spz_ax;
    g_Sc[8] += g_szz;
}

__device__ bool ours_plane_cond2d_bwd_fill(
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
    float* __restrict__ mu2_lat,
    float* __restrict__ mu2_ax,
    float* __restrict__ inv2_00,
    float* __restrict__ inv2_01,
    float* __restrict__ inv2_11,
    float* __restrict__ mu_z_out,
    float* __restrict__ s_zz_out,
    float* __restrict__ o_out,
    float* __restrict__ I_out) {
    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;
    float ox = pose[3], oy = pose[7], oz = pose[11];
    float dw_x = mptr[0] - ox, dw_y = mptr[1] - oy, dw_z = mptr[2] - oz;
    float mu_lat = dw_x * axis_lat.x + dw_y * axis_lat.y + dw_z * axis_lat.z;
    float mu_ax = dw_x * axis_ax.x + dw_y * axis_ax.y + dw_z * axis_ax.z;
    float mu_z = dw_x * axis_ele.x + dw_y * axis_ele.y + dw_z * axis_ele.z;

    float Rg[9];
    quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
    float sx = fmaxf(sptr[0], 1e-6f);
    float sy = fmaxf(sptr[1], 1e-6f);
    float sz = fmaxf(sptr[2], 1e-6f);
    float Sigma_w[9];
    mat3_sigma_world(sx, sy, sz, Rg, Sigma_w);
    float Sc[9];
    build_sigma_c_probe(Sigma_w, axis_lat, axis_ax, axis_ele, Sc);
    plane_cond2d_device::add_probe_cov_jitter(Sc);
    float s2_00 = 0.f, s2_01 = 0.f, s2_11 = 0.f;
    *mu_z_out = mu_z;
    *s_zz_out = Sc[8];
    *o_out = gauss_opacities[g];
    *I_out = gauss_intensities[g];
    return plane_cond2d_device::schur_conditional_plane(
        mu_lat, mu_ax, mu_z, Sc, mu2_lat, mu2_ax, &s2_00, &s2_01, &s2_11, inv2_00, inv2_01, inv2_11);
}

__device__ void plane_cond2d_sc_to_world_grads(
    const float g_Sc[9],
    float3 axis_lat,
    float3 axis_ax,
    float3 axis_ele,
    float gSigma_w[9]) {
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

/** dL/d(axis_i) for Sc_ij = axis_i^T Sigma_w axis_j (symmetric Sigma_w, g_Sc). */
__device__ void plane_cond2d_sc_to_axis_grads(
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
            const float3 Saj = plane_cond2d_device::mat3_vec_sym(Sigma_w, axes[j]);
            g_axes[i]->x += 2.f * gij * Saj.x;
            g_axes[i]->y += 2.f * gij * Saj.y;
            g_axes[i]->z += 2.f * gij * Saj.z;
        }
    }
}

}  // namespace plane_cond2d_cu

__device__ __forceinline__ float plane_cond2d_sanitize_grad(float g) {
    if (!isfinite(g)) {
        return 0.f;
    }
    if (fabsf(g) > plane_cond2d_device::kGradClamp) {
        return copysignf(plane_cond2d_device::kGradClamp, g);
    }
    return g;
}

__device__ void ours_plane_cond2d_bwd_wire(
    int col,
    int row,
    const float* __restrict__ slice_grid_pack,
    float mu_lat_2d,
    float mu_ax_2d,
    float s2d_inv00,
    float s2d_inv01,
    float s2d_inv11,
    float mu_z,
    float s_zz,
    float o,
    float Ir,
    float dL_dalpha,
    float dL_dI,
    float* __restrict__ wrow) {
    (void)Ir;
    float xi_lat = 0.f;
    float xi_ax = 0.f;
    sg_linear_pixel_to_plane_mm(slice_grid_pack, col, row, &xi_lat, &xi_ax);
    const float dlat = xi_lat - mu_lat_2d;
    const float dax = xi_ax - mu_ax_2d;
    const float quad_raw = dlat * (s2d_inv00 * dlat + s2d_inv01 * dax) + dax * (s2d_inv01 * dlat + s2d_inv11 * dax);
    const float base_plane = plane_cond2d_device::plane_cond2d_plane_weight(quad_raw);
    const float elev_quad = (mu_z * mu_z) / fmaxf(s_zz, 1e-12f);
    const float elev_atten = expf(-0.5f * elev_quad);
    const float base = base_plane * elev_atten;

    atomicAdd(wrow + PLANE_COND2D_WSP_G_OPACITY, dL_dalpha * base);
    atomicAdd(wrow + PLANE_COND2D_WSP_G_INTENSITY, dL_dI);

    const float dL_dbase_total = dL_dalpha * o;
    const float dL_dbase_plane = dL_dbase_total * elev_atten;
    const float dL_delev_atten = dL_dbase_total * base_plane;

    if (quad_raw > 0.f) {
        const float dL_dquad = dL_dbase_plane * (-0.5f * base_plane);
        const float g_delta_lat = 2.f * dL_dquad * (s2d_inv00 * dlat + s2d_inv01 * dax);
        const float g_delta_ax = 2.f * dL_dquad * (s2d_inv01 * dlat + s2d_inv11 * dax);
        if (isfinite(g_delta_lat)) {
            atomicAdd(wrow + PLANE_COND2D_WSP_G_MU2_LAT, -g_delta_lat);
        }
        if (isfinite(g_delta_ax)) {
            atomicAdd(wrow + PLANE_COND2D_WSP_G_MU2_AX, -g_delta_ax);
        }
        if (isfinite(dL_dquad)) {
            atomicAdd(wrow + PLANE_COND2D_WSP_G_I00, dL_dquad * dlat * dlat);
            atomicAdd(wrow + PLANE_COND2D_WSP_G_I01, dL_dquad * 2.f * dlat * dax);
            atomicAdd(wrow + PLANE_COND2D_WSP_G_I11, dL_dquad * dax * dax);
        }
    }

    if (isfinite(dL_delev_atten) && isfinite(elev_atten) && s_zz > 0.f) {
        const float g_mu_z = dL_delev_atten * elev_atten * (-mu_z / s_zz);
        const float g_szz = dL_delev_atten * elev_atten * (0.5f * elev_quad / s_zz);
        if (isfinite(g_mu_z)) {
            atomicAdd(wrow + PLANE_COND2D_WSP_G_MU_Z, g_mu_z);
        }
        if (isfinite(g_szz)) {
            atomicAdd(wrow + PLANE_COND2D_WSP_G_SZZ, g_szz);
        }
    }
}

namespace plane_cond2d_cu {

__global__ void kernel_plane_cond2d_alpha_back_pixel_csr(
    int n_pixels,
    int H,
    int W,
    float k_sigma,
    float fixed_bg_color,
    const float* __restrict__ slice_grid_pack,
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
    float3 o3 = make_float3(pose[3], pose[7], pose[11]);

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
        ours_plane_cond2d_eval_at_pixel(
            x,
            y,
            k_sigma,
            g,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            axis_lat,
            axis_ax,
            axis_ele,
            o3,
            slice_grid_pack,
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

        float mu2_lat, mu2_ax, i00, i01, i11, mu_z, s_zz, o, Ir;
        if (!ours_plane_cond2d_bwd_fill(
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
                &mu2_lat,
                &mu2_ax,
                &i00,
                &i01,
                &i11,
                &mu_z,
                &s_zz,
                &o,
                &Ir)) {
            continue;
        }
        float* wrow = workspace + g * PLANE_COND2D_WSP_STRIDE;
        ours_plane_cond2d_bwd_wire(
            x,
            y,
            slice_grid_pack,
            mu2_lat,
            mu2_ax,
            i00,
            i01,
            i11,
            mu_z,
            s_zz,
            o,
            Ir,
            dL_dalpha,
            dL_dI,
            wrow);
    }
}

__global__ void kernel_plane_cond2d_alpha_back_tile_csr(
    int n_pixels,
    int H,
    int W,
    int Tw,
    int tile_size_px,
    float k_sigma,
    float fixed_bg_color,
    const float* __restrict__ slice_grid_pack,
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
        ours_plane_cond2d_eval_at_pixel(
            x,
            y,
            k_sigma,
            g,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            axis_lat,
            axis_ax,
            axis_ele,
            o3,
            slice_grid_pack,
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

        float mu2_lat, mu2_ax, i00, i01, i11, mu_z, s_zz, o, Ir;
        if (!ours_plane_cond2d_bwd_fill(
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
                &mu2_lat,
                &mu2_ax,
                &i00,
                &i01,
                &i11,
                &mu_z,
                &s_zz,
                &o,
                &Ir)) {
            continue;
        }
        float* wrow = workspace + g * PLANE_COND2D_WSP_STRIDE;
        ours_plane_cond2d_bwd_wire(
            x,
            y,
            slice_grid_pack,
            mu2_lat,
            mu2_ax,
            i00,
            i01,
            i11,
            mu_z,
            s_zz,
            o,
            Ir,
            dL_dalpha,
            dL_dI,
            wrow);
    }
}

__global__ void kernel_plane_cond2d_additive_bwd(
    int n_pixels,
    int n_gauss,
    int H,
    int W,
    int tile_size_px,
    int gauss_rect_is_pixels,
    float k_sigma,
    const float* __restrict__ slice_grid_pack,
    const int* __restrict__ gauss_rects,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    const float* __restrict__ pose,
    const float* __restrict__ grad_pixel_colors,
    float* __restrict__ workspace) {
    (void)n_pixels;
    (void)H;
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

    float mu2_lat, mu2_ax, i00, i01, i11, mu_z, s_zz, o, Ir;
    if (!ours_plane_cond2d_bwd_fill(
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
            &mu2_lat,
            &mu2_ax,
            &i00,
            &i01,
            &i11,
            &mu_z,
            &s_zz,
            &o,
            &Ir)) {
        return;
    }

    float* wrow = workspace + g * PLANE_COND2D_WSP_STRIDE;

#define PLANE_BWD_ONE(pxv, pyv)                                                                 \
    do {                                                                                        \
        int p = (pyv)*W + (pxv);                                                                \
        float gc = grad_pixel_colors[p];                                                        \
        if (fabsf(gc) >= 1e-24f) {                                                              \
            float alpha = 0.f, Iv = 0.f;                                                      \
            float3 o3 = make_float3(pose[3], pose[7], pose[11]);                              \
            ours_plane_cond2d_eval_at_pixel(                                                    \
                (pxv),                                                                          \
                (pyv),                                                                          \
                k_sigma,                                                                        \
                g,                                                                              \
                gauss_means,                                                                    \
                gauss_scales,                                                                   \
                gauss_quats,                                                                    \
                gauss_opacities,                                                                \
                gauss_intensities,                                                              \
                axis_lat,                                                                       \
                axis_ax,                                                                        \
                axis_ele,                                                                       \
                o3,                                                                             \
                slice_grid_pack,                                                                \
                &alpha,                                                                         \
                &Iv);                                                                           \
            if (alpha >= OURS_ALPHA_MIN) {                                                      \
                ours_plane_cond2d_bwd_wire(                                                     \
                    (pxv),                                                                      \
                    (pyv),                                                                      \
                    slice_grid_pack,                                                            \
                    mu2_lat,                                                                    \
                    mu2_ax,                                                                     \
                    i00,                                                                        \
                    i01,                                                                        \
                    i11,                                                                        \
                    mu_z,                                                                       \
                    s_zz,                                                                       \
                    o,                                                                          \
                    Iv,                                                                         \
                    gc * Iv,                                                                    \
                    gc * alpha,                                                                 \
                    wrow);                                                                      \
            }                                                                                   \
        }                                                                                       \
    } while (0)

    if (gauss_rect_is_pixels != 0) {
        for (int py = r1; py < r3; ++py) {
            for (int px = r0; px < r2; ++px) {
                PLANE_BWD_ONE(px, py);
            }
        }
    } else {
        for (int tty = r1; tty < r3; ++tty) {
            for (int ttx = r0; ttx < r2; ++ttx) {
                int px0 = ttx * tile_size_px;
                int px1 = min(W - 1, (ttx + 1) * tile_size_px - 1);
                int py0 = tty * tile_size_px;
                int py1 = min(H - 1, (tty + 1) * tile_size_px - 1);
                for (int py = py0; py <= py1; ++py) {
                    for (int px = px0; px <= px1; ++px) {
                        PLANE_BWD_ONE(px, py);
                    }
                }
            }
        }
    }
#undef PLANE_BWD_ONE
}

__global__ void kernel_plane_cond2d_finalize_gauss(
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

    const float* wrow = workspace + g * PLANE_COND2D_WSP_STRIDE;
    float g_opacity = wrow[PLANE_COND2D_WSP_G_OPACITY];
    float g_intensity = wrow[PLANE_COND2D_WSP_G_INTENSITY];
    float g_mu2_lat = wrow[PLANE_COND2D_WSP_G_MU2_LAT];
    float g_mu2_ax = wrow[PLANE_COND2D_WSP_G_MU2_AX];
    float g_mu_z_elev = wrow[PLANE_COND2D_WSP_G_MU_Z];
    float g_szz_elev = wrow[PLANE_COND2D_WSP_G_SZZ];
    float g_i00 = wrow[PLANE_COND2D_WSP_G_I00];
    float g_i01 = wrow[PLANE_COND2D_WSP_G_I01];
    float g_i11 = wrow[PLANE_COND2D_WSP_G_I11];

    if (fabsf(g_opacity) + fabsf(g_intensity) + fabsf(g_mu2_lat) + fabsf(g_mu2_ax) + fabsf(g_mu_z_elev)
            + fabsf(g_szz_elev) + fabsf(g_i00) + fabsf(g_i01) + fabsf(g_i11)
        < 1e-30f) {
        return;
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
    float ox = pose[3], oy = pose[7], oz = pose[11];
    float dw_x = mptr[0] - ox, dw_y = mptr[1] - oy, dw_z = mptr[2] - oz;
    float mu_lat = dw_x * axis_lat.x + dw_y * axis_lat.y + dw_z * axis_lat.z;
    float mu_ax = dw_x * axis_ax.x + dw_y * axis_ax.y + dw_z * axis_ax.z;
    float mu_z = dw_x * axis_ele.x + dw_y * axis_ele.y + dw_z * axis_ele.z;

    float Rg[9];
    quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
    float sx = fmaxf(sptr[0], 1e-6f);
    float sy = fmaxf(sptr[1], 1e-6f);
    float sz = fmaxf(sptr[2], 1e-6f);
    float Sigma_w[9];
    mat3_sigma_world(sx, sy, sz, Rg, Sigma_w);
    float Sc[9];
    build_sigma_c_probe(Sigma_w, axis_lat, axis_ax, axis_ele, Sc);
    plane_cond2d_device::add_probe_cov_jitter(Sc);

    float mu2_lat = 0.f, mu2_ax = 0.f;
    float s2_00 = 0.f, s2_01 = 0.f, s2_11 = 0.f;
    float inv2_00 = 0.f, inv2_01 = 0.f, inv2_11 = 0.f;
    if (!plane_cond2d_device::schur_conditional_plane(
            mu_lat, mu_ax, mu_z, Sc, &mu2_lat, &mu2_ax, &s2_00, &s2_01, &s2_11, &inv2_00, &inv2_01, &inv2_11)) {
        grad_gauss_opacities[g] = g_opacity;
        grad_gauss_intensities[g] = g_intensity;
        return;
    }

    float g_s2_00 = 0.f, g_s2_01 = 0.f, g_s2_11 = 0.f;
    plane_cond2d_device::inv2_sym_bwd(s2_00, s2_01, s2_11, g_i00, g_i01, g_i11, &g_s2_00, &g_s2_01, &g_s2_11);

    float g_mu_probe[3] = {0.f, 0.f, 0.f};
    float g_Sc[9];
    schur_bwd(mu_lat, mu_ax, mu_z, Sc, g_mu2_lat, g_mu2_ax, g_s2_00, g_s2_01, g_s2_11, g_mu_probe, g_Sc);
    g_mu_probe[2] += g_mu_z_elev;
    g_Sc[8] += g_szz_elev;

    float gSigma_w[9];
    plane_cond2d_sc_to_world_grads(g_Sc, axis_lat, axis_ax, axis_ele, gSigma_w);

    float g_sx = 0.f, g_sy = 0.f, g_sz = 0.f;
    float grad_R[9];
    plane_cond2d_device::sigma_w_to_scale_rot_grads(gSigma_w, Rg, sx, sy, sz, &g_sx, &g_sy, &g_sz, grad_R);

    grad_gauss_opacities[g] = plane_cond2d_sanitize_grad(g_opacity);
    grad_gauss_intensities[g] = plane_cond2d_sanitize_grad(g_intensity);
    grad_gauss_scales[g * 3 + 0] = plane_cond2d_sanitize_grad(g_sx);
    grad_gauss_scales[g * 3 + 1] = plane_cond2d_sanitize_grad(g_sy);
    grad_gauss_scales[g * 3 + 2] = plane_cond2d_sanitize_grad(g_sz);

    float gq_local[4];
    grad_quat_finite_diff(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), grad_R, gq_local);
    grad_gauss_quats[g * 4 + 0] = plane_cond2d_sanitize_grad(gq_local[0]);
    grad_gauss_quats[g * 4 + 1] = plane_cond2d_sanitize_grad(gq_local[1]);
    grad_gauss_quats[g * 4 + 2] = plane_cond2d_sanitize_grad(gq_local[2]);
    grad_gauss_quats[g * 4 + 3] = plane_cond2d_sanitize_grad(gq_local[3]);

    float3 g_delta = make_float3(
        g_mu_probe[0] * axis_lat.x + g_mu_probe[1] * axis_ax.x + g_mu_probe[2] * axis_ele.x,
        g_mu_probe[0] * axis_lat.y + g_mu_probe[1] * axis_ax.y + g_mu_probe[2] * axis_ele.y,
        g_mu_probe[0] * axis_lat.z + g_mu_probe[1] * axis_ax.z + g_mu_probe[2] * axis_ele.z);
    grad_gauss_means[g * 3 + 0] = plane_cond2d_sanitize_grad(g_delta.x);
    grad_gauss_means[g * 3 + 1] = plane_cond2d_sanitize_grad(g_delta.y);
    grad_gauss_means[g * 3 + 2] = plane_cond2d_sanitize_grad(g_delta.z);

    atomicAdd(grad_pose + 3, -g_delta.x);
    atomicAdd(grad_pose + 7, -g_delta.y);
    atomicAdd(grad_pose + 11, -g_delta.z);

    float3 g_axis_lat = make_float3(0.f, 0.f, 0.f);
    float3 g_axis_ax = make_float3(0.f, 0.f, 0.f);
    float3 g_axis_ele = make_float3(0.f, 0.f, 0.f);
    plane_cond2d_sc_to_axis_grads(g_Sc, Sigma_w, axis_lat, axis_ax, axis_ele, &g_axis_lat, &g_axis_ax, &g_axis_ele);

    float3 g_lat_mu = make_float3(g_mu_probe[0] * dw_x, g_mu_probe[0] * dw_y, g_mu_probe[0] * dw_z);
    float3 g_ax_mu = make_float3(g_mu_probe[1] * dw_x, g_mu_probe[1] * dw_y, g_mu_probe[1] * dw_z);
    float3 g_ele_mu = make_float3(g_mu_probe[2] * dw_x, g_mu_probe[2] * dw_y, g_mu_probe[2] * dw_z);

    float3 g_raw_lat = normalize3_backward(
        raw_lat, axis_lat, make_float3(g_lat_mu.x + g_axis_lat.x, g_lat_mu.y + g_axis_lat.y, g_lat_mu.z + g_axis_lat.z));
    float3 g_raw_ax = normalize3_backward(
        raw_ax, axis_ax, make_float3(g_ax_mu.x + g_axis_ax.x, g_ax_mu.y + g_axis_ax.y, g_ax_mu.z + g_axis_ax.z));
    float3 g_raw_ele = normalize3_backward(
        raw_ele, axis_ele, make_float3(g_ele_mu.x + g_axis_ele.x, g_ele_mu.y + g_axis_ele.y, g_ele_mu.z + g_axis_ele.z));

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

}  // namespace plane_cond2d_cu

void RenderOursSliceBackwardPlaneCond2D_Impl(
    int n_pixels,
    int n_gauss,
    int H,
    int W,
    int tile_size_px,
    int gauss_rect_is_pixels,
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
    const float* slice_grid_pack,
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

    if (workspace == nullptr || workspace_floats < n_gauss * PLANE_COND2D_WSP_STRIDE) {
        return;
    }
    cudaMemsetAsync(workspace, 0, sizeof(float) * (size_t)n_gauss * PLANE_COND2D_WSP_STRIDE, stream);

    const int pb = 128;
    const int pg = (n_pixels + pb - 1) / pb;

    if (compositing_mode == 0) {
        const int blocks = (n_gauss + threads - 1) / threads;
        plane_cond2d_cu::kernel_plane_cond2d_additive_bwd<<<blocks, threads, 0, stream>>>(
            n_pixels,
            n_gauss,
            H,
            W,
            tile_size_px,
            gauss_rect_is_pixels,
            k_sigma,
            slice_grid_pack,
            gauss_rects,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose_probe_to_world,
            grad_pixel_colors,
            workspace);
    } else {
        if (mapper_kind == 0) {
            const int Tw = (W + tile_size_px - 1) / tile_size_px;
            plane_cond2d_cu::kernel_plane_cond2d_alpha_back_tile_csr<<<pg, pb, 0, stream>>>(
                n_pixels,
                H,
                W,
                Tw,
                tile_size_px,
                k_sigma,
                fixed_bg_color,
                slice_grid_pack,
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
            plane_cond2d_cu::kernel_plane_cond2d_alpha_back_pixel_csr<<<pg, pb, 0, stream>>>(
                n_pixels,
                H,
                W,
                k_sigma,
                fixed_bg_color,
                slice_grid_pack,
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
    }

    const int gb = 128;
    const int gg = (n_gauss + gb - 1) / gb;
    plane_cond2d_cu::kernel_plane_cond2d_finalize_gauss<<<gg, gb, 0, stream>>>(
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
