#include "covariance.cuh"
#include "shader/shared.cuh"
#include "load_utils.cuh"
#include "math_utils.cuh"

__device__ __forceinline__ int lower_bound_depth_bwd(
    const float* __restrict__ sorted_depths,
    int offset,
    int count,
    float value
) {
    int lo = 0;
    int hi = count;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (sorted_depths[offset + mid] < value) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ float sigmoid_prime(float sigma_val) {
    return sigma_val * (1.0f - sigma_val);
}

__device__ __forceinline__ void compute_integral_vjp(
    const float3& ray_o, const float3& ray_d,
    const float3& mu, const float3& s, const float* R,
    float l,
    float quadrature_window_radius_mm,
    float dL_dIntegral,
    float3& dL_dMean, float3& dL_dScale, float* dL_dR_buf
) {
    if (l <= 0.0f) return;

    float3 diff = ray_o - mu;
    float3 o_local;
    o_local.x = R[0] * diff.x + R[3] * diff.y + R[6] * diff.z;
    o_local.y = R[1] * diff.x + R[4] * diff.y + R[7] * diff.z;
    o_local.z = R[2] * diff.x + R[5] * diff.y + R[8] * diff.z;

    float3 d_local;
    d_local.x = R[0] * ray_d.x + R[3] * ray_d.y + R[6] * ray_d.z;
    d_local.y = R[1] * ray_d.x + R[4] * ray_d.y + R[7] * ray_d.z;
    d_local.z = R[2] * ray_d.x + R[5] * ray_d.y + R[8] * ray_d.z;

    float3 inv_s = make_float3(1.0f / s.x, 1.0f / s.y, 1.0f / s.z);
    float3 o_g = make_float3(o_local.x * inv_s.x, o_local.y * inv_s.y, o_local.z * inv_s.z);
    float3 d_g = make_float3(d_local.x * inv_s.x, d_local.y * inv_s.y, d_local.z * inv_s.z);

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
    if (t_end <= t_start) return;

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

    float coeff = dL_dIntegral * half;
    float3 dL_dog = make_float3(0.0f, 0.0f, 0.0f);
    float3 dL_ddg = make_float3(0.0f, 0.0f, 0.0f);
    dL_dog = dL_dog + (x0 * (-coeff * w0 * f0));
    dL_dog = dL_dog + (x1 * (-coeff * w1 * f1));
    dL_dog = dL_dog + (x2 * (-coeff * w2 * f2));
    dL_ddg = dL_ddg + (x0 * (-coeff * w0 * f0 * t0));
    dL_ddg = dL_ddg + (x1 * (-coeff * w1 * f1 * t1));
    dL_ddg = dL_ddg + (x2 * (-coeff * w2 * f2 * t2));
    float3 dL_dolocal = make_float3(dL_dog.x * inv_s.x, dL_dog.y * inv_s.y, dL_dog.z * inv_s.z);
    float3 dL_ddlocal = make_float3(dL_ddg.x * inv_s.x, dL_ddg.y * inv_s.y, dL_ddg.z * inv_s.z);

    dL_dScale.x += dL_dog.x * (-o_g.x * inv_s.x) + dL_ddg.x * (-d_g.x * inv_s.x);
    dL_dScale.y += dL_dog.y * (-o_g.y * inv_s.y) + dL_ddg.y * (-d_g.y * inv_s.y);
    dL_dScale.z += dL_dog.z * (-o_g.z * inv_s.z) + dL_ddg.z * (-d_g.z * inv_s.z);

    dL_dMean.x += -(R[0] * dL_dolocal.x + R[1] * dL_dolocal.y + R[2] * dL_dolocal.z);
    dL_dMean.y += -(R[3] * dL_dolocal.x + R[4] * dL_dolocal.y + R[5] * dL_dolocal.z);
    dL_dMean.z += -(R[6] * dL_dolocal.x + R[7] * dL_dolocal.y + R[8] * dL_dolocal.z);

    #pragma unroll
    for (int r = 0; r < 3; r++) {
        float diff_val = (r == 0) ? diff.x : (r == 1) ? diff.y : diff.z;
        float dir_val = (r == 0) ? ray_d.x : (r == 1) ? ray_d.y : ray_d.z;
        dL_dR_buf[r * 3 + 0] += dL_dolocal.x * diff_val + dL_ddlocal.x * dir_val;
        dL_dR_buf[r * 3 + 1] += dL_dolocal.y * diff_val + dL_ddlocal.y * dir_val;
        dL_dR_buf[r * 3 + 2] += dL_dolocal.z * diff_val + dL_ddlocal.z * dir_val;
    }
}

__device__ __forceinline__ void compute_weight_vjp(
    const float3& p, const float3& mu, const float* SigmaInv,
    float w, float dL_dw,
    float3& dL_dMean, float* dL_dSigmaInv
) {
    float dL_dQ = dL_dw * w * (-0.5f);
    float dx = p.x - mu.x;
    float dy = p.y - mu.y;
    float dz = p.z - mu.z;

    float v0 = SigmaInv[0] * dx + SigmaInv[1] * dy + SigmaInv[2] * dz;
    float v1 = SigmaInv[3] * dx + SigmaInv[4] * dy + SigmaInv[5] * dz;
    float v2 = SigmaInv[6] * dx + SigmaInv[7] * dy + SigmaInv[8] * dz;
    dL_dMean.x += dL_dQ * (-2.0f * v0);
    dL_dMean.y += dL_dQ * (-2.0f * v1);
    dL_dMean.z += dL_dQ * (-2.0f * v2);

    dL_dSigmaInv[0] += dL_dQ * dx * dx;
    dL_dSigmaInv[1] += dL_dQ * dx * dy;
    dL_dSigmaInv[2] += dL_dQ * dx * dz;
    dL_dSigmaInv[3] += dL_dQ * dy * dx;
    dL_dSigmaInv[4] += dL_dQ * dy * dy;
    dL_dSigmaInv[5] += dL_dQ * dy * dz;
    dL_dSigmaInv[6] += dL_dQ * dz * dx;
    dL_dSigmaInv[7] += dL_dQ * dz * dy;
    dL_dSigmaInv[8] += dL_dQ * dz * dz;
}

__global__ void render_bmode_backward_kernel(
    int W, int H,
    float4* __restrict__ dL_dMeans,
    float4* __restrict__ dL_dScales,
    float4* __restrict__ dL_dQuats,
    float* __restrict__ dL_dAttens,
    float4* __restrict__ dL_dSH,
    const float* __restrict__ dL_dImage,
    const int* __restrict__ ray_counts,
    const int* __restrict__ ray_offsets,
    const int* __restrict__ ray_indices,
    const float* __restrict__ sorted_depths,
    const float4* __restrict__ means,
    const float4* __restrict__ scales,
    const float4* __restrict__ quats,
    const float* __restrict__ attenuations,
    const float4* __restrict__ sh_coeffs,
    const float* __restrict__ pixel_positions,
    const float* __restrict__ ray_origins,
    const float* __restrict__ ray_dirs,
    const float* __restrict__ axis_elevational,
    const float* __restrict__ ray_perturbations,
    float k_sigma,
    float quadrature_window_radius_mm,
    bool local_transmission_window_only,
    float local_transmission_max_window_mm
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float dL_dB = dL_dImage[y * W + x];
    if (fabsf(dL_dB) < 1e-6f) return;
    int count = ray_counts[x];
    if (count == 0) return;
    int offset = ray_offsets[x];

    const float* p_ptr = pixel_positions + (x * H + y) * 3;
    float3 p = make_float3(p_ptr[0], p_ptr[1], p_ptr[2]);
    float3 ray_o = make_float3(ray_origins[x * 3], ray_origins[x * 3 + 1], ray_origins[x * 3 + 2]);
    float3 axis_ele = make_float3(axis_elevational[0], axis_elevational[1], axis_elevational[2]);
    ray_o = ray_o + axis_ele * ray_perturbations[x];
    float3 ray_d = make_float3(ray_dirs[x * 3], ray_dirs[x * 3 + 1], ray_dirs[x * 3 + 2]);
    float t_pixel = fmaxf(0.0f, dot_prod(p - ray_o, ray_d));

    float T_final = 1.0f;
    float E_num = 0.0f;
    float E_den = 0.0f;
    float S_accum = 0.0f;

    int begin_k = 0;
    float upper_depth = 1.0e30f;
    if (local_transmission_window_only) {
        begin_k = lower_bound_depth_bwd(sorted_depths, offset, count, t_pixel - local_transmission_max_window_mm);
        upper_depth = t_pixel + local_transmission_max_window_mm;
    }

    for (int k = begin_k; k < count; ++k) {
        if (local_transmission_window_only && sorted_depths[offset + k] > upper_depth) break;
        int idx = ray_indices[offset + k];
        float3 mu = xyz(means[idx]);
        float3 s = xyz(scales[idx]);
        float t_gauss = dot_prod(mu - ray_o, ray_d);
        float max_scale = fmaxf(s.x, fmaxf(s.y, s.z));
        float depth_radius = max_scale * k_sigma;
        bool needs_transmission = t_gauss < t_pixel;
        if (local_transmission_window_only) {
            needs_transmission = needs_transmission && ((t_pixel - t_gauss) <= depth_radius + quadrature_window_radius_mm);
        }
        bool needs_echo = fabsf(t_gauss - t_pixel) <= depth_radius;
        if (!needs_transmission && !needs_echo) continue;

        float4 q = quats[idx];
        float R[9]; compute_rot_from_quat(q, R);

        if (needs_transmission) {
            float psi = compute_ray_gaussian_integral(ray_o, ray_d, mu, s, R, t_pixel, quadrature_window_radius_mm);
            float raw_tau = attenuations[idx];
            float tau = 1.0f / (1.0f + __expf(-raw_tau));
            float T_i = tau + (1.0f - tau) * __expf(-psi);
            T_final *= T_i;
        }

        if (needs_echo) {
            float SigmaInv[9]; compute_cov3d_from_scale_rot(s, R, SigmaInv);
            float w = compute_gaussian_weight(p, mu, SigmaInv);
            float I = eval_sh_degree1(sh_coeffs[idx], ray_d);
            E_num += I * w;
            E_den += w;
            S_accum += w;
        }
    }

    float gamma = 1.0f - __expf(-S_accum);
    float epsilon = 1e-6f;
    float E_raw = E_num / (E_den + epsilon);
    float E_val = gamma * E_raw;

    float dL_dT = dL_dB * E_val;
    float dL_dE = dL_dB * T_final;
    float dL_dGamma = dL_dE * E_raw;
    float dL_dE_raw = dL_dE * gamma;
    float dL_dS_from_gamma = dL_dGamma * __expf(-S_accum);
    float dL_dNum = dL_dE_raw * (1.0f / (E_den + epsilon));
    float dL_dDen = dL_dE_raw * (-E_num / ((E_den + epsilon) * (E_den + epsilon)));

    for (int k = begin_k; k < count; ++k) {
        if (local_transmission_window_only && sorted_depths[offset + k] > upper_depth) break;
        int idx = ray_indices[offset + k];
        float3 mu = xyz(means[idx]);
        float3 s = xyz(scales[idx]);
        float t_gauss = dot_prod(mu - ray_o, ray_d);
        float max_scale = fmaxf(s.x, fmaxf(s.y, s.z));
        float depth_radius = max_scale * k_sigma;
        bool needs_transmission = t_gauss < t_pixel;
        if (local_transmission_window_only) {
            needs_transmission = needs_transmission && ((t_pixel - t_gauss) <= depth_radius + quadrature_window_radius_mm);
        }
        bool needs_echo = fabsf(t_gauss - t_pixel) <= depth_radius;
        if (!needs_transmission && !needs_echo) continue;

        float4 q = quats[idx];
        float R[9]; compute_rot_from_quat(q, R);

        float3 dL_dMu_acc = make_float3(0, 0, 0);
        float3 dL_dScale_acc = make_float3(0, 0, 0);
        float dL_dR_acc[9] = {0};

        if (needs_transmission) {
            float psi = compute_ray_gaussian_integral(ray_o, ray_d, mu, s, R, t_pixel, quadrature_window_radius_mm);
            float raw_tau = attenuations[idx];
            float sig_tau = 1.0f / (1.0f + __expf(-raw_tau));
            float exp_psi = __expf(-psi);
            float T_i = sig_tau + (1.0f - sig_tau) * exp_psi;

            float dL_dTi = 0.0f;
            if (T_i > 1e-6f) dL_dTi = dL_dT * (T_final / T_i);
            float dTi_dTau = 1.0f - exp_psi;
            float dTau_dRaw = sigmoid_prime(sig_tau);
            atomicAdd(dL_dAttens + idx, dL_dTi * dTi_dTau * dTau_dRaw);

            float dL_dPsi = dL_dTi * (1.0f - sig_tau) * exp_psi * (-1.0f);
            compute_integral_vjp(
                ray_o,
                ray_d,
                mu,
                s,
                R,
                t_pixel,
                quadrature_window_radius_mm,
                dL_dPsi,
                dL_dMu_acc,
                dL_dScale_acc,
                dL_dR_acc
            );
        }

        if (needs_echo) {
            float SigmaInv[9]; compute_cov3d_from_scale_rot(s, R, SigmaInv);
            float w = compute_gaussian_weight(p, mu, SigmaInv);
            float4 sh = sh_coeffs[idx];
            float I = eval_sh_degree1(sh, ray_d);
            atomicAdd(&(dL_dSH[idx].x), dL_dNum * w);
            atomicAdd(&(dL_dSH[idx].y), dL_dNum * w * ray_d.x);
            atomicAdd(&(dL_dSH[idx].z), dL_dNum * w * ray_d.y);
            atomicAdd(&(dL_dSH[idx].w), dL_dNum * w * ray_d.z);
            float dL_dw = (dL_dNum * I) + dL_dDen + dL_dS_from_gamma;

            float dL_dSigmaInv[9] = {0};
            compute_weight_vjp(p, mu, SigmaInv, w, dL_dw, dL_dMu_acc, dL_dSigmaInv);
            transform_grads_sigma_to_R_S(s, R, dL_dSigmaInv, dL_dR_acc, dL_dScale_acc);
        }

        atomicAdd(&(dL_dMeans[idx].x), dL_dMu_acc.x);
        atomicAdd(&(dL_dMeans[idx].y), dL_dMu_acc.y);
        atomicAdd(&(dL_dMeans[idx].z), dL_dMu_acc.z);
        atomicAdd(&(dL_dScales[idx].x), dL_dScale_acc.x);
        atomicAdd(&(dL_dScales[idx].y), dL_dScale_acc.y);
        atomicAdd(&(dL_dScales[idx].z), dL_dScale_acc.z);

        float4 dL_dQuat_local;
        transform_grads_R_to_Quat(q, dL_dR_acc, dL_dQuat_local);
        atomicAdd(&(dL_dQuats[idx].x), dL_dQuat_local.x);
        atomicAdd(&(dL_dQuats[idx].y), dL_dQuat_local.y);
        atomicAdd(&(dL_dQuats[idx].z), dL_dQuat_local.z);
        atomicAdd(&(dL_dQuats[idx].w), dL_dQuat_local.w);
    }
}
