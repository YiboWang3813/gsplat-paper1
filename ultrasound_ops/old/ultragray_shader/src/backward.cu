#include "covariance.cuh"
#include "shared.cuh"
#include "load_utils.cuh"
#include "math_utils.cuh"

// =========================================================
// Gradient Helpers: Manual Chain Rule
// =========================================================

// sigmoid 导数: sigma * (1 - sigma)
__device__ __forceinline__ float sigmoid_prime(float sigma_val) {
    return sigma_val * (1.0f - sigma_val);
}

// ---------------------------------------------------------
// Backward for: compute_ray_gaussian_integral
// ---------------------------------------------------------
__device__ __forceinline__ void compute_integral_vjp(
    // Primal Inputs (Forward 时的输入)
    const float3& ray_o, const float3& ray_d,
    const float3& mu, const float3& s, const float* R,
    // Primal Output & Gradients
    const float integral,     // Forward 算出的积分值
    const float dL_dIntegral, // 来自上层的梯度
    // Output Gradients (Accumulate)
    float3& dL_dMean, float3& dL_dScale, float* dL_dR_buf
) {
    // --- 1. Recompute Intermediates (必须重算一遍中间量) ---
    float3 diff = ray_o - mu;
    
    // R^T * diff
    float3 o_local;
    o_local.x = R[0]*diff.x + R[3]*diff.y + R[6]*diff.z;
    o_local.y = R[1]*diff.x + R[4]*diff.y + R[7]*diff.z;
    o_local.z = R[2]*diff.x + R[5]*diff.y + R[8]*diff.z;

    // R^T * dir
    float3 d_local;
    d_local.x = R[0]*ray_d.x + R[3]*ray_d.y + R[6]*ray_d.z;
    d_local.y = R[1]*ray_d.x + R[4]*ray_d.y + R[7]*ray_d.z;
    d_local.z = R[2]*ray_d.x + R[5]*ray_d.y + R[8]*ray_d.z;

    float3 inv_s = make_float3(1.0f/s.x, 1.0f/s.y, 1.0f/s.z);
    float3 o_g = make_float3(o_local.x*inv_s.x, o_local.y*inv_s.y, o_local.z*inv_s.z);
    float3 d_g = make_float3(d_local.x*inv_s.x, d_local.y*inv_s.y, d_local.z*inv_s.z);

    float A = dot_prod(d_g, d_g);
    float B = dot_prod(o_g, d_g);
    // float C = dot_prod(o_g, o_g); // Backward 不需要 C 本身，只需要导数

    // 保护：如果 A 太小，Forward 返回 0，梯度也为 0
    if (A < 1e-6f) return;

    // Integral = (sqrt(2pi) * A^-0.5) * exp(-0.5 * dist_sq)
    // Let Factor1 = sqrt(2pi) * A^-0.5
    // Let Factor2 = exp(-0.5 * dist_sq)
    // dist_sq = C - B^2/A
    
    // --- 2. Backprop Chain Rule ---
    
    // dL / d(Factor2)
    // Integral = Factor1 * Factor2 => dIntegral/dFactor2 = Factor1
    float factor1 = 2.506628f * rsqrtf(A);
    float dL_dFactor2 = dL_dIntegral * factor1;

    // dL / d(Factor1)
    // Factor1 = K * A^-0.5 => dFactor1/dA = -0.5 * K * A^-1.5 = -0.5 * Factor1 / A
    // Integral = Factor1 * Factor2 => dIntegral/dFactor1 = Factor2
    // dL/dA_from_factor1 = dL_dIntegral * Factor2 * (-0.5f * factor1 / A)
    //                    = dL_dIntegral * Integral * (-0.5f / A)
    float dL_dA_part1 = dL_dIntegral * integral * (-0.5f / A);

    // dL / d(dist_sq)
    // Factor2 = exp(-0.5 * dist_sq) => dFactor2 / d(dist_sq) = Factor2 * (-0.5)
    float dL_dDistSq = dL_dFactor2 * (integral / factor1) * (-0.5f); // integral/factor1 is Factor2

    // dist_sq = C - B^2 / A
    float dL_dC = dL_dDistSq; // 1.0
    float dL_dB = dL_dDistSq * (-2.0f * B / A);
    float dL_dA_part2 = dL_dDistSq * (B * B / (A * A));
    
    float dL_dA = dL_dA_part1 + dL_dA_part2;

    // Gradients w.r.t o_g and d_g
    // A = d_g . d_g => dA/d_dg = 2 * d_g
    // B = o_g . d_g => dB/d_dg = o_g, dB/d_og = d_g
    // C = o_g . o_g => dC/d_og = 2 * o_g
    
    float3 dL_dog = (d_g * dL_dB) + (o_g * (2.0f * dL_dC));
    float3 dL_ddg = (d_g * (2.0f * dL_dA)) + (o_g * dL_dB);

    // Gradients w.r.t o_local, d_local, s
    // o_g = o_local / s => d/do_local = 1/s, d/ds = -o_local/s^2 = -o_g/s
    float3 dL_dolocal = make_float3(dL_dog.x * inv_s.x, dL_dog.y * inv_s.y, dL_dog.z * inv_s.z);
    float3 dL_ddlocal = make_float3(dL_ddg.x * inv_s.x, dL_ddg.y * inv_s.y, dL_ddg.z * inv_s.z);

    dL_dScale.x += dL_dog.x * (-o_g.x * inv_s.x) + dL_ddg.x * (-d_g.x * inv_s.x);
    dL_dScale.y += dL_dog.y * (-o_g.y * inv_s.y) + dL_ddg.y * (-d_g.y * inv_s.y);
    dL_dScale.z += dL_dog.z * (-o_g.z * inv_s.z) + dL_ddg.z * (-d_g.z * inv_s.z);

    // Gradients w.r.t mu and R
    // o_local = R^T * (o - mu) => d/dmu = -R * d/dolocal
    // d_local = R^T * d        => d/dR...
    
    // 1. dL/dMu
    // dL_dMu = - R * dL_dolocal
    dL_dMean.x += -(R[0]*dL_dolocal.x + R[1]*dL_dolocal.y + R[2]*dL_dolocal.z);
    dL_dMean.y += -(R[3]*dL_dolocal.x + R[4]*dL_dolocal.y + R[5]*dL_dolocal.z);
    dL_dMean.z += -(R[6]*dL_dolocal.x + R[7]*dL_dolocal.y + R[8]*dL_dolocal.z);

    // 2. dL/dR
    // o_local_i = sum_j R_ji * diff_j
    // d(o_local_i) / d(R_ji) = diff_j
    // dL / dR_ji = dL/do_local_i * diff_j + dL/dd_local_i * ray_d_j
    
    // R[0] is R_00 (row 0, col 0). In our flatten: R0, R1, R2 is row 0.
    // But the formula used R[0]*diff.x + R[3]*diff.y ... this is M^T * vec.
    // So R[0] is M_00, R[3] is M_10.
    
    // Let's unroll carefully based on forward:
    // o_local.x = R[0]*diff.x + R[3]*diff.y + R[6]*diff.z
    // dL/dR[0] += dL_dolocal.x * diff.x
    // dL/dR[3] += dL_dolocal.x * diff.y
    
    #pragma unroll
    for (int r = 0; r < 3; r++) { // rows of R
        float diff_val = (r==0)?diff.x : (r==1)?diff.y : diff.z;
        float dir_val  = (r==0)?ray_d.x : (r==1)?ray_d.y : ray_d.z;
        
        dL_dR_buf[r*3 + 0] += dL_dolocal.x * diff_val + dL_ddlocal.x * dir_val;
        dL_dR_buf[r*3 + 1] += dL_dolocal.y * diff_val + dL_ddlocal.y * dir_val;
        dL_dR_buf[r*3 + 2] += dL_dolocal.z * diff_val + dL_ddlocal.z * dir_val;
    }
}

// ---------------------------------------------------------
// Backward for: compute_gaussian_weight (Mahalanobis)
// ---------------------------------------------------------
__device__ __forceinline__ void compute_weight_vjp(
    const float3& p, const float3& mu, const float* SigmaInv,
    const float w,        // Forward weight
    const float dL_dw,    // Incoming Gradient
    // Outputs
    float3& dL_dMean, float* dL_dSigmaInv
) {
    // w = exp(-0.5 * Q)
    // dL/dQ = dL/dw * w * (-0.5)
    float dL_dQ = dL_dw * w * (-0.5f);

    float dx = p.x - mu.x;
    float dy = p.y - mu.y;
    float dz = p.z - mu.z;

    // Q = dx*(S0*dx + S1*dy + S2*dz) + ...
    
    // 1. dL / dSigmaInv (Symmetric)
    // dQ / dS_00 = dx*dx
    // dQ / dS_01 = 2*dx*dy (if we consider S_01 and S_10 separate contributions in the formula)
    // The forward formula was written explicitly:
    // (S0*dx + S1*dy + S2*dz) * dx ...
    // Let's differentiate the forward expression directly.
    
    // Temp vector V = SigmaInv * d
    float v0 = SigmaInv[0]*dx + SigmaInv[1]*dy + SigmaInv[2]*dz;
    float v1 = SigmaInv[3]*dx + SigmaInv[4]*dy + SigmaInv[5]*dz;
    float v2 = SigmaInv[6]*dx + SigmaInv[7]*dy + SigmaInv[8]*dz;

    // dQ/dMu (dMu = -dX)
    // dQ/dX = 2 * SigmaInv * X = 2 * V
    // dL/dMu = dL/dQ * dQ/dX * (-1) = dL/dQ * 2 * V * (-1)
    dL_dMean.x += dL_dQ * (-2.0f * v0);
    dL_dMean.y += dL_dQ * (-2.0f * v1);
    dL_dMean.z += dL_dQ * (-2.0f * v2);

    // dL/dSigmaInv
    // dQ / dSigma_ij = x_i * x_j
    atomicAdd(dL_dSigmaInv + 0, dL_dQ * dx * dx);
    atomicAdd(dL_dSigmaInv + 1, dL_dQ * dx * dy);
    atomicAdd(dL_dSigmaInv + 2, dL_dQ * dx * dz);
    
    atomicAdd(dL_dSigmaInv + 3, dL_dQ * dy * dx);
    atomicAdd(dL_dSigmaInv + 4, dL_dQ * dy * dy);
    atomicAdd(dL_dSigmaInv + 5, dL_dQ * dy * dz);
    
    atomicAdd(dL_dSigmaInv + 6, dL_dQ * dz * dx);
    atomicAdd(dL_dSigmaInv + 7, dL_dQ * dz * dy);
    atomicAdd(dL_dSigmaInv + 8, dL_dQ * dz * dz);
}

// =========================================================
// Main Backward Kernel
// =========================================================
__global__ void render_bmode_backward_kernel(
    const int W, const int H,
    
    // Gradients (Outputs to Accumulate)
    float4* __restrict__ dL_dMeans,
    float4* __restrict__ dL_dScales,
    float4* __restrict__ dL_dQuats,
    float* __restrict__ dL_dAttens,
    float* __restrict__ dL_dIntensities,

    // Inputs (Same as Forward)
    const float* __restrict__ dL_dImage, // (W, H) Incoming Loss Gradient
    const int* __restrict__ ray_counts,
    const int* __restrict__ ray_offsets,
    const int* __restrict__ ray_indices,
    const float4* __restrict__ means,
    const float4* __restrict__ scales,
    const float4* __restrict__ quats,
    const float* __restrict__ attenuations,
    const float* __restrict__ intensities,
    const float* __restrict__ pixel_positions,
    const float* __restrict__ ray_origins,
    const float* __restrict__ ray_dirs
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    // Load Incoming Gradient
    float dL_dB = dL_dImage[y * W + x];
    if (abs(dL_dB) < 1e-6f) return; // Skip negligible gradients

    int count = ray_counts[x];
    if (count == 0) return;
    int offset = ray_offsets[x];

    // Load Geometry
    const float* p_ptr = pixel_positions + (x * H + y) * 3;
    float3 p = make_float3(p_ptr[0], p_ptr[1], p_ptr[2]);
    float3 ray_o = make_float3(ray_origins[x*3], ray_origins[x*3+1], ray_origins[x*3+2]);
    float3 ray_d = make_float3(ray_dirs[x*3], ray_dirs[x*3+1], ray_dirs[x*3+2]);
    float t_pixel = dot_prod(p - ray_o, ray_d);

    // =========================================================
    // Phase 1: Re-simulate Forward Pass
    // 目的：计算 gradients 依赖的全局聚合量 (T_total, E_num, E_den, S_total)
    // =========================================================
    
    float T_final = 1.0f;
    float E_num = 0.0f;
    float E_den = 0.0f;
    float S_accum = 0.0f;

    for (int k = 0; k < count; ++k) {
        int idx = ray_indices[offset + k];
        float3 mu = xyz(means[idx]);
        float3 s = xyz(scales[idx]);
        float4 q = quats[idx];
        
        float R[9]; compute_rot_from_quat(q, R);
        float SigmaInv[9]; compute_cov3d_from_scale_rot(s, R, SigmaInv);

        // 1. Attenuation Part
        float t_gauss = dot_prod(mu - ray_o, ray_d);
        if (t_gauss < t_pixel) {
            float psi = compute_ray_gaussian_integral(ray_o, ray_d, mu, s, R);
            float raw_tau = attenuations[idx];
            float tau = 1.0f / (1.0f + __expf(-raw_tau));
            float T_i = tau + (1.0f - tau) * __expf(-psi);
            T_final *= T_i;
        }

        // 2. Echo Part
        float w = compute_gaussian_weight(p, mu, SigmaInv);
        float I = intensities[idx];
        E_num += I * w;
        E_den += w;
        S_accum += w;
    }

    // Compute Final derived values
    float gamma = 1.0f - __expf(-S_accum);
    float epsilon = 1e-6f;
    float E_raw = E_num / (E_den + epsilon); // The weighted average
    float E_val = gamma * E_raw;             // The final Echo term
    
    // B = T * E
    // Gradients propagating back to T and E terms
    float dL_dT = dL_dB * E_val;
    float dL_dE = dL_dB * T_final;

    // Decompose dL_dE into parts
    // E = gamma * E_raw
    float dL_dGamma = dL_dE * E_raw;
    float dL_dE_raw = dL_dE * gamma;

    // gamma = 1 - exp(-S)
    float dL_dS_from_gamma = dL_dGamma * __expf(-S_accum); // dGamma/dS = exp(-S)

    // E_raw = Num / Den
    float dL_dNum = dL_dE_raw * (1.0f / (E_den + epsilon));
    float dL_dDen = dL_dE_raw * (-E_num / ((E_den + epsilon) * (E_den + epsilon)));

    // =========================================================
    // Phase 2: Backpropagate to per-Gaussian parameters
    // =========================================================

    for (int k = 0; k < count; ++k) {
        int idx = ray_indices[offset + k];

        // --- Re-load & Re-compute (Locals) ---
        float3 mu = xyz(means[idx]);
        float3 s = xyz(scales[idx]);
        float4 q = quats[idx];
        float R[9]; compute_rot_from_quat(q, R);
        float SigmaInv[9]; compute_cov3d_from_scale_rot(s, R, SigmaInv);

        // Temp gradients accumulators for this Gaussian
        float3 dL_dMu_acc = make_float3(0,0,0);
        float3 dL_dScale_acc = make_float3(0,0,0);
        float dL_dR_acc[9] = {0}; // Will accumulate rotation grads here

        // -----------------------------------------------------
        // A. Gradient from Attenuation (if applicable)
        // -----------------------------------------------------
        float t_gauss = dot_prod(mu - ray_o, ray_d);
        if (t_gauss < t_pixel) {
            float psi = compute_ray_gaussian_integral(ray_o, ray_d, mu, s, R);
            float raw_tau = attenuations[idx];
            float sig_tau = 1.0f / (1.0f + __expf(-raw_tau));
            float exp_psi = __expf(-psi);
            float T_i = sig_tau + (1.0f - sig_tau) * exp_psi;

            // dL/dT_i = dL/dT * (T_final / T_i)
            // Note: If T_i is very small, this division is unstable.
            // But T_final includes T_i, so it cancels out algebraically.
            // T_final / T_i = Product_{j!=i} T_j
            // We compute via division for simplicity, assuming T_i > epsilon.
            float dL_dTi = 0.0f;
            if (T_i > 1e-6f) {
                dL_dTi = dL_dT * (T_final / T_i);
            }

            // 1. dL/d(AttenParam) -> raw_tau
            // T_i = tau + (1-tau)A
            // dT_i/dtau = 1 - A = 1 - exp(-psi)
            float dTi_dTau = 1.0f - exp_psi;
            float dTau_dRaw = sigmoid_prime(sig_tau);
            atomicAdd(dL_dAttens + idx, dL_dTi * dTi_dTau * dTau_dRaw);

            // 2. dL/dPsi (Geometry)
            // T_i = ... + (1-tau)exp(-psi)
            // dT_i/dPsi = (1-tau) * exp(-psi) * (-1)
            float dL_dPsi = dL_dTi * (1.0f - sig_tau) * exp_psi * (-1.0f);

            // 3. dPsi / dGeom -> helper
            compute_integral_vjp(
                ray_o, ray_d, mu, s, R, 
                psi, dL_dPsi, 
                dL_dMu_acc, dL_dScale_acc, dL_dR_acc
            );
        }

        // -----------------------------------------------------
        // B. Gradient from Echo
        // -----------------------------------------------------
        float w = compute_gaussian_weight(p, mu, SigmaInv);
        float I = intensities[idx];

        // 1. dL / dIntensity
        // Affects E_num
        atomicAdd(dL_dIntensities + idx, dL_dNum * w);

        // 2. dL / dWeight
        // Weight affects Num, Den, and Gamma(via S)
        float dL_dw = (dL_dNum * I) + dL_dDen + dL_dS_from_gamma;

        // 3. dWeight / dGeom
        // Need gradients for Mu and SigmaInv
        float dL_dSigmaInv[9] = {0}; // Local buffer
        
        // This helper accumulates into dL_dMu_acc and dL_dSigmaInv
        compute_weight_vjp(p, mu, SigmaInv, w, dL_dw, dL_dMu_acc, dL_dSigmaInv);

        // Convert dL_dSigmaInv -> dL_dR, dL_dS
        transform_grads_sigma_to_R_S(s, R, dL_dSigmaInv, dL_dR_acc, dL_dScale_acc);

        // -----------------------------------------------------
        // C. Write Accumulators to Global Memory
        // -----------------------------------------------------
        
        // 1. Means
        atomicAdd(&(dL_dMeans[idx].x), dL_dMu_acc.x);
        atomicAdd(&(dL_dMeans[idx].y), dL_dMu_acc.y);
        atomicAdd(&(dL_dMeans[idx].z), dL_dMu_acc.z);

        // 2. Scales
        atomicAdd(&(dL_dScales[idx].x), dL_dScale_acc.x);
        atomicAdd(&(dL_dScales[idx].y), dL_dScale_acc.y);
        atomicAdd(&(dL_dScales[idx].z), dL_dScale_acc.z);

        // 3. Quaternions (From R grads)
        float4 dL_dQuat_local;
        transform_grads_R_to_Quat(q, dL_dR_acc, dL_dQuat_local);
        atomicAdd(&(dL_dQuats[idx].x), dL_dQuat_local.x);
        atomicAdd(&(dL_dQuats[idx].y), dL_dQuat_local.y);
        atomicAdd(&(dL_dQuats[idx].z), dL_dQuat_local.z);
        atomicAdd(&(dL_dQuats[idx].w), dL_dQuat_local.w);
    }
}