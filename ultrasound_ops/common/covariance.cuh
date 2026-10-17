#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include "math_utils.cuh"

// =========================================================
// Quaternion & Rotation (从 Covariance 中抽离)
// =========================================================

// Quat (w,x,y,z) -> RotMat (3x3 row-major)
__device__ __forceinline__ void compute_rot_from_quat(
    const float4 q, 
    float* R // Output 9 floats
) {
    float w = q.x; 
    float x = q.y; 
    float y = q.z; 
    float z = q.w;

    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;

    R[0] = 1.f - 2.f*(yy + zz);
    R[1] = 2.f*(xy - wz);
    R[2] = 2.f*(xz + wy);

    R[3] = 2.f*(xy + wz);
    R[4] = 1.f - 2.f*(xx + zz);
    R[5] = 2.f*(yz - wx);

    R[6] = 2.f*(xz - wy);
    R[7] = 2.f*(yz + wx);
    R[8] = 1.f - 2.f*(xx + yy);
}

// Scale + RotMat -> SigmaInv (3x3 row-major)
__device__ __forceinline__ void compute_cov3d_from_scale_rot(
    const float3 s, 
    const float* R, 
    float* SigmaInv // Output 9 floats
) {
    // 保护：防止 scale 变为 0
    constexpr float eps = 1e-6f;
    float idx = 1.0f / (fmaxf(s.x, eps) * fmaxf(s.x, eps));
    float idy = 1.0f / (fmaxf(s.y, eps) * fmaxf(s.y, eps));
    float idz = 1.0f / (fmaxf(s.z, eps) * fmaxf(s.z, eps));

    // M = R * D
    float M[9];
    M[0] = R[0] * idx; M[1] = R[1] * idy; M[2] = R[2] * idz;
    M[3] = R[3] * idx; M[4] = R[4] * idy; M[5] = R[5] * idz;
    M[6] = R[6] * idx; M[7] = R[7] * idy; M[8] = R[8] * idz;

    // S = M * R^T (直接算全矩阵)
    SigmaInv[0] = M[0]*R[0] + M[1]*R[1] + M[2]*R[2];
    SigmaInv[1] = M[0]*R[3] + M[1]*R[4] + M[2]*R[5];
    SigmaInv[2] = M[0]*R[6] + M[1]*R[7] + M[2]*R[8];

    SigmaInv[3] = M[3]*R[0] + M[4]*R[1] + M[5]*R[2]; 
    SigmaInv[4] = M[3]*R[3] + M[4]*R[4] + M[5]*R[5];
    SigmaInv[5] = M[3]*R[6] + M[4]*R[7] + M[5]*R[8];

    SigmaInv[6] = M[6]*R[0] + M[7]*R[1] + M[8]*R[2]; 
    SigmaInv[7] = M[6]*R[3] + M[7]*R[4] + M[8]*R[5]; 
    SigmaInv[8] = M[6]*R[6] + M[7]*R[7] + M[8]*R[8];
}

// ------------------------------------------------------------------
// 转换器 A: 几何梯度转换 (SigmaInv -> R, Scale)
// 这是所有 3DGS 变体都会用到的核心
// ------------------------------------------------------------------
__device__ __forceinline__ void transform_grads_sigma_to_R_S(
    // Inputs
    const float3 s,
    const float* R,
    const float* dL_dSigmaInv, // 来自 Rasterizer 的几何梯度
    // Outputs (Accumulate into these)
    float* dL_dR_accumulator,  // 累加到 R 的梯度
    float3& dL_dS_accumulator  // 累加到 Scale 的梯度
) {
    constexpr float eps = 1e-6f;
    float idx = 1.0f / (fmaxf(s.x, eps) * fmaxf(s.x, eps));
    float idy = 1.0f / (fmaxf(s.y, eps) * fmaxf(s.y, eps));
    float idz = 1.0f / (fmaxf(s.z, eps) * fmaxf(s.z, eps));

    // M = dL_dSig * R
    float M[9];
    mat3_mul(dL_dSigmaInv, R, M);

    // 1. Contribution to Scale
    // dL/dS = (dL/dSigma) * (dSigma/dS)
    // dL/d(D_ii) = col(R, i) . col(M, i)
    float dL_dDx = R[0]*M[0] + R[3]*M[3] + R[6]*M[6];
    float dL_dDy = R[1]*M[1] + R[4]*M[4] + R[7]*M[7];
    float dL_dDz = R[2]*M[2] + R[5]*M[5] + R[8]*M[8];
    
    // 累加 (+=)
    // dD/ds = -2 / s^3
    dL_dS_accumulator.x += dL_dDx * (-2.0f * idx / fmaxf(s.x, eps));
    dL_dS_accumulator.y += dL_dDy * (-2.0f * idy / fmaxf(s.y, eps));
    dL_dS_accumulator.z += dL_dDz * (-2.0f * idz / fmaxf(s.z, eps));

    // 2. Contribution to Rotation
    // dL/dR = 2 * M * D (assuming dL_dSigma is symmetrized)
    // 累加 (+=)
    dL_dR_accumulator[0] += 2.0f * M[0] * idx;
    dL_dR_accumulator[1] += 2.0f * M[1] * idy;
    dL_dR_accumulator[2] += 2.0f * M[2] * idz;
    
    dL_dR_accumulator[3] += 2.0f * M[3] * idx;
    dL_dR_accumulator[4] += 2.0f * M[4] * idy;
    dL_dR_accumulator[5] += 2.0f * M[5] * idz;
    
    dL_dR_accumulator[6] += 2.0f * M[6] * idx;
    dL_dR_accumulator[7] += 2.0f * M[7] * idy;
    dL_dR_accumulator[8] += 2.0f * M[8] * idz;
}

// ------------------------------------------------------------------
// 转换器 B: 旋转矩阵梯度转四元数梯度 (R -> Quat)
// [实现细节] 提取自 kernel_cov3d_backward
// ------------------------------------------------------------------
__device__ __forceinline__ void transform_grads_R_to_Quat(
    const float4 q,
    const float* dL_dR_total, // 包含了所有来源的总梯度
    float4& dL_dQuat_out
) {
    float w = q.x; 
    float x = q.y; 
    float y = q.z; 
    float z = q.w;

    const float* dL_dR = dL_dR_total;

    // Chain Rule: dL/dq = sum( (dL/dR)_ij * (dR_ij / dq) )
    // 矩阵 R 的元素索引：
    // 0 1 2
    // 3 4 5
    // 6 7 8

    // 1. dL/dw
    // dR/dw terms:
    // R[1]: -2z, R[2]: 2y
    // R[3]: 2z,  R[5]: -2x
    // R[6]: -2y, R[7]: 2x
    float g_w = 
        dL_dR[1] * (-2.f*z) + dL_dR[2] * (2.f*y) +
        dL_dR[3] * (2.f*z)  + dL_dR[5] * (-2.f*x) +
        dL_dR[6] * (-2.f*y) + dL_dR[7] * (2.f*x);

    // 2. dL/dx
    // dR/dx terms:
    // R[1]: 2y,  R[2]: 2z
    // R[4]: -4x, R[5]: -2w
    // R[7]: 2w,  R[8]: -4x
    float g_x = 
        dL_dR[1] * (2.f*y)  + dL_dR[2] * (2.f*z) +
        dL_dR[3] * (2.f*y)  + dL_dR[4] * (-4.f*x) + dL_dR[5] * (-2.f*w) +
        dL_dR[6] * (2.f*z)  + dL_dR[7] * (2.f*w)  + dL_dR[8] * (-4.f*x);

    // 3. dL/dy
    // dR/dy terms:
    // R[0]: -4y, R[1]: 2x, R[2]: 2w
    // R[3]: 2x,  R[5]: 2z
    // R[6]: -2w, R[7]: 2z, R[8]: -4y
    float g_y = 
        dL_dR[0] * (-4.f*y) + dL_dR[1] * (2.f*x)  + dL_dR[2] * (2.f*w) +
        dL_dR[3] * (2.f*x)  + dL_dR[5] * (2.f*z) +
        dL_dR[6] * (-2.f*w) + dL_dR[7] * (2.f*z)  + dL_dR[8] * (-4.f*y);

    // 4. dL/dz
    // dR/dz terms:
    // R[0]: -4z, R[1]: -2w, R[2]: 2x
    // R[3]: 2w,  R[4]: -4z, R[5]: 2y
    // R[6]: 2x,  R[7]: 2y
    float g_z = 
        dL_dR[0] * (-4.f*z) + dL_dR[1] * (-2.f*w) + dL_dR[2] * (2.f*x) +
        dL_dR[3] * (2.f*w)  + dL_dR[4] * (-4.f*z) + dL_dR[5] * (2.f*y) +
        dL_dR[6] * (2.f*x)  + dL_dR[7] * (2.f*y);

    // Write Output
    dL_dQuat_out = make_float4(g_w, g_x, g_y, g_z);
}
