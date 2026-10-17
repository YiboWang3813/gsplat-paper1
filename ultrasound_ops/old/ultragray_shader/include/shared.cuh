#pragma once

// 辅助：高斯几何权重计算
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

// =========================================================
// Helper: Ray-Gaussian Integral (Attenuation Physics)
// 计算射线穿过高斯的能量衰减积分 (Eq. 7)
// =========================================================
__device__ __forceinline__ float compute_ray_gaussian_integral(
    const float3& ray_origin,    // World Space
    const float3& ray_dir,       // World Space (Normalized)
    const float3& mu,            // Gaussian Mean
    const float3& s,             // Gaussian Scale
    const float* R               // Gaussian Rotation Matrix (3x3)
) {
    // 1. Transform Ray to Canonical Gaussian Space (Isotropic, Identity Cov)
    // o_local = R^T * (ray_origin - mu)
    // d_local = R^T * ray_dir
    
    float3 diff = ray_origin - mu;
    
    // MatMul R^T * diff
    float3 o_local;
    o_local.x = R[0]*diff.x + R[3]*diff.y + R[6]*diff.z;
    o_local.y = R[1]*diff.x + R[4]*diff.y + R[7]*diff.z;
    o_local.z = R[2]*diff.x + R[5]*diff.y + R[8]*diff.z;

    // MatMul R^T * dir
    float3 d_local;
    d_local.x = R[0]*ray_dir.x + R[3]*ray_dir.y + R[6]*ray_dir.z;
    d_local.y = R[1]*ray_dir.x + R[4]*ray_dir.y + R[7]*ray_dir.z;
    d_local.z = R[2]*ray_dir.x + R[5]*ray_dir.y + R[8]*ray_dir.z;

    // Apply Inverse Scale
    // o_g = o_local / s
    // d_g = d_local / s
    float3 o_g = make_float3(o_local.x/s.x, o_local.y/s.y, o_local.z/s.z);
    float3 d_g = make_float3(d_local.x/s.x, d_local.y/s.y, d_local.z/s.z);

    // 2. Analytic Integral of Gaussian along a line
    // Integral exp(-0.5 * ||o_g + t*d_g||^2) dt
    // Quadratic form inside exp: ||o_g||^2 + 2t(o_g.d_g) + t^2||d_g||^2
    // Let A = ||d_g||^2, B = o_g.d_g, C = ||o_g||^2
    
    float A = dot_prod(d_g, d_g);
    float B = dot_prod(o_g, d_g);
    float C = dot_prod(o_g, o_g);

    // Integral result over (-inf, inf) is: sqrt(2*pi / A) * exp(-0.5 * (C - B^2/A))
    // C - B^2/A is actually the squared distance from origin to the line in canonical space.
    
    // Numerical stability
    if (A < 1e-6f) return 0.0f;

    float dist_sq = C - (B * B) / A;
    float factor = __expf(-0.5f * dist_sq);
    
    // sqrt(2 * PI) = 2.506628f
    float integral = (2.506628f / sqrtf(A)) * factor;

    return integral;
}