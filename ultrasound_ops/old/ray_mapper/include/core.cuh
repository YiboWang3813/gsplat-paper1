#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

// ==========================================
// Math Helpers (Inline for performance)
// ==========================================

__device__ __forceinline__ float dot_prod(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 xyz(const float4& v) {
    return make_float3(v.x, v.y, v.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// 函数声明
__global__ void kernel_project_and_cull(
    const int n_gauss,
    const int image_width,
    const float pixel_spacing_mm, // 物理宽度 / 图像宽度
    const float near_clip_mm,     // 近平面裁剪距离
    const float elev_cutoff_mm,   // 切片厚度阈值
    const float k_sigma,          // 高斯包围盒倍率 (通常为3.0)
    
    // Data Inputs
    const float4* __restrict__ means_world,     // [N] 高斯中心 (World)
    const float4* __restrict__ scales_world,    // [N] 高斯缩放
    const float* __restrict__ elev_perturbations, // [N] 预计算的 Elevation 轴随机扰动

    // Probe Geometry (Ptrs to avoid constant memory limits)
    const float* __restrict__ probe_origin,
    const float* __restrict__ axis_lateral,
    const float* __restrict__ axis_elevational,
    const float* __restrict__ axis_axial,

    // Outputs
    int* num_rays_covered, // [N] 该高斯覆盖了多少条 Ray (用于 Scan)
    int2* ray_minmax,       // [N] {min_ray_idx, max_ray_idx}
    float* depth_probe       // [N] 高斯在探头坐标系下的深度 Z
);

__global__ void kernel_generate_sort_keys(
    const int n_gauss,
    const int* __restrict__ csr_offsets, // Scan 后的偏移量
    const int2* __restrict__ ray_minmax,
    const float* __restrict__ depths,
    
    uint64_t* __restrict__ sort_keys,    // Output Keys: [RayID (32b) | Depth (32b)]
    int* __restrict__ sort_values        // Output Values: Gaussian Index
);

__global__ void kernel_build_csr_structure(
    const int n_total_hits,
    const uint64_t* __restrict__ sorted_keys,
    int* __restrict__ ray_start_indices,
    int* __restrict__ ray_end_indices
);