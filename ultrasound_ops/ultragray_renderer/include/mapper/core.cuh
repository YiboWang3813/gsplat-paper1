#pragma once
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ float dot_prod(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 xyz(const float4& v) {
    return make_float3(v.x, v.y, v.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__global__ void kernel_project_and_cull(
    int n_gauss,
    int image_width,
    float pixel_spacing_mm,
    float lateral_col_offset,
    float near_clip_mm,
    float elev_cutoff_mm,
    float k_sigma,
    const float4* means_world,
    const float4* scales_world,
    const float* probe_origin,
    const float* axis_lateral,
    const float* axis_elevational,
    const float* axis_axial,
    int* num_rays_covered,
    int2* ray_minmax,
    float* depth_probe
);

__global__ void kernel_generate_sort_keys(
    int n_gauss,
    const int* csr_offsets,
    const int2* ray_minmax,
    const float* depths,
    uint64_t* sort_keys,
    int* sort_values
);

__global__ void kernel_build_csr_structure(
    int n_total_hits,
    const uint64_t* sorted_keys,
    int* ray_start_indices,
    int* ray_end_indices
);

__global__ void kernel_decode_sorted_depths(
    int n_total_hits,
    const uint64_t* sorted_keys,
    float* sorted_depths
);
