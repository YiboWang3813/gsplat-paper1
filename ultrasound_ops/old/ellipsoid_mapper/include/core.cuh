#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// 引入float3减法运算符重载
#include "math_utils.cuh"

#define SLICE_TILE_SIZE 16

// ============================================================================
// Coordinate / axis convention (CRITICAL)
//
// Probe coordinate system and any "probe-frame" quantities follow:
//   x = lateral, y = axial, z = elevational
//
// Throughout this mapper, we store axis-aligned triplets as float3 in the SAME
// component order:
//   .x = lateral, .y = axial, .z = elevational
//
// Units:
//   - positions: mm (world space inputs are in mm)
//   - projected_radii: mm (already multiplied by k_sigma)
//   - pixel coordinates: pix
// ============================================================================

// -------------------------------------------------------------------------
// [New] 投影半径计算 (Manhattan Bound)
// 计算 3D 椭球在给定方向上的投影“半宽”
// -------------------------------------------------------------------------
__device__ __forceinline__ float compute_projected_radius(
    const float3 scale,
    const float* R, // 3x3
    const float3 dir // Unit vector
) {
    // Proj = sum( |scale_i * (R_col_i dot dir)| )
    // R is row-major. Col 0 is R[0], R[3], R[6]
    
    float dot_0 = R[0]*dir.x + R[1]*dir.y + R[2]*dir.z; // Row 0 dot dir? NO!
    // Wait, R_col_0 = (R[0], R[3], R[6]). 
    // R * e_i is the i-th column.
    // The principal axes of the ellipsoid are the columns of R.
    
    // R stored as row-major in C usually means:
    // [ r00 r01 r02 ]
    // [ r10 r11 r12 ]
    // [ r20 r21 r22 ]
    // If standard 3DGS, R * Scale -> Columns are axes.
    // So Axis 0 = (r00, r10, r20)
    
    float3 axis0 = make_float3(R[0], R[3], R[6]);
    float3 axis1 = make_float3(R[1], R[4], R[7]);
    float3 axis2 = make_float3(R[2], R[5], R[8]);
    
    // Dot products with slice direction
    float p0 = axis0.x * dir.x + axis0.y * dir.y + axis0.z * dir.z;
    float p1 = axis1.x * dir.x + axis1.y * dir.y + axis1.z * dir.z;
    float p2 = axis2.x * dir.x + axis2.y * dir.y + axis2.z * dir.z;
    
    return fabsf(scale.x * p0) + fabsf(scale.y * p1) + fabsf(scale.z * p2);
}

// -------------------------------------------------------------------------
// [Optimized] 极简相交判定
// 直接使用预计算好的半径，不再重复投影
// -------------------------------------------------------------------------
struct IntersectionResult { bool hit; float depth; };

__device__ __forceinline__ 
IntersectionResult check_intersection_optimized(
    const float3 p,          // pixel position
    const float3 m,          // gauss mean
    // proj_radii is in mm and already multiplied by k_sigma.
    // Component order follows probe xyz:
    //   .x = lateral, .y = axial, .z = elevational
    const float3 proj_radii,
    const float3 axis_lateral,
    const float3 axis_axial
) {
    float3 v = m - p; // vector from pixel to mean

    // 1. Axial Check
    float d_ax = v.x*axis_axial.x + v.y*axis_axial.y + v.z*axis_axial.z;
    // 注意：这里 proj_radii 已经是 k_sigma * radius 了
    if (fabsf(d_ax) > proj_radii.y) return {false, 0.0f};

    // 2. Lateral Check
    float d_lat = v.x*axis_lateral.x + v.y*axis_lateral.y + v.z*axis_lateral.z;
    if (fabsf(d_lat) > proj_radii.x) return {false, 0.0f};

    return {true, d_ax};
}

// 函数声明
// 1) Precompute per-gauss projected halfwidths (mm, already *k_sigma)
__global__ void kernel_precompute_gauss_projected_radii(
    const int n_gauss,
    const float k_sigma,
    const float4* __restrict__ quats,
    const float4* __restrict__ scales_packed,
    // Slice Geometry
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ele,
    const float* __restrict__ ptr_vec_ax,
    // Output
    // projected_radii in mm (already *k_sigma), component order:
    //   .x = lateral, .y = axial, .z = elevational
    float3* __restrict__ projected_radii
);

// 2) Slab cull + project to pixel plane + compute tile rects
__global__ void kernel_project_gauss_and_compute_tile_rects(
    const int n_gauss,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii, // Precomputed
    // Geometry
    const int height, const int width,
    const float near_mm, const float delta_lat, const float delta_ax,
    const float slab_half_thickness_mm,
    const float lateral_col_offset,
    const float* __restrict__ ptr_origin,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ele,
    const float* __restrict__ ptr_vec_ax,
    // Output
    int* num_tiles,
    int4* rects,
    float* depths
);

// 3) Expand per-gauss rects into (tile_id, depth_key)->gauss_id instances
__global__ void kernel_generate_tile_depth_keys(
    const int n_gauss,
    const int grid_width_tiles,
    const bool sort_by_depth,
    const int4* rects,
    const float* depths,
    const int* offsets,
    uint64_t* keys_out,
    int* values_out
);

// 3.5) Identify [start,end) ranges for each tile_id in sorted keys
__global__ void kernel_identify_tile_ranges(
    const int num_instances,
    const uint64_t* sorted_keys,
    int* tile_ranges_start,
    int* tile_ranges_end
);

// 4) For each pixel, count candidate gaussians within its tile (lat/ax test)
__global__ void kernel_count_pixel_candidates(
    const int n_pixels, const int width, const int height,
    const float4* __restrict__ p_pos_packed,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii, // Precomputed
    // Selection
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    // Geometry
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ax,
    // Output
    int* counts
);

// 5) For each pixel, fill candidate gauss indices into output array
__global__ void kernel_fill_pixel_candidate_indices(
    const int n_pixels, const int width, const int height,
    const float4* __restrict__ p_pos_packed,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii, // Precomputed
    // Selection
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    // Geometry
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ax,
    // Offset
    const int* __restrict__ offsets,
    // Output
    int* indices
);
