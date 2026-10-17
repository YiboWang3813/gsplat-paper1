#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#include "math_utils.cuh"

#define SLICE_TILE_SIZE 16

__device__ __forceinline__ float compute_projected_radius(
    const float3 scale,
    const float* R,
    const float3 dir
) {
    float3 axis0 = make_float3(R[0], R[3], R[6]);
    float3 axis1 = make_float3(R[1], R[4], R[7]);
    float3 axis2 = make_float3(R[2], R[5], R[8]);

    float p0 = axis0.x * dir.x + axis0.y * dir.y + axis0.z * dir.z;
    float p1 = axis1.x * dir.x + axis1.y * dir.y + axis1.z * dir.z;
    float p2 = axis2.x * dir.x + axis2.y * dir.y + axis2.z * dir.z;

    return fabsf(scale.x * p0) + fabsf(scale.y * p1) + fabsf(scale.z * p2);
}

struct IntersectionResult {
    bool hit;
    float depth;
};

__device__ __forceinline__ IntersectionResult check_intersection_optimized(
    const float3 p,
    const float3 m,
    const float3 proj_radii,
    const float3 axis_lateral,
    const float3 axis_axial
) {
    float3 v = m - p;

    float d_ax = v.x * axis_axial.x + v.y * axis_axial.y + v.z * axis_axial.z;
    if (fabsf(d_ax) > proj_radii.y) {
        return {false, 0.0f};
    }

    float d_lat = v.x * axis_lateral.x + v.y * axis_lateral.y + v.z * axis_lateral.z;
    if (fabsf(d_lat) > proj_radii.x) {
        return {false, 0.0f};
    }

    return {true, d_ax};
}

__global__ void kernel_precompute_gauss_projected_radii(
    const int n_gauss,
    const float k_sigma,
    const float4* __restrict__ quats,
    const float4* __restrict__ scales_packed,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ele,
    const float* __restrict__ ptr_vec_ax,
    float3* __restrict__ projected_radii
);

__global__ void kernel_project_gauss_and_compute_tile_rects(
    const int n_gauss,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii,
    const int height,
    const int width,
    const float near_mm,
    const float delta_lat,
    const float delta_ax,
    const float slab_half_thickness_mm,
    const float lateral_col_offset,
    const float* __restrict__ ptr_origin,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ele,
    const float* __restrict__ ptr_vec_ax,
    int* num_tiles,
    int4* rects,
    float* depths
);

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

__global__ void kernel_identify_tile_ranges(
    const int num_instances,
    const uint64_t* sorted_keys,
    int* tile_ranges_start,
    int* tile_ranges_end
);

__global__ void kernel_count_pixel_candidates(
    const int n_pixels,
    const int width,
    const int height,
    const float4* __restrict__ p_pos_packed,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ax,
    int* counts
);

__global__ void kernel_fill_pixel_candidate_indices(
    const int n_pixels,
    const int width,
    const int height,
    const float4* __restrict__ p_pos_packed,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ax,
    const int* __restrict__ offsets,
    int* indices
);

// --- Batched (B slices, shared gaussians): geometry varies per batch -----------------

__global__ void kernel_precompute_gauss_projected_radii_batched(
    const int n_gauss,
    const int batch_size,
    const float k_sigma,
    const float4* __restrict__ quats,
    const float4* __restrict__ scales_packed,
    const float* __restrict__ vec_lat_batched,
    const float* __restrict__ vec_ele_batched,
    const float* __restrict__ vec_ax_batched,
    const uint8_t* __restrict__ active_gauss_mask_batched,
    float3* __restrict__ projected_radii_batched
);

__global__ void kernel_project_gauss_and_compute_tile_rects_batched(
    const int n_gauss,
    const int batch_size,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii_batched,
    const int height,
    const int width,
    const float near_mm,
    const float delta_lat,
    const float delta_ax,
    const float slab_half_thickness_mm,
    const float lateral_col_offset,
    const float* __restrict__ probe_org_batched,
    const float* __restrict__ vec_lat_batched,
    const float* __restrict__ vec_ele_batched,
    const float* __restrict__ vec_ax_batched,
    const uint8_t* __restrict__ active_gauss_mask_batched,
    int* __restrict__ tiles_touched_count_batched,
    int4* __restrict__ gauss_tile_rects_batched,
    float* __restrict__ gauss_axial_pix_batched
);

__global__ void kernel_generate_tile_depth_keys_batched(
    const int n_gauss,
    const int batch_size,
    const int grid_width_tiles,
    const int n_tiles,
    const bool sort_by_depth,
    const int4* __restrict__ rects_batched,
    const float* __restrict__ depths_batched,
    const int* __restrict__ gauss_instance_offsets_batched,
    uint64_t* __restrict__ keys_out,
    int* __restrict__ values_out
);

__global__ void kernel_count_pixel_candidates_batched(
    const int n_gauss,
    const int pixels_per_image,
    const int width,
    const int height,
    const int batch_size,
    const int n_tiles,
    const float4* __restrict__ p_pos_packed_batched,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii_batched,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    const float* __restrict__ vec_lat_batched,
    const float* __restrict__ vec_ax_batched,
    int* __restrict__ counts
);

__global__ void kernel_fill_pixel_candidate_indices_batched(
    const int n_gauss,
    const int pixels_per_image,
    const int width,
    const int height,
    const int batch_size,
    const int n_tiles,
    const float4* __restrict__ p_pos_packed_batched,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii_batched,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    const float* __restrict__ vec_lat_batched,
    const float* __restrict__ vec_ax_batched,
    const int* __restrict__ offsets,
    int* __restrict__ indices
);
