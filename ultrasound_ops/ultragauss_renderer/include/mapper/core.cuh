#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#define SLICE_TILE_SIZE 16

__device__ __forceinline__ float safe_diag_from_tri(float raw, float beta) {
    return raw * raw + beta;
}

__global__ void kernel_project_gauss_and_compute_tile_rects(
    const int n_gauss,
    const float4* __restrict__ means_packed,
    const float* __restrict__ precision_tril,
    const int height,
    const int width,
    const float near_mm,
    const float delta_lat,
    const float delta_ax,
    const float lateral_col_offset,
    const float chi2,
    const float beta,
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
    const float* __restrict__ precision_tril,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ax,
    const float* __restrict__ ptr_vec_ele,
    const float chi2,
    const float beta,
    int* counts
);

__global__ void kernel_fill_pixel_candidate_indices(
    const int n_pixels,
    const int width,
    const int height,
    const float4* __restrict__ p_pos_packed,
    const float4* __restrict__ means_packed,
    const float* __restrict__ precision_tril,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ax,
    const float* __restrict__ ptr_vec_ele,
    const float chi2,
    const float beta,
    const int* __restrict__ offsets,
    int* indices
);
