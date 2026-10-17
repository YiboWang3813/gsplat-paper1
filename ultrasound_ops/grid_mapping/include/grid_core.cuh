#pragma once 
#include <cuda_runtime.h>

// 控制空间哈希找邻域的半径
#define ND 1

// 控制一个高斯是否成为像素的候选 
#define E_CAND -6.0f

// ========== 常量 / 小工具 ==========
__device__ __forceinline__
int compute_tile(float x, float grid_min, float tile_size, int grid_size)
{
    int t = (int)floorf((x - grid_min) / tile_size);
    return max(0, min(t, grid_size - 1));
}


// ========== Tile-level API ==========
// tile_counts: (n_tiles,) int32
void count_gauss_per_tile(
    int n_gauss,
    const float* __restrict__ means,     // (n_gauss, 3)
    const float* __restrict__ grid_min,  // (3,)
    const float* __restrict__ tile_size, // (3,)
    const int*   __restrict__ grid_size, // (3,)
    int* __restrict__ tile_counts        // (n_tiles,)
);

// tile_indices: (sum tile_counts,) int32
// tile_offsets: (n_tiles,) int32  (exclusive scan result)
// tile_counters: (n_tiles,) int32 (zero init)
void fill_tile_indices(
    int n_gauss,
    const float* __restrict__ means,
    const float* __restrict__ grid_min,
    const float* __restrict__ tile_size,
    const int*   __restrict__ grid_size,
    const int* __restrict__ tile_offsets,
    int* __restrict__ tile_counters,
    int* __restrict__ tile_indices
);


// pixel_tile_ids: (n_pixels,) int32
// pixel_counts:   (n_pixels,) int32
void count_gauss_per_pixel(
    int n_pixels,
    const float* __restrict__ pixel_positions, // (n_pixels, 3)
    const float* __restrict__ means,           // (n_gauss, 3)
    const float* __restrict__ sigma_invs,      // (n_gauss, 9)
    const float* __restrict__ grid_min,        // (3,)
    const float* __restrict__ tile_size,       // (3,)
    const int*   __restrict__ grid_size,       // (3,)
    const int* __restrict__ tile_counts,       // (n_tiles,)
    const int* __restrict__ tile_offsets,      // (n_tiles,)
    const int* __restrict__ tile_indices,      // (sum tile_counts,)
    int* __restrict__ pixel_tile_ids,          // (n_pixels,)
    int* __restrict__ pixel_counts            // (n_pixels,)
);

// pixel_indices: (total_pairs,) int32
void fill_pixel_indices(
    int n_pixels,
    const float* __restrict__ pixel_positions,
    const float* __restrict__ means,
    const float* __restrict__ sigma_invs,
    const float* __restrict__ grid_min,
    const float* __restrict__ tile_size,
    const int*   __restrict__ grid_size,
    const int* __restrict__ tile_counts,
    const int* __restrict__ tile_offsets,
    const int* __restrict__ tile_indices,
    const int* __restrict__ pixel_offsets,   // (n_pixels,)
    int* __restrict__ pixel_indices          // (total_pairs,)
);
