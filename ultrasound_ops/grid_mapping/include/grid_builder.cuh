#pragma once
#include <torch/extension.h>

struct PixelGaussMappingGrid {
    // ---- tile level ----
    torch::Tensor tile_counts;    // (n_tiles,) int32
    torch::Tensor tile_offsets;   // (n_tiles,) int32
    torch::Tensor tile_indices;   // (sum tile_counts,) int32

    // ---- pixel level ----
    torch::Tensor pixel_tile_ids; // (n_pixels,) int32
    torch::Tensor pixel_counts;   // (n_pixels,) int32
    torch::Tensor pixel_offsets;  // (n_pixels,) int32
    torch::Tensor pixel_indices;  // (total_pairs,) int32
};

// 这个 builder 负责：
// 1) tile_counts -> tile_offsets (scan) -> tile_indices
// 2) pixel_counts -> pixel_offsets (scan) -> pixel_indices
PixelGaussMappingGrid BuildPixelGaussMappingGridCUDA(
    const torch::Tensor& grid_min,         // (3,) float32 cuda
    const torch::Tensor& tile_size,        // (3,) float32 cuda
    const torch::Tensor& grid_size,        // (3,) int32  cuda
    const torch::Tensor& pixel_positions,  // (P,3) float32 cuda
    const torch::Tensor& means,            // (G,3) float32 cuda
    const torch::Tensor& sigma_invs        // (G,9) float32 cuda
);
