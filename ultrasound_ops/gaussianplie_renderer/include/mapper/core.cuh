#pragma once
#include <torch/extension.h>

/// Tile–Gauss CSR (sorted gauss ids + per-tile [start,end) + per-gauss tile AABB in tile indices).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> OursTileGaussMapperCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    const torch::Tensor& us_focus_params,
    int image_height,
    int image_width,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    int tile_size_px,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    bool sort_by_v,
    int shader_weight_mode = 0);

/// Per-pixel CSR (counts, offsets, indices) using pixel-centered buckets (see plan §2.0).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> OursPixelGaussMapperCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    int image_height,
    int image_width,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    int num_pixels_lateral,
    int num_pixels_axial,
    bool sort_by_v,
    int shader_weight_mode = 0);
