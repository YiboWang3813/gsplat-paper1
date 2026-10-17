#pragma once
#include <torch/extension.h>

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
>
ShaderForwardCUDA(
    const int W,
    const float fixed_bg_color,
    const int compositing_mode,
    const torch::Tensor& gauss_means_intensities,
    const torch::Tensor& gauss_scales_opacities,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& pixel_positions_packed,
    const torch::Tensor& probe_origin,
    const torch::Tensor& probe_vec_elevational,
    const torch::Tensor& pixel_gauss_offsets,
    const torch::Tensor& pixel_gauss_counts,
    const torch::Tensor& pixel_gauss_indices,
    const torch::Tensor& gauss_intermediates
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ShaderBackwardCUDA(
    const int W,
    const float fixed_bg_color,
    const int compositing_mode,
    const torch::Tensor& gauss_means_intensities,
    const torch::Tensor& gauss_scales_opacities,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& pixel_positions_packed,
    const torch::Tensor& probe_origin,
    const torch::Tensor& probe_vec_elevational,
    const torch::Tensor& pixel_gauss_offsets,
    const torch::Tensor& pixel_gauss_counts,
    const torch::Tensor& pixel_gauss_indices,
    const torch::Tensor& gauss_Rs,
    const torch::Tensor& gauss_sigma_invs,
    const torch::Tensor& pixel_accum,
    const torch::Tensor& pixel_T,
    const torch::Tensor& dL_dPixelColors,
    const torch::Tensor& dL_dPixelDepths
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
>
ShaderForwardCUDABatched(
    const int W,
    const int batch_size,
    const int pixels_per_image,
    const float fixed_bg_color,
    const int compositing_mode,
    const torch::Tensor& gauss_means_intensities,
    const torch::Tensor& gauss_scales_opacities,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& pixel_positions_packed,
    const torch::Tensor& probe_origin,
    const torch::Tensor& probe_vec_elevational,
    const torch::Tensor& pixel_gauss_offsets,
    const torch::Tensor& pixel_gauss_counts,
    const torch::Tensor& pixel_gauss_indices,
    const torch::Tensor& gauss_intermediates
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ShaderBackwardCUDABatched(
    const int W,
    const int batch_size,
    const int pixels_per_image,
    const float fixed_bg_color,
    const int compositing_mode,
    const torch::Tensor& gauss_means_intensities,
    const torch::Tensor& gauss_scales_opacities,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& pixel_positions_packed,
    const torch::Tensor& probe_origin,
    const torch::Tensor& probe_vec_elevational,
    const torch::Tensor& pixel_gauss_offsets,
    const torch::Tensor& pixel_gauss_counts,
    const torch::Tensor& pixel_gauss_indices,
    const torch::Tensor& gauss_Rs,
    const torch::Tensor& gauss_sigma_invs,
    const torch::Tensor& pixel_accum,
    const torch::Tensor& pixel_T,
    const torch::Tensor& dL_dPixelColors,
    const torch::Tensor& dL_dPixelDepths
);
