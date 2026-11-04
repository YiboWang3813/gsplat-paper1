#pragma once
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ShaderForwardCUDA(
    const torch::Tensor& gauss_opacity_intensity,
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_precision_tril,
    const torch::Tensor& pixel_positions,
    const torch::Tensor& probe_origin,
    const torch::Tensor& probe_vec_elevational,
    const torch::Tensor& pixel_gauss_offsets,
    const torch::Tensor& pixel_gauss_counts,
    const torch::Tensor& pixel_gauss_indices,
    const float bg_alpha,
    const float bg_intensity,
    const float chi2_threshold,
    const float diag_beta
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ShaderBackwardCUDA(
    const torch::Tensor& dL_dImage,
    const torch::Tensor& dL_dAlpha,
    const torch::Tensor& dL_dDepth,
    const torch::Tensor& gauss_opacity_intensity,
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_precision_tril,
    const torch::Tensor& pixel_positions,
    const torch::Tensor& probe_origin,
    const torch::Tensor& probe_vec_elevational,
    const torch::Tensor& pixel_gauss_offsets,
    const torch::Tensor& pixel_gauss_counts,
    const torch::Tensor& pixel_gauss_indices,
    const torch::Tensor& sigma_inv_cache,
    const torch::Tensor& pixel_alpha,
    const float bg_alpha,
    const float bg_intensity,
    const float chi2_threshold,
    const float diag_beta
);
