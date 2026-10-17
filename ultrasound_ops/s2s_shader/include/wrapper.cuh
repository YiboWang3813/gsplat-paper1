#pragma once
#include <torch/extension.h>

// Forward 接口，对应 s2s_shader/__init__.py::_ShadeFunction.forward
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansForwardCUDA(
    const torch::Tensor& gauss_means,         // (G,3)
    const torch::Tensor& gauss_scales,        // (G,3)
    const torch::Tensor& gauss_quats,         // (G,4)
    const torch::Tensor& gauss_opacities,     // (G,)
    const torch::Tensor& gauss_intensities,   // (G,)
    const torch::Tensor& pixel_positions,     // (P,3)
    const torch::Tensor& pixel_gauss_offsets, // (P,)
    const torch::Tensor& pixel_gauss_counts,  // (P,)
    const torch::Tensor& pixel_gauss_indices  // (Total_Interactions,)
);

// Backward 接口，对应 s2s_shader/__init__.py::_ShadeFunction.backward
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor
>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& gauss_means,         // (G,3)
    const torch::Tensor& gauss_scales,        // (G,3)
    const torch::Tensor& gauss_quats,         // (G,4)
    const torch::Tensor& gauss_opacities,     // (G,)
    const torch::Tensor& gauss_intensities,   // (G,)
    const torch::Tensor& pixel_positions,     // (P,3)
    const torch::Tensor& pixel_gauss_offsets, // (P,)
    const torch::Tensor& pixel_gauss_counts,  // (P,)
    const torch::Tensor& pixel_gauss_indices, // (Total_Interactions,)
    const torch::Tensor& gauss_Rs,            // (G,9)
    const torch::Tensor& gauss_sigma_invs,    // (G,9)
    const torch::Tensor& dL_dPixelColors      // (P,)
);