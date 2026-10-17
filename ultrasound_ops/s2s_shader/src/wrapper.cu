#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <ATen/cuda/CUDAContext.h>

#include "wrapper.cuh"
#include "forward.cuh"
#include "backward.cuh"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansForwardCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_opacities,
    const torch::Tensor& gauss_intensities,
    const torch::Tensor& pixel_positions,
    const torch::Tensor& pixel_gauss_offsets,
    const torch::Tensor& pixel_gauss_counts,
    const torch::Tensor& pixel_gauss_indices
) {
    const int n_pixels = pixel_positions.size(0);
    const int n_gauss = gauss_means.size(0);

    auto opts_f = gauss_means.options().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor gauss_Rs        = torch::empty({n_gauss, 9}, opts_f);
    torch::Tensor gauss_sigma_inv = torch::empty({n_gauss, 9}, opts_f);
    torch::Tensor pixel_colors    = torch::empty({n_pixels},   opts_f);

    auto stream = at::cuda::getCurrentCUDAStream();

    RasterizeGaussiansForwardCUDA_Impl(
        n_pixels, n_gauss,
        gauss_means.data_ptr<float>(),
        gauss_scales.data_ptr<float>(),
        gauss_quats.data_ptr<float>(),
        gauss_opacities.data_ptr<float>(),
        gauss_intensities.data_ptr<float>(),
        pixel_positions.data_ptr<float>(),
        pixel_gauss_offsets.data_ptr<int>(),
        pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_indices.data_ptr<int>(),
        gauss_Rs.data_ptr<float>(),
        gauss_sigma_inv.data_ptr<float>(),
        pixel_colors.data_ptr<float>(),
        stream.stream()
    );

    return { pixel_colors, gauss_Rs, gauss_sigma_inv };
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor
>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_opacities,
    const torch::Tensor& gauss_intensities,
    const torch::Tensor& pixel_positions,
    const torch::Tensor& pixel_gauss_offsets,
    const torch::Tensor& pixel_gauss_counts,
    const torch::Tensor& pixel_gauss_indices,
    const torch::Tensor& gauss_Rs,
    const torch::Tensor& gauss_sigma_invs,
    const torch::Tensor& dL_dPixelColors
) {
    const int n_pixels = pixel_positions.size(0);
    const int n_gauss = gauss_means.size(0);

    auto opts_f = gauss_means.options().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor dL_dGaussMeans       = torch::zeros({n_gauss, 3}, opts_f);
    torch::Tensor dL_dGaussScales      = torch::zeros({n_gauss, 3}, opts_f);
    torch::Tensor dL_dGaussQuats       = torch::zeros({n_gauss, 4}, opts_f);
    torch::Tensor dL_dGaussOpacities   = torch::zeros({n_gauss},    opts_f);
    torch::Tensor dL_dGaussIntensities = torch::zeros({n_gauss},    opts_f);
    torch::Tensor dL_dGaussSigmaInvs   = torch::zeros({n_gauss, 9}, opts_f);

    auto stream = at::cuda::getCurrentCUDAStream();

    RasterizeGaussiansBackwardCUDA_Impl(
        n_pixels, n_gauss,
        gauss_means.data_ptr<float>(),
        gauss_scales.data_ptr<float>(),
        gauss_quats.data_ptr<float>(),
        gauss_opacities.data_ptr<float>(),
        gauss_intensities.data_ptr<float>(),
        pixel_positions.data_ptr<float>(),
        pixel_gauss_offsets.data_ptr<int>(),
        pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_indices.data_ptr<int>(),
        gauss_Rs.data_ptr<float>(),
        gauss_sigma_invs.data_ptr<float>(),
        dL_dPixelColors.data_ptr<float>(),
        dL_dGaussMeans.data_ptr<float>(),
        dL_dGaussScales.data_ptr<float>(),
        dL_dGaussQuats.data_ptr<float>(),
        dL_dGaussOpacities.data_ptr<float>(),
        dL_dGaussIntensities.data_ptr<float>(),
        dL_dGaussSigmaInvs.data_ptr<float>(),
        stream.stream()
    );

    return {
        dL_dGaussMeans,
        dL_dGaussScales,
        dL_dGaussQuats,
        dL_dGaussOpacities,
        dL_dGaussIntensities
    };
}