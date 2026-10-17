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

#include "forward.cuh"
#include "backward.cuh"

// =========================================================================
// Forward Interface (PyTorch Wrapper)
// =========================================================================
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
>
RasterizeGaussiansForwardCUDA(
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
)
{
    const int n_pixels = pixel_positions_packed.size(0);
    const int n_gauss  = gauss_means_intensities.size(0);

    auto float_opts = gauss_means_intensities.options().dtype(torch::kFloat32);

    // 主输出
    torch::Tensor pixel_colors = torch::zeros({n_pixels}, float_opts);
    torch::Tensor pixel_depths = torch::zeros({n_pixels}, float_opts);

    // 中间状态 (反向传播需要)
    torch::Tensor pixel_accum = torch::zeros({n_pixels}, float_opts);
    torch::Tensor pixel_T = torch::zeros({n_pixels}, float_opts);
    torch::Tensor pixel_contrib = torch::zeros({n_pixels}, float_opts);

    // 几何中间变量 (Cache)
    torch::Tensor gauss_Rs = torch::empty({n_gauss, 9}, float_opts);
    torch::Tensor gauss_sigma_invs = torch::empty({n_gauss, 9}, float_opts);

    // 是否填充 gauss_intermediates：仅当传入张量非空且 shape 为 (n_gauss, 3) 时写入
    const bool fill_gauss_intermediates = gauss_intermediates.defined()
        && gauss_intermediates.numel() == n_gauss * 3;

    RasterizeGaussiansForwardCUDA_Impl(
        n_pixels, n_gauss,
        fixed_bg_color, compositing_mode,
        reinterpret_cast<const float4*>(gauss_means_intensities.data_ptr<float>()),
        reinterpret_cast<const float4*>(gauss_scales_opacities.data_ptr<float>()),
        reinterpret_cast<const float4*>(gauss_quats.data_ptr<float>()),
        reinterpret_cast<const float4*>(pixel_positions_packed.data_ptr<float>()),
        probe_origin.data_ptr<float>(),
        probe_vec_elevational.data_ptr<float>(),
        pixel_gauss_offsets.data_ptr<int>(),
        pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_indices.data_ptr<int>(),
        gauss_Rs.data_ptr<float>(),
        gauss_sigma_invs.data_ptr<float>(),
        pixel_accum.data_ptr<float>(),
        pixel_T.data_ptr<float>(),
        pixel_contrib.data_ptr<float>(),
        pixel_depths.data_ptr<float>(),
        pixel_colors.data_ptr<float>(),
        fill_gauss_intermediates && gauss_intermediates.numel() > 0
            ? gauss_intermediates.data_ptr<float>() : nullptr,
        fill_gauss_intermediates,
        at::cuda::getCurrentCUDAStream()
    );

    return std::make_tuple(
        pixel_colors,
        pixel_depths,
        pixel_accum,
        pixel_T,
        pixel_contrib,
        gauss_Rs,
        gauss_sigma_invs
    );
}

// =========================================================================
// Backward Interface (PyTorch Wrapper)
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
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
)
{
    const int n_pixels = pixel_positions_packed.size(0);
    const int n_gauss = gauss_means_intensities.size(0);
    auto float_opts = gauss_means_intensities.options().dtype(torch::kFloat32);

    torch::Tensor dL_dGaussMeansIntensities = torch::zeros({n_gauss, 4}, float_opts);
    torch::Tensor dL_dGaussQuats = torch::zeros({n_gauss, 4}, float_opts);
    torch::Tensor dL_dGaussScalesOpacities = torch::zeros({n_gauss, 4}, float_opts);

    torch::Tensor dL_dGaussSigmaInvs = torch::zeros({n_gauss, 9}, float_opts);
    torch::Tensor dL_dPixelAccum = torch::zeros({n_pixels}, float_opts);
    torch::Tensor dL_dPixelT = torch::zeros({n_pixels}, float_opts);

    RasterizeGaussiansBackwardCUDA_Impl(
        n_pixels, n_gauss,
        fixed_bg_color, compositing_mode,
        reinterpret_cast<const float4*>(gauss_means_intensities.data_ptr<float>()),
        reinterpret_cast<const float4*>(gauss_scales_opacities.data_ptr<float>()),
        reinterpret_cast<const float4*>(gauss_quats.data_ptr<float>()),
        reinterpret_cast<const float4*>(pixel_positions_packed.data_ptr<float>()),
        probe_origin.data_ptr<float>(),
        probe_vec_elevational.data_ptr<float>(),
        pixel_gauss_offsets.data_ptr<int>(),
        pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_indices.data_ptr<int>(),
        gauss_Rs.data_ptr<float>(),
        gauss_sigma_invs.data_ptr<float>(),
        pixel_accum.data_ptr<float>(),
        pixel_T.data_ptr<float>(),
        dL_dPixelColors.data_ptr<float>(),
        dL_dPixelDepths.data_ptr<float>(),
        dL_dGaussSigmaInvs.data_ptr<float>(),
        dL_dPixelAccum.data_ptr<float>(),
        dL_dPixelT.data_ptr<float>(),
        reinterpret_cast<float4*>(dL_dGaussMeansIntensities.data_ptr<float>()),
        reinterpret_cast<float4*>(dL_dGaussQuats.data_ptr<float>()),
        reinterpret_cast<float4*>(dL_dGaussScalesOpacities.data_ptr<float>()),
        at::cuda::getCurrentCUDAStream()
    );

    return std::make_tuple(dL_dGaussMeansIntensities, dL_dGaussQuats, dL_dGaussScalesOpacities);
}
