#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "shader/forward.cuh"
#include "shader/backward.cuh"
#include "shader/wrapper_shader.cuh"

namespace {

void check_cuda_kernel(const char* name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "[ultragauss_renderer.shader] ", name, " failed: ", cudaGetErrorString(err));
}

}

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
) {
    int n_pixels = pixel_positions.size(0);
    int n_gauss = gauss_means.size(0);
    auto float_opts = gauss_means.options().dtype(torch::kFloat32);
    torch::Tensor pixel_image = torch::zeros({n_pixels}, float_opts);
    torch::Tensor pixel_alpha = torch::zeros({n_pixels}, float_opts);
    torch::Tensor pixel_depth = torch::zeros({n_pixels}, float_opts);
    torch::Tensor sigma_inv_cache = torch::empty({n_gauss, 9}, float_opts);

    RasterizeGaussiansForwardCUDA_Impl(
        n_pixels,
        n_gauss,
        reinterpret_cast<const float2*>(gauss_opacity_intensity.data_ptr<float>()),
        reinterpret_cast<const float4*>(gauss_means.data_ptr<float>()),
        gauss_precision_tril.data_ptr<float>(),
        reinterpret_cast<const float4*>(pixel_positions.data_ptr<float>()),
        probe_origin.data_ptr<float>(),
        probe_vec_elevational.data_ptr<float>(),
        pixel_gauss_offsets.data_ptr<int>(),
        pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_indices.data_ptr<int>(),
        bg_alpha,
        bg_intensity,
        chi2_threshold,
        diag_beta,
        pixel_image.data_ptr<float>(),
        pixel_alpha.data_ptr<float>(),
        pixel_depth.data_ptr<float>(),
        sigma_inv_cache.data_ptr<float>(),
        at::cuda::getCurrentCUDAStream()
    );
    check_cuda_kernel("RasterizeGaussiansForwardCUDA_Impl");

    return {pixel_image, pixel_alpha, pixel_depth, sigma_inv_cache};
}

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
) {
    int n_gauss = gauss_means.size(0);
    auto float_opts = gauss_means.options().dtype(torch::kFloat32);
    torch::Tensor dL_dMeans = torch::zeros({n_gauss, 4}, float_opts);
    torch::Tensor dL_dPrecision = torch::zeros({n_gauss, 6}, float_opts);
    torch::Tensor dL_dAttr = torch::zeros({n_gauss, 2}, float_opts);
    RasterizeGaussiansBackwardCUDA_Impl(
        pixel_positions.size(0),
        n_gauss,
        dL_dImage.data_ptr<float>(),
        dL_dAlpha.data_ptr<float>(),
        dL_dDepth.data_ptr<float>(),
        reinterpret_cast<const float2*>(gauss_opacity_intensity.data_ptr<float>()),
        reinterpret_cast<const float4*>(gauss_means.data_ptr<float>()),
        gauss_precision_tril.data_ptr<float>(),
        reinterpret_cast<const float4*>(pixel_positions.data_ptr<float>()),
        probe_origin.data_ptr<float>(),
        probe_vec_elevational.data_ptr<float>(),
        pixel_gauss_offsets.data_ptr<int>(),
        pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_indices.data_ptr<int>(),
        sigma_inv_cache.data_ptr<float>(),
        pixel_alpha.data_ptr<float>(),
        bg_alpha,
        bg_intensity,
        chi2_threshold,
        diag_beta,
        reinterpret_cast<float4*>(dL_dMeans.data_ptr<float>()),
        dL_dPrecision.data_ptr<float>(),
        reinterpret_cast<float2*>(dL_dAttr.data_ptr<float>()),
        at::cuda::getCurrentCUDAStream()
    );
    check_cuda_kernel("RasterizeGaussiansBackwardCUDA_Impl");
    return {dL_dMeans, dL_dPrecision, dL_dAttr};
}
