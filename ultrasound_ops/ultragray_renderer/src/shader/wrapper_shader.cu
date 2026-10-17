#include <torch/extension.h>

#include "shader/forward.cuh"
#include "shader/backward.cuh"
#include "shader/wrapper_shader.cuh"

torch::Tensor ShaderForwardCUDA(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& attenuations,
    const torch::Tensor& sh_coeffs,
    const torch::Tensor& ray_counts,
    const torch::Tensor& ray_offsets,
    const torch::Tensor& ray_indices,
    const torch::Tensor& sorted_depths,
    const torch::Tensor& pixel_positions,
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_dirs,
    const torch::Tensor& axis_elevational,
    const torch::Tensor& ray_perturbations,
    float k_sigma,
    float quadrature_window_radius_mm,
    bool local_transmission_window_only,
    float local_transmission_max_window_mm
) {
    int W = pixel_positions.size(0);
    int H = pixel_positions.size(1);
    auto opts = pixel_positions.options().dtype(torch::kFloat32);
    // Return standard image layout: (H, W).
    torch::Tensor output = torch::zeros({H, W}, opts);

    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);
    render_bmode_kernel<<<blocks, threads>>>(
        W, H,
        ray_counts.data_ptr<int>(),
        ray_offsets.data_ptr<int>(),
        ray_indices.data_ptr<int>(),
        sorted_depths.data_ptr<float>(),
        (float4*)means.data_ptr(),
        (float4*)scales.data_ptr(),
        (float4*)quats.data_ptr(),
        attenuations.data_ptr<float>(),
        (float4*)sh_coeffs.data_ptr(),
        pixel_positions.data_ptr<float>(),
        ray_origins.data_ptr<float>(),
        ray_dirs.data_ptr<float>(),
        axis_elevational.data_ptr<float>(),
        ray_perturbations.data_ptr<float>(),
        k_sigma,
        quadrature_window_radius_mm,
        local_transmission_window_only,
        local_transmission_max_window_mm,
        output.data_ptr<float>()
    );
    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ShaderBackwardCUDA(
    const torch::Tensor& dL_dImage,
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& attenuations,
    const torch::Tensor& sh_coeffs,
    const torch::Tensor& ray_counts,
    const torch::Tensor& ray_offsets,
    const torch::Tensor& ray_indices,
    const torch::Tensor& sorted_depths,
    const torch::Tensor& pixel_positions,
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_dirs,
    const torch::Tensor& axis_elevational,
    const torch::Tensor& ray_perturbations,
    float k_sigma,
    float quadrature_window_radius_mm,
    bool local_transmission_window_only,
    float local_transmission_max_window_mm
) {
    int W = pixel_positions.size(0);
    int H = pixel_positions.size(1);
    int G = means.size(0);

    auto fopts = means.options().dtype(torch::kFloat32);
    torch::Tensor dL_dMeans = torch::zeros({G, 4}, fopts);
    torch::Tensor dL_dScales = torch::zeros({G, 4}, fopts);
    torch::Tensor dL_dQuats = torch::zeros({G, 4}, fopts);
    torch::Tensor dL_dAttens = torch::zeros({G}, fopts);
    torch::Tensor dL_dSH = torch::zeros({G, 4}, fopts);

    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);
    render_bmode_backward_kernel<<<blocks, threads>>>(
        W, H,
        (float4*)dL_dMeans.data_ptr(),
        (float4*)dL_dScales.data_ptr(),
        (float4*)dL_dQuats.data_ptr(),
        dL_dAttens.data_ptr<float>(),
        (float4*)dL_dSH.data_ptr(),
        dL_dImage.data_ptr<float>(),
        ray_counts.data_ptr<int>(),
        ray_offsets.data_ptr<int>(),
        ray_indices.data_ptr<int>(),
        sorted_depths.data_ptr<float>(),
        (float4*)means.data_ptr(),
        (float4*)scales.data_ptr(),
        (float4*)quats.data_ptr(),
        attenuations.data_ptr<float>(),
        (float4*)sh_coeffs.data_ptr(),
        pixel_positions.data_ptr<float>(),
        ray_origins.data_ptr<float>(),
        ray_dirs.data_ptr<float>(),
        axis_elevational.data_ptr<float>(),
        ray_perturbations.data_ptr<float>(),
        k_sigma,
        quadrature_window_radius_mm,
        local_transmission_window_only,
        local_transmission_max_window_mm
    );

    return {dL_dMeans, dL_dScales, dL_dQuats, dL_dAttens, dL_dSH};
}
