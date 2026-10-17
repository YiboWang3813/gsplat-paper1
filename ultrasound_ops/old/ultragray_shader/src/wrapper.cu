#include <torch/extension.h>
#include <vector>

#include "forward.cuh"
#include "backward.cuh"


torch::Tensor shade_forward(
    // Gaussian Attributes
    const torch::Tensor& means,           // (G, 3)
    const torch::Tensor& scales,          // (G, 3)
    const torch::Tensor& quats,           // (G, 4)
    const torch::Tensor& attenuations,    // (G)
    const torch::Tensor& intensities,     // (G)
    
    // CSR Structure
    const torch::Tensor& ray_counts,      // (W)
    const torch::Tensor& ray_offsets,     // (W)
    const torch::Tensor& ray_indices,     // (TotalHits)
    
    // Scene Data
    const torch::Tensor& pixel_positions, // (W, H, 3)
    const torch::Tensor& ray_origins,     // (W, 3)
    const torch::Tensor& ray_dirs         // (W, 3)
) {
    int W = pixel_positions.size(0);
    int H = pixel_positions.size(1);
    
    // Output Tensor
    auto opts = pixel_positions.options().dtype(torch::kFloat32);
    torch::Tensor output = torch::zeros({W, H}, opts);

    // Launch Config
    // Grid covers (Width, Height)
    dim3 threads(16, 16); 
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    render_bmode_kernel<<<blocks, threads>>>(
        W, H,
        ray_counts.data_ptr<int>(),
        ray_offsets.data_ptr<int>(),
        ray_indices.data_ptr<int>(),
        
        (float4*)means.data_ptr(),
        (float4*)scales.data_ptr(),
        (float4*)quats.data_ptr(),
        
        attenuations.data_ptr<float>(),
        intensities.data_ptr<float>(),
        
        pixel_positions.data_ptr<float>(),
        ray_origins.data_ptr<float>(),
        ray_dirs.data_ptr<float>(),
        
        output.data_ptr<float>()
    );

    return output;
}

void shade_backward(
    // Gradients Inputs
    const torch::Tensor& dL_dImage,       // (W, H)
    
    // Forward Inputs (Must match forward pass)
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& attenuations,
    const torch::Tensor& intensities,
    
    const torch::Tensor& ray_counts,
    const torch::Tensor& ray_offsets,
    const torch::Tensor& ray_indices,
    
    const torch::Tensor& pixel_positions,
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_dirs,
    
    // Gradients Outputs (Accumulators)
    torch::Tensor& dL_dMeans,
    torch::Tensor& dL_dScales,
    torch::Tensor& dL_dQuats,
    torch::Tensor& dL_dAttens,
    torch::Tensor& dL_dIntensities
) {
    int W = pixel_positions.size(0);
    int H = pixel_positions.size(1);
    
    dim3 threads(16, 16); 
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    render_bmode_backward_kernel<<<blocks, threads>>>(
        W, H,
        // Grads Out
        (float4*)dL_dMeans.data_ptr(),
        (float4*)dL_dScales.data_ptr(),
        (float4*)dL_dQuats.data_ptr(),
        dL_dAttens.data_ptr<float>(),
        dL_dIntensities.data_ptr<float>(),
        
        // Grads In & Data
        dL_dImage.data_ptr<float>(),
        ray_counts.data_ptr<int>(),
        ray_offsets.data_ptr<int>(),
        ray_indices.data_ptr<int>(),
        
        (float4*)means.data_ptr(),
        (float4*)scales.data_ptr(),
        (float4*)quats.data_ptr(),
        attenuations.data_ptr<float>(),
        intensities.data_ptr<float>(),
        
        pixel_positions.data_ptr<float>(),
        ray_origins.data_ptr<float>(),
        ray_dirs.data_ptr<float>()
    );
}