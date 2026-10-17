
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
);

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
);