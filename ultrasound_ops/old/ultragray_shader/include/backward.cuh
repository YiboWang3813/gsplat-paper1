

__global__ void render_bmode_backward_kernel(
    const int W, const int H,
    
    // Gradients (Outputs to Accumulate)
    float4* __restrict__ dL_dMeans,
    float4* __restrict__ dL_dScales,
    float4* __restrict__ dL_dQuats,
    float* __restrict__ dL_dAttens,
    float* __restrict__ dL_dIntensities,

    // Inputs (Same as Forward)
    const float* __restrict__ dL_dImage, // (W, H) Incoming Loss Gradient
    const int* __restrict__ ray_counts,
    const int* __restrict__ ray_offsets,
    const int* __restrict__ ray_indices,
    const float4* __restrict__ means,
    const float4* __restrict__ scales,
    const float4* __restrict__ quats,
    const float* __restrict__ attenuations,
    const float* __restrict__ intensities,
    const float* __restrict__ pixel_positions,
    const float* __restrict__ ray_origins,
    const float* __restrict__ ray_dirs
);