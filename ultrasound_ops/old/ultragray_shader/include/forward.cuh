

__global__ void render_bmode_kernel(
    // Grid: (ImageWidth, ImageHeight)
    const int W, const int H,
    
    // CSR Inputs (from Mapper)
    const int* __restrict__ ray_counts,   // (W)
    const int* __restrict__ ray_offsets,  // (W)
    const int* __restrict__ ray_indices,  // (TotalHits) Sorted by depth

    // Geometry Inputs
    const float4* __restrict__ means,     // (G, 3)
    const float4* __restrict__ scales,    // (G, 3)
    const float4* __restrict__ quats,     // (G, 4)
    
    // Physics Parameters (Learnable)
    const float* __restrict__ attenuations, // (G) Range [0, 1] usually passed through sigmoid
    const float* __restrict__ intensities,  // (G) Echo strength

    // Scene Inputs
    const float* __restrict__ pixel_positions, // (W, H, 3) Flat
    const float* __restrict__ ray_origins,     // (W, 3)
    const float* __restrict__ ray_dirs,        // (W, 3) Normalized

    // Output
    float* __restrict__ output_image           // (W, H)
);