#pragma once

__global__ void render_bmode_kernel(
    int W,
    int H,
    const int* __restrict__ ray_counts,
    const int* __restrict__ ray_offsets,
    const int* __restrict__ ray_indices,
    const float* __restrict__ sorted_depths,
    const float4* __restrict__ means,
    const float4* __restrict__ scales,
    const float4* __restrict__ quats,
    const float* __restrict__ attenuations,
    const float4* __restrict__ sh_coeffs,
    const float* __restrict__ pixel_positions,
    const float* __restrict__ ray_origins,
    const float* __restrict__ ray_dirs,
    const float* __restrict__ axis_elevational,
    const float* __restrict__ ray_perturbations,
    float k_sigma,
    float quadrature_window_radius_mm,
    bool local_transmission_window_only,
    float local_transmission_max_window_mm,
    float* __restrict__ output_image
);
