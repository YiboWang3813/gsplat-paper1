#include "mapper/core.cuh"

__global__ void kernel_project_and_cull(
    int n_gauss,
    int image_width,
    float pixel_spacing_mm,
    float lateral_col_offset,
    float near_clip_mm,
    float elev_cutoff_mm,
    float k_sigma,
    const float4* means_world,
    const float4* scales_world,
    const float* probe_origin,
    const float* axis_lateral,
    const float* axis_elevational,
    const float* axis_axial,
    int* num_rays_covered,
    int2* ray_minmax,
    float* depth_probe
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gauss) return;

    float3 origin = make_float3(probe_origin[0], probe_origin[1], probe_origin[2]);
    float3 axis_u = make_float3(axis_lateral[0], axis_lateral[1], axis_lateral[2]);
    float3 axis_v = make_float3(axis_elevational[0], axis_elevational[1], axis_elevational[2]);
    float3 axis_w = make_float3(axis_axial[0], axis_axial[1], axis_axial[2]);

    float3 p_world = xyz(means_world[idx]);
    float3 s_world = xyz(scales_world[idx]);
    float3 diff = p_world - origin;

    float3 p_local;
    p_local.x = dot_prod(diff, axis_u);
    p_local.y = dot_prod(diff, axis_v);
    p_local.z = dot_prod(diff, axis_w);

    float max_scale = fmaxf(s_world.x, fmaxf(s_world.y, s_world.z));
    float radius_safe = max_scale * k_sigma;

    if (p_local.z + radius_safe < near_clip_mm) {
        num_rays_covered[idx] = 0;
        return;
    }

    float dist_ele = fabsf(p_local.y);
    if (dist_ele > (radius_safe + elev_cutoff_mm)) {
        num_rays_covered[idx] = 0;
        return;
    }

    float center_pixel = (p_local.x / pixel_spacing_mm) + lateral_col_offset;
    float radius_pixel = radius_safe / pixel_spacing_mm;

    int min_ray = (int)floorf(center_pixel - radius_pixel);
    int max_ray = (int)ceilf(center_pixel + radius_pixel);
    min_ray = max(0, min_ray);
    max_ray = min(image_width - 1, max_ray);

    if (min_ray > max_ray) {
        num_rays_covered[idx] = 0;
        return;
    }

    num_rays_covered[idx] = (max_ray - min_ray + 1);
    ray_minmax[idx] = make_int2(min_ray, max_ray);
    depth_probe[idx] = p_local.z;
}

__global__ void kernel_generate_sort_keys(
    int n_gauss,
    const int* csr_offsets,
    const int2* ray_minmax,
    const float* depths,
    uint64_t* sort_keys,
    int* sort_values
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gauss) return;

    int start_offset = csr_offsets[idx];
    int count = csr_offsets[idx + 1] - start_offset;
    if (count <= 0) return;

    int2 range = ray_minmax[idx];
    float d_clamped = fmaxf(0.0f, depths[idx]);
    uint32_t depth_quantized = (uint32_t)(d_clamped * 1000.0f);

    int writer = start_offset;
    for (int r = range.x; r <= range.y; ++r) {
        uint64_t key = ((uint64_t)r << 32) | depth_quantized;
        sort_keys[writer] = key;
        sort_values[writer] = idx;
        writer++;
    }
}

__global__ void kernel_build_csr_structure(
    int n_total_hits,
    const uint64_t* sorted_keys,
    int* ray_start_indices,
    int* ray_end_indices
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total_hits) return;

    int cur_ray = (int)(sorted_keys[idx] >> 32);
    bool is_start = (idx == 0);
    if (idx > 0) {
        int prev_ray = (int)(sorted_keys[idx - 1] >> 32);
        if (cur_ray != prev_ray) {
            is_start = true;
            ray_end_indices[prev_ray] = idx;
        }
    }
    if (is_start) ray_start_indices[cur_ray] = idx;
    if (idx == n_total_hits - 1) ray_end_indices[cur_ray] = n_total_hits;
}

__global__ void kernel_decode_sorted_depths(
    int n_total_hits,
    const uint64_t* sorted_keys,
    float* sorted_depths
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total_hits) return;
    uint32_t depth_quantized = (uint32_t)(sorted_keys[idx] & 0xffffffffu);
    sorted_depths[idx] = ((float)depth_quantized) * 0.001f;
}
