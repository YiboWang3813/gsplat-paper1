#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "covariance.cuh"
#include "load_utils.cuh"
#include "math_utils.cuh"

#include "mapper/core.cuh"

__global__ void kernel_precompute_gauss_projected_radii(
    const int n_gauss,
    const float k_sigma,
    const float4* __restrict__ quats,
    const float4* __restrict__ scales_packed,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ele,
    const float* __restrict__ ptr_vec_ax,
    float3* __restrict__ projected_radii
) {
    __shared__ float s_vec_lat[3];
    __shared__ float s_vec_ax[3];
    __shared__ float s_vec_ele[3];
    if (threadIdx.x == 0) {
        vec3_copy(ptr_vec_lat, s_vec_lat);
        vec3_copy(ptr_vec_ax, s_vec_ax);
        vec3_copy(ptr_vec_ele, s_vec_ele);
    }
    __syncthreads();
    float3 axis_lateral = make_float3(s_vec_lat[0], s_vec_lat[1], s_vec_lat[2]);
    float3 axis_axial = make_float3(s_vec_ax[0], s_vec_ax[1], s_vec_ax[2]);
    float3 axis_elevational = make_float3(s_vec_ele[0], s_vec_ele[1], s_vec_ele[2]);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gauss) return;

    float4 q = quats[idx];
    float3 s = unpack_float4(scales_packed[idx]);

    float R[9];
    compute_rot_from_quat(q, R);

    float r_lat = compute_projected_radius(s, R, axis_lateral) * k_sigma;
    float r_ax = compute_projected_radius(s, R, axis_axial) * k_sigma;
    float r_ele = compute_projected_radius(s, R, axis_elevational) * k_sigma;

    projected_radii[idx] = make_float3(r_lat, r_ax, r_ele);
}

__global__ void kernel_project_gauss_and_compute_tile_rects(
    const int n_gauss,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii,
    const int height,
    const int width,
    const float near_mm,
    const float delta_lat,
    const float delta_ax,
    const float slab_half_thickness_mm,
    const float lateral_col_offset,
    const float* __restrict__ ptr_origin,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ele,
    const float* __restrict__ ptr_vec_ax,
    int* num_tiles,
    int4* rects,
    float* depths
) {
    __shared__ float s_origin[3];
    __shared__ float s_vec_lat[3];
    __shared__ float s_vec_ax[3];
    __shared__ float s_vec_ele[3];
    if (threadIdx.x == 0) {
        vec3_copy(ptr_origin, s_origin);
        vec3_copy(ptr_vec_lat, s_vec_lat);
        vec3_copy(ptr_vec_ax, s_vec_ax);
        vec3_copy(ptr_vec_ele, s_vec_ele);
    }
    __syncthreads();
    float3 origin = make_float3(s_origin[0], s_origin[1], s_origin[2]);
    float3 axis_lateral = make_float3(s_vec_lat[0], s_vec_lat[1], s_vec_lat[2]);
    float3 axis_axial = make_float3(s_vec_ax[0], s_vec_ax[1], s_vec_ax[2]);
    float3 axis_elevational = make_float3(s_vec_ele[0], s_vec_ele[1], s_vec_ele[2]);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gauss) return;

    float3 m = unpack_float4(means_packed[idx]);
    float3 radii = projected_radii[idx];

    float3 v = m - origin;
    float d_ele = dot_prod(v, axis_elevational);
    if (fabsf(d_ele) > radii.z + slab_half_thickness_mm) {
        num_tiles[idx] = 0;
        return;
    }

    float x_lat_mm = dot_prod(v, axis_lateral);
    float z_ax_mm = dot_prod(v, axis_axial);

    float u_pix = x_lat_mm / delta_lat + lateral_col_offset;
    float v_pix = (z_ax_mm - near_mm) / delta_ax;

    float r_lat_pix = fmaxf(1.0f, radii.x / delta_lat);
    float r_ax_pix = fmaxf(1.0f, radii.y / delta_ax);

    int min_tx = (int)floorf((u_pix - r_lat_pix) / SLICE_TILE_SIZE);
    int min_ty = (int)floorf((v_pix - r_ax_pix) / SLICE_TILE_SIZE);
    int max_tx = (int)ceilf((u_pix + r_lat_pix) / SLICE_TILE_SIZE);
    int max_ty = (int)ceilf((v_pix + r_ax_pix) / SLICE_TILE_SIZE);

    min_tx = max(0, min_tx);
    min_ty = max(0, min_ty);
    max_tx = min((width + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE, max_tx);
    max_ty = min((height + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE, max_ty);

    if (min_tx >= max_tx || min_ty >= max_ty) {
        num_tiles[idx] = 0;
        return;
    }

    rects[idx] = make_int4(min_tx, min_ty, max_tx, max_ty);
    depths[idx] = v_pix;
    num_tiles[idx] = (max_tx - min_tx) * (max_ty - min_ty);
}

__global__ void kernel_generate_tile_depth_keys(
    const int n_gauss,
    const int grid_width_tiles,
    const bool sort_by_depth,
    const int4* rects,
    const float* depths,
    const int* offsets,
    uint64_t* keys_out,
    int* values_out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gauss) return;

    int offset = offsets[idx];
    int count = offsets[idx + 1] - offset;
    if (count <= 0) return;

    int4 r = rects[idx];
    float d = depths[idx];

    uint32_t depth_key = 0u;
    if (sort_by_depth) {
        const float kDepthKeyOffset = 500.0f;
        const float kDepthKeyScale = 1000.0f;
        depth_key = (uint32_t)((d + kDepthKeyOffset) * kDepthKeyScale);
    }

    for (int y = r.y; y < r.w; ++y) {
        for (int x = r.x; x < r.z; ++x) {
            uint64_t tile_id = y * grid_width_tiles + x;
            keys_out[offset] = (tile_id << 32) | depth_key;
            values_out[offset] = idx;
            offset++;
        }
    }
}

__global__ void kernel_identify_tile_ranges(
    const int num_instances,
    const uint64_t* sorted_keys,
    int* tile_ranges_start,
    int* tile_ranges_end
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;

    uint64_t key = sorted_keys[idx];
    uint32_t cur_tile = (uint32_t)(key >> 32);

    if (idx == 0) {
        tile_ranges_start[cur_tile] = 0;
    } else {
        uint32_t prev_tile = (uint32_t)(sorted_keys[idx - 1] >> 32);
        if (cur_tile != prev_tile) {
            tile_ranges_end[prev_tile] = idx;
            tile_ranges_start[cur_tile] = idx;
        }
    }

    if (idx == num_instances - 1) {
        tile_ranges_end[cur_tile] = num_instances;
    }
}

__global__ void kernel_count_pixel_candidates(
    const int n_pixels,
    const int width,
    const int height,
    const float4* __restrict__ p_pos_packed,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ax,
    int* counts
) {
    __shared__ float s_vec_lat[3];
    __shared__ float s_vec_ax[3];
    if (threadIdx.x == 0) {
        vec3_copy(ptr_vec_lat, s_vec_lat);
        vec3_copy(ptr_vec_ax, s_vec_ax);
    }
    __syncthreads();
    float3 axis_lateral = make_float3(s_vec_lat[0], s_vec_lat[1], s_vec_lat[2]);
    float3 axis_axial = make_float3(s_vec_ax[0], s_vec_ax[1], s_vec_ax[2]);

    int px_x = blockIdx.x * blockDim.x + threadIdx.x;
    int px_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (px_x >= width || px_y >= height) return;

    int p = px_y * width + px_x;
    if (p >= n_pixels) return;

    int tile_x = px_x / SLICE_TILE_SIZE;
    int tile_y = px_y / SLICE_TILE_SIZE;
    int grid_w = (width + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE;
    int tile_id = tile_y * grid_w + tile_x;

    int start = tile_ranges_start[tile_id];
    int end = tile_ranges_end[tile_id];
    if (start == -1 || start >= end) {
        counts[p] = 0;
        return;
    }

    float3 p_vec = make_float3(p_pos_packed[p].x, p_pos_packed[p].y, p_pos_packed[p].z);

    int counter = 0;
    for (int i = start; i < end; ++i) {
        int g_idx = sorted_gauss_ids[i];
        float3 m = unpack_float4(means_packed[g_idx]);
        float3 r = projected_radii[g_idx];

        IntersectionResult res = check_intersection_optimized(
            p_vec, m, r, axis_lateral, axis_axial
        );

        if (res.hit) counter++;
    }
    counts[p] = counter;
}

__global__ void kernel_fill_pixel_candidate_indices(
    const int n_pixels,
    const int width,
    const int height,
    const float4* __restrict__ p_pos_packed,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ax,
    const int* __restrict__ offsets,
    int* indices
) {
    __shared__ float s_vec_lat[3];
    __shared__ float s_vec_ax[3];
    if (threadIdx.x == 0) {
        vec3_copy(ptr_vec_lat, s_vec_lat);
        vec3_copy(ptr_vec_ax, s_vec_ax);
    }
    __syncthreads();
    float3 axis_lateral = make_float3(s_vec_lat[0], s_vec_lat[1], s_vec_lat[2]);
    float3 axis_axial = make_float3(s_vec_ax[0], s_vec_ax[1], s_vec_ax[2]);

    int px_x = blockIdx.x * blockDim.x + threadIdx.x;
    int px_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (px_x >= width || px_y >= height) return;

    int p = px_y * width + px_x;
    if (p >= n_pixels) return;

    int tile_x = px_x / SLICE_TILE_SIZE;
    int tile_y = px_y / SLICE_TILE_SIZE;
    int grid_w = (width + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE;
    int tile_id = tile_y * grid_w + tile_x;

    int start = tile_ranges_start[tile_id];
    int end = tile_ranges_end[tile_id];
    if (start == -1 || start >= end) return;

    float3 p_vec = make_float3(p_pos_packed[p].x, p_pos_packed[p].y, p_pos_packed[p].z);

    int write_ptr = offsets[p];
    for (int i = start; i < end; ++i) {
        int g_idx = sorted_gauss_ids[i];
        float3 m = unpack_float4(means_packed[g_idx]);
        float3 r = projected_radii[g_idx];

        IntersectionResult res = check_intersection_optimized(
            p_vec, m, r, axis_lateral, axis_axial
        );

        if (res.hit) {
            indices[write_ptr] = g_idx;
            write_ptr++;
        }
    }
}

// ======================= Batched kernels ====================================

__global__ void kernel_precompute_gauss_projected_radii_batched(
    const int n_gauss,
    const int batch_size,
    const float k_sigma,
    const float4* __restrict__ quats,
    const float4* __restrict__ scales_packed,
    const float* __restrict__ vec_lat_batched,
    const float* __restrict__ vec_ele_batched,
    const float* __restrict__ vec_ax_batched,
    const uint8_t* __restrict__ active_gauss_mask_batched,
    float3* __restrict__ projected_radii_batched
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (g >= n_gauss || b >= batch_size) return;

    __shared__ float s_vec_lat[3];
    __shared__ float s_vec_ax[3];
    __shared__ float s_vec_ele[3];
    if (threadIdx.x == 0) {
        const float* ptr_vec_lat = vec_lat_batched + b * 3;
        const float* ptr_vec_ele = vec_ele_batched + b * 3;
        const float* ptr_vec_ax = vec_ax_batched + b * 3;
        vec3_copy(ptr_vec_lat, s_vec_lat);
        vec3_copy(ptr_vec_ax, s_vec_ax);
        vec3_copy(ptr_vec_ele, s_vec_ele);
    }
    __syncthreads();
    float3 axis_lateral = make_float3(s_vec_lat[0], s_vec_lat[1], s_vec_lat[2]);
    float3 axis_axial = make_float3(s_vec_ax[0], s_vec_ax[1], s_vec_ax[2]);
    float3 axis_elevational = make_float3(s_vec_ele[0], s_vec_ele[1], s_vec_ele[2]);
    if (active_gauss_mask_batched && active_gauss_mask_batched[(size_t)b * (size_t)n_gauss + (size_t)g] == 0) {
        projected_radii_batched[(size_t)b * (size_t)n_gauss + (size_t)g] = make_float3(0.f, 0.f, 0.f);
        return;
    }

    float4 q = quats[g];
    float3 s = unpack_float4(scales_packed[g]);

    float R[9];
    compute_rot_from_quat(q, R);

    float r_lat = compute_projected_radius(s, R, axis_lateral) * k_sigma;
    float r_ax = compute_projected_radius(s, R, axis_axial) * k_sigma;
    float r_ele = compute_projected_radius(s, R, axis_elevational) * k_sigma;

    projected_radii_batched[(size_t)b * (size_t)n_gauss + (size_t)g] = make_float3(r_lat, r_ax, r_ele);
}

__global__ void kernel_project_gauss_and_compute_tile_rects_batched(
    const int n_gauss,
    const int batch_size,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii_batched,
    const int height,
    const int width,
    const float near_mm,
    const float delta_lat,
    const float delta_ax,
    const float slab_half_thickness_mm,
    const float lateral_col_offset,
    const float* __restrict__ probe_org_batched,
    const float* __restrict__ vec_lat_batched,
    const float* __restrict__ vec_ele_batched,
    const float* __restrict__ vec_ax_batched,
    const uint8_t* __restrict__ active_gauss_mask_batched,
    int* __restrict__ tiles_touched_count_batched,
    int4* __restrict__ gauss_tile_rects_batched,
    float* __restrict__ gauss_axial_pix_batched
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (g >= n_gauss || b >= batch_size) return;

    __shared__ float s_origin[3];
    __shared__ float s_vec_lat[3];
    __shared__ float s_vec_ax[3];
    __shared__ float s_vec_ele[3];
    if (threadIdx.x == 0) {
        const float* ptr_origin = probe_org_batched + b * 3;
        const float* ptr_vec_lat = vec_lat_batched + b * 3;
        const float* ptr_vec_ax = vec_ax_batched + b * 3;
        const float* ptr_vec_ele = vec_ele_batched + b * 3;
        vec3_copy(ptr_origin, s_origin);
        vec3_copy(ptr_vec_lat, s_vec_lat);
        vec3_copy(ptr_vec_ax, s_vec_ax);
        vec3_copy(ptr_vec_ele, s_vec_ele);
    }
    __syncthreads();
    float3 origin = make_float3(s_origin[0], s_origin[1], s_origin[2]);
    float3 axis_lateral = make_float3(s_vec_lat[0], s_vec_lat[1], s_vec_lat[2]);
    float3 axis_axial = make_float3(s_vec_ax[0], s_vec_ax[1], s_vec_ax[2]);
    float3 axis_elevational = make_float3(s_vec_ele[0], s_vec_ele[1], s_vec_ele[2]);
    if (active_gauss_mask_batched && active_gauss_mask_batched[(size_t)b * (size_t)n_gauss + (size_t)g] == 0) {
        tiles_touched_count_batched[(size_t)b * (size_t)(n_gauss + 1) + (size_t)g] = 0;
        return;
    }

    float3 m = unpack_float4(means_packed[g]);
    float3 radii = projected_radii_batched[(size_t)b * (size_t)n_gauss + (size_t)g];

    float3 v = m - origin;
    float d_ele = dot_prod(v, axis_elevational);
    if (fabsf(d_ele) > radii.z + slab_half_thickness_mm) {
        tiles_touched_count_batched[(size_t)b * (size_t)(n_gauss + 1) + (size_t)g] = 0;
        return;
    }

    float x_lat_mm = dot_prod(v, axis_lateral);
    float z_ax_mm = dot_prod(v, axis_axial);

    float u_pix = x_lat_mm / delta_lat + lateral_col_offset;
    float v_pix = (z_ax_mm - near_mm) / delta_ax;

    float r_lat_pix = fmaxf(1.0f, radii.x / delta_lat);
    float r_ax_pix = fmaxf(1.0f, radii.y / delta_ax);

    int min_tx = (int)floorf((u_pix - r_lat_pix) / SLICE_TILE_SIZE);
    int min_ty = (int)floorf((v_pix - r_ax_pix) / SLICE_TILE_SIZE);
    int max_tx = (int)ceilf((u_pix + r_lat_pix) / SLICE_TILE_SIZE);
    int max_ty = (int)ceilf((v_pix + r_ax_pix) / SLICE_TILE_SIZE);

    min_tx = max(0, min_tx);
    min_ty = max(0, min_ty);
    max_tx = min((width + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE, max_tx);
    max_ty = min((height + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE, max_ty);

    if (min_tx >= max_tx || min_ty >= max_ty) {
        tiles_touched_count_batched[(size_t)b * (size_t)(n_gauss + 1) + (size_t)g] = 0;
        return;
    }

    gauss_tile_rects_batched[(size_t)b * (size_t)n_gauss + (size_t)g] = make_int4(min_tx, min_ty, max_tx, max_ty);
    gauss_axial_pix_batched[(size_t)b * (size_t)n_gauss + (size_t)g] = v_pix;
    tiles_touched_count_batched[(size_t)b * (size_t)(n_gauss + 1) + (size_t)g] =
        (max_tx - min_tx) * (max_ty - min_ty);
}

__global__ void kernel_generate_tile_depth_keys_batched(
    const int n_gauss,
    const int batch_size,
    const int grid_width_tiles,
    const int n_tiles,
    const bool sort_by_depth,
    const int4* __restrict__ rects_batched,
    const float* __restrict__ depths_batched,
    const int* __restrict__ gauss_instance_offsets_batched,
    uint64_t* __restrict__ keys_out,
    int* __restrict__ values_out
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (g >= n_gauss || b >= batch_size) return;

    int offset = gauss_instance_offsets_batched[(size_t)b * (size_t)(n_gauss + 1) + (size_t)g];
    int count =
        gauss_instance_offsets_batched[(size_t)b * (size_t)(n_gauss + 1) + (size_t)g + 1] - offset;
    if (count <= 0) return;

    int4 r = rects_batched[(size_t)b * (size_t)n_gauss + (size_t)g];
    float d = depths_batched[(size_t)b * (size_t)n_gauss + (size_t)g];

    uint32_t depth_key = 0u;
    if (sort_by_depth) {
        const float kDepthKeyOffset = 500.0f;
        const float kDepthKeyScale = 1000.0f;
        depth_key = (uint32_t)((d + kDepthKeyOffset) * kDepthKeyScale);
    }

    for (int y = r.y; y < r.w; ++y) {
        for (int x = r.x; x < r.z; ++x) {
            uint32_t tile_local = (uint32_t)(y * grid_width_tiles + x);
            uint32_t composite_tile = (uint32_t)b * (uint32_t)n_tiles + tile_local;
            keys_out[offset] = ((uint64_t)composite_tile << 32) | (uint64_t)depth_key;
            values_out[offset] = b * n_gauss + g;
            offset++;
        }
    }
}

__global__ void kernel_count_pixel_candidates_batched(
    const int n_gauss,
    const int pixels_per_image,
    const int width,
    const int height,
    const int batch_size,
    const int n_tiles,
    const float4* __restrict__ p_pos_packed_batched,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii_batched,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    const float* __restrict__ vec_lat_batched,
    const float* __restrict__ vec_ax_batched,
    int* __restrict__ counts
) {
    int px_x = blockIdx.x * blockDim.x + threadIdx.x;
    int px_y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (px_x >= width || px_y >= height || b >= batch_size) return;

    int p_local = px_y * width + px_x;
    if (p_local >= pixels_per_image) return;

    int gp = b * pixels_per_image + p_local;

    __shared__ float s_vec_lat[3];
    __shared__ float s_vec_ax[3];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        const float* ptr_vec_lat = vec_lat_batched + b * 3;
        const float* ptr_vec_ax = vec_ax_batched + b * 3;
        vec3_copy(ptr_vec_lat, s_vec_lat);
        vec3_copy(ptr_vec_ax, s_vec_ax);
    }
    __syncthreads();
    float3 axis_lateral = make_float3(s_vec_lat[0], s_vec_lat[1], s_vec_lat[2]);
    float3 axis_axial = make_float3(s_vec_ax[0], s_vec_ax[1], s_vec_ax[2]);

    int tile_x = px_x / SLICE_TILE_SIZE;
    int tile_y = px_y / SLICE_TILE_SIZE;
    int tile_local = tile_y * ((width + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE) + tile_x;
    int composite_tile = b * n_tiles + tile_local;

    int start = tile_ranges_start[composite_tile];
    int end = tile_ranges_end[composite_tile];
    if (start == -1 || start >= end) {
        counts[gp] = 0;
        return;
    }

    float3 p_vec = make_float3(
        p_pos_packed_batched[gp].x,
        p_pos_packed_batched[gp].y,
        p_pos_packed_batched[gp].z
    );

    int counter = 0;
    for (int i = start; i < end; ++i) {
        int bg = sorted_gauss_ids[i];
        int g_idx = bg % n_gauss;
        float3 m = unpack_float4(means_packed[g_idx]);
        float3 r = projected_radii_batched[(size_t)b * (size_t)n_gauss + (size_t)g_idx];

        IntersectionResult res = check_intersection_optimized(
            p_vec, m, r, axis_lateral, axis_axial
        );

        if (res.hit) counter++;
    }
    counts[gp] = counter;
}

__global__ void kernel_fill_pixel_candidate_indices_batched(
    const int n_gauss,
    const int pixels_per_image,
    const int width,
    const int height,
    const int batch_size,
    const int n_tiles,
    const float4* __restrict__ p_pos_packed_batched,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii_batched,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    const float* __restrict__ vec_lat_batched,
    const float* __restrict__ vec_ax_batched,
    const int* __restrict__ offsets,
    int* __restrict__ indices
) {
    int px_x = blockIdx.x * blockDim.x + threadIdx.x;
    int px_y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (px_x >= width || px_y >= height || b >= batch_size) return;

    int p_local = px_y * width + px_x;
    if (p_local >= pixels_per_image) return;

    int gp = b * pixels_per_image + p_local;

    __shared__ float s_vec_lat[3];
    __shared__ float s_vec_ax[3];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        const float* ptr_vec_lat = vec_lat_batched + b * 3;
        const float* ptr_vec_ax = vec_ax_batched + b * 3;
        vec3_copy(ptr_vec_lat, s_vec_lat);
        vec3_copy(ptr_vec_ax, s_vec_ax);
    }
    __syncthreads();
    float3 axis_lateral = make_float3(s_vec_lat[0], s_vec_lat[1], s_vec_lat[2]);
    float3 axis_axial = make_float3(s_vec_ax[0], s_vec_ax[1], s_vec_ax[2]);

    int tile_x = px_x / SLICE_TILE_SIZE;
    int tile_y = px_y / SLICE_TILE_SIZE;
    int tile_local = tile_y * ((width + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE) + tile_x;
    int composite_tile = b * n_tiles + tile_local;

    int start = tile_ranges_start[composite_tile];
    int end = tile_ranges_end[composite_tile];
    if (start == -1 || start >= end) return;

    float3 p_vec = make_float3(
        p_pos_packed_batched[gp].x,
        p_pos_packed_batched[gp].y,
        p_pos_packed_batched[gp].z
    );

    int write_ptr = offsets[gp];
    for (int i = start; i < end; ++i) {
        int bg = sorted_gauss_ids[i];
        int g_idx = bg % n_gauss;
        float3 m = unpack_float4(means_packed[g_idx]);
        float3 r = projected_radii_batched[(size_t)b * (size_t)n_gauss + (size_t)g_idx];

        IntersectionResult res = check_intersection_optimized(
            p_vec, m, r, axis_lateral, axis_axial
        );

        if (res.hit) {
            indices[write_ptr] = g_idx;
            write_ptr++;
        }
    }
}
