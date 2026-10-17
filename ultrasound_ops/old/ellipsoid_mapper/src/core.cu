#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "covariance.cuh"
#include "load_utils.cuh"
#include "math_utils.cuh"

#include "core.cuh"


// -------------------------------------------------------------------------
// Kernel 1: 预计算投影半径 (Precompute Projection Radii)
// Quat/Scale -> R -> Projected Radii (Lat, Ax, Ele)
//
// IMPORTANT:
// - projected_radii is stored in probe-axis component order:
//     .x = lateral, .y = axial, .z = elevational
// - projected_radii is in mm and already multiplied by k_sigma.
// -------------------------------------------------------------------------
__global__ void kernel_precompute_gauss_projected_radii(
    const int n_gauss,
    const float k_sigma,
    const float4* __restrict__ quats,
    const float4* __restrict__ scales_packed,
    // Slice Geometry
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ele,
    const float* __restrict__ ptr_vec_ax,
    // Output
    float3* __restrict__ projected_radii
) {
    // 使用 Shared Memory 加载3个轴的方向向量
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

    // Load
    float4 q = quats[idx];
    float3 s = unpack_float4(scales_packed[idx]);

    // Compute R
    float R[9];
    compute_rot_from_quat(q, R);

    // Compute Projected Radii (Manhattan Bound)
    // 乘上 k_sigma，后续比较直接用
    float r_lat = compute_projected_radius(s, R, axis_lateral) * k_sigma;
    float r_ax  = compute_projected_radius(s, R, axis_axial)  * k_sigma;
    float r_ele = compute_projected_radius(s, R, axis_elevational) * k_sigma;

    // Store in probe xyz order: (lat, ax, ele)
    projected_radii[idx] = make_float3(r_lat, r_ax, r_ele);
}

// -------------------------------------------------------------------------
// Kernel 2: 粗筛与计数 (Cull & Count)
// [Updated] 读取预计算半径，移除复杂的 Rot 计算
// -------------------------------------------------------------------------
__global__ void kernel_project_gauss_and_compute_tile_rects(
    const int n_gauss,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii, // Precomputed
    // Geometry
    const int height, const int width,
    const float near_mm, const float delta_lat, const float delta_ax,
    const float slab_half_thickness_mm,
    const float lateral_col_offset,
    const float* __restrict__ ptr_origin,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ele,
    const float* __restrict__ ptr_vec_ax,
    // Output
    int* num_tiles,
    int4* rects,
    float* depths
) {
    // 使用 Shared Memory 加载3个轴的方向向量+探头原点坐标
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

    // Load
    float3 m = unpack_float4(means_packed[idx]);
    // projected_radii component order: .x=lat, .y=ax, .z=ele
    float3 radii = projected_radii[idx];

    // 1. Elevational Culling
    float3 v = m - origin;
    float d_ele = dot_prod(v, axis_elevational);
    
    // [old] if (fabsf(d_ele) > radii.y)  // when .y was elevational
    // [修改后] 物理相交测试：中心距离 > (Slab半厚度 + 高斯投影半长)
    // NOTE: slab_half_thickness_mm is half thickness (per project convention)
    if (fabsf(d_ele) > radii.z + slab_half_thickness_mm) {
        num_tiles[idx] = 0;
        return;
    }

    // 2. Project to Pixel Space
    float x_lat_mm = dot_prod(v, axis_lateral);
    float z_ax_mm  = dot_prod(v, axis_axial);

    float u_pix = x_lat_mm / delta_lat + lateral_col_offset;
    float v_pix = (z_ax_mm - near_mm) / delta_ax;

    // Radius in Pixels
    float r_lat_pix = radii.x / delta_lat;
    float r_ax_pix  = radii.y / delta_ax;

    // Safe Guard
    r_lat_pix = fmaxf(1.0f, r_lat_pix);
    r_ax_pix  = fmaxf(1.0f, r_ax_pix);

    // 3. Tile AABB
    int min_tx = (int)floorf((u_pix - r_lat_pix) / SLICE_TILE_SIZE);
    int min_ty = (int)floorf((v_pix - r_ax_pix) / SLICE_TILE_SIZE);
    int max_tx = (int)ceilf((u_pix + r_lat_pix) / SLICE_TILE_SIZE);
    int max_ty = (int)ceilf((v_pix + r_ax_pix) / SLICE_TILE_SIZE);

    // Clamp
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

// ... (kernel_generate_tile_depth_keys, kernel_identify_tile_ranges 保持不变) ...
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
    
    // Depth key for sorting within each tile.
    // Convention: smaller axial (shallower, encountered earlier by the ultrasound beam)
    // comes first. We currently use v_pix (axial pixel coordinate) as the sort value.
    //
    // Map float -> uint32 key (quantize for stable radix sort).
    //
    // NOTE:
    // - In this project, axial pixel coordinate v_pix is expected to be non-negative
    //   (near_mm >= 0 and we mainly keep gaussians inside the image region).
    // - We STILL keep a positive offset as a defensive guard against out-of-range / jitter,
    //   so the key never underflows when cast to uint32.
    uint32_t depth_key = 0u;
    if (sort_by_depth) {
        const float kDepthKeyOffset = 500.0f;   // in pixel units (since d is v_pix)
        const float kDepthKeyScale  = 1000.0f;  // fixed-point scale
        depth_key = (uint32_t)((d + kDepthKeyOffset) * kDepthKeyScale);
    }

    for (int y = r.y; y < r.w; ++y) {
        for (int x = r.x; x < r.z; ++x) {
            uint64_t tile_id = y * grid_width_tiles + x;
            // High 32 bits: Tile ID, Low 32 bits: Depth Key
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

// -------------------------------------------------------------------------
// Kernel 4 & 5: Hit Counting & Index Filling
// [Updated] 使用 Optimized Intersection Check
// -------------------------------------------------------------------------
__global__ void kernel_count_pixel_candidates(
    const int n_pixels, const int width, const int height,
    const float4* __restrict__ p_pos_packed,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii, // Precomputed
    // Selection
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    // Geometry
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ax,
    // Output
    int* counts
) {
    // 使用 Shared Memory 加载2个轴的方向向量
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

    // Tile Logic
    int tile_x = px_x / SLICE_TILE_SIZE;
    int tile_y = px_y / SLICE_TILE_SIZE;
    int grid_w = (width + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE;
    int tile_id = tile_y * grid_w + tile_x;

    int start = tile_ranges_start[tile_id];
    int end   = tile_ranges_end[tile_id];
    if (start == -1 || start >= end) { counts[p] = 0; return; }

    // Load Pixel Pos (Assuming Packed or Float4, adapt load_vec3)
    float3 p_vec = make_float3(p_pos_packed[p].x, p_pos_packed[p].y, p_pos_packed[p].z);

    int counter = 0;
    for (int i = start; i < end; ++i) {
        int g_idx = sorted_gauss_ids[i];
        
        // 极简加载：只读 Mean 和 Radii (6 floats total vs 48 floats before)
        float3 m = unpack_float4(means_packed[g_idx]);
        float3 r = projected_radii[g_idx];

        IntersectionResult res = check_intersection_optimized(
            p_vec, m, r, axis_lateral, axis_axial
        );

        if (res.hit) counter++;
    }
    counts[p] = counter;
}

// (kernel_fill_pixel_candidate_indices 逻辑同上，只是写入 indices)
__global__ void kernel_fill_pixel_candidate_indices(
    const int n_pixels, const int width, const int height,
    const float4* __restrict__ p_pos_packed,
    const float4* __restrict__ means_packed,
    const float3* __restrict__ projected_radii, // Precomputed
    // Selection
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    // Geometry
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ax,
    // Offset
    const int* __restrict__ offsets,
    // Output
    int* indices
) {
    // 使用 Shared Memory 加载2个轴的方向向量
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

    // Tile Logic
    int tile_x = px_x / SLICE_TILE_SIZE;
    int tile_y = px_y / SLICE_TILE_SIZE;
    int grid_w = (width + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE;
    int tile_id = tile_y * grid_w + tile_x;

    int start = tile_ranges_start[tile_id];
    int end   = tile_ranges_end[tile_id];
    if (start == -1 || start >= end) return;

    // Load Pixel Pos (Assuming Packed or Float4, adapt load_vec3)
    float3 p_vec = make_float3(p_pos_packed[p].x, p_pos_packed[p].y, p_pos_packed[p].z);

    int write_ptr = offsets[p];
    for (int i = start; i < end; ++i) {
        int g_idx = sorted_gauss_ids[i];
        
        // 极简加载：只读 Mean 和 Radii (6 floats total vs 48 floats before)
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
