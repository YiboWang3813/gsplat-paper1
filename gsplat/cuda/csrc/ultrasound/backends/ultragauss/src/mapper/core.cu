#include <cstdint>
#include <cuda_runtime.h>

#include "math_utils.cuh"
#include "load_utils.cuh"

#include "mapper/core.cuh"

__device__ __forceinline__ void compute_probe_bbox_radii(
    const float* tri,
    const float chi2,
    const float beta,
    const float3 axis_lat,
    const float3 axis_ax,
    const float3 axis_ele,
    float& r_lat,
    float& r_ax,
    float& r_ele
) {
    const float l11 = safe_diag_from_tri(tri[0], beta);
    const float l12 = tri[1];
    const float l13 = tri[2];
    const float l22 = safe_diag_from_tri(tri[3], beta);
    const float l23 = tri[4];
    const float l33 = safe_diag_from_tri(tri[5], beta);
    const float y_lat0 = axis_lat.x / fmaxf(l11, 1e-10f);
    const float y_lat1 = (axis_lat.y - l12 * y_lat0) / fmaxf(l22, 1e-10f);
    const float y_lat2 = (axis_lat.z - l13 * y_lat0 - l23 * y_lat1) / fmaxf(l33, 1e-10f);
    const float sigma_lat = fmaxf(y_lat0 * y_lat0 + y_lat1 * y_lat1 + y_lat2 * y_lat2, 1e-12f);

    const float y_ax0 = axis_ax.x / fmaxf(l11, 1e-10f);
    const float y_ax1 = (axis_ax.y - l12 * y_ax0) / fmaxf(l22, 1e-10f);
    const float y_ax2 = (axis_ax.z - l13 * y_ax0 - l23 * y_ax1) / fmaxf(l33, 1e-10f);
    const float sigma_ax = fmaxf(y_ax0 * y_ax0 + y_ax1 * y_ax1 + y_ax2 * y_ax2, 1e-12f);

    const float y_ele0 = axis_ele.x / fmaxf(l11, 1e-10f);
    const float y_ele1 = (axis_ele.y - l12 * y_ele0) / fmaxf(l22, 1e-10f);
    const float y_ele2 = (axis_ele.z - l13 * y_ele0 - l23 * y_ele1) / fmaxf(l33, 1e-10f);
    const float sigma_ele = fmaxf(y_ele0 * y_ele0 + y_ele1 * y_ele1 + y_ele2 * y_ele2, 1e-12f);
    const float k = sqrtf(chi2);
    r_lat = fmaxf(1e-3f, k * sqrtf(sigma_lat));
    r_ax = fmaxf(1e-3f, k * sqrtf(sigma_ax));
    r_ele = fmaxf(1e-3f, k * sqrtf(sigma_ele));
}

__global__ void kernel_project_gauss_and_compute_tile_rects(
    const int n_gauss,
    const float4* __restrict__ means_packed,
    const float* __restrict__ precision_tril,
    const int height,
    const int width,
    const float near_mm,
    const float delta_lat,
    const float delta_ax,
    const float lateral_col_offset,
    const float chi2,
    const float beta,
    const float* __restrict__ ptr_origin,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ele,
    const float* __restrict__ ptr_vec_ax,
    int* num_tiles,
    int4* rects,
    float* depths
) {
    __shared__ float s_origin[3], s_lat[3], s_ele[3], s_ax[3];
    if (threadIdx.x == 0) {
        vec3_copy(ptr_origin, s_origin);
        vec3_copy(ptr_vec_lat, s_lat);
        vec3_copy(ptr_vec_ele, s_ele);
        vec3_copy(ptr_vec_ax, s_ax);
    }
    __syncthreads();
    float3 origin = make_float3(s_origin[0], s_origin[1], s_origin[2]);
    float3 axis_lat = make_float3(s_lat[0], s_lat[1], s_lat[2]);
    float3 axis_ele = make_float3(s_ele[0], s_ele[1], s_ele[2]);
    float3 axis_ax = make_float3(s_ax[0], s_ax[1], s_ax[2]);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gauss) return;

    float3 m = make_float3(means_packed[idx].x, means_packed[idx].y, means_packed[idx].z);
    float3 v = m - origin;
    float lat = dot_prod(v, axis_lat);
    float ele = dot_prod(v, axis_ele);
    float ax = dot_prod(v, axis_ax);

    const float* tri = precision_tril + idx * 6;
    float r_lat, r_ax, r_ele;
    compute_probe_bbox_radii(tri, chi2, beta, axis_lat, axis_ax, axis_ele, r_lat, r_ax, r_ele);
    if (fabsf(ele) > r_ele) {
        num_tiles[idx] = 0;
        return;
    }

    float u_pix = lat / delta_lat + lateral_col_offset;
    float v_pix = (ax - near_mm) / delta_ax;
    float r_u = fmaxf(1.0f, r_lat / delta_lat);
    float r_v = fmaxf(1.0f, r_ax / delta_ax);

    int min_tx = (int)floorf((u_pix - r_u) / SLICE_TILE_SIZE);
    int min_ty = (int)floorf((v_pix - r_v) / SLICE_TILE_SIZE);
    int max_tx = (int)ceilf((u_pix + r_u) / SLICE_TILE_SIZE);
    int max_ty = (int)ceilf((v_pix + r_v) / SLICE_TILE_SIZE);
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gauss) return;
    int offset = offsets[idx];
    int count = offsets[idx + 1] - offset;
    if (count <= 0) return;
    int4 r = rects[idx];
    uint32_t depth_key = sort_by_depth ? (uint32_t)((depths[idx] + 500.0f) * 1000.0f) : 0u;
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;
    uint32_t cur_tile = (uint32_t)(sorted_keys[idx] >> 32);
    if (idx == 0) {
        tile_ranges_start[cur_tile] = 0;
    } else {
        uint32_t prev_tile = (uint32_t)(sorted_keys[idx - 1] >> 32);
        if (cur_tile != prev_tile) {
            tile_ranges_end[prev_tile] = idx;
            tile_ranges_start[cur_tile] = idx;
        }
    }
    if (idx == num_instances - 1) tile_ranges_end[cur_tile] = num_instances;
}

__device__ __forceinline__ bool pixel_hits_gaussian_probe_window(
    const float3 p_vec,
    const float3 m_vec,
    const float* tri,
    const float3 axis_lat,
    const float3 axis_ax,
    const float3 axis_ele,
    const float chi2,
    const float beta
) {
    float r_lat, r_ax, r_ele;
    compute_probe_bbox_radii(tri, chi2, beta, axis_lat, axis_ax, axis_ele, r_lat, r_ax, r_ele);
    float3 d = p_vec - m_vec;
    float dl = fabsf(dot_prod(d, axis_lat));
    float da = fabsf(dot_prod(d, axis_ax));
    return dl <= r_lat && da <= r_ax;
}

__global__ void kernel_count_pixel_candidates(
    const int n_pixels,
    const int width,
    const int height,
    const float4* __restrict__ p_pos_packed,
    const float4* __restrict__ means_packed,
    const float* __restrict__ precision_tril,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ax,
    const float* __restrict__ ptr_vec_ele,
    const float chi2,
    const float beta,
    int* counts
) {
    __shared__ float s_lat[3], s_ax[3], s_ele[3];
    if (threadIdx.x == 0) {
        vec3_copy(ptr_vec_lat, s_lat);
        vec3_copy(ptr_vec_ax, s_ax);
        vec3_copy(ptr_vec_ele, s_ele);
    }
    __syncthreads();
    float3 axis_lat = make_float3(s_lat[0], s_lat[1], s_lat[2]);
    float3 axis_ax = make_float3(s_ax[0], s_ax[1], s_ax[2]);
    float3 axis_ele = make_float3(s_ele[0], s_ele[1], s_ele[2]);

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
    float3 pv = make_float3(p_pos_packed[p].x, p_pos_packed[p].y, p_pos_packed[p].z);
    int c = 0;
    for (int i = start; i < end; ++i) {
        int g = sorted_gauss_ids[i];
        float3 mv = make_float3(means_packed[g].x, means_packed[g].y, means_packed[g].z);
        if (pixel_hits_gaussian_probe_window(pv, mv, precision_tril + g * 6, axis_lat, axis_ax, axis_ele, chi2, beta)) c++;
    }
    counts[p] = c;
}

__global__ void kernel_fill_pixel_candidate_indices(
    const int n_pixels,
    const int width,
    const int height,
    const float4* __restrict__ p_pos_packed,
    const float4* __restrict__ means_packed,
    const float* __restrict__ precision_tril,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_ranges_start,
    const int* __restrict__ tile_ranges_end,
    const float* __restrict__ ptr_vec_lat,
    const float* __restrict__ ptr_vec_ax,
    const float* __restrict__ ptr_vec_ele,
    const float chi2,
    const float beta,
    const int* __restrict__ offsets,
    int* indices
) {
    __shared__ float s_lat[3], s_ax[3], s_ele[3];
    if (threadIdx.x == 0) {
        vec3_copy(ptr_vec_lat, s_lat);
        vec3_copy(ptr_vec_ax, s_ax);
        vec3_copy(ptr_vec_ele, s_ele);
    }
    __syncthreads();
    float3 axis_lat = make_float3(s_lat[0], s_lat[1], s_lat[2]);
    float3 axis_ax = make_float3(s_ax[0], s_ax[1], s_ax[2]);
    float3 axis_ele = make_float3(s_ele[0], s_ele[1], s_ele[2]);

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
    float3 pv = make_float3(p_pos_packed[p].x, p_pos_packed[p].y, p_pos_packed[p].z);
    int write_ptr = offsets[p];
    for (int i = start; i < end; ++i) {
        int g = sorted_gauss_ids[i];
        float3 mv = make_float3(means_packed[g].x, means_packed[g].y, means_packed[g].z);
        if (pixel_hits_gaussian_probe_window(pv, mv, precision_tril + g * 6, axis_lat, axis_ax, axis_ele, chi2, beta)) {
            indices[write_ptr++] = g;
        }
    }
}
