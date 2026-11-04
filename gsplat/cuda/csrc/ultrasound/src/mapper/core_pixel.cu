#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

#include "mapper/core.cuh"
#include "shader/plane_cond2d_device.cuh"
#include "shader/shared.cuh"
#include "shader/slice_grid_geometry.cuh"

namespace {

template <typename T>
T copy_scalar_d2h_pixel(const T* device_ptr, cudaStream_t stream) {
    T host_v{};
    TORCH_CHECK(
        cudaMemcpyAsync(&host_v, device_ptr, sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess,
        "ours_pixel_mapper D2H copy failed");
    TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "ours_pixel_mapper stream sync failed");
    return host_v;
}

__global__ void kernel_fill_v_pix_depths(
    int n_gauss,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    const float* __restrict__ gauss_means,
    const float* __restrict__ pose,
    const int* __restrict__ hits_count,
    float* __restrict__ depths) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) return;
    if (hits_count[g] <= 0) {
        depths[g] = 0.f;
        return;
    }
    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 axis_lat = ours_normalize3(raw_lat);
    float3 axis_ax = ours_normalize3(raw_ax);
    float3 origin = make_float3(pose[3], pose[7], pose[11]);
    const float* mptr = gauss_means + g * 3;
    float3 mw = make_float3(mptr[0], mptr[1], mptr[2]);
    float3 d = make_float3(mw.x - origin.x, mw.y - origin.y, mw.z - origin.z);
    float mu_x = ours_dot3(d, axis_lat);
    float mu_y = ours_dot3(d, axis_ax);
    (void)mu_x;
    depths[g] = (mu_y - axial_near_mm) / axial_spacing_mm_per_pix;
}

__global__ void kernel_count_bucket_hits_per_gauss(
    const int n_gauss,
    const int H,
    const int W,
    const float lateral_spacing_mm_per_pix,
    const float axial_spacing_mm_per_pix,
    const float axial_near_mm,
    const float elev_corridor_half_mm,
    const float slab_half_thickness_mm,
    const float k_sigma,
    const int shader_weight_mode,
    const int num_pixels_lateral,
    const int num_pixels_axial,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ pose,
    const float* __restrict__ slice_grid_pack,
    int* __restrict__ hits_count,
    int4* __restrict__ gauss_pixel_rect) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) return;

    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 raw_ele = make_float3(pose[2], pose[6], pose[10]);
    float3 axis_lat = ours_normalize3(raw_lat);
    float3 axis_ax = ours_normalize3(raw_ax);
    float3 axis_ele = ours_normalize3(raw_ele);
    float3 origin = make_float3(pose[3], pose[7], pose[11]);

    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;

    float3 mw = make_float3(mptr[0], mptr[1], mptr[2]);
    float3 d = make_float3(mw.x - origin.x, mw.y - origin.y, mw.z - origin.z);
    float mu_x = ours_dot3(d, axis_lat);
    float mu_y = ours_dot3(d, axis_ax);
    float mu_z = ours_dot3(d, axis_ele);

    float Rg[9];
    ours_quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
    float sx = fmaxf(sptr[0], 1e-6f);
    float sy = fmaxf(sptr[1], 1e-6f);
    float sz = fmaxf(sptr[2], 1e-6f);

    float ez0 = sx * (Rg[0] * axis_ele.x + Rg[3] * axis_ele.y + Rg[6] * axis_ele.z);
    float ez1 = sy * (Rg[1] * axis_ele.x + Rg[4] * axis_ele.y + Rg[7] * axis_ele.z);
    float ez2 = sz * (Rg[2] * axis_ele.x + Rg[5] * axis_ele.y + Rg[8] * axis_ele.z);
    float r_ele = sqrtf(fmaxf(1e-12f, ez0 * ez0 + ez1 * ez1 + ez2 * ez2)) * k_sigma;
    float elev_pass = elev_corridor_half_mm + slab_half_thickness_mm + r_ele;
    if (fabsf(mu_z) >= elev_pass) {
        hits_count[g] = 0;
        gauss_pixel_rect[g] = make_int4(0, 0, 0, 0);
        return;
    }

    float map_mu_lat = mu_x;
    float map_mu_ax = mu_y;

    float ex0 = sx * (Rg[0] * axis_lat.x + Rg[3] * axis_lat.y + Rg[6] * axis_lat.z);
    float ex1 = sy * (Rg[1] * axis_lat.x + Rg[4] * axis_lat.y + Rg[7] * axis_lat.z);
    float ex2 = sz * (Rg[2] * axis_lat.x + Rg[5] * axis_lat.y + Rg[8] * axis_lat.z);
    float ey0 = sx * (Rg[0] * axis_ax.x + Rg[3] * axis_ax.y + Rg[6] * axis_ax.z);
    float ey1 = sy * (Rg[1] * axis_ax.x + Rg[4] * axis_ax.y + Rg[7] * axis_ax.z);
    float ey2 = sz * (Rg[2] * axis_ax.x + Rg[5] * axis_ax.y + Rg[8] * axis_ax.z);
    float r_lat = sqrtf(fmaxf(1e-12f, ex0 * ex0 + ex1 * ex1 + ex2 * ex2)) * k_sigma;
    float r_ax = sqrtf(fmaxf(1e-12f, ey0 * ey0 + ey1 * ey1 + ey2 * ey2)) * k_sigma;
    if (shader_weight_mode == SHADER_WEIGHT_PLANE_COND2D) {
        float mu2_lat = 0.f;
        float mu2_ax = 0.f;
        float r2_lat = 0.f;
        float r2_ax = 0.f;
        if (plane_cond2d_device::plane_cond2d_mapper_footprint_mm(
                mptr,
                sptr,
                qptr,
                axis_lat,
                axis_ax,
                axis_ele,
                origin,
                k_sigma,
                &mu2_lat,
                &mu2_ax,
                &r2_lat,
                &r2_ax)) {
            map_mu_lat = mu2_lat;
            map_mu_ax = mu2_ax;
            r_lat = r2_lat;
            r_ax = r2_ax;
        }
    }

    const float lat_col_offset = slice_grid_pack[SG_LAT_COL_OFFSET];
    float u_pix = map_mu_lat / lateral_spacing_mm_per_pix + lat_col_offset;
    float v_pix = (map_mu_ax - axial_near_mm) / axial_spacing_mm_per_pix;
    float r_lat_pix = fmaxf(1.0f, r_lat / fmaxf(1e-8f, lateral_spacing_mm_per_pix));
    float r_ax_pix = fmaxf(1.0f, r_ax / fmaxf(1e-8f, axial_spacing_mm_per_pix));
    (void)num_pixels_lateral;
    (void)num_pixels_axial;

    int x0 = max(0, (int)floorf(u_pix - r_lat_pix));
    int x1 = min(W - 1, (int)ceilf(u_pix + r_lat_pix));
    int y0 = max(0, (int)floorf(v_pix - r_ax_pix));
    int y1 = min(H - 1, (int)ceilf(v_pix + r_ax_pix));
    if (x0 > x1 || y0 > y1) {
        hits_count[g] = 0;
        gauss_pixel_rect[g] = make_int4(0, 0, 0, 0);
        return;
    }
    // backward: exclusive max pixel index +1 for [x0,x1] inclusive
    gauss_pixel_rect[g] = make_int4(x0, y0, x1 + 1, y1 + 1);
    hits_count[g] = (x1 - x0 + 1) * (y1 - y0 + 1);
}

__global__ void kernel_emit_pixel_keys(
    const int n_gauss,
    const int W,
    const bool sort_by_v,
    const int* __restrict__ hits_offsets,
    const int* __restrict__ hits_count,
    const int4* __restrict__ rects,
    const float* __restrict__ depths,
    uint64_t* __restrict__ keys_out,
    int* __restrict__ vals_out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) return;
    int cnt = hits_count[g];
    int off = hits_offsets[g];
    if (cnt <= 0) return;

    int4 r = rects[g];
    float v_depth = depths[g];
    uint32_t vk = sort_by_v ? ours_v_to_depth_key(v_depth) : 0u;

    int k = 0;
    for (int y = r.y; y < r.w; ++y) {
        for (int x = r.x; x < r.z; ++x) {
            int pid = y * W + x;
            uint64_t key = ((uint64_t)(uint32_t)pid << 32) | (uint64_t)vk;
            keys_out[off + k] = key;
            vals_out[off + k] = g;
            ++k;
        }
    }
}

__global__ void kernel_identify_pixel_key_ranges(
    const int num_instances,
    const int n_pixels,
    const uint64_t* __restrict__ sorted_keys,
    int* __restrict__ range_start,
    int* __restrict__ range_end) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;

    uint32_t cur = (uint32_t)(sorted_keys[idx] >> 32);
    if (idx == 0) {
        range_start[cur] = 0;
    } else {
        uint32_t prev = (uint32_t)(sorted_keys[idx - 1] >> 32);
        if (cur != prev) {
            range_end[prev] = idx;
            range_start[cur] = idx;
        }
    }
    if (idx == num_instances - 1) {
        range_end[cur] = num_instances;
    }
    (void)n_pixels;
}

__global__ void kernel_fill_pixel_counts_from_ranges(
    const int n_pixels,
    const int* __restrict__ range_start,
    const int* __restrict__ range_end,
    int* __restrict__ counts) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    int s = range_start[p];
    if (s < 0) {
        counts[p] = 0;
        return;
    }
    int e = range_end[p];
    counts[p] = max(0, e - s);
}

__global__ void kernel_scatter_sorted_vals_to_csr(
    const int n_pixels,
    const int* __restrict__ range_start,
    const int* __restrict__ range_end,
    const int* __restrict__ pixel_offsets,
    const int* __restrict__ sorted_vals,
    int* __restrict__ indices) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    int s = range_start[p];
    if (s < 0) return;
    int e = range_end[p];
    int o = pixel_offsets[p];
    for (int j = s; j < e; ++j) {
        indices[o + (j - s)] = sorted_vals[j];
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> OursPixelGaussMapperCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    int image_height,
    int image_width,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    int num_pixels_lateral,
    int num_pixels_axial,
    bool sort_by_v,
    int shader_weight_mode) {
    TORCH_CHECK(gauss_means.is_cuda(), "OursPixelGaussMapperCUDA expects CUDA tensors.");
    TORCH_CHECK(slice_grid_geometry.is_cuda(), "OursPixelGaussMapperCUDA expects CUDA slice_grid_geometry.");
    const int n_gauss = (int)gauss_means.size(0);
    const int n_pixels = image_height * image_width;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_l = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    torch::Tensor empty_rects = torch::zeros({0, 4}, opts_i);
    if (n_gauss <= 0 || n_pixels <= 0) {
        torch::Tensor zc = torch::zeros({n_pixels}, opts_i);
        torch::Tensor zo = torch::zeros({n_pixels}, opts_i);
        torch::Tensor ei = torch::empty({0}, opts_i);
        return {zc, zo, ei, empty_rects};
    }

    torch::Tensor hits_count = torch::zeros({n_gauss}, opts_i);
    torch::Tensor gauss_pixel_rect = torch::zeros({n_gauss, 4}, opts_i);
    torch::Tensor depths = torch::zeros({n_gauss}, opts_f);

    const int block = 256;
    const int grid = (n_gauss + block - 1) / block;

    // depths = v_pix for sorting (reuse kernel path: compute v in separate small kernel or inline - emit reads mu)
    // Precompute depths on host from same geometry in count kernel - duplicate: add v_pix write in count kernel
    // Simpler: second kernel to fill depths from gauss centers - skip, embed in count kernel by writing depths[g]=v_pix

    // Patch: add depth output to count kernel - I'll launch a tiny kernel after count to fill depths from rect centers
    // Actually use same kernel - add float* depths param and set depths[g]=v_pix when hits>0

    // Re-launch: replace kernel with version that writes depths - edit file inline below by rewriting count kernel call

    kernel_count_bucket_hits_per_gauss<<<grid, block, 0, stream>>>(
        n_gauss,
        image_height,
        image_width,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        elev_corridor_half_mm,
        slab_half_thickness_mm,
        k_sigma,
        shader_weight_mode,
        num_pixels_lateral,
        num_pixels_axial,
        gauss_means.data_ptr<float>(),
        gauss_scales.data_ptr<float>(),
        gauss_quats.data_ptr<float>(),
        pose_probe_to_world.data_ptr<float>(),
        slice_grid_geometry.data_ptr<float>(),
        hits_count.data_ptr<int>(),
        reinterpret_cast<int4*>(gauss_pixel_rect.data_ptr<int>()));

    kernel_fill_v_pix_depths<<<grid, block, 0, stream>>>(
        n_gauss,
        image_width,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        gauss_means.data_ptr<float>(),
        pose_probe_to_world.data_ptr<float>(),
        hits_count.data_ptr<int>(),
        depths.data_ptr<float>());

    torch::Tensor hits_prefix = torch::empty({n_gauss + 1}, opts_i);
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, temp_bytes, hits_count.data_ptr<int>(), hits_prefix.data_ptr<int>(), n_gauss + 1, stream);
    torch::Tensor cub_tmp = torch::empty({(long)temp_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    d_temp = cub_tmp.data_ptr();
    cub::DeviceScan::ExclusiveSum(
        d_temp, temp_bytes, hits_count.data_ptr<int>(), hits_prefix.data_ptr<int>(), n_gauss + 1, stream);

    const int total_pairs = copy_scalar_d2h_pixel<int>(hits_prefix.data_ptr<int>() + n_gauss, stream);
    if (total_pairs <= 0) {
        torch::Tensor zc = torch::zeros({n_pixels}, opts_i);
        torch::Tensor zo = torch::zeros({n_pixels}, opts_i);
        torch::Tensor ei = torch::empty({0}, opts_i);
        return {zc, zo, ei, gauss_pixel_rect};
    }

    torch::Tensor keys1 = torch::empty({total_pairs}, opts_l);
    torch::Tensor keys2 = torch::empty({total_pairs}, opts_l);
    torch::Tensor vals1 = torch::empty({total_pairs}, opts_i);
    torch::Tensor vals2 = torch::empty({total_pairs}, opts_i);

    kernel_emit_pixel_keys<<<grid, block, 0, stream>>>(
        n_gauss,
        image_width,
        sort_by_v,
        hits_prefix.data_ptr<int>(),
        hits_count.data_ptr<int>(),
        reinterpret_cast<const int4*>(gauss_pixel_rect.data_ptr<int>()),
        depths.data_ptr<float>(),
        reinterpret_cast<uint64_t*>(keys1.data_ptr<int64_t>()),
        vals1.data_ptr<int>());

    uint64_t* d_k_in = reinterpret_cast<uint64_t*>(keys1.data_ptr<int64_t>());
    uint64_t* d_k_out = reinterpret_cast<uint64_t*>(keys2.data_ptr<int64_t>());
    int* d_v_in = vals1.data_ptr<int>();
    int* d_v_out = vals2.data_ptr<int>();

    cub::DoubleBuffer<uint64_t> d_keys(d_k_in, d_k_out);
    cub::DoubleBuffer<int> d_vals(d_v_in, d_v_out);
    temp_bytes = 0;
    const int sort_begin_bit = sort_by_v ? 0 : 32;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes, d_keys, d_vals, total_pairs, sort_begin_bit, 64, stream);
    if (cub_tmp.numel() < (long)temp_bytes) {
        cub_tmp = torch::empty({(long)temp_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        d_temp = cub_tmp.data_ptr();
    }
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes, d_keys, d_vals, total_pairs, sort_begin_bit, 64, stream);

    torch::Tensor range_start = torch::full({n_pixels}, -1, opts_i);
    torch::Tensor range_end = torch::full({n_pixels}, -1, opts_i);
    const int rg = (total_pairs + block - 1) / block;
    kernel_identify_pixel_key_ranges<<<rg, block, 0, stream>>>(
        total_pairs, n_pixels, d_keys.Current(), range_start.data_ptr<int>(), range_end.data_ptr<int>());

    torch::Tensor pixel_gauss_counts = torch::zeros({n_pixels}, opts_i);
    const int pg = (n_pixels + block - 1) / block;
    kernel_fill_pixel_counts_from_ranges<<<pg, block, 0, stream>>>(
        n_pixels, range_start.data_ptr<int>(), range_end.data_ptr<int>(), pixel_gauss_counts.data_ptr<int>());

    torch::Tensor pixel_gauss_offsets = torch::zeros({n_pixels}, opts_i);
    temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, temp_bytes, pixel_gauss_counts.data_ptr<int>(), pixel_gauss_offsets.data_ptr<int>(), n_pixels, stream);
    if (cub_tmp.numel() < (long)temp_bytes) {
        cub_tmp = torch::empty({(long)temp_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        d_temp = cub_tmp.data_ptr();
    }
    cub::DeviceScan::ExclusiveSum(
        d_temp, temp_bytes, pixel_gauss_counts.data_ptr<int>(), pixel_gauss_offsets.data_ptr<int>(), n_pixels, stream);

    int last_off = 0, last_cnt = 0;
    cudaMemcpyAsync(&last_off, pixel_gauss_offsets.data_ptr<int>() + (n_pixels - 1), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_cnt, pixel_gauss_counts.data_ptr<int>() + (n_pixels - 1), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    long long nnz = (long long)last_off + last_cnt;
    torch::Tensor pixel_gauss_indices = torch::empty({nnz}, opts_i);

    torch::Tensor sorted_vals = torch::empty({total_pairs}, opts_i);
    cudaMemcpyAsync(
        sorted_vals.data_ptr<int>(),
        d_vals.Current(),
        sizeof(int) * (size_t)total_pairs,
        cudaMemcpyDeviceToDevice,
        stream);

    kernel_scatter_sorted_vals_to_csr<<<pg, block, 0, stream>>>(
        n_pixels,
        range_start.data_ptr<int>(),
        range_end.data_ptr<int>(),
        pixel_gauss_offsets.data_ptr<int>(),
        sorted_vals.data_ptr<int>(),
        pixel_gauss_indices.data_ptr<int>());

    return {pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices, gauss_pixel_rect};
}
