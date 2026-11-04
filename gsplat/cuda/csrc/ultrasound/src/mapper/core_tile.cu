#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

#include "mapper/core.cuh"
#include "shader/plane_cond2d_device.cuh"
#include "shader/shared.cuh"
#include "shader/slice_grid_geometry.cuh"
#include "shader/us_focus_device.cuh"

namespace {

template <typename T>
T copy_scalar_d2h(const T* device_ptr, cudaStream_t stream) {
    T host_v{};
    TORCH_CHECK(
        cudaMemcpyAsync(&host_v, device_ptr, sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess,
        "ours_tile_mapper D2H copy failed");
    TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "ours_tile_mapper stream sync failed");
    return host_v;
}

__global__ void kernel_ours_count_tile_rects(
    const int n_gauss,
    const int height,
    const int width,
    const int tile_size,
    const int grid_w_tiles,
    const int grid_h_tiles,
    const float lateral_spacing_mm_per_pix,
    const float axial_spacing_mm_per_pix,
    const float axial_near_mm,
    const float elev_corridor_half_mm,
    const float slab_half_thickness_mm,
    const float k_sigma,
    const int shader_weight_mode,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ pose,
    const float* __restrict__ slice_grid_pack,
    const float* __restrict__ us_focus_params,
    int* __restrict__ tiles_touched_count,
    int4* __restrict__ gauss_tile_rects,
    float* __restrict__ gauss_depth_pix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gauss) return;
    (void)slab_half_thickness_mm;

    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 raw_ele = make_float3(pose[2], pose[6], pose[10]);
    float3 axis_lat = ours_normalize3(raw_lat);
    float3 axis_ax = ours_normalize3(raw_ax);
    float3 axis_ele = ours_normalize3(raw_ele);
    float3 origin = make_float3(pose[3], pose[7], pose[11]);

    const float* mptr = gauss_means + idx * 3;
    const float* sptr = gauss_scales + idx * 3;
    const float* qptr = gauss_quats + idx * 4;

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

    float ex0 = sx * (Rg[0] * axis_lat.x + Rg[3] * axis_lat.y + Rg[6] * axis_lat.z);
    float ex1 = sy * (Rg[1] * axis_lat.x + Rg[4] * axis_lat.y + Rg[7] * axis_lat.z);
    float ex2 = sz * (Rg[2] * axis_lat.x + Rg[5] * axis_lat.y + Rg[8] * axis_lat.z);
    float ey0 = sx * (Rg[0] * axis_ax.x + Rg[3] * axis_ax.y + Rg[6] * axis_ax.z);
    float ey1 = sy * (Rg[1] * axis_ax.x + Rg[4] * axis_ax.y + Rg[7] * axis_ax.z);
    float ey2 = sz * (Rg[2] * axis_ax.x + Rg[5] * axis_ax.y + Rg[8] * axis_ax.z);
    float ez0 = sx * (Rg[0] * axis_ele.x + Rg[3] * axis_ele.y + Rg[6] * axis_ele.z);
    float ez1 = sy * (Rg[1] * axis_ele.x + Rg[4] * axis_ele.y + Rg[7] * axis_ele.z);
    float ez2 = sz * (Rg[2] * axis_ele.x + Rg[5] * axis_ele.y + Rg[8] * axis_ele.z);

    float r_ele = sqrtf(fmaxf(1e-12f, ez0 * ez0 + ez1 * ez1 + ez2 * ez2)) * k_sigma;
    float elev_pass = elev_corridor_half_mm + slab_half_thickness_mm + r_ele;
    if (shader_weight_mode != SHADER_WEIGHT_US_FOCUS && fabsf(mu_z) >= elev_pass) {
        tiles_touched_count[idx] = 0;
        gauss_tile_rects[idx] = make_int4(0, 0, 0, 0);
        gauss_depth_pix[idx] = 0.f;
        return;
    }

    float map_mu_lat = mu_x;
    float map_mu_ax = mu_y;
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
    } else if (shader_weight_mode == SHADER_WEIGHT_US_FOCUS) {
        us_focus_device::UsFocusFootprintOut fp;
        if (!us_focus_device::compute_us_focus_footprint_mm(
                mptr,
                sptr,
                qptr,
                axis_lat,
                axis_ax,
                axis_ele,
                origin,
                axial_near_mm,
                k_sigma,
                us_focus_params,
                &fp)) {
            tiles_touched_count[idx] = 0;
            gauss_tile_rects[idx] = make_int4(0, 0, 0, 0);
            gauss_depth_pix[idx] = 0.f;
            return;
        }
        map_mu_lat = fp.mu2_lat;
        map_mu_ax = fp.mu2_ax;
        r_lat = fp.r_lat_mm;
        r_ax = fp.r_ax_mm;
    }

    const float lat_col_offset = slice_grid_pack[SG_LAT_COL_OFFSET];
    float u_pix = map_mu_lat / lateral_spacing_mm_per_pix + lat_col_offset;
    float v_pix = (map_mu_ax - axial_near_mm) / axial_spacing_mm_per_pix;
    float r_lat_pix = fmaxf(1.0f, r_lat / fmaxf(1e-8f, lateral_spacing_mm_per_pix));
    float r_ax_pix = fmaxf(1.0f, r_ax / fmaxf(1e-8f, axial_spacing_mm_per_pix));

    float ts = (float)tile_size;
    int min_tx = (int)floorf((u_pix - r_lat_pix) / ts);
    int min_ty = (int)floorf((v_pix - r_ax_pix) / ts);
    int max_tx = (int)ceilf((u_pix + r_lat_pix) / ts);
    int max_ty = (int)ceilf((v_pix + r_ax_pix) / ts);

    min_tx = max(0, min_tx);
    min_ty = max(0, min_ty);
    max_tx = min(grid_w_tiles, max_tx);
    max_ty = min(grid_h_tiles, max_ty);

    if (min_tx >= max_tx || min_ty >= max_ty) {
        tiles_touched_count[idx] = 0;
        gauss_tile_rects[idx] = make_int4(0, 0, 0, 0);
        gauss_depth_pix[idx] = v_pix;
        return;
    }

    gauss_tile_rects[idx] = make_int4(min_tx, min_ty, max_tx, max_ty);
    gauss_depth_pix[idx] = v_pix;
    tiles_touched_count[idx] = (max_tx - min_tx) * (max_ty - min_ty);
}

__global__ void kernel_ours_emit_tile_keys(
    const int n_gauss,
    const int grid_width_tiles,
    const bool sort_by_v,
    const int4* __restrict__ rects,
    const float* __restrict__ depths,
    const int* __restrict__ offsets,
    uint64_t* __restrict__ keys_out,
    int* __restrict__ values_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gauss) return;

    int offset = offsets[idx];
    int count = offsets[idx + 1] - offset;
    if (count <= 0) return;

    int4 r = rects[idx];
    float d = depths[idx];

    uint32_t depth_key = 0u;
    if (sort_by_v) {
        depth_key = ours_v_to_depth_key(d);
    }

    for (int y = r.y; y < r.w; ++y) {
        for (int x = r.x; x < r.z; ++x) {
            uint64_t tile_id = (uint64_t)(y * grid_width_tiles + x);
            keys_out[offset] = (tile_id << 32) | (uint64_t)depth_key;
            values_out[offset] = idx;
            offset++;
        }
    }
}

__global__ void kernel_ours_identify_tile_ranges(
    const int num_instances,
    const uint64_t* __restrict__ sorted_keys,
    int* __restrict__ tile_ranges_start,
    int* __restrict__ tile_ranges_end) {
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

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> OursTileGaussMapperCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    const torch::Tensor& us_focus_params,
    int image_height,
    int image_width,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    int tile_size_px,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    bool sort_by_v,
    int shader_weight_mode) {
    TORCH_CHECK(gauss_means.is_cuda(), "OursTileGaussMapperCUDA expects CUDA tensors.");
    TORCH_CHECK(slice_grid_geometry.is_cuda(), "OursTileGaussMapperCUDA expects CUDA slice_grid_geometry.");
    TORCH_CHECK(
        shader_weight_mode != SHADER_WEIGHT_US_FOCUS || us_focus_params.is_cuda(),
        "OursTileGaussMapperCUDA expects CUDA us_focus_params for us_focus.");
    const int n_gauss = (int)gauss_means.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_l = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    const int grid_w = (image_width + tile_size_px - 1) / tile_size_px;
    const int grid_h = (image_height + tile_size_px - 1) / tile_size_px;
    const int n_tiles = grid_w * grid_h;

    if (n_gauss <= 0) {
        torch::Tensor empty_sorted = torch::empty({0}, opts_i);
        torch::Tensor range_start = torch::full({n_tiles}, -1, opts_i);
        torch::Tensor range_end = torch::full({n_tiles}, -1, opts_i);
        torch::Tensor empty_rects = torch::zeros({0, 4}, opts_i);
        return {empty_sorted, range_start, range_end, empty_rects};
    }

    torch::Tensor gauss_rects = torch::zeros({n_gauss, 4}, opts_i);
    torch::Tensor depths = torch::zeros({n_gauss}, opts_f);
    torch::Tensor tiles_touched = torch::zeros({n_gauss + 1}, opts_i);

    const int block = 256;
    const int grid = (n_gauss + block - 1) / block;

    kernel_ours_count_tile_rects<<<grid, block, 0, stream>>>(
        n_gauss,
        image_height,
        image_width,
        tile_size_px,
        grid_w,
        grid_h,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        elev_corridor_half_mm,
        slab_half_thickness_mm,
        k_sigma,
        shader_weight_mode,
        gauss_means.data_ptr<float>(),
        gauss_scales.data_ptr<float>(),
        gauss_quats.data_ptr<float>(),
        pose_probe_to_world.data_ptr<float>(),
        slice_grid_geometry.data_ptr<float>(),
        us_focus_params.numel() > 0 ? us_focus_params.data_ptr<float>() : nullptr,
        tiles_touched.data_ptr<int>(),
        reinterpret_cast<int4*>(gauss_rects.data_ptr<int>()),
        depths.data_ptr<float>());

    torch::Tensor instance_offsets = torch::empty({n_gauss + 1}, opts_i);
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, temp_bytes, tiles_touched.data_ptr<int>(), instance_offsets.data_ptr<int>(), n_gauss + 1, stream);
    torch::Tensor cub_tmp = torch::empty({(long)temp_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    d_temp = cub_tmp.data_ptr();
    cub::DeviceScan::ExclusiveSum(
        d_temp, temp_bytes, tiles_touched.data_ptr<int>(), instance_offsets.data_ptr<int>(), n_gauss + 1, stream);

    const int total_instances = copy_scalar_d2h<int>(instance_offsets.data_ptr<int>() + n_gauss, stream);

    torch::Tensor range_start = torch::full({n_tiles}, -1, opts_i);
    torch::Tensor range_end = torch::full({n_tiles}, -1, opts_i);

    if (total_instances <= 0 || n_gauss == 0) {
        torch::Tensor empty_sorted = torch::empty({0}, opts_i);
        return {empty_sorted, range_start, range_end, gauss_rects};
    }

    torch::Tensor keys1 = torch::empty({total_instances}, opts_l);
    torch::Tensor keys2 = torch::empty({total_instances}, opts_l);
    torch::Tensor vals1 = torch::empty({total_instances}, opts_i);
    torch::Tensor vals2 = torch::empty({total_instances}, opts_i);

    uint64_t* d_k_in = reinterpret_cast<uint64_t*>(keys1.data_ptr<int64_t>());
    uint64_t* d_k_out = reinterpret_cast<uint64_t*>(keys2.data_ptr<int64_t>());
    int* d_v_in = vals1.data_ptr<int>();
    int* d_v_out = vals2.data_ptr<int>();

    kernel_ours_emit_tile_keys<<<grid, block, 0, stream>>>(
        n_gauss,
        grid_w,
        sort_by_v,
        reinterpret_cast<const int4*>(gauss_rects.data_ptr<int>()),
        depths.data_ptr<float>(),
        instance_offsets.data_ptr<int>(),
        d_k_in,
        d_v_in);

    cub::DoubleBuffer<uint64_t> d_keys(d_k_in, d_k_out);
    cub::DoubleBuffer<int> d_vals(d_v_in, d_v_out);

    temp_bytes = 0;
    const int sort_begin_bit = sort_by_v ? 0 : 32;
    const int sort_end_bit = 64;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes, d_keys, d_vals, total_instances, sort_begin_bit, sort_end_bit, stream);
    if (cub_tmp.numel() < (long)temp_bytes) {
        cub_tmp = torch::empty({(long)temp_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        d_temp = cub_tmp.data_ptr();
    }
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes, d_keys, d_vals, total_instances, sort_begin_bit, sort_end_bit, stream);

    const int range_grid = (total_instances + block - 1) / block;
    kernel_ours_identify_tile_ranges<<<range_grid, block, 0, stream>>>(
        total_instances, d_keys.Current(), range_start.data_ptr<int>(), range_end.data_ptr<int>());

    torch::Tensor sorted_ids = torch::empty({total_instances}, opts_i);
    cudaMemcpyAsync(
        sorted_ids.data_ptr<int>(),
        d_vals.Current(),
        sizeof(int) * (size_t)total_instances,
        cudaMemcpyDeviceToDevice,
        stream);

    return {sorted_ids, range_start, range_end, gauss_rects};
}
