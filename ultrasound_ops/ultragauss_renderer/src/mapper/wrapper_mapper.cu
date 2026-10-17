#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>

#include "mapper/wrapper_mapper.cuh"
#include "mapper/core.cuh"

namespace {

void check_cuda_kernel(const char* name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "[ultragauss_renderer.mapper] ", name, " failed: ", cudaGetErrorString(err));
}

}

template <typename T>
T wait_and_copy_scalar_from_device(const T* device_ptr, cudaStream_t stream, const char* debug_name) {
    T host_value{};
    cudaEvent_t done_event = nullptr;
    TORCH_CHECK(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming) == cudaSuccess,
                "[ultragauss_renderer.mapper] failed to create event for ", debug_name);
    TORCH_CHECK(cudaMemcpyAsync(&host_value, device_ptr, sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess,
                "[ultragauss_renderer.mapper] failed to schedule D2H copy for ", debug_name);
    TORCH_CHECK(cudaEventRecord(done_event, stream) == cudaSuccess,
                "[ultragauss_renderer.mapper] failed to record event for ", debug_name);
    TORCH_CHECK(cudaEventSynchronize(done_event) == cudaSuccess,
                "[ultragauss_renderer.mapper] failed waiting on event for ", debug_name);
    TORCH_CHECK(cudaEventDestroy(done_event) == cudaSuccess,
                "[ultragauss_renderer.mapper] failed to destroy event for ", debug_name);
    return host_value;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MapperForwardCUDA(
    const torch::Tensor& p_pos_packed,
    const torch::Tensor& means_packed,
    const torch::Tensor& precision_tril,
    const torch::Tensor& probe_org,
    const torch::Tensor& vec_lat,
    const torch::Tensor& vec_ax,
    const torch::Tensor& vec_ele,
    const int image_height,
    const int image_width,
    const float delta_lat,
    const float delta_ax,
    const float near_mm,
    const float lateral_col_offset,
    const float chi2_threshold,
    const bool sort_by_depth,
    const float diag_beta
) {
    const int n_pixels = p_pos_packed.size(0);
    const int n_gauss = means_packed.size(0);
    auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto opts_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_long = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    if (n_pixels == 0 || n_gauss == 0 || image_width <= 0 || image_height <= 0) {
        return {torch::zeros({n_pixels}, opts_int), torch::zeros({n_pixels}, opts_int), torch::empty({0}, opts_int)};
    }
    TORCH_CHECK(n_pixels == image_height * image_width, "n_pixels must equal H*W");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int block_size = 256;
    int grid_size = (n_gauss + block_size - 1) / block_size;

    torch::Tensor tiles_touched_count = torch::zeros({n_gauss + 1}, opts_int);
    torch::Tensor gauss_tile_rects = torch::empty({n_gauss * 4}, opts_int);
    torch::Tensor gauss_axial_pix = torch::empty({n_gauss}, opts_float);
    kernel_project_gauss_and_compute_tile_rects<<<grid_size, block_size, 0, stream>>>(
        n_gauss,
        reinterpret_cast<float4*>(means_packed.data_ptr<float>()),
        precision_tril.data_ptr<float>(),
        image_height,
        image_width,
        near_mm,
        delta_lat,
        delta_ax,
        lateral_col_offset,
        chi2_threshold,
        diag_beta,
        probe_org.data_ptr<float>(),
        vec_lat.data_ptr<float>(),
        vec_ele.data_ptr<float>(),
        vec_ax.data_ptr<float>(),
        tiles_touched_count.data_ptr<int>(),
        reinterpret_cast<int4*>(gauss_tile_rects.data_ptr<int>()),
        gauss_axial_pix.data_ptr<float>()
    );
    check_cuda_kernel("kernel_project_gauss_and_compute_tile_rects");

    torch::Tensor gauss_instance_offsets = torch::empty({n_gauss + 1}, opts_int);
    torch::Tensor temp_storage;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
        tiles_touched_count.data_ptr<int>(), gauss_instance_offsets.data_ptr<int>(), n_gauss + 1, stream);
    temp_storage = torch::empty({(long)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    d_temp_storage = temp_storage.data_ptr();
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
        tiles_touched_count.data_ptr<int>(), gauss_instance_offsets.data_ptr<int>(), n_gauss + 1, stream);

    int total_mapping_instances = wait_and_copy_scalar_from_device<int>(
        gauss_instance_offsets.data_ptr<int>() + n_gauss, stream, "total_mapping_instances");
    if (total_mapping_instances == 0) {
        return {torch::zeros({n_pixels}, opts_int), torch::zeros({n_pixels}, opts_int), torch::empty({0}, opts_int)};
    }

    int grid_w = (image_width + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE;
    int grid_h = (image_height + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE;
    int n_tiles = grid_w * grid_h;
    torch::Tensor tile_ranges_start = torch::full({n_tiles}, -1, opts_int);
    torch::Tensor tile_ranges_end = torch::full({n_tiles}, -1, opts_int);

    torch::Tensor keys_buf_1 = torch::empty({total_mapping_instances}, opts_long);
    torch::Tensor keys_buf_2 = torch::empty({total_mapping_instances}, opts_long);
    torch::Tensor vals_buf_1 = torch::empty({total_mapping_instances}, opts_int);
    torch::Tensor vals_buf_2 = torch::empty({total_mapping_instances}, opts_int);
    uint64_t* d_keys_in = (uint64_t*)keys_buf_1.data_ptr<long>();
    uint64_t* d_keys_out = (uint64_t*)keys_buf_2.data_ptr<long>();
    int* d_vals_in = vals_buf_1.data_ptr<int>();
    int* d_vals_out = vals_buf_2.data_ptr<int>();

    kernel_generate_tile_depth_keys<<<grid_size, block_size, 0, stream>>>(
        n_gauss, grid_w, sort_by_depth,
        reinterpret_cast<int4*>(gauss_tile_rects.data_ptr<int>()),
        gauss_axial_pix.data_ptr<float>(),
        gauss_instance_offsets.data_ptr<int>(),
        d_keys_in, d_vals_in
    );
    check_cuda_kernel("kernel_generate_tile_depth_keys");

    cub::DoubleBuffer<uint64_t> d_keys(d_keys_in, d_keys_out);
    cub::DoubleBuffer<int> d_values(d_vals_in, d_vals_out);
    temp_storage_bytes = 0;
    const int sort_begin_bit = sort_by_depth ? 0 : 32;
    const int sort_end_bit = 64;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, d_keys, d_values, total_mapping_instances,
        sort_begin_bit, sort_end_bit, stream);
    if (temp_storage.numel() < temp_storage_bytes) {
        temp_storage = torch::empty({(long)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
        d_temp_storage = temp_storage.data_ptr();
    }
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, total_mapping_instances,
        sort_begin_bit, sort_end_bit, stream);

    int range_grid_size = (total_mapping_instances + block_size - 1) / block_size;
    kernel_identify_tile_ranges<<<range_grid_size, block_size, 0, stream>>>(
        total_mapping_instances, d_keys.Current(), tile_ranges_start.data_ptr<int>(), tile_ranges_end.data_ptr<int>());
    check_cuda_kernel("kernel_identify_tile_ranges");
    int* sorted_gauss_ids_ptr = d_values.Current();

    torch::Tensor pixel_gauss_counts = torch::zeros({n_pixels}, opts_int);
    torch::Tensor pixel_gauss_offsets = torch::zeros({n_pixels}, opts_int);
    dim3 block_dim(SLICE_TILE_SIZE, SLICE_TILE_SIZE);
    dim3 grid_dim(grid_w, grid_h);
    kernel_count_pixel_candidates<<<grid_dim, block_dim, 0, stream>>>(
        n_pixels, image_width, image_height,
        reinterpret_cast<float4*>(p_pos_packed.data_ptr<float>()),
        reinterpret_cast<float4*>(means_packed.data_ptr<float>()),
        precision_tril.data_ptr<float>(),
        sorted_gauss_ids_ptr,
        tile_ranges_start.data_ptr<int>(),
        tile_ranges_end.data_ptr<int>(),
        vec_lat.data_ptr<float>(),
        vec_ax.data_ptr<float>(),
        vec_ele.data_ptr<float>(),
        chi2_threshold,
        diag_beta,
        pixel_gauss_counts.data_ptr<int>());
    check_cuda_kernel("kernel_count_pixel_candidates");

    size_t scan_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_storage_bytes, pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_offsets.data_ptr<int>(), n_pixels, stream);
    if (temp_storage.numel() < scan_temp_storage_bytes) {
        temp_storage = torch::empty({(long)scan_temp_storage_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    }
    d_temp_storage = temp_storage.data_ptr();
    cub::DeviceScan::ExclusiveSum(d_temp_storage, scan_temp_storage_bytes, pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_offsets.data_ptr<int>(), n_pixels, stream);
    int last_pixel_offset = wait_and_copy_scalar_from_device<int>(pixel_gauss_offsets.data_ptr<int>() + (n_pixels - 1), stream, "last_pixel_offset");
    int last_pixel_count = wait_and_copy_scalar_from_device<int>(pixel_gauss_counts.data_ptr<int>() + (n_pixels - 1), stream, "last_pixel_count");
    long long total_pairs = (long long)last_pixel_offset + last_pixel_count;
    torch::Tensor pixel_gauss_indices = torch::empty({total_pairs}, opts_int);
    if (total_pairs > 0) {
        kernel_fill_pixel_candidate_indices<<<grid_dim, block_dim, 0, stream>>>(
            n_pixels, image_width, image_height,
            reinterpret_cast<float4*>(p_pos_packed.data_ptr<float>()),
            reinterpret_cast<float4*>(means_packed.data_ptr<float>()),
            precision_tril.data_ptr<float>(),
            sorted_gauss_ids_ptr,
            tile_ranges_start.data_ptr<int>(),
            tile_ranges_end.data_ptr<int>(),
                vec_lat.data_ptr<float>(),
                vec_ax.data_ptr<float>(),
                vec_ele.data_ptr<float>(),
                chi2_threshold,
            diag_beta,
            pixel_gauss_offsets.data_ptr<int>(),
            pixel_gauss_indices.data_ptr<int>());
        check_cuda_kernel("kernel_fill_pixel_candidate_indices");
    }
    return {pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices};
}
