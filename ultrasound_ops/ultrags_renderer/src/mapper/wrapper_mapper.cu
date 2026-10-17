#include <algorithm>

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>

#include "mapper/wrapper_mapper.cuh"
#include "mapper/core.cuh"

template <typename T>
T copy_scalar_from_device_and_sync(const T* device_ptr, cudaStream_t stream, const char* debug_name) {
    T host_value{};
    TORCH_CHECK(cudaMemcpyAsync(&host_value, device_ptr, sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess,
                "[ultrags_renderer.mapper] failed to schedule D2H copy for ", debug_name);
    TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess,
                "[ultrags_renderer.mapper] failed waiting stream for ", debug_name);
    return host_value;
}

__global__ void kernel_extract_batch_totals(
    const int* __restrict__ gauss_instance_offsets,
    const int batch_size,
    const int n_gauss,
    int* __restrict__ batch_totals
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    const int row_stride = n_gauss + 1;
    batch_totals[b] = gauss_instance_offsets[b * row_stride + n_gauss];
}

__global__ void kernel_add_batch_prefix_to_offsets(
    int* __restrict__ gauss_instance_offsets,
    const int batch_size,
    const int n_gauss,
    const int* __restrict__ batch_prefix
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (b >= batch_size || g > n_gauss) return;
    const int row_stride = n_gauss + 1;
    gauss_instance_offsets[b * row_stride + g] += batch_prefix[b];
}

__global__ void kernel_fill_pair_batch_offsets(
    const int* __restrict__ pixel_gauss_offsets,
    const int batch_size,
    const int pixels_per_image,
    const int64_t total_pairs,
    int64_t* __restrict__ pair_batch_offsets
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    pair_batch_offsets[b] = static_cast<int64_t>(pixel_gauss_offsets[b * pixels_per_image]);
    if (b == batch_size - 1) {
        pair_batch_offsets[batch_size] = total_pairs;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MapperForwardCUDA(
    const torch::Tensor& p_pos_packed,
    const torch::Tensor& means_packed,
    const torch::Tensor& quats,
    const torch::Tensor& scales_packed,
    const torch::Tensor& probe_org,
    const torch::Tensor& vec_ele,
    const torch::Tensor& vec_lat,
    const torch::Tensor& vec_ax,
    const int image_height,
    const int image_width,
    const float delta_lat,
    const float delta_ax,
    const float slab_half_thickness_mm,
    const float near_mm,
    const float k_sigma,
    const float lateral_col_offset,
    const bool sort_by_depth
) {
    const int n_pixels = p_pos_packed.size(0);
    const int n_gauss = means_packed.size(0);

    auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto opts_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_long = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    if (n_pixels == 0 || n_gauss == 0 || image_width <= 0 || image_height <= 0) {
        torch::Tensor pixel_gauss_counts = torch::zeros({n_pixels}, opts_int);
        torch::Tensor pixel_gauss_offsets = torch::zeros({n_pixels}, opts_int);
        torch::Tensor pixel_gauss_indices = torch::empty({0}, opts_int);
        return {pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices};
    }

    TORCH_CHECK(
        n_pixels == image_height * image_width,
        "[ultrags_renderer.mapper] n_pixels mismatch: got n_pixels=", n_pixels,
        ", but H*W=", image_height, "*", image_width, "=", image_height * image_width
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor temp_storage;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    torch::Tensor projected_radii = torch::empty({n_gauss, 3}, opts_float);
    int block_size = 256;
    int grid_size = (n_gauss + block_size - 1) / block_size;

    kernel_precompute_gauss_projected_radii<<<grid_size, block_size, 0, stream>>>(
        n_gauss, k_sigma,
        reinterpret_cast<float4*>(quats.data_ptr<float>()),
        reinterpret_cast<float4*>(scales_packed.data_ptr<float>()),
        vec_lat.data_ptr<float>(),
        vec_ele.data_ptr<float>(),
        vec_ax.data_ptr<float>(),
        reinterpret_cast<float3*>(projected_radii.data_ptr<float>())
    );

    torch::Tensor tiles_touched_count = torch::zeros({n_gauss + 1}, opts_int);
    torch::Tensor gauss_tile_rects = torch::empty({n_gauss * 4}, opts_int);
    torch::Tensor gauss_axial_pix = torch::empty({n_gauss}, opts_float);

    kernel_project_gauss_and_compute_tile_rects<<<grid_size, block_size, 0, stream>>>(
        n_gauss,
        reinterpret_cast<float4*>(means_packed.data_ptr<float>()),
        reinterpret_cast<float3*>(projected_radii.data_ptr<float>()),
        image_height, image_width, near_mm,
        delta_lat, delta_ax, slab_half_thickness_mm,
        lateral_col_offset,
        probe_org.data_ptr<float>(),
        vec_lat.data_ptr<float>(),
        vec_ele.data_ptr<float>(),
        vec_ax.data_ptr<float>(),
        tiles_touched_count.data_ptr<int>(),
        reinterpret_cast<int4*>(gauss_tile_rects.data_ptr<int>()),
        gauss_axial_pix.data_ptr<float>()
    );

    torch::Tensor gauss_instance_offsets = torch::empty({n_gauss + 1}, opts_int);
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
        tiles_touched_count.data_ptr<int>(), gauss_instance_offsets.data_ptr<int>(), n_gauss + 1, stream);
    temp_storage = torch::empty({(long)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    d_temp_storage = temp_storage.data_ptr();
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
        tiles_touched_count.data_ptr<int>(), gauss_instance_offsets.data_ptr<int>(), n_gauss + 1, stream);

    int total_mapping_instances = copy_scalar_from_device_and_sync<int>(
        gauss_instance_offsets.data_ptr<int>() + n_gauss,
        stream,
        "total_mapping_instances"
    );

    if (total_mapping_instances == 0) {
        torch::Tensor pixel_gauss_counts = torch::zeros({n_pixels}, opts_int);
        torch::Tensor pixel_gauss_offsets = torch::zeros({n_pixels}, opts_int);
        torch::Tensor pixel_gauss_indices = torch::empty({0}, opts_int);
        return {pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices};
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
        n_gauss, grid_w,
        sort_by_depth,
        reinterpret_cast<int4*>(gauss_tile_rects.data_ptr<int>()),
        gauss_axial_pix.data_ptr<float>(),
        gauss_instance_offsets.data_ptr<int>(),
        d_keys_in, d_vals_in
    );

    cub::DoubleBuffer<uint64_t> d_keys(d_keys_in, d_keys_out);
    cub::DoubleBuffer<int> d_values(d_vals_in, d_vals_out);

    temp_storage_bytes = 0;
    const int sort_begin_bit = sort_by_depth ? 0 : 32;
    const int sort_end_bit = 64;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes, d_keys, d_values, total_mapping_instances,
        sort_begin_bit, sort_end_bit, stream
    );
    if (temp_storage.numel() < temp_storage_bytes) {
        temp_storage = torch::empty({(long)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
        d_temp_storage = temp_storage.data_ptr();
    }
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, d_keys, d_values, total_mapping_instances,
        sort_begin_bit, sort_end_bit, stream
    );

    int range_grid_size = (total_mapping_instances + block_size - 1) / block_size;
    kernel_identify_tile_ranges<<<range_grid_size, block_size, 0, stream>>>(
        total_mapping_instances,
        d_keys.Current(),
        tile_ranges_start.data_ptr<int>(),
        tile_ranges_end.data_ptr<int>()
    );

    int* sorted_gauss_ids_ptr = d_values.Current();

    torch::Tensor pixel_gauss_counts = torch::zeros({n_pixels}, opts_int);
    torch::Tensor pixel_gauss_offsets = torch::zeros({n_pixels}, opts_int);
    dim3 block_dim(SLICE_TILE_SIZE, SLICE_TILE_SIZE);
    dim3 grid_dim(grid_w, grid_h);

    kernel_count_pixel_candidates<<<grid_dim, block_dim, 0, stream>>>(
        n_pixels, image_width, image_height,
        reinterpret_cast<float4*>(p_pos_packed.data_ptr<float>()),
        reinterpret_cast<float4*>(means_packed.data_ptr<float>()),
        reinterpret_cast<float3*>(projected_radii.data_ptr<float>()),
        sorted_gauss_ids_ptr,
        tile_ranges_start.data_ptr<int>(),
        tile_ranges_end.data_ptr<int>(),
        vec_lat.data_ptr<float>(),
        vec_ax.data_ptr<float>(),
        pixel_gauss_counts.data_ptr<int>()
    );

    size_t scan_pixel_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        scan_pixel_bytes,
        pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_offsets.data_ptr<int>(),
        n_pixels,
        stream
    );
    if (scan_pixel_bytes > temp_storage_bytes) {
        temp_storage_bytes = scan_pixel_bytes;
        temp_storage = torch::empty(
            {(long)temp_storage_bytes},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
        );
        d_temp_storage = temp_storage.data_ptr();
    }
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_offsets.data_ptr<int>(),
        n_pixels,
        stream
    );

    int last_pixel_offset = 0;
    int last_pixel_count = 0;
    TORCH_CHECK(
        cudaMemcpyAsync(
            &last_pixel_offset,
            pixel_gauss_offsets.data_ptr<int>() + (n_pixels - 1),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream
        ) == cudaSuccess,
        "[ultrags_renderer.mapper] failed to schedule D2H copy for last_pixel_offset"
    );
    TORCH_CHECK(
        cudaMemcpyAsync(
            &last_pixel_count,
            pixel_gauss_counts.data_ptr<int>() + (n_pixels - 1),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream
        ) == cudaSuccess,
        "[ultrags_renderer.mapper] failed to schedule D2H copy for last_pixel_count"
    );
    TORCH_CHECK(
        cudaStreamSynchronize(stream) == cudaSuccess,
        "[ultrags_renderer.mapper] failed waiting stream for last pixel stats"
    );

    long long total_pairs = (long long)last_pixel_offset + last_pixel_count;
    torch::Tensor pixel_gauss_indices = torch::empty({total_pairs}, opts_int);

    if (total_pairs > 0) {
        kernel_fill_pixel_candidate_indices<<<grid_dim, block_dim, 0, stream>>>(
            n_pixels, image_width, image_height,
            reinterpret_cast<float4*>(p_pos_packed.data_ptr<float>()),
            reinterpret_cast<float4*>(means_packed.data_ptr<float>()),
            reinterpret_cast<float3*>(projected_radii.data_ptr<float>()),
            sorted_gauss_ids_ptr,
            tile_ranges_start.data_ptr<int>(),
            tile_ranges_end.data_ptr<int>(),
            vec_lat.data_ptr<float>(),
            vec_ax.data_ptr<float>(),
            pixel_gauss_offsets.data_ptr<int>(),
            pixel_gauss_indices.data_ptr<int>()
        );
    }

    return {pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MapperForwardCUDABatched(
    const torch::Tensor& p_pos_packed,
    const torch::Tensor& means_packed,
    const torch::Tensor& quats,
    const torch::Tensor& scales_packed,
    const torch::Tensor& probe_org,
    const torch::Tensor& vec_ele,
    const torch::Tensor& vec_lat,
    const torch::Tensor& vec_ax,
    const int image_height,
    const int image_width,
    const float delta_lat,
    const float delta_ax,
    const float slab_half_thickness_mm,
    const float near_mm,
    const float k_sigma,
    const float lateral_col_offset,
    const bool sort_by_depth,
    const torch::Tensor& active_gauss_mask_batched,
    const bool return_pair_batch_offsets
) {
    const int batch_size = static_cast<int>(p_pos_packed.size(0));
    const int pixels_per_image = static_cast<int>(p_pos_packed.size(1));
    const int n_gauss = static_cast<int>(means_packed.size(0));

    auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto opts_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_long = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    auto opts_long_cuda = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor pair_batch_offsets = return_pair_batch_offsets
        ? torch::zeros({batch_size + 1}, opts_long_cuda)
        : torch::empty({0}, opts_long_cuda);

    const int n_total_pixels = batch_size * pixels_per_image;
    const bool has_active_mask = active_gauss_mask_batched.defined() && active_gauss_mask_batched.numel() > 0;
    const uint8_t* active_mask_ptr = nullptr;
    if (has_active_mask) {
        TORCH_CHECK(
            active_gauss_mask_batched.is_cuda(),
            "[ultrags_renderer.mapper_batched] active_gauss_mask_batched must be CUDA tensor"
        );
        TORCH_CHECK(
            active_gauss_mask_batched.dtype() == torch::kUInt8,
            "[ultrags_renderer.mapper_batched] active_gauss_mask_batched must be uint8"
        );
        TORCH_CHECK(
            active_gauss_mask_batched.dim() == 2
                && active_gauss_mask_batched.size(0) == batch_size
                && active_gauss_mask_batched.size(1) == n_gauss,
            "[ultrags_renderer.mapper_batched] active_gauss_mask_batched shape must be [B, G]"
        );
        TORCH_CHECK(
            active_gauss_mask_batched.is_contiguous(),
            "[ultrags_renderer.mapper_batched] active_gauss_mask_batched must be contiguous"
        );
        active_mask_ptr = active_gauss_mask_batched.data_ptr<uint8_t>();
    }

    if (batch_size <= 0 || pixels_per_image <= 0 || n_gauss == 0 || image_width <= 0 || image_height <= 0) {
        const int npx = std::max(0, n_total_pixels);
        torch::Tensor pixel_gauss_counts = torch::zeros({npx}, opts_int);
        torch::Tensor pixel_gauss_offsets = torch::zeros({npx}, opts_int);
        torch::Tensor pixel_gauss_indices = torch::empty({0}, opts_int);
        return {pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices, pair_batch_offsets};
    }

    TORCH_CHECK(
        pixels_per_image == image_height * image_width,
        "[ultrags_renderer.mapper_batched] pixels_per_image mismatch H*W"
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor temp_storage;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    torch::Tensor projected_radii_batched = torch::empty({batch_size, n_gauss, 3}, opts_float);

    dim3 block_size_pre(256, 1, 1);
    dim3 grid_size_pre((n_gauss + block_size_pre.x - 1) / block_size_pre.x, batch_size, 1);

    kernel_precompute_gauss_projected_radii_batched<<<grid_size_pre, block_size_pre, 0, stream>>>(
        n_gauss,
        batch_size,
        k_sigma,
        reinterpret_cast<float4*>(quats.data_ptr<float>()),
        reinterpret_cast<float4*>(scales_packed.data_ptr<float>()),
        vec_lat.data_ptr<float>(),
        vec_ele.data_ptr<float>(),
        vec_ax.data_ptr<float>(),
        active_mask_ptr,
        reinterpret_cast<float3*>(projected_radii_batched.data_ptr<float>())
    );

    torch::Tensor tiles_touched_count =
        torch::zeros({batch_size, n_gauss + 1}, opts_int);
    torch::Tensor gauss_tile_rects = torch::empty({batch_size * n_gauss, 4}, opts_int);
    torch::Tensor gauss_axial_pix = torch::empty({batch_size, n_gauss}, opts_float);

    kernel_project_gauss_and_compute_tile_rects_batched<<<grid_size_pre, block_size_pre, 0, stream>>>(
        n_gauss,
        batch_size,
        reinterpret_cast<float4*>(means_packed.data_ptr<float>()),
        reinterpret_cast<float3*>(projected_radii_batched.data_ptr<float>()),
        image_height,
        image_width,
        near_mm,
        delta_lat,
        delta_ax,
        slab_half_thickness_mm,
        lateral_col_offset,
        probe_org.data_ptr<float>(),
        vec_lat.data_ptr<float>(),
        vec_ele.data_ptr<float>(),
        vec_ax.data_ptr<float>(),
        active_mask_ptr,
        tiles_touched_count.data_ptr<int>(),
        reinterpret_cast<int4*>(gauss_tile_rects.data_ptr<int>()),
        gauss_axial_pix.data_ptr<float>()
    );

    torch::Tensor gauss_instance_offsets = torch::empty({batch_size, n_gauss + 1}, opts_int);

    size_t scan_gauss_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        scan_gauss_bytes,
        tiles_touched_count.data_ptr<int>(),
        gauss_instance_offsets.data_ptr<int>(),
        n_gauss + 1,
        stream
    );
    temp_storage_bytes = scan_gauss_bytes;
    temp_storage = torch::empty(
        {(long)temp_storage_bytes},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
    );
    d_temp_storage = temp_storage.data_ptr();

    for (int b = 0; b < batch_size; ++b) {
        cub::DeviceScan::ExclusiveSum(
            d_temp_storage,
            temp_storage_bytes,
            tiles_touched_count.data_ptr<int>() + b * (n_gauss + 1),
            gauss_instance_offsets.data_ptr<int>() + b * (n_gauss + 1),
            n_gauss + 1,
            stream
        );
    }

    torch::Tensor batch_totals = torch::empty({batch_size}, opts_int);
    torch::Tensor batch_prefix = torch::empty({batch_size}, opts_int);
    {
        const int tpb = 256;
        const int blocks = (batch_size + tpb - 1) / tpb;
        kernel_extract_batch_totals<<<blocks, tpb, 0, stream>>>(
            gauss_instance_offsets.data_ptr<int>(),
            batch_size,
            n_gauss,
            batch_totals.data_ptr<int>()
        );
    }
    size_t scan_batch_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        scan_batch_bytes,
        batch_totals.data_ptr<int>(),
        batch_prefix.data_ptr<int>(),
        batch_size,
        stream
    );
    if (scan_batch_bytes > temp_storage_bytes) {
        temp_storage_bytes = scan_batch_bytes;
        temp_storage = torch::empty(
            {(long)temp_storage_bytes},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
        );
        d_temp_storage = temp_storage.data_ptr();
    }
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        batch_totals.data_ptr<int>(),
        batch_prefix.data_ptr<int>(),
        batch_size,
        stream
    );
    {
        dim3 tpb(256, 1, 1);
        dim3 blocks((n_gauss + 1 + tpb.x - 1) / tpb.x, batch_size, 1);
        kernel_add_batch_prefix_to_offsets<<<blocks, tpb, 0, stream>>>(
            gauss_instance_offsets.data_ptr<int>(),
            batch_size,
            n_gauss,
            batch_prefix.data_ptr<int>()
        );
    }
    int total_mapping_instances = 0;
    if (batch_size > 0) {
        const int last_prefix = copy_scalar_from_device_and_sync<int>(
            batch_prefix.data_ptr<int>() + (batch_size - 1),
            stream,
            "last_batch_prefix"
        );
        const int last_total = copy_scalar_from_device_and_sync<int>(
            batch_totals.data_ptr<int>() + (batch_size - 1),
            stream,
            "last_batch_total"
        );
        total_mapping_instances = last_prefix + last_total;
    }

    if (total_mapping_instances == 0) {
        torch::Tensor pixel_gauss_counts = torch::zeros({n_total_pixels}, opts_int);
        torch::Tensor pixel_gauss_offsets = torch::zeros({n_total_pixels}, opts_int);
        torch::Tensor pixel_gauss_indices = torch::empty({0}, opts_int);
        return {pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices, pair_batch_offsets};
    }

    int grid_w = (image_width + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE;
    int grid_h = (image_height + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE;
    int n_tiles = grid_w * grid_h;

    torch::Tensor tile_ranges_start = torch::full({batch_size * n_tiles}, -1, opts_int);
    torch::Tensor tile_ranges_end = torch::full({batch_size * n_tiles}, -1, opts_int);

    torch::Tensor keys_buf_1 = torch::empty({total_mapping_instances}, opts_long);
    torch::Tensor keys_buf_2 = torch::empty({total_mapping_instances}, opts_long);
    torch::Tensor vals_buf_1 = torch::empty({total_mapping_instances}, opts_int);
    torch::Tensor vals_buf_2 = torch::empty({total_mapping_instances}, opts_int);

    uint64_t* d_keys_in = (uint64_t*)keys_buf_1.data_ptr<long>();
    uint64_t* d_keys_out = (uint64_t*)keys_buf_2.data_ptr<long>();
    int* d_vals_in = vals_buf_1.data_ptr<int>();
    int* d_vals_out = vals_buf_2.data_ptr<int>();

    int block_k = 256;
    dim3 grid_k((n_gauss + block_k - 1) / block_k, batch_size, 1);

    kernel_generate_tile_depth_keys_batched<<<grid_k, dim3(block_k, 1, 1), 0, stream>>>(
        n_gauss,
        batch_size,
        grid_w,
        n_tiles,
        sort_by_depth,
        reinterpret_cast<int4*>(gauss_tile_rects.data_ptr<int>()),
        gauss_axial_pix.data_ptr<float>(),
        gauss_instance_offsets.data_ptr<int>(),
        d_keys_in,
        d_vals_in
    );

    cub::DoubleBuffer<uint64_t> d_keys(d_keys_in, d_keys_out);
    cub::DoubleBuffer<int> d_values(d_vals_in, d_vals_out);

    temp_storage_bytes = 0;
    const int sort_begin_bit = sort_by_depth ? 0 : 32;
    const int sort_end_bit = 64;
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        temp_storage_bytes,
        d_keys,
        d_values,
        total_mapping_instances,
        sort_begin_bit,
        sort_end_bit,
        stream
    );
    if (temp_storage.numel() < (long)temp_storage_bytes) {
        temp_storage = torch::empty(
            {(long)temp_storage_bytes},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
        );
        d_temp_storage = temp_storage.data_ptr();
    }
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys,
        d_values,
        total_mapping_instances,
        sort_begin_bit,
        sort_end_bit,
        stream
    );

    int range_grid_size = (total_mapping_instances + block_k - 1) / block_k;
    kernel_identify_tile_ranges<<<range_grid_size, block_k, 0, stream>>>(
        total_mapping_instances,
        d_keys.Current(),
        tile_ranges_start.data_ptr<int>(),
        tile_ranges_end.data_ptr<int>()
    );

    int* sorted_gauss_ids_ptr = d_values.Current();

    float4* p_pos_flat = reinterpret_cast<float4*>(p_pos_packed.data_ptr<float>());

    dim3 block_dim(SLICE_TILE_SIZE, SLICE_TILE_SIZE, 1);
    dim3 grid_dim(grid_w, grid_h, batch_size);

    torch::Tensor pixel_gauss_counts = torch::zeros({n_total_pixels}, opts_int);
    torch::Tensor pixel_gauss_offsets = torch::zeros({n_total_pixels}, opts_int);

    kernel_count_pixel_candidates_batched<<<grid_dim, block_dim, 0, stream>>>(
        n_gauss,
        pixels_per_image,
        image_width,
        image_height,
        batch_size,
        n_tiles,
        p_pos_flat,
        reinterpret_cast<float4*>(means_packed.data_ptr<float>()),
        reinterpret_cast<float3*>(projected_radii_batched.data_ptr<float>()),
        sorted_gauss_ids_ptr,
        tile_ranges_start.data_ptr<int>(),
        tile_ranges_end.data_ptr<int>(),
        vec_lat.data_ptr<float>(),
        vec_ax.data_ptr<float>(),
        pixel_gauss_counts.data_ptr<int>()
    );

    size_t scan_pixel_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        scan_pixel_bytes,
        pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_offsets.data_ptr<int>(),
        n_total_pixels,
        stream
    );
    if (scan_pixel_bytes > temp_storage_bytes) {
        temp_storage_bytes = scan_pixel_bytes;
        temp_storage = torch::empty(
            {(long)temp_storage_bytes},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
        );
        d_temp_storage = temp_storage.data_ptr();
    }
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_offsets.data_ptr<int>(),
        n_total_pixels,
        stream
    );

    int last_pixel_offset = 0;
    int last_pixel_count = 0;
    TORCH_CHECK(
        cudaMemcpyAsync(
            &last_pixel_offset,
            pixel_gauss_offsets.data_ptr<int>() + (n_total_pixels - 1),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream
        ) == cudaSuccess,
        "[ultrags_renderer.mapper_batched] last_pixel_offset copy failed"
    );
    TORCH_CHECK(
        cudaMemcpyAsync(
            &last_pixel_count,
            pixel_gauss_counts.data_ptr<int>() + (n_total_pixels - 1),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream
        ) == cudaSuccess,
        "[ultrags_renderer.mapper_batched] last_pixel_count copy failed"
    );
    TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "[ultrags_renderer.mapper_batched] sync failed");

    long long total_pairs = (long long)last_pixel_offset + last_pixel_count;
    torch::Tensor pixel_gauss_indices = torch::empty({total_pairs}, opts_int);

    if (total_pairs > 0) {
        kernel_fill_pixel_candidate_indices_batched<<<grid_dim, block_dim, 0, stream>>>(
            n_gauss,
            pixels_per_image,
            image_width,
            image_height,
            batch_size,
            n_tiles,
            p_pos_flat,
            reinterpret_cast<float4*>(means_packed.data_ptr<float>()),
            reinterpret_cast<float3*>(projected_radii_batched.data_ptr<float>()),
            sorted_gauss_ids_ptr,
            tile_ranges_start.data_ptr<int>(),
            tile_ranges_end.data_ptr<int>(),
            vec_lat.data_ptr<float>(),
            vec_ax.data_ptr<float>(),
            pixel_gauss_offsets.data_ptr<int>(),
            pixel_gauss_indices.data_ptr<int>()
        );
    }

    if (return_pair_batch_offsets) {
        const int tpb = 256;
        const int blocks = (batch_size + tpb - 1) / tpb;
        kernel_fill_pair_batch_offsets<<<blocks, tpb, 0, stream>>>(
            pixel_gauss_offsets.data_ptr<int>(),
            batch_size,
            pixels_per_image,
            static_cast<int64_t>(total_pairs),
            pair_batch_offsets.data_ptr<int64_t>()
        );
    }

    return {
        pixel_gauss_counts,
        pixel_gauss_offsets,
        pixel_gauss_indices,
        pair_batch_offsets
    };
}
