#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh> 
// 辅助：获取 PyTorch 当前流
#include <ATen/cuda/CUDAContext.h>

#include "wrapper.cuh"
#include "core.cuh"


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launch_ellipsoid_mapper(
    // 几何输入
    const torch::Tensor& p_pos_packed, 
    const torch::Tensor& means_packed,     
    const torch::Tensor& quats,
    const torch::Tensor& scales_packed,
    // 探头参数
    const torch::Tensor& probe_org,    
    const torch::Tensor& vec_ele,    
    const torch::Tensor& vec_lat,    
    const torch::Tensor& vec_ax,      
    // 配置
    const int   image_height,
    const int   image_width,
    const float delta_lat,       
    const float delta_ax,
    const float slab_half_thickness_mm,
    const float near_mm,
    const float k_sigma,
    const float lateral_col_offset,
    const bool  sort_by_depth
) {
    const int n_pixels = p_pos_packed.size(0);
    const int n_gauss  = means_packed.size(0);
    
    // 选项
    auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto opts_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_long = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    // 0. 基础检查
    if (n_pixels == 0 || n_gauss == 0 || image_width <= 0 || image_height <= 0) {
        torch::Tensor pixel_gauss_counts  = torch::zeros({n_pixels}, opts_int);
        torch::Tensor pixel_gauss_offsets = torch::zeros({n_pixels}, opts_int);
        torch::Tensor pixel_gauss_indices = torch::empty({0}, opts_int);
        return {pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices};
    }

    TORCH_CHECK(
        n_pixels == image_height * image_width,
        "[ellipsoid_mapper] n_pixels mismatch: got n_pixels=", n_pixels,
        ", but H*W=", image_height, "*", image_width, "=", image_height * image_width,
        ". Expect pixel_positions_world to be a row-major flatten of an (H,W) slice."
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 定义复用的临时存储变量
    torch::Tensor temp_storage;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // -------------------------------------------------------------------------
    // 1. Precompute Radii (New Step)
    // -------------------------------------------------------------------------
    // projected_radii component order: (lateral, axial, elevational), in mm and already *k_sigma.
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

    // -------------------------------------------------------------------------
    // 2. Cull & Count Tiles
    // -------------------------------------------------------------------------
    // Per-gauss: number of tiles touched, tile rect (min/max), and axial pixel depth for sorting.
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

    // -------------------------------------------------------------------------
    // 3. Scan (Prefix Sum)
    // -------------------------------------------------------------------------
    torch::Tensor gauss_instance_offsets = torch::empty({n_gauss + 1}, opts_int);

    // Query
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
        tiles_touched_count.data_ptr<int>(), gauss_instance_offsets.data_ptr<int>(), n_gauss + 1, stream);
    
    // Allocate
    temp_storage = torch::empty({(long)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    d_temp_storage = temp_storage.data_ptr();

    // Run
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
        tiles_touched_count.data_ptr<int>(), gauss_instance_offsets.data_ptr<int>(), n_gauss + 1, stream);

    // Read total
    int total_mapping_instances = 0;
    cudaMemcpyAsync(&total_mapping_instances, gauss_instance_offsets.data_ptr<int>() + n_gauss, sizeof(int), cudaMemcpyDeviceToHost, stream);
    // 这里必须同步一下，因为后续显存分配依赖 total_mapping_instances
    // 如果想要完全异步，可以在 Python 端分配一个足够大的 Max Buffer，但这很难估计
    cudaStreamSynchronize(stream); 

    if (total_mapping_instances == 0) {
        torch::Tensor pixel_gauss_counts  = torch::zeros({n_pixels}, opts_int);
        torch::Tensor pixel_gauss_offsets = torch::zeros({n_pixels}, opts_int);
        torch::Tensor pixel_gauss_indices = torch::empty({0}, opts_int);
        return {pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices};
    }

    // -------------------------------------------------------------------------
    // 4. Generate Keys & Sort
    // -------------------------------------------------------------------------
    int grid_w = (image_width + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE;
    int grid_h = (image_height + SLICE_TILE_SIZE - 1) / SLICE_TILE_SIZE;
    int n_tiles = grid_w * grid_h;
    
    torch::Tensor tile_ranges_start = torch::full({n_tiles}, -1, opts_int);
    torch::Tensor tile_ranges_end   = torch::full({n_tiles}, -1, opts_int);

    // Double Buffers for Sort
    torch::Tensor keys_buf_1 = torch::empty({total_mapping_instances}, opts_long);
    torch::Tensor keys_buf_2 = torch::empty({total_mapping_instances}, opts_long);
    torch::Tensor vals_buf_1 = torch::empty({total_mapping_instances}, opts_int);
    torch::Tensor vals_buf_2 = torch::empty({total_mapping_instances}, opts_int);

    uint64_t* d_keys_in = (uint64_t*)keys_buf_1.data_ptr<long>();
    uint64_t* d_keys_out = (uint64_t*)keys_buf_2.data_ptr<long>();
    int* d_vals_in = vals_buf_1.data_ptr<int>();
    int* d_vals_out = vals_buf_2.data_ptr<int>();

    // 4.1 Generate
    kernel_generate_tile_depth_keys<<<grid_size, block_size, 0, stream>>>(
        n_gauss, grid_w, 
        sort_by_depth,
        reinterpret_cast<int4*>(gauss_tile_rects.data_ptr<int>()), 
        gauss_axial_pix.data_ptr<float>(), 
        gauss_instance_offsets.data_ptr<int>(), 
        d_keys_in, d_vals_in
    );

    // 4.2 Radix Sort
    cub::DoubleBuffer<uint64_t> d_keys(d_keys_in, d_keys_out);
    cub::DoubleBuffer<int>      d_values(d_vals_in, d_vals_out);

    // Query Size (Sort 往往需要更大的 temp storage)
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

    // Re-Allocate if needed
    if (temp_storage.numel() < temp_storage_bytes) {
        temp_storage = torch::empty({(long)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
        d_temp_storage = temp_storage.data_ptr();
    }

    // Run Sort
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

    // 4.3 Identify Ranges
    int range_grid_size = (total_mapping_instances + block_size - 1) / block_size;
    kernel_identify_tile_ranges<<<range_grid_size, block_size, 0, stream>>>(
        total_mapping_instances, 
        d_keys.Current(), 
        tile_ranges_start.data_ptr<int>(), 
        tile_ranges_end.data_ptr<int>()
    );

    // 获取排序后的 ID
    int* sorted_gauss_ids_ptr = d_values.Current();

    // -------------------------------------------------------------------------
    // 5. Construct Output (Count & Fill)
    // -------------------------------------------------------------------------
    torch::Tensor pixel_gauss_counts  = torch::zeros({n_pixels}, opts_int);
    torch::Tensor pixel_gauss_offsets = torch::zeros({n_pixels}, opts_int);

    dim3 block_dim(SLICE_TILE_SIZE, SLICE_TILE_SIZE); 
    dim3 grid_dim(grid_w, grid_h);

    // 5.1 Count Hits
    kernel_count_pixel_candidates<<<grid_dim, block_dim, 0, stream>>>(
        n_pixels, image_width, image_height,
        reinterpret_cast<float4*>(p_pos_packed.data_ptr<float>()), // 假设 float4
        reinterpret_cast<float4*>(means_packed.data_ptr<float>()),
        reinterpret_cast<float3*>(projected_radii.data_ptr<float>()),
        sorted_gauss_ids_ptr,
        tile_ranges_start.data_ptr<int>(),
        tile_ranges_end.data_ptr<int>(),
        vec_lat.data_ptr<float>(),
        vec_ax.data_ptr<float>(),
        pixel_gauss_counts.data_ptr<int>()
    );

    // 5.2 Scan Offsets
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
        pixel_gauss_counts.data_ptr<int>(),
        pixel_gauss_offsets.data_ptr<int>(),
        n_pixels, stream);

    // 5.3 Alloc Indices
    int last_pixel_offset, last_pixel_count;
    cudaMemcpyAsync(&last_pixel_offset, pixel_gauss_offsets.data_ptr<int>() + (n_pixels - 1), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_pixel_count, pixel_gauss_counts.data_ptr<int>() + (n_pixels - 1), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); 
    
    long long total_pairs = (long long)last_pixel_offset + last_pixel_count;
    torch::Tensor pixel_gauss_indices = torch::empty({total_pairs}, opts_int);

    // 5.4 Fill Indices
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
