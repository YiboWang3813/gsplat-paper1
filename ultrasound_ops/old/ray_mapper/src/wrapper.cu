#include <torch/extension.h>
#include <cub/cub.cuh>
#include <tuple>

#include "wrapper.cuh"
#include "core.cuh"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> launch_ray_mapper(
    // Geometry Tensors
    const torch::Tensor& means,           // (N, 4)
    const torch::Tensor& scales,          // (N, 4)
    const torch::Tensor& perturbations,   // (N,) Python 预计算的随机扰动
    const torch::Tensor& origin,          // (3,)
    const torch::Tensor& axis_lat,        // (3,)
    const torch::Tensor& axis_ele,        // (3,)
    const torch::Tensor& axis_ax,         // (3,)
    // Scalar Params
    const int image_width,
    const float width_mm,
    const float near_mm,
    const float elev_cutoff_mm,
    const float k_sigma
) {
    const int n_gauss = means.size(0);
    const float pixel_spacing = width_mm / (float)image_width;

    // Helper for tensor options
    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    // -------------------------------------------------------------------------
    // Phase 1: Projection & Culling
    // -------------------------------------------------------------------------
    
    torch::Tensor num_rays = torch::zeros({n_gauss}, int_opts);
    torch::Tensor ray_minmax = torch::empty({n_gauss, 2}, int_opts);
    torch::Tensor depths = torch::empty({n_gauss}, float_opts);

    int block = 256;
    int grid = (n_gauss + block - 1) / block;

    kernel_project_and_cull<<<grid, block>>>(
        n_gauss, image_width, pixel_spacing, near_mm, elev_cutoff_mm, k_sigma,
        (float4*)means.data_ptr(),
        (float4*)scales.data_ptr(),
        perturbations.data_ptr<float>(),
        origin.data_ptr<float>(),
        axis_lat.data_ptr<float>(),
        axis_ele.data_ptr<float>(),
        axis_ax.data_ptr<float>(),
        // Outputs
        num_rays.data_ptr<int>(),
        (int2*)ray_minmax.data_ptr(),
        depths.data_ptr<float>()
    );

    // -------------------------------------------------------------------------
    // Phase 2: Prefix Sum (Scan) to get offsets
    // -------------------------------------------------------------------------
    
    // offsets 需要 N+1 大小
    torch::Tensor offsets = torch::empty({n_gauss + 1}, int_opts);
    
    // CUB Scan
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, num_rays.data_ptr<int>(), offsets.data_ptr<int>(), n_gauss + 1);
    
    torch::Tensor temp_storage = torch::empty({(long)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    cub::DeviceScan::ExclusiveSum(temp_storage.data_ptr(), temp_storage_bytes, num_rays.data_ptr<int>(), offsets.data_ptr<int>(), n_gauss + 1);

    // 获取 Total Hits (位于 offsets 的最后一个位置)
    int total_hits = 0;
    cudaMemcpy(&total_hits, offsets.data_ptr<int>() + n_gauss, sizeof(int), cudaMemcpyDeviceToHost);

    // 如果没有任何高斯命中，直接返回空结果
    if (total_hits == 0) {
        return std::make_tuple(
            torch::zeros({image_width}, int_opts), // counts
            torch::zeros({image_width}, int_opts), // offsets
            torch::empty({0}, int_opts)            // indices
        );
    }

    // -------------------------------------------------------------------------
    // Phase 3: Generate Sort Keys
    // -------------------------------------------------------------------------

    torch::Tensor keys_in = torch::empty({total_hits}, long_opts);
    torch::Tensor vals_in = torch::empty({total_hits}, int_opts);
    torch::Tensor keys_out = torch::empty({total_hits}, long_opts);
    torch::Tensor vals_out = torch::empty({total_hits}, int_opts); // 最终排序后的 indices

    kernel_generate_sort_keys<<<grid, block>>>(
        n_gauss,
        offsets.data_ptr<int>(),
        (int2*)ray_minmax.data_ptr(),
        depths.data_ptr<float>(),
        (uint64_t*)keys_in.data_ptr<int64_t>(),
        vals_in.data_ptr<int>()
    );

    // -------------------------------------------------------------------------
    // Phase 4: Radix Sort (RayID first, then Depth)
    // -------------------------------------------------------------------------
    
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, (uint64_t*)keys_in.data_ptr<int64_t>(), (uint64_t*)keys_out.data_ptr<int64_t>(), vals_in.data_ptr<int>(), vals_out.data_ptr<int>(), total_hits);
    
    torch::Tensor sort_storage = torch::empty({(long)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    cub::DeviceRadixSort::SortPairs(sort_storage.data_ptr(), temp_storage_bytes, (uint64_t*)keys_in.data_ptr<int64_t>(), (uint64_t*)keys_out.data_ptr<int64_t>(), vals_in.data_ptr<int>(), vals_out.data_ptr<int>(), total_hits);

    // -------------------------------------------------------------------------
    // Phase 5: Build CSR Format
    // -------------------------------------------------------------------------

    // 初始化为 -1 表示无高斯覆盖该 Ray
    torch::Tensor ray_starts = torch::full({image_width}, -1, int_opts);
    torch::Tensor ray_ends   = torch::full({image_width}, -1, int_opts);

    int grid_hits = (total_hits + block - 1) / block;
    kernel_build_csr_structure<<<grid_hits, block>>>(
        total_hits,
        (uint64_t*)keys_out.data_ptr<int64_t>(),
        ray_starts.data_ptr<int>(),
        ray_ends.data_ptr<int>()
    );

    // Post-process to clean CSR arrays
    // 如果 ray_starts == -1 (empty ray), 则设 count=0, offset=0
    torch::Tensor mask = (ray_starts != -1);
    torch::Tensor final_offsets = torch::where(mask, ray_starts, torch::zeros_like(ray_starts));
    torch::Tensor final_counts = torch::where(mask, ray_ends - ray_starts, torch::zeros_like(ray_starts));

    return std::make_tuple(final_counts, final_offsets, vals_out);
}