#include <torch/extension.h>
#include <cub/cub.cuh>

#include "mapper/wrapper_mapper.cuh"
#include "mapper/core.cuh"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MapperForwardCUDA(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& origin,
    const torch::Tensor& axis_lat,
    const torch::Tensor& axis_ele,
    const torch::Tensor& axis_ax,
    int image_width,
    float width_mm,
    float near_mm,
    float lateral_col_offset,
    float elev_cutoff_mm,
    float k_sigma
) {
    const int n_gauss = means.size(0);
    const float pixel_spacing = width_mm / (float)image_width;

    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    torch::Tensor num_rays = torch::zeros({n_gauss}, int_opts);
    torch::Tensor ray_minmax = torch::empty({n_gauss, 2}, int_opts);
    torch::Tensor depths = torch::empty({n_gauss}, float_opts);

    const int block = 256;
    const int grid = (n_gauss + block - 1) / block;

    kernel_project_and_cull<<<grid, block>>>(
        n_gauss, image_width, pixel_spacing, lateral_col_offset, near_mm, elev_cutoff_mm, k_sigma,
        (float4*)means.data_ptr(),
        (float4*)scales.data_ptr(),
        origin.data_ptr<float>(),
        axis_lat.data_ptr<float>(),
        axis_ele.data_ptr<float>(),
        axis_ax.data_ptr<float>(),
        num_rays.data_ptr<int>(),
        (int2*)ray_minmax.data_ptr(),
        depths.data_ptr<float>()
    );

    // Correct prefix-sum layout: exclusive scan of length N into offsets[0:N],
    // and manually write offsets[N] = total_hits.
    torch::Tensor offsets = torch::empty({n_gauss + 1}, int_opts);
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, num_rays.data_ptr<int>(), offsets.data_ptr<int>(), n_gauss);
    torch::Tensor temp = torch::empty({(long)temp_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    cub::DeviceScan::ExclusiveSum(temp.data_ptr(), temp_bytes, num_rays.data_ptr<int>(), offsets.data_ptr<int>(), n_gauss);

    int last_off = 0;
    int last_cnt = 0;
    if (n_gauss > 0) {
        cudaMemcpy(&last_off, offsets.data_ptr<int>() + (n_gauss - 1), sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_cnt, num_rays.data_ptr<int>() + (n_gauss - 1), sizeof(int), cudaMemcpyDeviceToHost);
    }
    int total_hits = last_off + last_cnt;
    cudaMemcpy(offsets.data_ptr<int>() + n_gauss, &total_hits, sizeof(int), cudaMemcpyHostToDevice);

    if (total_hits == 0) {
        return std::make_tuple(
            torch::zeros({image_width}, int_opts),
            torch::zeros({image_width}, int_opts),
            torch::empty({0}, int_opts),
            torch::empty({0}, float_opts)
        );
    }

    torch::Tensor keys_in = torch::empty({total_hits}, long_opts);
    torch::Tensor vals_in = torch::empty({total_hits}, int_opts);
    torch::Tensor keys_out = torch::empty({total_hits}, long_opts);
    torch::Tensor vals_out = torch::empty({total_hits}, int_opts);

    kernel_generate_sort_keys<<<grid, block>>>(
        n_gauss,
        offsets.data_ptr<int>(),
        (int2*)ray_minmax.data_ptr(),
        depths.data_ptr<float>(),
        (uint64_t*)keys_in.data_ptr<int64_t>(),
        vals_in.data_ptr<int>()
    );

    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        (uint64_t*)keys_in.data_ptr<int64_t>(),
        (uint64_t*)keys_out.data_ptr<int64_t>(),
        vals_in.data_ptr<int>(),
        vals_out.data_ptr<int>(),
        total_hits
    );
    torch::Tensor sort_temp = torch::empty({(long)temp_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    cub::DeviceRadixSort::SortPairs(
        sort_temp.data_ptr(), temp_bytes,
        (uint64_t*)keys_in.data_ptr<int64_t>(),
        (uint64_t*)keys_out.data_ptr<int64_t>(),
        vals_in.data_ptr<int>(),
        vals_out.data_ptr<int>(),
        total_hits
    );

    torch::Tensor ray_starts = torch::full({image_width}, -1, int_opts);
    torch::Tensor ray_ends = torch::full({image_width}, -1, int_opts);
    int grid_hits = (total_hits + block - 1) / block;
    kernel_build_csr_structure<<<grid_hits, block>>>(
        total_hits,
        (uint64_t*)keys_out.data_ptr<int64_t>(),
        ray_starts.data_ptr<int>(),
        ray_ends.data_ptr<int>()
    );

    torch::Tensor sorted_depths = torch::empty({total_hits}, float_opts);
    kernel_decode_sorted_depths<<<grid_hits, block>>>(
        total_hits,
        (uint64_t*)keys_out.data_ptr<int64_t>(),
        sorted_depths.data_ptr<float>()
    );

    torch::Tensor mask = (ray_starts != -1);
    torch::Tensor final_offsets = torch::where(mask, ray_starts, torch::zeros_like(ray_starts));
    torch::Tensor final_counts = torch::where(mask, ray_ends - ray_starts, torch::zeros_like(ray_starts));
    return std::make_tuple(final_counts, final_offsets, vals_out, sorted_depths);
}
