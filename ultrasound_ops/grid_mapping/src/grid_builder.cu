// grid_builder.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "grid_builder.cuh"
#include "grid_core.cuh"


PixelGaussMappingGrid BuildPixelGaussMappingGridCUDA(
    const torch::Tensor& grid_min,
    const torch::Tensor& tile_size,
    const torch::Tensor& grid_size,
    const torch::Tensor& pixel_positions,
    const torch::Tensor& means,
    const torch::Tensor& sigma_invs
)
{
    const int n_pixels = (int)pixel_positions.size(0);
    const int n_gauss  = (int)means.size(0);

    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    // ---- n_tiles ----
    const int grid_size_x = grid_size[0].item<int>();
    const int grid_size_y = grid_size[1].item<int>();
    const int grid_size_z = grid_size[2].item<int>();
    const int n_tiles = grid_size_x * grid_size_y * grid_size_z;

    PixelGaussMappingGrid out;

    // ============================================================
    // Tile level
    // ============================================================
    out.tile_counts  = torch::zeros({n_tiles}, options_int);
    out.tile_offsets = torch::zeros({n_tiles}, options_int);

    count_gauss_per_tile(
        n_gauss,
        means.data_ptr<float>(),
        grid_min.data_ptr<float>(),
        tile_size.data_ptr<float>(),
        grid_size.data_ptr<int>(),
        out.tile_counts.data_ptr<int>()
    );

    // exclusive scan: tile_offsets
    {
        thrust::device_ptr<int> counts_ptr(out.tile_counts.data_ptr<int>());
        thrust::device_ptr<int> offsets_ptr(out.tile_offsets.data_ptr<int>());
        thrust::exclusive_scan(counts_ptr, counts_ptr + n_tiles, offsets_ptr);
    }

    // total_tile_indices = last_offset + last_count
    int last_tile_offset = 0, last_tile_count = 0;
    // CUDA_CHECK(cudaMemcpy(...))
    cudaMemcpy(&last_tile_offset, out.tile_offsets.data_ptr<int>() + (n_tiles - 1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_tile_count,  out.tile_counts.data_ptr<int>()  + (n_tiles - 1), sizeof(int), cudaMemcpyDeviceToHost);
    const int total_tile_indices = last_tile_offset + last_tile_count;

    out.tile_indices  = torch::full({total_tile_indices}, -1, options_int);
    torch::Tensor tile_counters = torch::zeros({n_tiles}, options_int);

    fill_tile_indices(
        n_gauss,
        means.data_ptr<float>(),
        grid_min.data_ptr<float>(),
        tile_size.data_ptr<float>(),
        grid_size.data_ptr<int>(),
        out.tile_offsets.data_ptr<int>(),
        tile_counters.data_ptr<int>(),
        out.tile_indices.data_ptr<int>()
    );

    // ============================================================
    // Pixel level
    // ============================================================
    out.pixel_tile_ids = torch::zeros({n_pixels}, options_int);
    out.pixel_counts   = torch::zeros({n_pixels}, options_int);
    out.pixel_offsets  = torch::zeros({n_pixels}, options_int);

    count_gauss_per_pixel(
        n_pixels,
        pixel_positions.data_ptr<float>(),
        means.data_ptr<float>(),
        sigma_invs.data_ptr<float>(),
        grid_min.data_ptr<float>(),
        tile_size.data_ptr<float>(),
        grid_size.data_ptr<int>(),
        out.tile_counts.data_ptr<int>(),
        out.tile_offsets.data_ptr<int>(),
        out.tile_indices.data_ptr<int>(),
        out.pixel_tile_ids.data_ptr<int>(),
        out.pixel_counts.data_ptr<int>()
    );

    // exclusive scan: pixel_offsets
    {
        thrust::device_ptr<int> counts_ptr(out.pixel_counts.data_ptr<int>());
        thrust::device_ptr<int> offsets_ptr(out.pixel_offsets.data_ptr<int>());
        thrust::exclusive_scan(counts_ptr, counts_ptr + n_pixels, offsets_ptr);
    }

    int last_pixel_offset = 0, last_pixel_count = 0;
    cudaMemcpy(&last_pixel_offset, out.pixel_offsets.data_ptr<int>() + (n_pixels - 1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_pixel_count,  out.pixel_counts.data_ptr<int>()  + (n_pixels - 1), sizeof(int), cudaMemcpyDeviceToHost);
    const int total_pairs = last_pixel_offset + last_pixel_count;

    out.pixel_indices = torch::full({total_pairs}, -1, options_int);

    fill_pixel_indices(
        n_pixels,
        pixel_positions.data_ptr<float>(),
        means.data_ptr<float>(),
        sigma_invs.data_ptr<float>(),
        grid_min.data_ptr<float>(),
        tile_size.data_ptr<float>(),
        grid_size.data_ptr<int>(),
        out.tile_counts.data_ptr<int>(),
        out.tile_offsets.data_ptr<int>(),
        out.tile_indices.data_ptr<int>(),
        out.pixel_offsets.data_ptr<int>(),
        out.pixel_indices.data_ptr<int>()
    );

    return out;
}
