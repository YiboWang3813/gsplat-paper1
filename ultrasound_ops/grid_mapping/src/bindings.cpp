#include <torch/extension.h>
#include <vector>
#include "grid_builder.cuh"

// 包装器：将 C++ 结构体拆解为 Tensor 列表返回给 Python
std::vector<torch::Tensor> build_mapping_wrapper(
    const torch::Tensor& grid_min,
    const torch::Tensor& tile_size,
    const torch::Tensor& grid_size,
    const torch::Tensor& pixel_positions,
    const torch::Tensor& means,
    const torch::Tensor& sigma_invs
) {
    // 调用核心 CUDA Builder
    PixelGaussMappingGrid res = BuildPixelGaussMappingGridCUDA(
        grid_min, 
        tile_size, 
        grid_size, 
        pixel_positions, 
        means, 
        sigma_invs
    );

    // 按照约定顺序组装返回列表
    return {
        res.tile_counts,
        res.tile_offsets,
        res.tile_indices,
        res.pixel_tile_ids,
        res.pixel_counts,
        res.pixel_offsets,
        res.pixel_indices
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_mapping", &build_mapping_wrapper, "Build Grid-based Pixel-Gaussian Mapping (CUDA)");
}