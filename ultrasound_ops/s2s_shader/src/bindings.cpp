#include <torch/extension.h>
#include "wrapper.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &RasterizeGaussiansForwardCUDA, 
          "Rasterize Gaussians Forward (CUDA)");
          
    m.def("backward", &RasterizeGaussiansBackwardCUDA, 
          "Rasterize Gaussians Backward (CUDA)");
}