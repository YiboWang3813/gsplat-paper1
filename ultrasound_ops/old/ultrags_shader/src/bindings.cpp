#include <torch/extension.h>
#include "wrapper.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shade_forward", &RasterizeGaussiansForwardCUDA, 
          "Rasterize Gaussians Forward (CUDA)");
          
    m.def("shade_backward", &RasterizeGaussiansBackwardCUDA, 
          "Rasterize Gaussians Backward (CUDA)");
}