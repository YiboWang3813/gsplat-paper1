#include <torch/extension.h>
#include "wrapper.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shade_forward", &shade_forward, 
          "Rasterize Gaussians Forward (CUDA)");
          
    m.def("shade_backward", &shade_backward, 
          "Rasterize Gaussians Backward (CUDA)");
}