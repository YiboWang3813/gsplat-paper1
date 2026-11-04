#include <torch/extension.h>

#include "shader/wrapper_shader.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ProjectedSliceForwardCUDA", &ProjectedSliceForwardCUDA, "Ours slice forward (CUDA, mm world)");
    m.def("ProjectedSliceBackwardCUDA", &ProjectedSliceBackwardCUDA, "Ours slice backward (CUDA, mm world)");
}
