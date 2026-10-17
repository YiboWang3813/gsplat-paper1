#include <torch/extension.h>

#include "mapper/wrapper_mapper.cuh"
#include "shader/wrapper_shader.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("MapperForwardCUDA", &MapperForwardCUDA, "Mapper forward (CUDA)");
    m.def("MapperForwardCUDABatched", &MapperForwardCUDABatched, "Mapper forward batched (CUDA)");
    m.def("ShaderForwardCUDA", &ShaderForwardCUDA, "Shader forward (CUDA)");
    m.def("ShaderForwardCUDABatched", &ShaderForwardCUDABatched, "Shader forward batched (CUDA)");
    m.def("ShaderBackwardCUDA", &ShaderBackwardCUDA, "Shader backward (CUDA)");
    m.def("ShaderBackwardCUDABatched", &ShaderBackwardCUDABatched, "Shader backward batched (CUDA)");
}
