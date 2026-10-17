#include <torch/extension.h>
#include "wrapper.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_ray_mapper", &launch_ray_mapper, 
          "Launch mapper.");
}