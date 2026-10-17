#include <torch/extension.h>
#include "wrapper.cuh"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_ellipsoid_mapper", &launch_ellipsoid_mapper, 
          "Launch mapper.");
}