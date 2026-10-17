#include <torch/extension.h>

std::vector<torch::Tensor> additive_splat_forward_cuda(
    torch::Tensor pixels,
    torch::Tensor means2d,
    torch::Tensor inv2d,
    torch::Tensor weights,
    torch::Tensor colors);

std::vector<torch::Tensor> additive_splat_backward_cuda(
    torch::Tensor pixels,
    torch::Tensor means2d,
    torch::Tensor inv2d,
    torch::Tensor weights,
    torch::Tensor colors,
    torch::Tensor grad_image,
    torch::Tensor grad_alpha);

std::vector<torch::Tensor> additive_splat_bounded_forward_cuda(
    int64_t image_height,
    int64_t image_width,
    float half_width,
    float half_height,
    torch::Tensor means2d,
    torch::Tensor inv2d,
    torch::Tensor weights,
    torch::Tensor colors,
    float radius_sigma);

std::vector<torch::Tensor> additive_splat_bounded_backward_cuda(
    int64_t image_height,
    int64_t image_width,
    float half_width,
    float half_height,
    torch::Tensor means2d,
    torch::Tensor inv2d,
    torch::Tensor weights,
    torch::Tensor colors,
    torch::Tensor grad_image,
    torch::Tensor grad_alpha,
    float radius_sigma);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &additive_splat_forward_cuda, "slice-adapted additive splat forward");
  m.def("backward", &additive_splat_backward_cuda, "slice-adapted additive splat backward");
  m.def("bounded_forward", &additive_splat_bounded_forward_cuda, "bounded slice-adapted additive splat forward");
  m.def("bounded_backward", &additive_splat_bounded_backward_cuda, "bounded slice-adapted additive splat backward");
}
