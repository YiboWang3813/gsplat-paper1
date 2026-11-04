#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "shader/plane_cond2d_device.cuh"

namespace {
__global__ void covariance_kernel(int n, const float* s, const float* q, float* out) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    float R[9];
    ours_quat_to_rot(make_float4(q[4*g], q[4*g+1], q[4*g+2], q[4*g+3]), R);
    plane_cond2d_device::mat3_sigma_world(s[3*g], s[3*g+1], s[3*g+2], R, out + 9*g);
}
}

torch::Tensor CovarianceDiagnosticCUDA(torch::Tensor scales, torch::Tensor quats) {
    TORCH_CHECK(scales.is_cuda() && quats.is_cuda() && scales.device() == quats.device(), "same CUDA device required");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32 && quats.scalar_type() == torch::kFloat32, "float32 required");
    TORCH_CHECK(scales.is_contiguous() && quats.is_contiguous(), "contiguous inputs required");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3 && quats.dim() == 2 && quats.size(1) == 4 && scales.size(0) == quats.size(0), "expected N x 3 scales and N x 4 quaternions");
    const c10::cuda::CUDAGuard guard(scales.device());
    auto out = torch::empty({scales.size(0), 3, 3}, scales.options());
    if (scales.size(0)) {
        covariance_kernel<<<(scales.size(0)+255)/256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(scales.size(0), scales.data_ptr<float>(), quats.data_ptr<float>(), out.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
