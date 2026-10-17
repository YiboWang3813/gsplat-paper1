#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

#define SLICE_3DGS_TILE_SIZE 16
#define SLICE_3DGS_TILE_BIN_MIN_GAUSSIANS 16384

namespace {

void check_cuda_kernel(const char* name) {
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "[slice_adapted_3dgs_renderer] ", name, " failed: ", cudaGetErrorString(err));
}

template <typename T>
T wait_and_copy_scalar_from_device(const T* device_ptr, cudaStream_t stream, const char* debug_name) {
  T host_value{};
  cudaEvent_t done_event = nullptr;
  TORCH_CHECK(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming) == cudaSuccess,
              "[slice_adapted_3dgs_renderer] failed to create event for ", debug_name);
  TORCH_CHECK(cudaMemcpyAsync(&host_value, device_ptr, sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess,
              "[slice_adapted_3dgs_renderer] failed to schedule D2H copy for ", debug_name);
  TORCH_CHECK(cudaEventRecord(done_event, stream) == cudaSuccess,
              "[slice_adapted_3dgs_renderer] failed to record event for ", debug_name);
  TORCH_CHECK(cudaEventSynchronize(done_event) == cudaSuccess,
              "[slice_adapted_3dgs_renderer] failed waiting on event for ", debug_name);
  TORCH_CHECK(cudaEventDestroy(done_event) == cudaSuccess,
              "[slice_adapted_3dgs_renderer] failed to destroy event for ", debug_name);
  return host_value;
}

__global__ void forward_kernel(
    const float* __restrict__ pixels,
    const float* __restrict__ means,
    const float* __restrict__ invs,
    const float* __restrict__ weights,
    const float* __restrict__ colors,
    int64_t P,
    int64_t N,
    float* __restrict__ image,
    float* __restrict__ alpha) {
  int64_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) {
    return;
  }

  float px = pixels[p * 2 + 0];
  float py = pixels[p * 2 + 1];
  float out0 = 0.0f;
  float out1 = 0.0f;
  float out2 = 0.0f;
  float aout = 0.0f;

  for (int64_t g = 0; g < N; ++g) {
    float dx = px - means[g * 2 + 0];
    float dy = py - means[g * 2 + 1];
    const float* inv = invs + g * 4;
    float qx = inv[0] * dx + inv[1] * dy;
    float qy = inv[2] * dx + inv[3] * dy;
    float power = -0.5f * (dx * qx + dy * qy);
    if (power < -50.0f) {
      continue;
    }
    if (power > 0.0f) {
      power = 0.0f;
    }
    float e = __expf(power);
    float w = e * weights[g];
    out0 += w * colors[g * 3 + 0];
    out1 += w * colors[g * 3 + 1];
    out2 += w * colors[g * 3 + 2];
    aout += w;
  }

  image[p * 3 + 0] = out0;
  image[p * 3 + 1] = out1;
  image[p * 3 + 2] = out2;
  alpha[p] = aout;
}

__global__ void backward_kernel(
    const float* __restrict__ pixels,
    const float* __restrict__ means,
    const float* __restrict__ invs,
    const float* __restrict__ weights,
    const float* __restrict__ colors,
    const float* __restrict__ grad_image,
    const float* __restrict__ grad_alpha,
    int64_t P,
    int64_t N,
    float* __restrict__ grad_means,
    float* __restrict__ grad_invs,
    float* __restrict__ grad_weights,
    float* __restrict__ grad_colors) {
  int64_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) {
    return;
  }

  float px = pixels[p * 2 + 0];
  float py = pixels[p * 2 + 1];
  float go0 = grad_image[p * 3 + 0];
  float go1 = grad_image[p * 3 + 1];
  float go2 = grad_image[p * 3 + 2];
  float ga = grad_alpha[p];

  for (int64_t g = 0; g < N; ++g) {
    float dx = px - means[g * 2 + 0];
    float dy = py - means[g * 2 + 1];
    const float* inv = invs + g * 4;
    float qx = inv[0] * dx + inv[1] * dy;
    float qy = inv[2] * dx + inv[3] * dy;
    float quad = dx * qx + dy * qy;
    float power = -0.5f * quad;
    if (power < -50.0f) {
      continue;
    }
    if (power > 0.0f) {
      power = 0.0f;
    }
    float e = __expf(power);
    float weight = weights[g];
    float c0 = colors[g * 3 + 0];
    float c1 = colors[g * 3 + 1];
    float c2 = colors[g * 3 + 2];
    float dot_color = go0 * c0 + go1 * c1 + go2 * c2;
    float dL_dw = e * (dot_color + ga);
    float dL_de = weight * (dot_color + ga);

    atomicAdd(grad_weights + g, dL_dw);
    atomicAdd(grad_colors + g * 3 + 0, e * weight * go0);
    atomicAdd(grad_colors + g * 3 + 1, e * weight * go1);
    atomicAdd(grad_colors + g * 3 + 2, e * weight * go2);

    float dL_dpower = dL_de * e;
    float common = -0.5f * dL_dpower;
    atomicAdd(grad_invs + g * 4 + 0, common * dx * dx);
    atomicAdd(grad_invs + g * 4 + 1, common * dx * dy);
    atomicAdd(grad_invs + g * 4 + 2, common * dy * dx);
    atomicAdd(grad_invs + g * 4 + 3, common * dy * dy);

    float sym_qx = (inv[0] + inv[0]) * dx + (inv[1] + inv[2]) * dy;
    float sym_qy = (inv[2] + inv[1]) * dx + (inv[3] + inv[3]) * dy;
    atomicAdd(grad_means + g * 2 + 0, 0.5f * dL_dpower * sym_qx);
    atomicAdd(grad_means + g * 2 + 1, 0.5f * dL_dpower * sym_qy);
  }
}

__device__ __forceinline__ float pixel_x_from_col(int x, int W, float half_width) {
  if (W <= 1) {
    return 0.0f;
  }
  return -half_width + (2.0f * half_width) * float(x) / float(W - 1);
}

__device__ __forceinline__ float pixel_y_from_row(int y, int H, float half_height) {
  if (H <= 1) {
    return 0.0f;
  }
  return -half_height + (2.0f * half_height) * float(y) / float(H - 1);
}

__device__ __forceinline__ void gaussian_bbox(
    const float* inv,
    float mx,
    float my,
    int H,
    int W,
    float half_width,
    float half_height,
    float radius_sigma,
    int& xmin,
    int& xmax,
    int& ymin,
    int& ymax) {
  float det = inv[0] * inv[3] - inv[1] * inv[2];
  det = fmaxf(det, 1.0e-8f);
  float sigma_x = sqrtf(fmaxf(inv[3] / det, 1.0e-8f));
  float sigma_y = sqrtf(fmaxf(inv[0] / det, 1.0e-8f));
  float rx = radius_sigma * sigma_x;
  float ry = radius_sigma * sigma_y;
  float sx = (W > 1) ? float(W - 1) / (2.0f * half_width) : 0.0f;
  float sy = (H > 1) ? float(H - 1) / (2.0f * half_height) : 0.0f;
  xmin = max(0, int(floorf((mx - rx + half_width) * sx)));
  xmax = min(W - 1, int(ceilf((mx + rx + half_width) * sx)));
  ymin = max(0, int(floorf((my - ry + half_height) * sy)));
  ymax = min(H - 1, int(ceilf((my + ry + half_height) * sy)));
}

__global__ void bounded_forward_atomic_kernel(
    int H,
    int W,
    float half_width,
    float half_height,
    const float* __restrict__ means,
    const float* __restrict__ invs,
    const float* __restrict__ weights,
    const float* __restrict__ colors,
    int64_t N,
    float radius_sigma,
    float* __restrict__ image,
    float* __restrict__ alpha) {
  int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= N) {
    return;
  }

  const float* inv = invs + g * 4;
  float mx = means[g * 2 + 0];
  float my = means[g * 2 + 1];
  int xmin, xmax, ymin, ymax;
  gaussian_bbox(inv, mx, my, H, W, half_width, half_height, radius_sigma, xmin, xmax, ymin, ymax);
  if (xmax < xmin || ymax < ymin) {
    return;
  }

  float weight = weights[g];
  float c0 = colors[g * 3 + 0];
  float c1 = colors[g * 3 + 1];
  float c2 = colors[g * 3 + 2];
  float radius2 = radius_sigma * radius_sigma;
  for (int y = ymin; y <= ymax; ++y) {
    float py = pixel_y_from_row(y, H, half_height);
    for (int x = xmin; x <= xmax; ++x) {
      float px = pixel_x_from_col(x, W, half_width);
      float dx = px - mx;
      float dy = py - my;
      float qx = inv[0] * dx + inv[1] * dy;
      float qy = inv[2] * dx + inv[3] * dy;
      float quad = dx * qx + dy * qy;
      if (quad > radius2) {
        continue;
      }
      float power = -0.5f * quad;
      if (power < -50.0f) {
        continue;
      }
      if (power > 0.0f) {
        power = 0.0f;
      }
      float e = __expf(power);
      float w = e * weight;
      int64_t p = int64_t(y) * int64_t(W) + int64_t(x);
      atomicAdd(image + p * 3 + 0, w * c0);
      atomicAdd(image + p * 3 + 1, w * c1);
      atomicAdd(image + p * 3 + 2, w * c2);
      atomicAdd(alpha + p, w);
    }
  }
}

__global__ void compute_tile_rects_kernel(
    int H,
    int W,
    int grid_w,
    int grid_h,
    float half_width,
    float half_height,
    const float* __restrict__ means,
    const float* __restrict__ invs,
    int64_t N,
    float radius_sigma,
    int* __restrict__ tiles_touched,
    int4* __restrict__ tile_rects) {
  int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= N) {
    return;
  }

  const float* inv = invs + g * 4;
  float mx = means[g * 2 + 0];
  float my = means[g * 2 + 1];
  int xmin, xmax, ymin, ymax;
  gaussian_bbox(inv, mx, my, H, W, half_width, half_height, radius_sigma, xmin, xmax, ymin, ymax);
  if (xmax < xmin || ymax < ymin) {
    tiles_touched[g] = 0;
    tile_rects[g] = make_int4(0, 0, 0, 0);
    return;
  }

  int min_tx = max(0, xmin / SLICE_3DGS_TILE_SIZE);
  int min_ty = max(0, ymin / SLICE_3DGS_TILE_SIZE);
  int max_tx = min(grid_w, xmax / SLICE_3DGS_TILE_SIZE + 1);
  int max_ty = min(grid_h, ymax / SLICE_3DGS_TILE_SIZE + 1);
  if (min_tx >= max_tx || min_ty >= max_ty) {
    tiles_touched[g] = 0;
    tile_rects[g] = make_int4(0, 0, 0, 0);
    return;
  }
  tile_rects[g] = make_int4(min_tx, min_ty, max_tx, max_ty);
  tiles_touched[g] = (max_tx - min_tx) * (max_ty - min_ty);
}

__global__ void generate_tile_keys_kernel(
    int64_t N,
    int grid_w,
    const int4* __restrict__ tile_rects,
    const int* __restrict__ tile_offsets,
    uint64_t* __restrict__ keys_out,
    int* __restrict__ values_out) {
  int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= N) {
    return;
  }
  int offset = tile_offsets[g];
  int count = tile_offsets[g + 1] - offset;
  if (count <= 0) {
    return;
  }
  int4 rect = tile_rects[g];
  for (int ty = rect.y; ty < rect.w; ++ty) {
    for (int tx = rect.x; tx < rect.z; ++tx) {
      uint64_t tile_id = static_cast<uint64_t>(ty * grid_w + tx);
      keys_out[offset] = tile_id << 32;
      values_out[offset] = static_cast<int>(g);
      offset++;
    }
  }
}

__global__ void identify_tile_ranges_kernel(
    int num_instances,
    const uint64_t* __restrict__ sorted_keys,
    int* __restrict__ tile_range_starts,
    int* __restrict__ tile_range_ends) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_instances) {
    return;
  }
  int cur_tile = static_cast<int>(sorted_keys[idx] >> 32);
  if (idx == 0) {
    tile_range_starts[cur_tile] = 0;
  } else {
    int prev_tile = static_cast<int>(sorted_keys[idx - 1] >> 32);
    if (cur_tile != prev_tile) {
      tile_range_ends[prev_tile] = idx;
      tile_range_starts[cur_tile] = idx;
    }
  }
  if (idx == num_instances - 1) {
    tile_range_ends[cur_tile] = num_instances;
  }
}

__global__ void tile_binned_forward_kernel(
    int H,
    int W,
    int grid_w,
    float half_width,
    float half_height,
    const float* __restrict__ means,
    const float* __restrict__ invs,
    const float* __restrict__ weights,
    const float* __restrict__ colors,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_range_starts,
    const int* __restrict__ tile_range_ends,
    float radius_sigma,
    float* __restrict__ image,
    float* __restrict__ alpha) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) {
    return;
  }

  int tile_id = (y / SLICE_3DGS_TILE_SIZE) * grid_w + (x / SLICE_3DGS_TILE_SIZE);
  int start = tile_range_starts[tile_id];
  int end = tile_range_ends[tile_id];
  int64_t p = int64_t(y) * int64_t(W) + int64_t(x);
  if (start < 0 || start >= end) {
    image[p * 3 + 0] = 0.0f;
    image[p * 3 + 1] = 0.0f;
    image[p * 3 + 2] = 0.0f;
    alpha[p] = 0.0f;
    return;
  }

  float px = pixel_x_from_col(x, W, half_width);
  float py = pixel_y_from_row(y, H, half_height);
  float radius2 = radius_sigma * radius_sigma;
  float out0 = 0.0f;
  float out1 = 0.0f;
  float out2 = 0.0f;
  float aout = 0.0f;

  for (int i = start; i < end; ++i) {
    int g = sorted_gauss_ids[i];
    const float* inv = invs + g * 4;
    float dx = px - means[g * 2 + 0];
    float dy = py - means[g * 2 + 1];
    float qx = inv[0] * dx + inv[1] * dy;
    float qy = inv[2] * dx + inv[3] * dy;
    float quad = dx * qx + dy * qy;
    if (quad > radius2) {
      continue;
    }
    float power = -0.5f * quad;
    if (power < -50.0f) {
      continue;
    }
    if (power > 0.0f) {
      power = 0.0f;
    }
    float e = __expf(power);
    float w = e * weights[g];
    out0 += w * colors[g * 3 + 0];
    out1 += w * colors[g * 3 + 1];
    out2 += w * colors[g * 3 + 2];
    aout += w;
  }

  image[p * 3 + 0] = out0;
  image[p * 3 + 1] = out1;
  image[p * 3 + 2] = out2;
  alpha[p] = aout;
}

__global__ void tile_binned_backward_kernel(
    int H,
    int W,
    int grid_w,
    float half_width,
    float half_height,
    const float* __restrict__ means,
    const float* __restrict__ invs,
    const float* __restrict__ weights,
    const float* __restrict__ colors,
    const float* __restrict__ grad_image,
    const float* __restrict__ grad_alpha,
    const int* __restrict__ sorted_gauss_ids,
    const int* __restrict__ tile_range_starts,
    const int* __restrict__ tile_range_ends,
    float radius_sigma,
    float* __restrict__ grad_means,
    float* __restrict__ grad_invs,
    float* __restrict__ grad_weights,
    float* __restrict__ grad_colors) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) {
    return;
  }

  int tile_id = (y / SLICE_3DGS_TILE_SIZE) * grid_w + (x / SLICE_3DGS_TILE_SIZE);
  int start = tile_range_starts[tile_id];
  int end = tile_range_ends[tile_id];
  if (start < 0 || start >= end) {
    return;
  }

  int64_t p = int64_t(y) * int64_t(W) + int64_t(x);
  float px = pixel_x_from_col(x, W, half_width);
  float py = pixel_y_from_row(y, H, half_height);
  float go0 = grad_image[p * 3 + 0];
  float go1 = grad_image[p * 3 + 1];
  float go2 = grad_image[p * 3 + 2];
  float ga = grad_alpha[p];
  float radius2 = radius_sigma * radius_sigma;

  for (int i = start; i < end; ++i) {
    int g = sorted_gauss_ids[i];
    const float* inv = invs + g * 4;
    float dx = px - means[g * 2 + 0];
    float dy = py - means[g * 2 + 1];
    float qx = inv[0] * dx + inv[1] * dy;
    float qy = inv[2] * dx + inv[3] * dy;
    float quad = dx * qx + dy * qy;
    if (quad > radius2) {
      continue;
    }
    float power = -0.5f * quad;
    if (power < -50.0f) {
      continue;
    }
    if (power > 0.0f) {
      power = 0.0f;
    }

    float e = __expf(power);
    float weight = weights[g];
    float c0 = colors[g * 3 + 0];
    float c1 = colors[g * 3 + 1];
    float c2 = colors[g * 3 + 2];
    float dot_color = go0 * c0 + go1 * c1 + go2 * c2;
    float dL_dw = e * (dot_color + ga);
    float dL_de = weight * (dot_color + ga);

    atomicAdd(grad_weights + g, dL_dw);
    atomicAdd(grad_colors + g * 3 + 0, e * weight * go0);
    atomicAdd(grad_colors + g * 3 + 1, e * weight * go1);
    atomicAdd(grad_colors + g * 3 + 2, e * weight * go2);

    float dL_dpower = dL_de * e;
    float common = -0.5f * dL_dpower;
    atomicAdd(grad_invs + g * 4 + 0, common * dx * dx);
    atomicAdd(grad_invs + g * 4 + 1, common * dx * dy);
    atomicAdd(grad_invs + g * 4 + 2, common * dy * dx);
    atomicAdd(grad_invs + g * 4 + 3, common * dy * dy);

    float sym_qx = (inv[0] + inv[0]) * dx + (inv[1] + inv[2]) * dy;
    float sym_qy = (inv[2] + inv[1]) * dx + (inv[3] + inv[3]) * dy;
    atomicAdd(grad_means + g * 2 + 0, 0.5f * dL_dpower * sym_qx);
    atomicAdd(grad_means + g * 2 + 1, 0.5f * dL_dpower * sym_qy);
  }
}

__global__ void bounded_backward_kernel(
    int H,
    int W,
    float half_width,
    float half_height,
    const float* __restrict__ means,
    const float* __restrict__ invs,
    const float* __restrict__ weights,
    const float* __restrict__ colors,
    const float* __restrict__ grad_image,
    const float* __restrict__ grad_alpha,
    int64_t N,
    float radius_sigma,
    float* __restrict__ grad_means,
    float* __restrict__ grad_invs,
    float* __restrict__ grad_weights,
    float* __restrict__ grad_colors) {
  int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= N) {
    return;
  }

  const float* inv = invs + g * 4;
  float mx = means[g * 2 + 0];
  float my = means[g * 2 + 1];
  int xmin, xmax, ymin, ymax;
  gaussian_bbox(inv, mx, my, H, W, half_width, half_height, radius_sigma, xmin, xmax, ymin, ymax);
  if (xmax < xmin || ymax < ymin) {
    return;
  }

  float weight = weights[g];
  float c0 = colors[g * 3 + 0];
  float c1 = colors[g * 3 + 1];
  float c2 = colors[g * 3 + 2];
  float radius2 = radius_sigma * radius_sigma;
  float gm0 = 0.0f;
  float gm1 = 0.0f;
  float gi0 = 0.0f;
  float gi1 = 0.0f;
  float gi2 = 0.0f;
  float gi3 = 0.0f;
  float gw = 0.0f;
  float gc0 = 0.0f;
  float gc1 = 0.0f;
  float gc2 = 0.0f;

  for (int y = ymin; y <= ymax; ++y) {
    float py = pixel_y_from_row(y, H, half_height);
    for (int x = xmin; x <= xmax; ++x) {
      float px = pixel_x_from_col(x, W, half_width);
      float dx = px - mx;
      float dy = py - my;
      float qx = inv[0] * dx + inv[1] * dy;
      float qy = inv[2] * dx + inv[3] * dy;
      float quad = dx * qx + dy * qy;
      if (quad > radius2) {
        continue;
      }
      float power = -0.5f * quad;
      if (power < -50.0f) {
        continue;
      }
      if (power > 0.0f) {
        power = 0.0f;
      }
      float e = __expf(power);
      int64_t p = int64_t(y) * int64_t(W) + int64_t(x);
      float go0 = grad_image[p * 3 + 0];
      float go1 = grad_image[p * 3 + 1];
      float go2 = grad_image[p * 3 + 2];
      float ga = grad_alpha[p];
      float dot_color = go0 * c0 + go1 * c1 + go2 * c2;
      float dL_dw = e * (dot_color + ga);
      float dL_de = weight * (dot_color + ga);

      gw += dL_dw;
      gc0 += e * weight * go0;
      gc1 += e * weight * go1;
      gc2 += e * weight * go2;

      float dL_dpower = dL_de * e;
      float common = -0.5f * dL_dpower;
      gi0 += common * dx * dx;
      gi1 += common * dx * dy;
      gi2 += common * dy * dx;
      gi3 += common * dy * dy;

      float sym_qx = (inv[0] + inv[0]) * dx + (inv[1] + inv[2]) * dy;
      float sym_qy = (inv[2] + inv[1]) * dx + (inv[3] + inv[3]) * dy;
      gm0 += 0.5f * dL_dpower * sym_qx;
      gm1 += 0.5f * dL_dpower * sym_qy;
    }
  }

  grad_means[g * 2 + 0] = gm0;
  grad_means[g * 2 + 1] = gm1;
  grad_invs[g * 4 + 0] = gi0;
  grad_invs[g * 4 + 1] = gi1;
  grad_invs[g * 4 + 2] = gi2;
  grad_invs[g * 4 + 3] = gi3;
  grad_weights[g] = gw;
  grad_colors[g * 3 + 0] = gc0;
  grad_colors[g * 3 + 1] = gc1;
  grad_colors[g * 3 + 2] = gc2;
}

struct TileBinning {
  int grid_w;
  int grid_h;
  int n_tiles;
  int total_instances;
  torch::Tensor sorted_gauss_ids;
  torch::Tensor tile_range_starts;
  torch::Tensor tile_range_ends;
};

TileBinning build_tile_binning(
    int image_height,
    int image_width,
    float half_width,
    float half_height,
    torch::Tensor means2d,
    torch::Tensor inv2d,
    int64_t N,
    float radius_sigma,
    cudaStream_t stream) {
  int threads = 256;
  int blocks = static_cast<int>((N + threads - 1) / threads);
  int grid_w = (image_width + SLICE_3DGS_TILE_SIZE - 1) / SLICE_3DGS_TILE_SIZE;
  int grid_h = (image_height + SLICE_3DGS_TILE_SIZE - 1) / SLICE_3DGS_TILE_SIZE;
  int n_tiles = grid_w * grid_h;
  auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(means2d.device());
  auto opts_long = torch::TensorOptions().dtype(torch::kInt64).device(means2d.device());
  auto opts_byte = torch::TensorOptions().dtype(torch::kByte).device(means2d.device());

  auto tiles_touched = torch::zeros({N + 1}, opts_int);
  auto tile_rects = torch::empty({N * 4}, opts_int);
  compute_tile_rects_kernel<<<blocks, threads, 0, stream>>>(
      image_height,
      image_width,
      grid_w,
      grid_h,
      half_width,
      half_height,
      means2d.data_ptr<float>(),
      inv2d.data_ptr<float>(),
      N,
      radius_sigma,
      tiles_touched.data_ptr<int>(),
      reinterpret_cast<int4*>(tile_rects.data_ptr<int>()));
  check_cuda_kernel("compute_tile_rects_kernel");

  auto tile_offsets = torch::empty({N + 1}, opts_int);
  void* temp_storage_ptr = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr,
      temp_storage_bytes,
      tiles_touched.data_ptr<int>(),
      tile_offsets.data_ptr<int>(),
      N + 1,
      stream);
  auto temp_storage = torch::empty({static_cast<long>(temp_storage_bytes)}, opts_byte);
  temp_storage_ptr = temp_storage.data_ptr();
  cub::DeviceScan::ExclusiveSum(
      temp_storage_ptr,
      temp_storage_bytes,
      tiles_touched.data_ptr<int>(),
      tile_offsets.data_ptr<int>(),
      N + 1,
      stream);

  int total_instances = wait_and_copy_scalar_from_device<int>(
      tile_offsets.data_ptr<int>() + N, stream, "total_tile_instances");
  auto tile_range_starts = torch::full({n_tiles}, -1, opts_int);
  auto tile_range_ends = torch::full({n_tiles}, -1, opts_int);
  if (total_instances == 0) {
    return {grid_w, grid_h, n_tiles, total_instances, torch::empty({0}, opts_int), tile_range_starts, tile_range_ends};
  }

  auto keys_buf_1 = torch::empty({total_instances}, opts_long);
  auto keys_buf_2 = torch::empty({total_instances}, opts_long);
  auto vals_buf_1 = torch::empty({total_instances}, opts_int);
  auto vals_buf_2 = torch::empty({total_instances}, opts_int);
  uint64_t* keys_in = reinterpret_cast<uint64_t*>(keys_buf_1.data_ptr<int64_t>());
  uint64_t* keys_out = reinterpret_cast<uint64_t*>(keys_buf_2.data_ptr<int64_t>());
  int* vals_in = vals_buf_1.data_ptr<int>();
  int* vals_out = vals_buf_2.data_ptr<int>();

  generate_tile_keys_kernel<<<blocks, threads, 0, stream>>>(
      N,
      grid_w,
      reinterpret_cast<int4*>(tile_rects.data_ptr<int>()),
      tile_offsets.data_ptr<int>(),
      keys_in,
      vals_in);
  check_cuda_kernel("generate_tile_keys_kernel");

  cub::DoubleBuffer<uint64_t> keys(keys_in, keys_out);
  cub::DoubleBuffer<int> values(vals_in, vals_out);
  size_t sort_temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr,
      sort_temp_storage_bytes,
      keys,
      values,
      total_instances,
      32,
      64,
      stream);
  if (temp_storage.numel() < static_cast<int64_t>(sort_temp_storage_bytes)) {
    temp_storage = torch::empty({static_cast<long>(sort_temp_storage_bytes)}, opts_byte);
  }
  temp_storage_ptr = temp_storage.data_ptr();
  cub::DeviceRadixSort::SortPairs(
      temp_storage_ptr,
      sort_temp_storage_bytes,
      keys,
      values,
      total_instances,
      32,
      64,
      stream);

  int range_blocks = (total_instances + threads - 1) / threads;
  identify_tile_ranges_kernel<<<range_blocks, threads, 0, stream>>>(
      total_instances,
      keys.Current(),
      tile_range_starts.data_ptr<int>(),
      tile_range_ends.data_ptr<int>());
  check_cuda_kernel("identify_tile_ranges_kernel");

  torch::Tensor sorted_gauss_ids = values.Current() == vals_buf_1.data_ptr<int>() ? vals_buf_1 : vals_buf_2;
  return {grid_w, grid_h, n_tiles, total_instances, sorted_gauss_ids, tile_range_starts, tile_range_ends};
}

}  // namespace

std::vector<torch::Tensor> additive_splat_forward_cuda(
    torch::Tensor pixels,
    torch::Tensor means2d,
    torch::Tensor inv2d,
    torch::Tensor weights,
    torch::Tensor colors) {
  CHECK_INPUT(pixels);
  CHECK_INPUT(means2d);
  CHECK_INPUT(inv2d);
  CHECK_INPUT(weights);
  CHECK_INPUT(colors);
  TORCH_CHECK(pixels.dim() == 2 && pixels.size(1) == 2, "pixels must have shape (P, 2)");
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "means2d must have shape (N, 2)");
  TORCH_CHECK(inv2d.dim() == 3 && inv2d.size(1) == 2 && inv2d.size(2) == 2, "inv2d must have shape (N, 2, 2)");
  TORCH_CHECK(weights.dim() == 2 && weights.size(1) == 1, "weights must have shape (N, 1)");
  TORCH_CHECK(colors.dim() == 2 && colors.size(1) == 3, "colors must have shape (N, 3)");
  int64_t P = pixels.size(0);
  int64_t N = means2d.size(0);
  TORCH_CHECK(inv2d.size(0) == N && weights.size(0) == N && colors.size(0) == N, "Gaussian dimension mismatch");

  auto image = torch::empty({P, 3}, pixels.options());
  auto alpha = torch::empty({P, 1}, pixels.options());
  int threads = 128;
  int blocks = static_cast<int>((P + threads - 1) / threads);
  forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      pixels.data_ptr<float>(),
      means2d.data_ptr<float>(),
      inv2d.data_ptr<float>(),
      weights.data_ptr<float>(),
      colors.data_ptr<float>(),
      P,
      N,
      image.data_ptr<float>(),
      alpha.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {image, alpha};
}

std::vector<torch::Tensor> additive_splat_backward_cuda(
    torch::Tensor pixels,
    torch::Tensor means2d,
    torch::Tensor inv2d,
    torch::Tensor weights,
    torch::Tensor colors,
    torch::Tensor grad_image,
    torch::Tensor grad_alpha) {
  CHECK_INPUT(pixels);
  CHECK_INPUT(means2d);
  CHECK_INPUT(inv2d);
  CHECK_INPUT(weights);
  CHECK_INPUT(colors);
  CHECK_INPUT(grad_image);
  CHECK_INPUT(grad_alpha);
  int64_t P = pixels.size(0);
  int64_t N = means2d.size(0);

  auto grad_means = torch::zeros_like(means2d);
  auto grad_invs = torch::zeros_like(inv2d);
  auto grad_weights = torch::zeros_like(weights);
  auto grad_colors = torch::zeros_like(colors);
  int threads = 128;
  int blocks = static_cast<int>((P + threads - 1) / threads);
  backward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      pixels.data_ptr<float>(),
      means2d.data_ptr<float>(),
      inv2d.data_ptr<float>(),
      weights.data_ptr<float>(),
      colors.data_ptr<float>(),
      grad_image.data_ptr<float>(),
      grad_alpha.data_ptr<float>(),
      P,
      N,
      grad_means.data_ptr<float>(),
      grad_invs.data_ptr<float>(),
      grad_weights.data_ptr<float>(),
      grad_colors.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_means, grad_invs, grad_weights, grad_colors};
}

std::vector<torch::Tensor> additive_splat_bounded_forward_cuda(
    int64_t image_height,
    int64_t image_width,
    float half_width,
    float half_height,
    torch::Tensor means2d,
    torch::Tensor inv2d,
    torch::Tensor weights,
    torch::Tensor colors,
    float radius_sigma) {
  CHECK_INPUT(means2d);
  CHECK_INPUT(inv2d);
  CHECK_INPUT(weights);
  CHECK_INPUT(colors);
  TORCH_CHECK(image_height > 0 && image_width > 0, "image dimensions must be positive");
  TORCH_CHECK(half_width > 0.0f && half_height > 0.0f, "physical half extents must be positive");
  TORCH_CHECK(radius_sigma > 0.0f, "radius_sigma must be positive");
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "means2d must have shape (N, 2)");
  TORCH_CHECK(inv2d.dim() == 3 && inv2d.size(1) == 2 && inv2d.size(2) == 2, "inv2d must have shape (N, 2, 2)");
  TORCH_CHECK(weights.dim() == 2 && weights.size(1) == 1, "weights must have shape (N, 1)");
  TORCH_CHECK(colors.dim() == 2 && colors.size(1) == 3, "colors must have shape (N, 3)");
  int64_t N = means2d.size(0);
  TORCH_CHECK(inv2d.size(0) == N && weights.size(0) == N && colors.size(0) == N, "Gaussian dimension mismatch");
  int64_t P = image_height * image_width;
  auto image = torch::zeros({P, 3}, means2d.options());
  auto alpha = torch::zeros({P, 1}, means2d.options());
  if (N == 0 || P == 0) {
    return {image, alpha};
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int threads = 256;
  int blocks = static_cast<int>((N + threads - 1) / threads);
  if (N < SLICE_3DGS_TILE_BIN_MIN_GAUSSIANS) {
    bounded_forward_atomic_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        static_cast<int>(image_height),
        static_cast<int>(image_width),
        half_width,
        half_height,
        means2d.data_ptr<float>(),
        inv2d.data_ptr<float>(),
        weights.data_ptr<float>(),
        colors.data_ptr<float>(),
        N,
        radius_sigma,
        image.data_ptr<float>(),
        alpha.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {image, alpha};
  }

  TileBinning binning = build_tile_binning(
      static_cast<int>(image_height),
      static_cast<int>(image_width),
      half_width,
      half_height,
      means2d,
      inv2d,
      N,
      radius_sigma,
      stream);
  if (binning.total_instances == 0) {
    return {image, alpha};
  }

  dim3 block_dim(SLICE_3DGS_TILE_SIZE, SLICE_3DGS_TILE_SIZE);
  dim3 grid_dim(binning.grid_w, binning.grid_h);
  tile_binned_forward_kernel<<<grid_dim, block_dim, 0, stream>>>(
      static_cast<int>(image_height),
      static_cast<int>(image_width),
      binning.grid_w,
      half_width,
      half_height,
      means2d.data_ptr<float>(),
      inv2d.data_ptr<float>(),
      weights.data_ptr<float>(),
      colors.data_ptr<float>(),
      binning.sorted_gauss_ids.data_ptr<int>(),
      binning.tile_range_starts.data_ptr<int>(),
      binning.tile_range_ends.data_ptr<int>(),
      radius_sigma,
      image.data_ptr<float>(),
      alpha.data_ptr<float>());
  check_cuda_kernel("tile_binned_forward_kernel");
  return {image, alpha};
}

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
    float radius_sigma) {
  CHECK_INPUT(means2d);
  CHECK_INPUT(inv2d);
  CHECK_INPUT(weights);
  CHECK_INPUT(colors);
  CHECK_INPUT(grad_image);
  CHECK_INPUT(grad_alpha);
  int64_t N = means2d.size(0);
  auto grad_means = torch::zeros_like(means2d);
  auto grad_invs = torch::zeros_like(inv2d);
  auto grad_weights = torch::zeros_like(weights);
  auto grad_colors = torch::zeros_like(colors);
  if (N == 0 || image_height * image_width == 0) {
    return {grad_means, grad_invs, grad_weights, grad_colors};
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int threads = 128;
  int blocks = static_cast<int>((N + threads - 1) / threads);
  if (N < SLICE_3DGS_TILE_BIN_MIN_GAUSSIANS) {
    bounded_backward_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<int>(image_height),
        static_cast<int>(image_width),
        half_width,
        half_height,
        means2d.data_ptr<float>(),
        inv2d.data_ptr<float>(),
        weights.data_ptr<float>(),
        colors.data_ptr<float>(),
        grad_image.data_ptr<float>(),
        grad_alpha.data_ptr<float>(),
        N,
        radius_sigma,
        grad_means.data_ptr<float>(),
        grad_invs.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        grad_colors.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_means, grad_invs, grad_weights, grad_colors};
  }

  TileBinning binning = build_tile_binning(
      static_cast<int>(image_height),
      static_cast<int>(image_width),
      half_width,
      half_height,
      means2d,
      inv2d,
      N,
      radius_sigma,
      stream);
  if (binning.total_instances == 0) {
    return {grad_means, grad_invs, grad_weights, grad_colors};
  }

  dim3 block_dim(SLICE_3DGS_TILE_SIZE, SLICE_3DGS_TILE_SIZE);
  dim3 grid_dim(binning.grid_w, binning.grid_h);
  tile_binned_backward_kernel<<<grid_dim, block_dim, 0, stream>>>(
      static_cast<int>(image_height),
      static_cast<int>(image_width),
      binning.grid_w,
      half_width,
      half_height,
      means2d.data_ptr<float>(),
      inv2d.data_ptr<float>(),
      weights.data_ptr<float>(),
      colors.data_ptr<float>(),
      grad_image.data_ptr<float>(),
      grad_alpha.data_ptr<float>(),
      binning.sorted_gauss_ids.data_ptr<int>(),
      binning.tile_range_starts.data_ptr<int>(),
      binning.tile_range_ends.data_ptr<int>(),
      radius_sigma,
      grad_means.data_ptr<float>(),
      grad_invs.data_ptr<float>(),
      grad_weights.data_ptr<float>(),
      grad_colors.data_ptr<float>());
  check_cuda_kernel("tile_binned_backward_kernel");
  return {grad_means, grad_invs, grad_weights, grad_colors};
}
