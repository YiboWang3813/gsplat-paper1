#include "covariance.cuh"
#include "shader/shared.cuh"
#include "load_utils.cuh"
#include "math_utils.cuh"

__device__ __forceinline__ int lower_bound_depth(
    const float* __restrict__ sorted_depths,
    int offset,
    int count,
    float value
) {
    int lo = 0;
    int hi = count;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (sorted_depths[offset + mid] < value) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void render_bmode_kernel(
    int W,
    int H,
    const int* __restrict__ ray_counts,
    const int* __restrict__ ray_offsets,
    const int* __restrict__ ray_indices,
    const float* __restrict__ sorted_depths,
    const float4* __restrict__ means,
    const float4* __restrict__ scales,
    const float4* __restrict__ quats,
    const float* __restrict__ attenuations,
    const float4* __restrict__ sh_coeffs,
    const float* __restrict__ pixel_positions,
    const float* __restrict__ ray_origins,
    const float* __restrict__ ray_dirs,
    const float* __restrict__ axis_elevational,
    const float* __restrict__ ray_perturbations,
    float k_sigma,
    float quadrature_window_radius_mm,
    bool local_transmission_window_only,
    float local_transmission_max_window_mm,
    float* __restrict__ output_image
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int count = ray_counts[x];
    if (count == 0) {
        output_image[y * W + x] = 0.0f;
        return;
    }
    int offset = ray_offsets[x];

    const float* p_ptr = pixel_positions + (x * H + y) * 3;
    float3 p = make_float3(p_ptr[0], p_ptr[1], p_ptr[2]);
    float3 ray_o = make_float3(ray_origins[x * 3 + 0], ray_origins[x * 3 + 1], ray_origins[x * 3 + 2]);
    float3 axis_ele = make_float3(axis_elevational[0], axis_elevational[1], axis_elevational[2]);
    ray_o = ray_o + axis_ele * ray_perturbations[x];
    float3 ray_d = make_float3(ray_dirs[x * 3 + 0], ray_dirs[x * 3 + 1], ray_dirs[x * 3 + 2]);
    float t_pixel = fmaxf(0.0f, dot_prod(p - ray_o, ray_d));

    float transmission = 1.0f;
    float echo_num = 0.0f;
    float echo_den = 0.0f;
    float s_accum = 0.0f;

    int begin_k = 0;
    float upper_depth = 1.0e30f;
    if (local_transmission_window_only) {
        begin_k = lower_bound_depth(sorted_depths, offset, count, t_pixel - local_transmission_max_window_mm);
        upper_depth = t_pixel + local_transmission_max_window_mm;
    }

    for (int k = begin_k; k < count; ++k) {
        if (local_transmission_window_only && sorted_depths[offset + k] > upper_depth) break;
        int idx = ray_indices[offset + k];
        float3 mu = xyz(means[idx]);
        float3 s = xyz(scales[idx]);
        float t_gauss = dot_prod(mu - ray_o, ray_d);
        float max_scale = fmaxf(s.x, fmaxf(s.y, s.z));
        float depth_radius = max_scale * k_sigma;
        bool needs_transmission = t_gauss < t_pixel;
        if (local_transmission_window_only) {
            needs_transmission = needs_transmission && ((t_pixel - t_gauss) <= depth_radius + quadrature_window_radius_mm);
        }
        bool needs_echo = fabsf(t_gauss - t_pixel) <= depth_radius;
        if (!needs_transmission && !needs_echo) continue;

        float4 q = quats[idx];

        float R[9];
        compute_rot_from_quat(q, R);

        float raw_tau = attenuations[idx];
        float tau = 1.0f / (1.0f + __expf(-raw_tau));
        if (needs_transmission) {
            float psi = compute_ray_gaussian_integral(ray_o, ray_d, mu, s, R, t_pixel, quadrature_window_radius_mm);
            float T_i = tau + (1.0f - tau) * __expf(-psi);
            transmission *= T_i;
        }

        if (needs_echo) {
            float SigmaInv[9];
            compute_cov3d_from_scale_rot(s, R, SigmaInv);
            float w = compute_gaussian_weight(p, mu, SigmaInv);
            float I = eval_sh_degree1(sh_coeffs[idx], ray_d);
            echo_num += I * w;
            echo_den += w;
            s_accum += w;
        }
    }

    float gamma = 1.0f - __expf(-s_accum);
    float echo_signal = echo_num / (echo_den + 1e-6f);
    float b_mode_val = transmission * (gamma * echo_signal);
    output_image[y * W + x] = b_mode_val;
}
