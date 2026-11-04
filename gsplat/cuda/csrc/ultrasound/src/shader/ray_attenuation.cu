#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

#include "covariance.cuh"
#include "mapper/core.cuh"
#include "shader/ray_attenuation.cuh"
#include "shader/shared.cuh"
#include "shader/slice_grid_geometry.cuh"
#include "shader/us_focus_device.cuh"

namespace {

template <typename T>
T copy_scalar_d2h_ray(const T* device_ptr, cudaStream_t stream) {
    T host_v{};
    TORCH_CHECK(
        cudaMemcpyAsync(&host_v, device_ptr, sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess,
        "ray_csr_mapper D2H copy failed");
    TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "ray_csr_mapper stream sync failed");
    return host_v;
}

int ray_fixed_capacity_per_ray_from_env() {
    const char* raw = std::getenv("OURS_RAY_CSR_FIXED_CAPACITY_PER_RAY");
    if (raw == nullptr || raw[0] == '\0') {
        return 0;
    }
    const int value = std::atoi(raw);
    return value > 0 ? value : 0;
}

int ray_fixed_total_capacity_from_env() {
    const char* raw = std::getenv("OURS_RAY_CSR_FIXED_TOTAL_CAPACITY");
    if (raw == nullptr || raw[0] == '\0') {
        return 0;
    }
    const int value = std::atoi(raw);
    return value > 0 ? value : 0;
}

torch::Tensor ensure_slice_grid_pack_ray(
    const torch::Tensor& slice_grid_geometry,
    const torch::Tensor& ref,
    int H,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm) {
    auto float_opts = ref.options().dtype(torch::kFloat32);
    if (slice_grid_geometry.numel() >= SLICE_GRID_PACK_LEN) {
        return slice_grid_geometry.to(torch::kFloat32).contiguous().view({-1});
    }
    torch::Tensor pack = torch::zeros({SLICE_GRID_PACK_LEN}, float_opts);
    pack[SG_PROBE_TYPE] = static_cast<float>(SLICE_GRID_PROBE_LINEAR);
    pack[SG_H] = static_cast<float>(H);
    pack[SG_W] = static_cast<float>(W);
    pack[SG_DELTA_LAT_MM] = lateral_spacing_mm_per_pix;
    pack[SG_DELTA_AX_MM] = axial_spacing_mm_per_pix;
    pack[SG_NEAR_MM] = axial_near_mm;
    pack[SG_LAT_COL_OFFSET] = static_cast<float>(W - 1) * 0.5f;
    pack[SG_CENTER_LATERAL] = 1.f;
    return pack;
}

torch::Tensor ensure_us_focus_params_ray(const torch::Tensor& us_focus_params, const torch::Tensor& ref) {
    auto float_opts = ref.options().dtype(torch::kFloat32);
    torch::Tensor params = torch::zeros({us_focus_device::US_FOCUS_PARAMS_LEN}, float_opts);
    if (us_focus_params.numel() > 0) {
        const int n = std::min<int>(us_focus_device::US_FOCUS_PARAMS_LEN, static_cast<int>(us_focus_params.numel()));
        params.slice(0, 0, n).copy_(us_focus_params.to(torch::kFloat32).contiguous().view({-1}).slice(0, 0, n));
    }
    return params;
}

__device__ __forceinline__ float3 ray_make_sub3(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 ray_make_add3(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 ray_make_mul3(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float compute_ray_gaussian_integral_ours(
    const float3& ray_origin,
    const float3& ray_dir,
    const float3& mu,
    const float3& scale,
    const float* __restrict__ R,
    float length_mm,
    float quadrature_window_radius_mm,
    int quadrature_points) {
    if (length_mm <= 0.f) {
        return 0.f;
    }

    const float3 diff = ray_make_sub3(ray_origin, mu);
    const float3 o_local = make_float3(
        R[0] * diff.x + R[3] * diff.y + R[6] * diff.z,
        R[1] * diff.x + R[4] * diff.y + R[7] * diff.z,
        R[2] * diff.x + R[5] * diff.y + R[8] * diff.z);
    const float3 d_local = make_float3(
        R[0] * ray_dir.x + R[3] * ray_dir.y + R[6] * ray_dir.z,
        R[1] * ray_dir.x + R[4] * ray_dir.y + R[7] * ray_dir.z,
        R[2] * ray_dir.x + R[5] * ray_dir.y + R[8] * ray_dir.z);

    const float3 inv_s = make_float3(
        1.f / fmaxf(scale.x, 1e-6f),
        1.f / fmaxf(scale.y, 1e-6f),
        1.f / fmaxf(scale.z, 1e-6f));
    const float3 o_g = make_float3(o_local.x * inv_s.x, o_local.y * inv_s.y, o_local.z * inv_s.z);
    const float3 d_g = make_float3(d_local.x * inv_s.x, d_local.y * inv_s.y, d_local.z * inv_s.z);

    const float dg_norm2 = ours_dot3(d_g, d_g);
    const float t_center = -ours_dot3(o_g, d_g) / (dg_norm2 + 1e-8f);
    float t_start = 0.f;
    float t_end = length_mm;
    if (quadrature_window_radius_mm > 0.f) {
        t_start = fmaxf(0.f, t_center - quadrature_window_radius_mm);
        t_end = fminf(length_mm, t_center + quadrature_window_radius_mm);
    }
    if (t_end <= t_start) {
        return 0.f;
    }

    const float half = 0.5f * (t_end - t_start);
    const float center = 0.5f * (t_end + t_start);
    if (quadrature_points <= 1) {
        const float3 x = ray_make_add3(o_g, ray_make_mul3(d_g, center));
        const float f = expf(-0.5f * ours_dot3(x, x));
        return (t_end - t_start) * f;
    }
    if (quadrature_points == 2) {
        constexpr float xi = 0.5773502692f;
        const float t0 = center - half * xi;
        const float t1 = center + half * xi;
        const float3 x0 = ray_make_add3(o_g, ray_make_mul3(d_g, t0));
        const float3 x1 = ray_make_add3(o_g, ray_make_mul3(d_g, t1));
        const float f0 = expf(-0.5f * ours_dot3(x0, x0));
        const float f1 = expf(-0.5f * ours_dot3(x1, x1));
        return half * (f0 + f1);
    }
    constexpr float xi0 = -0.7745966692f;
    constexpr float xi1 = 0.f;
    constexpr float xi2 = 0.7745966692f;
    constexpr float w0 = 0.5555555556f;
    constexpr float w1 = 0.8888888889f;
    constexpr float w2 = 0.5555555556f;
    const float t0 = half * xi0 + center;
    const float t1 = half * xi1 + center;
    const float t2 = half * xi2 + center;
    const float3 x0 = ray_make_add3(o_g, ray_make_mul3(d_g, t0));
    const float3 x1 = ray_make_add3(o_g, ray_make_mul3(d_g, t1));
    const float3 x2 = ray_make_add3(o_g, ray_make_mul3(d_g, t2));
    const float f0 = expf(-0.5f * ours_dot3(x0, x0));
    const float f1 = expf(-0.5f * ours_dot3(x1, x1));
    const float f2 = expf(-0.5f * ours_dot3(x2, x2));
    return half * (w0 * f0 + w1 * f1 + w2 * f2);
}

__device__ __forceinline__ float compute_ray_slice_depth_response_ours(
    const float3& axis_lat,
    const float3& axis_ele,
    const float3& origin,
    float xi_lat,
    const float3& mu,
    const float3& scale) {
    const float3 delta = make_float3(mu.x - origin.x, mu.y - origin.y, mu.z - origin.z);
    const float mu_lat = ours_dot3(delta, axis_lat);
    const float mu_ele = ours_dot3(delta, axis_ele);
    const float d_lat = mu_lat - xi_lat;
    const float d_ele = mu_ele;
    const float sx = fmaxf(scale.x, 1e-6f);
    const float sy = fmaxf(scale.y, 1e-6f);
    const float sz = fmaxf(scale.z, 1e-6f);
    const float sigma = sqrtf(fmaxf(1e-12f, (sx * sx + sy * sy + sz * sz) * (1.f / 3.f)));
    const float inv_sigma = 1.f / fmaxf(sigma, 1e-6f);
    const float q = (d_lat * d_lat + d_ele * d_ele) * inv_sigma * inv_sigma;
    return sigma * expf(-0.5f * q);
}

__device__ __forceinline__ void compute_ray_integral_vjp_ours(
    const float3& ray_origin,
    const float3& ray_dir,
    const float3& mu,
    const float3& scale,
    const float* __restrict__ R,
    float length_mm,
    float quadrature_window_radius_mm,
    int quadrature_points,
    float dL_dIntegral,
    float3& dL_dMean,
    float3& dL_dScale,
    float* __restrict__ dL_dR) {
    if (length_mm <= 0.f || dL_dIntegral == 0.f) {
        return;
    }

    const float3 diff = ray_make_sub3(ray_origin, mu);
    const float3 o_local = make_float3(
        R[0] * diff.x + R[3] * diff.y + R[6] * diff.z,
        R[1] * diff.x + R[4] * diff.y + R[7] * diff.z,
        R[2] * diff.x + R[5] * diff.y + R[8] * diff.z);
    const float3 d_local = make_float3(
        R[0] * ray_dir.x + R[3] * ray_dir.y + R[6] * ray_dir.z,
        R[1] * ray_dir.x + R[4] * ray_dir.y + R[7] * ray_dir.z,
        R[2] * ray_dir.x + R[5] * ray_dir.y + R[8] * ray_dir.z);

    const float3 inv_s = make_float3(
        1.f / fmaxf(scale.x, 1e-6f),
        1.f / fmaxf(scale.y, 1e-6f),
        1.f / fmaxf(scale.z, 1e-6f));
    const float3 o_g = make_float3(o_local.x * inv_s.x, o_local.y * inv_s.y, o_local.z * inv_s.z);
    const float3 d_g = make_float3(d_local.x * inv_s.x, d_local.y * inv_s.y, d_local.z * inv_s.z);

    const float dg_norm2 = ours_dot3(d_g, d_g);
    const float t_center = -ours_dot3(o_g, d_g) / (dg_norm2 + 1e-8f);
    float t_start = 0.f;
    float t_end = length_mm;
    float d_tstart_dtcenter = 0.f;
    float d_tend_dtcenter = 0.f;
    if (quadrature_window_radius_mm > 0.f) {
        const float raw_start = t_center - quadrature_window_radius_mm;
        const float raw_end = t_center + quadrature_window_radius_mm;
        t_start = fmaxf(0.f, raw_start);
        t_end = fminf(length_mm, raw_end);
        d_tstart_dtcenter = raw_start > 0.f ? 1.f : 0.f;
        d_tend_dtcenter = raw_end < length_mm ? 1.f : 0.f;
    }
    if (t_end <= t_start) {
        return;
    }

    const float half = 0.5f * (t_end - t_start);
    const float center = 0.5f * (t_end + t_start);
    float3 dL_dog = make_float3(0.f, 0.f, 0.f);
    float3 dL_ddg = make_float3(0.f, 0.f, 0.f);
    float sample_sum = 0.f;
    float sample_dfdt_xi_sum = 0.f;
    float sample_dfdt_sum = 0.f;

    if (quadrature_points <= 1) {
        const float xi = 0.f;
        const float w = 2.f;
        const float t = center;
        const float3 x = ray_make_add3(o_g, ray_make_mul3(d_g, t));
        const float f = expf(-0.5f * ours_dot3(x, x));
        const float coeff = dL_dIntegral * half * w;
        dL_dog = ray_make_add3(dL_dog, ray_make_mul3(x, -coeff * f));
        dL_ddg = ray_make_add3(dL_ddg, ray_make_mul3(x, -coeff * f * t));
        const float dfdt = -f * ours_dot3(x, d_g);
        sample_sum += w * f;
        sample_dfdt_xi_sum += w * dfdt * xi;
        sample_dfdt_sum += w * dfdt;
    } else if (quadrature_points == 2) {
        constexpr float xi = 0.5773502692f;
        const float xis[2] = {-xi, xi};
        for (int i = 0; i < 2; ++i) {
            const float t = half * xis[i] + center;
            const float3 x = ray_make_add3(o_g, ray_make_mul3(d_g, t));
            const float f = expf(-0.5f * ours_dot3(x, x));
            const float coeff = dL_dIntegral * half;
            dL_dog = ray_make_add3(dL_dog, ray_make_mul3(x, -coeff * f));
            dL_ddg = ray_make_add3(dL_ddg, ray_make_mul3(x, -coeff * f * t));
            const float dfdt = -f * ours_dot3(x, d_g);
            sample_sum += f;
            sample_dfdt_xi_sum += dfdt * xis[i];
            sample_dfdt_sum += dfdt;
        }
    } else {
        constexpr float xis[3] = {-0.7745966692f, 0.f, 0.7745966692f};
        constexpr float ws[3] = {0.5555555556f, 0.8888888889f, 0.5555555556f};
        for (int i = 0; i < 3; ++i) {
            const float t = half * xis[i] + center;
            const float3 x = ray_make_add3(o_g, ray_make_mul3(d_g, t));
            const float f = expf(-0.5f * ours_dot3(x, x));
            const float coeff = dL_dIntegral * half * ws[i];
            dL_dog = ray_make_add3(dL_dog, ray_make_mul3(x, -coeff * f));
            dL_ddg = ray_make_add3(dL_ddg, ray_make_mul3(x, -coeff * f * t));
            const float dfdt = -f * ours_dot3(x, d_g);
            sample_sum += ws[i] * f;
            sample_dfdt_xi_sum += ws[i] * dfdt * xis[i];
            sample_dfdt_sum += ws[i] * dfdt;
        }
    }

    if (d_tstart_dtcenter != 0.f || d_tend_dtcenter != 0.f) {
        const float dI_dhalf = sample_sum + half * sample_dfdt_xi_sum;
        const float dI_dc = half * sample_dfdt_sum;
        const float dI_dtstart = -0.5f * dI_dhalf + 0.5f * dI_dc;
        const float dI_dtend = 0.5f * dI_dhalf + 0.5f * dI_dc;
        const float dL_dtcenter = dL_dIntegral * (dI_dtstart * d_tstart_dtcenter + dI_dtend * d_tend_dtcenter);
        const float den = dg_norm2 + 1e-8f;
        const float num = ours_dot3(o_g, d_g);
        const float inv_den = 1.f / den;
        const float inv_den2 = inv_den * inv_den;
        dL_dog = ray_make_add3(dL_dog, ray_make_mul3(d_g, -dL_dtcenter * inv_den));
        dL_ddg = ray_make_add3(
            dL_ddg,
            make_float3(
                dL_dtcenter * (-o_g.x * inv_den + 2.f * num * d_g.x * inv_den2),
                dL_dtcenter * (-o_g.y * inv_den + 2.f * num * d_g.y * inv_den2),
                dL_dtcenter * (-o_g.z * inv_den + 2.f * num * d_g.z * inv_den2)));
    }

    const float3 dL_dolocal = make_float3(dL_dog.x * inv_s.x, dL_dog.y * inv_s.y, dL_dog.z * inv_s.z);
    const float3 dL_ddlocal = make_float3(dL_ddg.x * inv_s.x, dL_ddg.y * inv_s.y, dL_ddg.z * inv_s.z);

    dL_dScale.x += dL_dog.x * (-o_g.x * inv_s.x) + dL_ddg.x * (-d_g.x * inv_s.x);
    dL_dScale.y += dL_dog.y * (-o_g.y * inv_s.y) + dL_ddg.y * (-d_g.y * inv_s.y);
    dL_dScale.z += dL_dog.z * (-o_g.z * inv_s.z) + dL_ddg.z * (-d_g.z * inv_s.z);

    dL_dMean.x += -(R[0] * dL_dolocal.x + R[1] * dL_dolocal.y + R[2] * dL_dolocal.z);
    dL_dMean.y += -(R[3] * dL_dolocal.x + R[4] * dL_dolocal.y + R[5] * dL_dolocal.z);
    dL_dMean.z += -(R[6] * dL_dolocal.x + R[7] * dL_dolocal.y + R[8] * dL_dolocal.z);

    dL_dR[0] += dL_dolocal.x * diff.x + dL_ddlocal.x * ray_dir.x;
    dL_dR[1] += dL_dolocal.y * diff.x + dL_ddlocal.y * ray_dir.x;
    dL_dR[2] += dL_dolocal.z * diff.x + dL_ddlocal.z * ray_dir.x;
    dL_dR[3] += dL_dolocal.x * diff.y + dL_ddlocal.x * ray_dir.y;
    dL_dR[4] += dL_dolocal.y * diff.y + dL_ddlocal.y * ray_dir.y;
    dL_dR[5] += dL_dolocal.z * diff.y + dL_ddlocal.z * ray_dir.y;
    dL_dR[6] += dL_dolocal.x * diff.z + dL_ddlocal.x * ray_dir.z;
    dL_dR[7] += dL_dolocal.y * diff.z + dL_ddlocal.y * ray_dir.z;
    dL_dR[8] += dL_dolocal.z * diff.z + dL_ddlocal.z * ray_dir.z;
}

__device__ __forceinline__ void compute_ray_slice_depth_response_vjp_ours(
    const float3& axis_lat,
    const float3& axis_ele,
    const float3& origin,
    float xi_lat,
    const float3& mu,
    const float3& scale,
    float dL_dResponse,
    float3& dL_dMean,
    float3& dL_dScale) {
    if (dL_dResponse == 0.f) {
        return;
    }
    const float3 delta = make_float3(mu.x - origin.x, mu.y - origin.y, mu.z - origin.z);
    const float mu_lat = ours_dot3(delta, axis_lat);
    const float mu_ele = ours_dot3(delta, axis_ele);
    const float d_lat = mu_lat - xi_lat;
    const float d_ele = mu_ele;
    const float sx = fmaxf(scale.x, 1e-6f);
    const float sy = fmaxf(scale.y, 1e-6f);
    const float sz = fmaxf(scale.z, 1e-6f);
    const float sigma = sqrtf(fmaxf(1e-12f, (sx * sx + sy * sy + sz * sz) * (1.f / 3.f)));
    const float inv_sigma = 1.f / fmaxf(sigma, 1e-6f);
    const float inv_sigma2 = inv_sigma * inv_sigma;
    const float q = (d_lat * d_lat + d_ele * d_ele) * inv_sigma2;
    const float e = expf(-0.5f * q);

    const float dI_dlat = -e * d_lat * inv_sigma;
    const float dI_dele = -e * d_ele * inv_sigma;
    dL_dMean.x += dL_dResponse * (dI_dlat * axis_lat.x + dI_dele * axis_ele.x);
    dL_dMean.y += dL_dResponse * (dI_dlat * axis_lat.y + dI_dele * axis_ele.y);
    dL_dMean.z += dL_dResponse * (dI_dlat * axis_lat.z + dI_dele * axis_ele.z);

    const float dI_dsigma = e * (1.f + q);
    const float dL_dsigma = dL_dResponse * dI_dsigma;
    const float inv_three_sigma = 1.f / fmaxf(3e-6f, 3.f * sigma);
    dL_dScale.x += dL_dsigma * sx * inv_three_sigma;
    dL_dScale.y += dL_dsigma * sy * inv_three_sigma;
    dL_dScale.z += dL_dsigma * sz * inv_three_sigma;
}

__device__ __forceinline__ void gaussian_probe_stats(
    int g,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    float3 axis_lat,
    float3 axis_ax,
    float3 axis_ele,
    float3 origin,
    float* __restrict__ mu_lat,
    float* __restrict__ mu_ax,
    float* __restrict__ mu_ele,
    float* __restrict__ sigma_lat,
    float* __restrict__ sigma_ele) {
    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;

    float3 mw = make_float3(mptr[0], mptr[1], mptr[2]);
    float3 delta = make_float3(mw.x - origin.x, mw.y - origin.y, mw.z - origin.z);
    *mu_lat = ours_dot3(delta, axis_lat);
    *mu_ax = ours_dot3(delta, axis_ax);
    *mu_ele = ours_dot3(delta, axis_ele);

    float Rg[9];
    ours_quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
    const float sx = fmaxf(sptr[0], 1e-6f);
    const float sy = fmaxf(sptr[1], 1e-6f);
    const float sz = fmaxf(sptr[2], 1e-6f);

    const float lat0 = sx * (Rg[0] * axis_lat.x + Rg[3] * axis_lat.y + Rg[6] * axis_lat.z);
    const float lat1 = sy * (Rg[1] * axis_lat.x + Rg[4] * axis_lat.y + Rg[7] * axis_lat.z);
    const float lat2 = sz * (Rg[2] * axis_lat.x + Rg[5] * axis_lat.y + Rg[8] * axis_lat.z);
    const float ele0 = sx * (Rg[0] * axis_ele.x + Rg[3] * axis_ele.y + Rg[6] * axis_ele.z);
    const float ele1 = sy * (Rg[1] * axis_ele.x + Rg[4] * axis_ele.y + Rg[7] * axis_ele.z);
    const float ele2 = sz * (Rg[2] * axis_ele.x + Rg[5] * axis_ele.y + Rg[8] * axis_ele.z);

    *sigma_lat = sqrtf(fmaxf(1e-12f, lat0 * lat0 + lat1 * lat1 + lat2 * lat2));
    *sigma_ele = sqrtf(fmaxf(1e-12f, ele0 * ele0 + ele1 * ele1 + ele2 * ele2));
}

__global__ void kernel_ray_csr_count(
    int n_gauss,
    int Tw,
    int tile_size_px,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    const float* __restrict__ slice_grid_pack,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ pose,
    int2* __restrict__ ray_ranges,
    int* __restrict__ ray_touched_count,
    float* __restrict__ depth_pix) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) {
        return;
    }

    const float3 axis_lat = make_float3(pose[0], pose[4], pose[8]);
    const float3 axis_ax = make_float3(pose[1], pose[5], pose[9]);
    const float3 axis_ele = ours_normalize3(make_float3(pose[2], pose[6], pose[10]));
    const float3 origin = make_float3(pose[3], pose[7], pose[11]);

    float mu_lat = 0.f;
    float mu_ax = 0.f;
    float mu_ele = 0.f;
    float sigma_lat = 1.f;
    float sigma_ele = 1.f;
    gaussian_probe_stats(
        g,
        gauss_means,
        gauss_scales,
        gauss_quats,
        axis_lat,
        axis_ax,
        axis_ele,
        origin,
        &mu_lat,
        &mu_ax,
        &mu_ele,
        &sigma_lat,
        &sigma_ele);

    const float r_lat = sigma_lat * k_sigma;
    const float r_ele = sigma_ele * k_sigma;
    const float elev_pass = elev_corridor_half_mm + slab_half_thickness_mm + r_ele;
    const float v_pix = (mu_ax - axial_near_mm) / fmaxf(1e-8f, axial_spacing_mm_per_pix);
    depth_pix[g] = v_pix;

    if (fabsf(mu_ele) > elev_pass || mu_ax < axial_near_mm - fabsf(axial_spacing_mm_per_pix * tile_size_px)) {
        ray_ranges[g] = make_int2(0, 0);
        ray_touched_count[g] = 0;
        return;
    }

    const float u_pix = mu_lat / fmaxf(1e-8f, lateral_spacing_mm_per_pix) + slice_grid_pack[SG_LAT_COL_OFFSET];
    const float r_pix = fmaxf(1.0f, r_lat / fmaxf(1e-8f, lateral_spacing_mm_per_pix));
    int min_ray = static_cast<int>(floorf((u_pix - r_pix) / static_cast<float>(tile_size_px)));
    int max_ray = static_cast<int>(ceilf((u_pix + r_pix) / static_cast<float>(tile_size_px)));
    min_ray = max(0, min_ray);
    max_ray = min(Tw, max_ray);

    if (min_ray >= max_ray || u_pix + r_pix < 0.f || u_pix - r_pix >= static_cast<float>(W)) {
        ray_ranges[g] = make_int2(0, 0);
        ray_touched_count[g] = 0;
        return;
    }

    ray_ranges[g] = make_int2(min_ray, max_ray);
    ray_touched_count[g] = max_ray - min_ray;
}

__global__ void kernel_ray_csr_emit(
    int n_gauss,
    const int2* __restrict__ ray_ranges,
    const int* __restrict__ offsets,
    const float* __restrict__ depth_pix,
    int64_t* __restrict__ keys,
    int* __restrict__ values) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) {
        return;
    }
    const int2 rr = ray_ranges[g];
    const int base = offsets[g];
    const uint32_t depth_key = ours_v_to_depth_key(depth_pix[g]);
    for (int ray = rr.x; ray < rr.y; ++ray) {
        const int out = base + (ray - rr.x);
        keys[out] = (static_cast<int64_t>(ray) << 32) | static_cast<int64_t>(depth_key);
        values[out] = g;
    }
}

__global__ void kernel_ray_csr_emit_fixed_per_ray(
    int n_gauss,
    int capacity_per_ray,
    const int2* __restrict__ ray_ranges,
    const float* __restrict__ depth_pix,
    int* __restrict__ ray_counts,
    int* __restrict__ total_count,
    int* __restrict__ overflow_count,
    int64_t* __restrict__ keys,
    int* __restrict__ values) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) {
        return;
    }
    const int2 rr = ray_ranges[g];
    if (rr.y <= rr.x) {
        return;
    }
    const uint32_t depth_key = ours_v_to_depth_key(depth_pix[g]);
    for (int ray = rr.x; ray < rr.y; ++ray) {
        atomicAdd(total_count, 1);
        const int slot = atomicAdd(ray_counts + ray, 1);
        if (slot >= capacity_per_ray) {
            atomicAdd(overflow_count, 1);
            continue;
        }
        const int out = ray * capacity_per_ray + slot;
        keys[out] = (static_cast<int64_t>(ray) << 32) | static_cast<int64_t>(depth_key);
        values[out] = g;
    }
}

__global__ void kernel_ray_csr_emit_fixed_total(
    int n_gauss,
    int capacity,
    const int2* __restrict__ ray_ranges,
    const float* __restrict__ depth_pix,
    int* __restrict__ total_count,
    int* __restrict__ overflow_count,
    int64_t* __restrict__ keys,
    int* __restrict__ values) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gauss) {
        return;
    }
    const int2 rr = ray_ranges[g];
    if (rr.y <= rr.x) {
        return;
    }
    const uint32_t depth_key = ours_v_to_depth_key(depth_pix[g]);
    for (int ray = rr.x; ray < rr.y; ++ray) {
        const int slot = atomicAdd(total_count, 1);
        if (slot >= capacity) {
            atomicAdd(overflow_count, 1);
            continue;
        }
        keys[slot] = (static_cast<int64_t>(ray) << 32) | static_cast<int64_t>(depth_key);
        values[slot] = g;
    }
}

__global__ void kernel_ray_csr_ranges(
    int total,
    int Tw,
    const int64_t* __restrict__ sorted_keys,
    int* __restrict__ ray_start,
    int* __restrict__ ray_end) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int ray = static_cast<int>(sorted_keys[idx] >> 32);
    if (ray < 0 || ray >= Tw) {
        return;
    }
    if (idx == 0 || static_cast<int>(sorted_keys[idx - 1] >> 32) != ray) {
        ray_start[ray] = idx;
    }
    if (idx == total - 1 || static_cast<int>(sorted_keys[idx + 1] >> 32) != ray) {
        ray_end[ray] = idx + 1;
    }
}

__global__ void kernel_ray_csr_sorted_depths(
    int total,
    const int* __restrict__ sorted_ids,
    const float* __restrict__ depth_pix,
    float* __restrict__ sorted_depth_pix) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int g = sorted_ids[idx];
    sorted_depth_pix[idx] = g >= 0 ? depth_pix[g] : CUDART_INF_F;
}

__global__ void kernel_ray_csr_attenuation(
    int H,
    int W,
    int Tw,
    int Th,
    int tile_size_px,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float k_sigma,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm,
    float quadrature_window_radius_mm,
    int quadrature_points,
    int attenuation_formula,
    const float* __restrict__ slice_grid_pack,
    const int* __restrict__ ray_start,
    const int* __restrict__ ray_end,
    const int* __restrict__ sorted_gauss_ids,
    const float* __restrict__ sorted_depth_pix,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_attenuations,
    const float* __restrict__ pose,
    float* __restrict__ modulation) {
    (void)Th;
    (void)k_sigma;
    const int pidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_pixels = H * W;
    if (pidx >= n_pixels) {
        return;
    }

    const int y = pidx / W;
    const int x = pidx - y * W;

    float xi_lat = 0.f;
    float xi_ax = 0.f;
    sg_linear_pixel_to_plane_mm(slice_grid_pack, x, y, &xi_lat, &xi_ax);
    const float xi_v = static_cast<float>(y);
    const float margin_pix = axial_margin_mm / fmaxf(1e-8f, axial_spacing_mm_per_pix);

    const float3 axis_lat = make_float3(pose[0], pose[4], pose[8]);
    const float3 axis_ax = make_float3(pose[1], pose[5], pose[9]);
    const float3 axis_ele = make_float3(pose[2], pose[6], pose[10]);
    const float3 origin = make_float3(pose[3], pose[7], pose[11]);
    const float3 ray_origin =
        ray_make_add3(ray_make_add3(origin, ray_make_mul3(axis_lat, xi_lat)), ray_make_mul3(axis_ax, axial_near_mm));
    const float3 ray_dir = axis_ax;
    const float t_pixel = fmaxf(0.f, xi_ax - axial_near_mm);

    float transmission = 1.f;
    const int safe_tile = max(tile_size_px, 1);
    const int ray_bin = min(max(x / safe_tile, 0), max(Tw - 1, 0));
    const int s = ray_start[ray_bin];
    const int e = ray_end[ray_bin];
    if (s >= 0 && e > s) {
        for (int j = s; j < e; ++j) {
            if (sorted_depth_pix[j] > xi_v + margin_pix) {
                break;
            }
            const int g = sorted_gauss_ids[j];
            const float3 mu = make_float3(gauss_means[g * 3 + 0], gauss_means[g * 3 + 1], gauss_means[g * 3 + 2]);
            const float3 scale = make_float3(
                fmaxf(gauss_scales[g * 3 + 0], 1e-6f),
                fmaxf(gauss_scales[g * 3 + 1], 1e-6f),
                fmaxf(gauss_scales[g * 3 + 2], 1e-6f));
            float response = 0.f;
            if (attenuation_formula == 1) {
                response = compute_ray_slice_depth_response_ours(axis_lat, axis_ele, origin, xi_lat, mu, scale);
            } else {
                const float* qptr = gauss_quats + g * 4;
                float Rg[9];
                ours_quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
                response = compute_ray_gaussian_integral_ours(
                    ray_origin, ray_dir, mu, scale, Rg, t_pixel, quadrature_window_radius_mm, quadrature_points);
            }
            const float psi = response * fmaxf(strength, 0.f);
            if (!isfinite(psi) || psi <= weight_min) {
                continue;
            }
            const float tau = fminf(1.f, fmaxf(0.f, gauss_attenuations[g]));
            const float Ti = tau + (1.f - tau) * expf(-psi);
            transmission *= fminf(1.f, fmaxf(0.f, Ti));
        }
    }

    transmission = fminf(1.f, fmaxf(min_transmission, transmission));
    modulation[pidx] = 1.f - beta + beta * transmission;
}

__global__ void kernel_ray_csr_path_contribution_stats(
    int H,
    int W,
    int Tw,
    int tile_size_px,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float strength,
    float weight_min,
    float axial_margin_mm,
    float quadrature_window_radius_mm,
    int quadrature_points,
    int attenuation_formula,
    const float* __restrict__ slice_grid_pack,
    const int* __restrict__ ray_start,
    const int* __restrict__ ray_end,
    const int* __restrict__ sorted_gauss_ids,
    const float* __restrict__ sorted_depth_pix,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_attenuations,
    const float* __restrict__ pose,
    float* __restrict__ path_contribution,
    float* __restrict__ path_support) {
    const int pidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_pixels = H * W;
    if (pidx >= n_pixels) {
        return;
    }

    const int y = pidx / W;
    const int x = pidx - y * W;
    float xi_lat = 0.f;
    float xi_ax = 0.f;
    sg_linear_pixel_to_plane_mm(slice_grid_pack, x, y, &xi_lat, &xi_ax);
    const float xi_v = static_cast<float>(y);
    const float margin_pix = axial_margin_mm / fmaxf(1e-8f, axial_spacing_mm_per_pix);

    const float3 axis_lat = make_float3(pose[0], pose[4], pose[8]);
    const float3 axis_ax = make_float3(pose[1], pose[5], pose[9]);
    const float3 axis_ele = make_float3(pose[2], pose[6], pose[10]);
    const float3 origin = make_float3(pose[3], pose[7], pose[11]);
    const float3 ray_origin =
        ray_make_add3(ray_make_add3(origin, ray_make_mul3(axis_lat, xi_lat)), ray_make_mul3(axis_ax, axial_near_mm));
    const float3 ray_dir = axis_ax;
    const float t_pixel = fmaxf(0.f, xi_ax - axial_near_mm);

    const int safe_tile = max(tile_size_px, 1);
    const int ray_bin = min(max(x / safe_tile, 0), max(Tw - 1, 0));
    const int s = ray_start[ray_bin];
    const int e = ray_end[ray_bin];
    if (s < 0 || e <= s) {
        return;
    }

    for (int j = s; j < e; ++j) {
        if (sorted_depth_pix[j] > xi_v + margin_pix) {
            break;
        }
        const int g = sorted_gauss_ids[j];
        const float3 mu = make_float3(
            gauss_means[g * 3 + 0], gauss_means[g * 3 + 1], gauss_means[g * 3 + 2]);
        const float3 scale = make_float3(
            fmaxf(gauss_scales[g * 3 + 0], 1e-6f),
            fmaxf(gauss_scales[g * 3 + 1], 1e-6f),
            fmaxf(gauss_scales[g * 3 + 2], 1e-6f));
        float response = 0.f;
        if (attenuation_formula == 1) {
            response = compute_ray_slice_depth_response_ours(axis_lat, axis_ele, origin, xi_lat, mu, scale);
        } else {
            const float* qptr = gauss_quats + g * 4;
            float Rg[9];
            ours_quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
            response = compute_ray_gaussian_integral_ours(
                ray_origin,
                ray_dir,
                mu,
                scale,
                Rg,
                t_pixel,
                quadrature_window_radius_mm,
                quadrature_points);
        }
        const float psi = response * fmaxf(strength, 0.f);
        if (!isfinite(psi) || psi <= weight_min) {
            continue;
        }
        const float tau = fminf(1.f, fmaxf(0.f, gauss_attenuations[g]));
        const float Ti = fminf(1.f, fmaxf(0.f, tau + (1.f - tau) * expf(-psi)));
        const float effect = 1.f - Ti;
        atomicAdd(path_support + g, 1.f);
        if (effect > 0.f && isfinite(effect)) {
            atomicAdd(path_contribution + g, effect);
        }
    }
}

__global__ void kernel_ray_csr_attenuation_prefix_slice_depth(
    int H,
    int W,
    int Tw,
    int tile_size_px,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm,
    float axial_spacing_mm_per_pix,
    const float* __restrict__ slice_grid_pack,
    const int* __restrict__ ray_start,
    const int* __restrict__ ray_end,
    const int* __restrict__ sorted_gauss_ids,
    const float* __restrict__ sorted_depth_pix,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_attenuations,
    const float* __restrict__ pose,
    float* __restrict__ modulation) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= W) {
        return;
    }

    float xi_lat = 0.f;
    float xi_ax = 0.f;
    sg_linear_pixel_to_plane_mm(slice_grid_pack, x, 0, &xi_lat, &xi_ax);
    const float margin_pix = axial_margin_mm / fmaxf(1e-8f, axial_spacing_mm_per_pix);

    const float3 axis_lat = make_float3(pose[0], pose[4], pose[8]);
    const float3 axis_ele = make_float3(pose[2], pose[6], pose[10]);
    const float3 origin = make_float3(pose[3], pose[7], pose[11]);

    const int safe_tile = max(tile_size_px, 1);
    const int ray_bin = min(max(x / safe_tile, 0), max(Tw - 1, 0));
    const int s = ray_start[ray_bin];
    const int e = ray_end[ray_bin];
    if (s < 0 || e <= s) {
        for (int y = 0; y < H; ++y) {
            modulation[y * W + x] = 1.f;
        }
        return;
    }

    float transmission = 1.f;
    int j = s;
    for (int y = 0; y < H; ++y) {
        const float y_limit = static_cast<float>(y) + margin_pix;
        while (j < e && sorted_depth_pix[j] <= y_limit) {
            const int g = sorted_gauss_ids[j];
            const float3 mu = make_float3(gauss_means[g * 3 + 0], gauss_means[g * 3 + 1], gauss_means[g * 3 + 2]);
            const float3 scale = make_float3(
                fmaxf(gauss_scales[g * 3 + 0], 1e-6f),
                fmaxf(gauss_scales[g * 3 + 1], 1e-6f),
                fmaxf(gauss_scales[g * 3 + 2], 1e-6f));
            const float response = compute_ray_slice_depth_response_ours(axis_lat, axis_ele, origin, xi_lat, mu, scale);
            const float psi = response * fmaxf(strength, 0.f);
            if (isfinite(psi) && psi > weight_min) {
                const float tau = fminf(1.f, fmaxf(0.f, gauss_attenuations[g]));
                const float Ti = tau + (1.f - tau) * expf(-psi);
                transmission *= fminf(1.f, fmaxf(0.f, Ti));
            }
            ++j;
        }
        const float T = fminf(1.f, fmaxf(min_transmission, transmission));
        modulation[y * W + x] = 1.f - beta + beta * T;
    }
}

__global__ void kernel_ray_csr_attenuation_backward(
    int H,
    int W,
    int Tw,
    int Th,
    int tile_size_px,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float k_sigma,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm,
    float quadrature_window_radius_mm,
    int quadrature_points,
    int attenuation_formula,
    const float* __restrict__ slice_grid_pack,
    const int* __restrict__ ray_start,
    const int* __restrict__ ray_end,
    const int* __restrict__ sorted_gauss_ids,
    const float* __restrict__ sorted_depth_pix,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_attenuations,
    const float* __restrict__ pose,
    const float* __restrict__ local_colors,
    const float* __restrict__ modulation,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_attenuations,
    float* __restrict__ grad_means,
    float* __restrict__ grad_scales,
    float* __restrict__ grad_quats) {
    (void)Th;
    (void)k_sigma;
    const int pidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_pixels = H * W;
    if (pidx >= n_pixels) {
        return;
    }

    const int y = pidx / W;
    const int x = pidx - y * W;
    const float grad_mod = grad_output[pidx] * local_colors[pidx];
    if (grad_mod == 0.f || beta == 0.f || strength == 0.f) {
        return;
    }

    float xi_lat = 0.f;
    float xi_ax = 0.f;
    sg_linear_pixel_to_plane_mm(slice_grid_pack, x, y, &xi_lat, &xi_ax);
    const float xi_v = static_cast<float>(y);
    const float margin_pix = axial_margin_mm / fmaxf(1e-8f, axial_spacing_mm_per_pix);

    const float3 axis_lat = make_float3(pose[0], pose[4], pose[8]);
    const float3 axis_ax = make_float3(pose[1], pose[5], pose[9]);
    const float3 axis_ele = make_float3(pose[2], pose[6], pose[10]);
    const float3 origin = make_float3(pose[3], pose[7], pose[11]);
    const float3 ray_origin =
        ray_make_add3(ray_make_add3(origin, ray_make_mul3(axis_lat, xi_lat)), ray_make_mul3(axis_ax, axial_near_mm));
    const float3 ray_dir = axis_ax;
    const float t_pixel = fmaxf(0.f, xi_ax - axial_near_mm);

    const int safe_tile = max(tile_size_px, 1);
    const int ray_bin = min(max(x / safe_tile, 0), max(Tw - 1, 0));
    const int s = ray_start[ray_bin];
    const int e = ray_end[ray_bin];
    if (s < 0 || e <= s) {
        return;
    }

    const float transmission_product =
        fminf(1.f, fmaxf(0.f, (modulation[pidx] - (1.f - beta)) / fmaxf(beta, 1e-12f)));
    if (transmission_product <= min_transmission + 1e-6f) {
        return;
    }
    const float dL_dT = grad_mod * beta;
    for (int j = s; j < e; ++j) {
        if (sorted_depth_pix[j] > xi_v + margin_pix) {
            break;
        }
        const int g = sorted_gauss_ids[j];
        const float3 mu = make_float3(gauss_means[g * 3 + 0], gauss_means[g * 3 + 1], gauss_means[g * 3 + 2]);
        const float3 scale = make_float3(
            fmaxf(gauss_scales[g * 3 + 0], 1e-6f),
            fmaxf(gauss_scales[g * 3 + 1], 1e-6f),
            fmaxf(gauss_scales[g * 3 + 2], 1e-6f));
        const float* qptr = gauss_quats + g * 4;
        float Rg[9] = {0.f};
        float psi_base = 0.f;
        if (attenuation_formula == 1) {
            psi_base = compute_ray_slice_depth_response_ours(axis_lat, axis_ele, origin, xi_lat, mu, scale);
        } else {
            ours_quat_to_rot(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), Rg);
            psi_base = compute_ray_gaussian_integral_ours(
                ray_origin, ray_dir, mu, scale, Rg, t_pixel, quadrature_window_radius_mm, quadrature_points);
        }
        const float psi = psi_base * fmaxf(strength, 0.f);
        if (!isfinite(psi) || psi <= weight_min) {
            continue;
        }
        const float tau = fminf(1.f, fmaxf(0.f, gauss_attenuations[g]));
        const float exp_neg_psi = expf(-psi);
        const float Ti = tau + (1.f - tau) * exp_neg_psi;
        if (Ti <= 1e-8f) {
            continue;
        }
        const float dL_dTi = dL_dT * (transmission_product / Ti);
        atomicAdd(grad_attenuations + g, dL_dTi * (1.f - exp_neg_psi));
        const float dL_dPsiBase = dL_dTi * (1.f - tau) * exp_neg_psi * (-fmaxf(strength, 0.f));
        float3 g_mu = make_float3(0.f, 0.f, 0.f);
        float3 g_scale = make_float3(0.f, 0.f, 0.f);
        if (attenuation_formula == 1) {
            compute_ray_slice_depth_response_vjp_ours(
                axis_lat,
                axis_ele,
                origin,
                xi_lat,
                mu,
                scale,
                dL_dPsiBase,
                g_mu,
                g_scale);
        } else {
            float g_R[9] = {0.f};
            compute_ray_integral_vjp_ours(
                ray_origin,
                ray_dir,
                mu,
                scale,
                Rg,
                t_pixel,
                quadrature_window_radius_mm,
                quadrature_points,
                dL_dPsiBase,
                g_mu,
                g_scale,
                g_R);
            float4 gq;
            transform_grads_R_to_Quat(make_float4(qptr[0], qptr[1], qptr[2], qptr[3]), g_R, gq);
            atomicAdd(grad_quats + g * 4 + 0, gq.x);
            atomicAdd(grad_quats + g * 4 + 1, gq.y);
            atomicAdd(grad_quats + g * 4 + 2, gq.z);
            atomicAdd(grad_quats + g * 4 + 3, gq.w);
        }
        atomicAdd(grad_means + g * 3 + 0, g_mu.x);
        atomicAdd(grad_means + g * 3 + 1, g_mu.y);
        atomicAdd(grad_means + g * 3 + 2, g_mu.z);
        atomicAdd(grad_scales + g * 3 + 0, g_scale.x);
        atomicAdd(grad_scales + g * 3 + 1, g_scale.y);
        atomicAdd(grad_scales + g * 3 + 2, g_scale.z);
    }
}

__global__ void kernel_ray_csr_attenuation_backward_prefix_slice_depth(
    int H,
    int W,
    int Tw,
    int tile_size_px,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm,
    float axial_spacing_mm_per_pix,
    const float* __restrict__ slice_grid_pack,
    const int* __restrict__ ray_start,
    const int* __restrict__ ray_end,
    const int* __restrict__ sorted_gauss_ids,
    const float* __restrict__ sorted_depth_pix,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_attenuations,
    const float* __restrict__ pose,
    const float* __restrict__ local_colors,
    const float* __restrict__ modulation,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_attenuations,
    float* __restrict__ grad_means,
    float* __restrict__ grad_scales) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= W || beta == 0.f || strength == 0.f) {
        return;
    }

    float xi_lat = 0.f;
    float xi_ax = 0.f;
    sg_linear_pixel_to_plane_mm(slice_grid_pack, x, 0, &xi_lat, &xi_ax);
    const float margin_pix = axial_margin_mm / fmaxf(1e-8f, axial_spacing_mm_per_pix);

    const float3 axis_lat = make_float3(pose[0], pose[4], pose[8]);
    const float3 axis_ele = make_float3(pose[2], pose[6], pose[10]);
    const float3 origin = make_float3(pose[3], pose[7], pose[11]);

    const int safe_tile = max(tile_size_px, 1);
    const int ray_bin = min(max(x / safe_tile, 0), max(Tw - 1, 0));
    const int s = ray_start[ray_bin];
    const int e = ray_end[ray_bin];
    if (s < 0 || e <= s) {
        return;
    }

    const float beta_safe = fmaxf(beta, 1e-12f);
    const float strength_pos = fmaxf(strength, 0.f);
    float suffix_dL_dT_times_T = 0.f;
    int y_cursor = H - 1;

    for (int j = e - 1; j >= s; --j) {
        const float depth_limit = sorted_depth_pix[j] - margin_pix;
        const int y_min_raw = static_cast<int>(ceilf(depth_limit));
        if (y_min_raw <= H - 1) {
            const int y_min = max(0, y_min_raw);
            while (y_cursor >= y_min) {
                const int pidx = y_cursor * W + x;
                const float grad_mod = grad_output[pidx] * local_colors[pidx];
                if (grad_mod != 0.f) {
                    const float transmission_product =
                        fminf(1.f, fmaxf(0.f, (modulation[pidx] - (1.f - beta)) / beta_safe));
                    if (transmission_product > min_transmission + 1e-6f) {
                        suffix_dL_dT_times_T += grad_mod * beta * transmission_product;
                    }
                }
                --y_cursor;
            }
        }

        if (suffix_dL_dT_times_T == 0.f) {
            continue;
        }

        const int g = sorted_gauss_ids[j];
        const float3 mu = make_float3(gauss_means[g * 3 + 0], gauss_means[g * 3 + 1], gauss_means[g * 3 + 2]);
        const float3 scale = make_float3(
            fmaxf(gauss_scales[g * 3 + 0], 1e-6f),
            fmaxf(gauss_scales[g * 3 + 1], 1e-6f),
            fmaxf(gauss_scales[g * 3 + 2], 1e-6f));
        const float psi_base = compute_ray_slice_depth_response_ours(axis_lat, axis_ele, origin, xi_lat, mu, scale);
        const float psi = psi_base * strength_pos;
        if (!isfinite(psi) || psi <= weight_min) {
            continue;
        }

        const float tau = fminf(1.f, fmaxf(0.f, gauss_attenuations[g]));
        const float exp_neg_psi = expf(-psi);
        const float Ti = tau + (1.f - tau) * exp_neg_psi;
        if (Ti <= 1e-8f) {
            continue;
        }

        const float summed_dL_dTi = suffix_dL_dT_times_T / Ti;
        atomicAdd(grad_attenuations + g, summed_dL_dTi * (1.f - exp_neg_psi));
        const float dL_dPsiBase = summed_dL_dTi * (1.f - tau) * exp_neg_psi * (-strength_pos);
        float3 g_mu = make_float3(0.f, 0.f, 0.f);
        float3 g_scale = make_float3(0.f, 0.f, 0.f);
        compute_ray_slice_depth_response_vjp_ours(
            axis_lat,
            axis_ele,
            origin,
            xi_lat,
            mu,
            scale,
            dL_dPsiBase,
            g_mu,
            g_scale);
        atomicAdd(grad_means + g * 3 + 0, g_mu.x);
        atomicAdd(grad_means + g * 3 + 1, g_mu.y);
        atomicAdd(grad_means + g * 3 + 2, g_mu.z);
        atomicAdd(grad_scales + g * 3 + 0, g_scale.x);
        atomicAdd(grad_scales + g * 3 + 1, g_scale.y);
        atomicAdd(grad_scales + g * 3 + 2, g_scale.z);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> build_ray_csr(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    int H,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    int tile_size_px,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    int fixed_total_capacity,
    int fixed_capacity_per_ray) {
    (void)H;
    const int n_gauss = static_cast<int>(gauss_means.size(0));
    const int Tw = (W + tile_size_px - 1) / tile_size_px;
    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(gauss_means.device());
    auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(gauss_means.device());
    auto float_opts = gauss_means.options().dtype(torch::kFloat32);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor ray_start = torch::full({Tw}, -1, int_opts);
    torch::Tensor ray_end = torch::full({Tw}, -1, int_opts);
    torch::Tensor empty_i = torch::zeros({0}, int_opts);
    torch::Tensor empty_f = torch::zeros({0}, float_opts);
    if (n_gauss == 0 || Tw <= 0) {
        return {empty_i, ray_start, ray_end, empty_f};
    }

    torch::Tensor ray_ranges = torch::empty({n_gauss, 2}, int_opts);
    torch::Tensor counts = torch::zeros({n_gauss + 1}, int_opts);
    torch::Tensor offsets = torch::zeros({n_gauss + 1}, int_opts);
    torch::Tensor depth_pix = torch::zeros({n_gauss}, float_opts);

    const int block = 128;
    const int grid = (n_gauss + block - 1) / block;
    kernel_ray_csr_count<<<grid, block, 0, stream>>>(
        n_gauss,
        Tw,
        tile_size_px,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        elev_corridor_half_mm,
        slab_half_thickness_mm,
        k_sigma,
        slice_grid_geometry.data_ptr<float>(),
        gauss_means.data_ptr<float>(),
        gauss_scales.data_ptr<float>(),
        gauss_quats.data_ptr<float>(),
        pose_probe_to_world.data_ptr<float>(),
        reinterpret_cast<int2*>(ray_ranges.data_ptr<int>()),
        counts.data_ptr<int>(),
        depth_pix.data_ptr<float>());

    if (fixed_total_capacity > 0) {
        const int capacity = fixed_total_capacity;
        const int64_t invalid_key = (static_cast<int64_t>(Tw) << 32) | static_cast<int64_t>(0xffffffffu);
        torch::Tensor total_count = torch::zeros({1}, int_opts);
        torch::Tensor overflow_count = torch::zeros({1}, int_opts);
        torch::Tensor keys = torch::full({capacity}, invalid_key, long_opts);
        torch::Tensor keys_sorted = torch::empty({capacity}, long_opts);
        torch::Tensor values = torch::full({capacity}, -1, int_opts);
        torch::Tensor values_sorted = torch::empty({capacity}, int_opts);
        kernel_ray_csr_emit_fixed_total<<<grid, block, 0, stream>>>(
            n_gauss,
            capacity,
            reinterpret_cast<const int2*>(ray_ranges.data_ptr<int>()),
            depth_pix.data_ptr<float>(),
            total_count.data_ptr<int>(),
            overflow_count.data_ptr<int>(),
            keys.data_ptr<int64_t>(),
            values.data_ptr<int>());

        size_t sort_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_bytes,
            keys.data_ptr<int64_t>(),
            keys_sorted.data_ptr<int64_t>(),
            values.data_ptr<int>(),
            values_sorted.data_ptr<int>(),
            capacity,
            0,
            64,
            stream);
        torch::Tensor sort_temp = torch::empty({static_cast<long>(sort_bytes)}, gauss_means.options().dtype(torch::kUInt8));
        cub::DeviceRadixSort::SortPairs(
            sort_temp.data_ptr(),
            sort_bytes,
            keys.data_ptr<int64_t>(),
            keys_sorted.data_ptr<int64_t>(),
            values.data_ptr<int>(),
            values_sorted.data_ptr<int>(),
            capacity,
            0,
            64,
            stream);

        const int range_grid = (capacity + block - 1) / block;
        kernel_ray_csr_ranges<<<range_grid, block, 0, stream>>>(
            capacity,
            Tw,
            keys_sorted.data_ptr<int64_t>(),
            ray_start.data_ptr<int>(),
            ray_end.data_ptr<int>());
        torch::Tensor sorted_depth_pix = torch::empty({capacity}, float_opts);
        kernel_ray_csr_sorted_depths<<<range_grid, block, 0, stream>>>(
            capacity,
            values_sorted.data_ptr<int>(),
            depth_pix.data_ptr<float>(),
            sorted_depth_pix.data_ptr<float>());
        return {values_sorted, ray_start, ray_end, sorted_depth_pix};
    }

    if (fixed_capacity_per_ray > 0) {
        const int64_t capacity = static_cast<int64_t>(Tw) * static_cast<int64_t>(fixed_capacity_per_ray);
        const int64_t invalid_key = (static_cast<int64_t>(Tw) << 32) | static_cast<int64_t>(0xffffffffu);
        torch::Tensor ray_counts = torch::zeros({Tw}, int_opts);
        torch::Tensor total_count = torch::zeros({1}, int_opts);
        torch::Tensor overflow_count = torch::zeros({1}, int_opts);
        torch::Tensor keys = torch::full({capacity}, invalid_key, long_opts);
        torch::Tensor keys_sorted = torch::empty({capacity}, long_opts);
        torch::Tensor values = torch::full({capacity}, -1, int_opts);
        torch::Tensor values_sorted = torch::empty({capacity}, int_opts);
        kernel_ray_csr_emit_fixed_per_ray<<<grid, block, 0, stream>>>(
            n_gauss,
            fixed_capacity_per_ray,
            reinterpret_cast<const int2*>(ray_ranges.data_ptr<int>()),
            depth_pix.data_ptr<float>(),
            ray_counts.data_ptr<int>(),
            total_count.data_ptr<int>(),
            overflow_count.data_ptr<int>(),
            keys.data_ptr<int64_t>(),
            values.data_ptr<int>());

        size_t sort_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_bytes,
            keys.data_ptr<int64_t>(),
            keys_sorted.data_ptr<int64_t>(),
            values.data_ptr<int>(),
            values_sorted.data_ptr<int>(),
            static_cast<int>(capacity),
            0,
            64,
            stream);
        torch::Tensor sort_temp = torch::empty({static_cast<long>(sort_bytes)}, gauss_means.options().dtype(torch::kUInt8));
        cub::DeviceRadixSort::SortPairs(
            sort_temp.data_ptr(),
            sort_bytes,
            keys.data_ptr<int64_t>(),
            keys_sorted.data_ptr<int64_t>(),
            values.data_ptr<int>(),
            values_sorted.data_ptr<int>(),
            static_cast<int>(capacity),
            0,
            64,
            stream);

        const int range_grid = (static_cast<int>(capacity) + block - 1) / block;
        kernel_ray_csr_ranges<<<range_grid, block, 0, stream>>>(
            static_cast<int>(capacity),
            Tw,
            keys_sorted.data_ptr<int64_t>(),
            ray_start.data_ptr<int>(),
            ray_end.data_ptr<int>());
        torch::Tensor sorted_depth_pix = torch::empty({capacity}, float_opts);
        kernel_ray_csr_sorted_depths<<<range_grid, block, 0, stream>>>(
            static_cast<int>(capacity),
            values_sorted.data_ptr<int>(),
            depth_pix.data_ptr<float>(),
            sorted_depth_pix.data_ptr<float>());
        return {values_sorted, ray_start, ray_end, sorted_depth_pix};
    }

    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, counts.data_ptr<int>(), offsets.data_ptr<int>(), n_gauss + 1, stream);
    torch::Tensor temp = torch::empty({static_cast<long>(temp_bytes)}, gauss_means.options().dtype(torch::kUInt8));
    cub::DeviceScan::ExclusiveSum(
        temp.data_ptr(),
        temp_bytes,
        counts.data_ptr<int>(),
        offsets.data_ptr<int>(),
        n_gauss + 1,
        stream);

    const int total = copy_scalar_d2h_ray<int>(offsets.data_ptr<int>() + n_gauss, stream);
    if (total <= 0) {
        return {empty_i, ray_start, ray_end, empty_f};
    }

    torch::Tensor keys = torch::empty({total}, long_opts);
    torch::Tensor keys_sorted = torch::empty({total}, long_opts);
    torch::Tensor values = torch::empty({total}, int_opts);
    torch::Tensor values_sorted = torch::empty({total}, int_opts);
    kernel_ray_csr_emit<<<grid, block, 0, stream>>>(
        n_gauss,
        reinterpret_cast<const int2*>(ray_ranges.data_ptr<int>()),
        offsets.data_ptr<int>(),
        depth_pix.data_ptr<float>(),
        keys.data_ptr<int64_t>(),
        values.data_ptr<int>());

    size_t sort_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_bytes,
        keys.data_ptr<int64_t>(),
        keys_sorted.data_ptr<int64_t>(),
        values.data_ptr<int>(),
        values_sorted.data_ptr<int>(),
        total,
        0,
        64,
        stream);
    torch::Tensor sort_temp = torch::empty({static_cast<long>(sort_bytes)}, gauss_means.options().dtype(torch::kUInt8));
    cub::DeviceRadixSort::SortPairs(
        sort_temp.data_ptr(),
        sort_bytes,
        keys.data_ptr<int64_t>(),
        keys_sorted.data_ptr<int64_t>(),
        values.data_ptr<int>(),
        values_sorted.data_ptr<int>(),
        total,
        0,
        64,
        stream);

    const int range_grid = (total + block - 1) / block;
    kernel_ray_csr_ranges<<<range_grid, block, 0, stream>>>(
        total,
        Tw,
        keys_sorted.data_ptr<int64_t>(),
        ray_start.data_ptr<int>(),
        ray_end.data_ptr<int>());
    torch::Tensor sorted_depth_pix = torch::empty({total}, float_opts);
    kernel_ray_csr_sorted_depths<<<range_grid, block, 0, stream>>>(
        total,
        values_sorted.data_ptr<int>(),
        depth_pix.data_ptr<float>(),
        sorted_depth_pix.data_ptr<float>());

    return {values_sorted, ray_start, ray_end, sorted_depth_pix};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_ray_csr_profile(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    int H,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    int tile_size_px,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    int fixed_total_capacity,
    int fixed_capacity_per_ray) {
    (void)H;
    const int n_gauss = static_cast<int>(gauss_means.size(0));
    const int Tw = (W + tile_size_px - 1) / tile_size_px;
    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(gauss_means.device());
    auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(gauss_means.device());
    auto float_opts = gauss_means.options().dtype(torch::kFloat32);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor ray_start = torch::full({Tw}, -1, int_opts);
    torch::Tensor ray_end = torch::full({Tw}, -1, int_opts);
    torch::Tensor empty_i = torch::zeros({0}, int_opts);
    torch::Tensor empty_f = torch::zeros({0}, float_opts);
    torch::Tensor timings = torch::zeros({11}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor stats = torch::zeros({8}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    stats[0] = static_cast<float>(n_gauss);
    stats[1] = static_cast<float>(W);
    stats[2] = static_cast<float>(Tw);
    if (n_gauss == 0 || Tw <= 0) {
        return {empty_i, ray_start, ray_end, empty_f, timings, stats};
    }

    cudaEvent_t e0, e1, e2, e2_after_d2h, e3, e4, e5, e6, e7;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e2_after_d2h);
    cudaEventCreate(&e3);
    cudaEventCreate(&e4);
    cudaEventCreate(&e5);
    cudaEventCreate(&e6);
    cudaEventCreate(&e7);

    torch::Tensor ray_ranges = torch::empty({n_gauss, 2}, int_opts);
    torch::Tensor counts = torch::zeros({n_gauss + 1}, int_opts);
    torch::Tensor offsets = torch::zeros({n_gauss + 1}, int_opts);
    torch::Tensor depth_pix = torch::zeros({n_gauss}, float_opts);

    const int block = 128;
    const int grid = (n_gauss + block - 1) / block;
    cudaEventRecord(e0, stream);
    kernel_ray_csr_count<<<grid, block, 0, stream>>>(
        n_gauss,
        Tw,
        tile_size_px,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        elev_corridor_half_mm,
        slab_half_thickness_mm,
        k_sigma,
        slice_grid_geometry.data_ptr<float>(),
        gauss_means.data_ptr<float>(),
        gauss_scales.data_ptr<float>(),
        gauss_quats.data_ptr<float>(),
        pose_probe_to_world.data_ptr<float>(),
        reinterpret_cast<int2*>(ray_ranges.data_ptr<int>()),
        counts.data_ptr<int>(),
        depth_pix.data_ptr<float>());
    cudaEventRecord(e1, stream);

    if (fixed_total_capacity > 0) {
        const int capacity = fixed_total_capacity;
        const int64_t invalid_key = (static_cast<int64_t>(Tw) << 32) | static_cast<int64_t>(0xffffffffu);
        torch::Tensor total_count = torch::zeros({1}, int_opts);
        torch::Tensor overflow_count = torch::zeros({1}, int_opts);
        torch::Tensor keys = torch::full({capacity}, invalid_key, long_opts);
        torch::Tensor keys_sorted = torch::empty({capacity}, long_opts);
        torch::Tensor values = torch::full({capacity}, -1, int_opts);
        torch::Tensor values_sorted = torch::empty({capacity}, int_opts);
        cudaEventRecord(e2, stream);
        cudaEventRecord(e2_after_d2h, stream);
        kernel_ray_csr_emit_fixed_total<<<grid, block, 0, stream>>>(
            n_gauss,
            capacity,
            reinterpret_cast<const int2*>(ray_ranges.data_ptr<int>()),
            depth_pix.data_ptr<float>(),
            total_count.data_ptr<int>(),
            overflow_count.data_ptr<int>(),
            keys.data_ptr<int64_t>(),
            values.data_ptr<int>());
        cudaEventRecord(e3, stream);

        size_t sort_bytes = 0;
        const auto sort_query_start = std::chrono::high_resolution_clock::now();
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_bytes,
            keys.data_ptr<int64_t>(),
            keys_sorted.data_ptr<int64_t>(),
            values.data_ptr<int>(),
            values_sorted.data_ptr<int>(),
            capacity,
            0,
            64,
            stream);
        const auto sort_query_end = std::chrono::high_resolution_clock::now();
        torch::Tensor sort_temp = torch::empty({static_cast<long>(sort_bytes)}, gauss_means.options().dtype(torch::kUInt8));
        cub::DeviceRadixSort::SortPairs(
            sort_temp.data_ptr(),
            sort_bytes,
            keys.data_ptr<int64_t>(),
            keys_sorted.data_ptr<int64_t>(),
            values.data_ptr<int>(),
            values_sorted.data_ptr<int>(),
            capacity,
            0,
            64,
            stream);
        cudaEventRecord(e4, stream);

        const int range_grid = (capacity + block - 1) / block;
        kernel_ray_csr_ranges<<<range_grid, block, 0, stream>>>(
            capacity,
            Tw,
            keys_sorted.data_ptr<int64_t>(),
            ray_start.data_ptr<int>(),
            ray_end.data_ptr<int>());
        cudaEventRecord(e5, stream);
        torch::Tensor sorted_depth_pix = torch::empty({capacity}, float_opts);
        kernel_ray_csr_sorted_depths<<<range_grid, block, 0, stream>>>(
            capacity,
            values_sorted.data_ptr<int>(),
            depth_pix.data_ptr<float>(),
            sorted_depth_pix.data_ptr<float>());
        cudaEventRecord(e6, stream);
        cudaEventRecord(e7, stream);
        cudaEventSynchronize(e7);

        float count_ms = 0.f;
        float emit_ms = 0.f;
        float sort_ms = 0.f;
        float ranges_ms = 0.f;
        float depths_ms = 0.f;
        float total_gpu_ms = 0.f;
        cudaEventElapsedTime(&count_ms, e0, e1);
        cudaEventElapsedTime(&emit_ms, e2_after_d2h, e3);
        cudaEventElapsedTime(&sort_ms, e3, e4);
        cudaEventElapsedTime(&ranges_ms, e4, e5);
        cudaEventElapsedTime(&depths_ms, e5, e6);
        cudaEventElapsedTime(&total_gpu_ms, e0, e6);
        timings[0] = count_ms;
        timings[1] = 0.f;
        timings[2] = 0.f;
        timings[3] = emit_ms;
        timings[4] = sort_ms;
        timings[5] = ranges_ms;
        timings[6] = depths_ms;
        timings[8] = total_gpu_ms;
        timings[9] = 0.f;
        timings[10] = std::chrono::duration<float, std::milli>(sort_query_end - sort_query_start).count();

        int total_host = 0;
        int overflow_host = 0;
        TORCH_CHECK(
            cudaMemcpy(&total_host, total_count.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess,
            "ray_csr_mapper fixed-total total copy failed");
        TORCH_CHECK(
            cudaMemcpy(&overflow_host, overflow_count.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess,
            "ray_csr_mapper fixed-total overflow copy failed");
        stats[3] = static_cast<float>(total_host);
        stats[4] = 0.f;
        stats[5] = static_cast<float>(sort_bytes);
        stats[6] = static_cast<float>(overflow_host);
        stats[7] = static_cast<float>(capacity);

        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
        cudaEventDestroy(e2_after_d2h);
        cudaEventDestroy(e3);
        cudaEventDestroy(e4);
        cudaEventDestroy(e5);
        cudaEventDestroy(e6);
        cudaEventDestroy(e7);
        return {values_sorted, ray_start, ray_end, sorted_depth_pix, timings, stats};
    }

    if (fixed_capacity_per_ray > 0) {
        const int64_t capacity = static_cast<int64_t>(Tw) * static_cast<int64_t>(fixed_capacity_per_ray);
        const int64_t invalid_key = (static_cast<int64_t>(Tw) << 32) | static_cast<int64_t>(0xffffffffu);
        torch::Tensor ray_counts = torch::zeros({Tw}, int_opts);
        torch::Tensor total_count = torch::zeros({1}, int_opts);
        torch::Tensor overflow_count = torch::zeros({1}, int_opts);
        torch::Tensor keys = torch::full({capacity}, invalid_key, long_opts);
        torch::Tensor keys_sorted = torch::empty({capacity}, long_opts);
        torch::Tensor values = torch::full({capacity}, -1, int_opts);
        torch::Tensor values_sorted = torch::empty({capacity}, int_opts);
        cudaEventRecord(e2, stream);
        cudaEventRecord(e2_after_d2h, stream);
        kernel_ray_csr_emit_fixed_per_ray<<<grid, block, 0, stream>>>(
            n_gauss,
            fixed_capacity_per_ray,
            reinterpret_cast<const int2*>(ray_ranges.data_ptr<int>()),
            depth_pix.data_ptr<float>(),
            ray_counts.data_ptr<int>(),
            total_count.data_ptr<int>(),
            overflow_count.data_ptr<int>(),
            keys.data_ptr<int64_t>(),
            values.data_ptr<int>());
        cudaEventRecord(e3, stream);

        size_t sort_bytes = 0;
        const auto sort_query_start = std::chrono::high_resolution_clock::now();
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_bytes,
            keys.data_ptr<int64_t>(),
            keys_sorted.data_ptr<int64_t>(),
            values.data_ptr<int>(),
            values_sorted.data_ptr<int>(),
            static_cast<int>(capacity),
            0,
            64,
            stream);
        const auto sort_query_end = std::chrono::high_resolution_clock::now();
        torch::Tensor sort_temp = torch::empty({static_cast<long>(sort_bytes)}, gauss_means.options().dtype(torch::kUInt8));
        cub::DeviceRadixSort::SortPairs(
            sort_temp.data_ptr(),
            sort_bytes,
            keys.data_ptr<int64_t>(),
            keys_sorted.data_ptr<int64_t>(),
            values.data_ptr<int>(),
            values_sorted.data_ptr<int>(),
            static_cast<int>(capacity),
            0,
            64,
            stream);
        cudaEventRecord(e4, stream);

        const int range_grid = (static_cast<int>(capacity) + block - 1) / block;
        kernel_ray_csr_ranges<<<range_grid, block, 0, stream>>>(
            static_cast<int>(capacity),
            Tw,
            keys_sorted.data_ptr<int64_t>(),
            ray_start.data_ptr<int>(),
            ray_end.data_ptr<int>());
        cudaEventRecord(e5, stream);
        torch::Tensor sorted_depth_pix = torch::empty({capacity}, float_opts);
        kernel_ray_csr_sorted_depths<<<range_grid, block, 0, stream>>>(
            static_cast<int>(capacity),
            values_sorted.data_ptr<int>(),
            depth_pix.data_ptr<float>(),
            sorted_depth_pix.data_ptr<float>());
        cudaEventRecord(e6, stream);
        cudaEventRecord(e7, stream);
        cudaEventSynchronize(e7);

        float count_ms = 0.f;
        float emit_ms = 0.f;
        float sort_ms = 0.f;
        float ranges_ms = 0.f;
        float depths_ms = 0.f;
        float total_gpu_ms = 0.f;
        cudaEventElapsedTime(&count_ms, e0, e1);
        cudaEventElapsedTime(&emit_ms, e2_after_d2h, e3);
        cudaEventElapsedTime(&sort_ms, e3, e4);
        cudaEventElapsedTime(&ranges_ms, e4, e5);
        cudaEventElapsedTime(&depths_ms, e5, e6);
        cudaEventElapsedTime(&total_gpu_ms, e0, e6);
        timings[0] = count_ms;
        timings[1] = 0.f;
        timings[2] = 0.f;
        timings[3] = emit_ms;
        timings[4] = sort_ms;
        timings[5] = ranges_ms;
        timings[6] = depths_ms;
        timings[8] = total_gpu_ms;
        timings[9] = 0.f;
        timings[10] = std::chrono::duration<float, std::milli>(sort_query_end - sort_query_start).count();

        int total_host = 0;
        int overflow_host = 0;
        TORCH_CHECK(
            cudaMemcpy(&total_host, total_count.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess,
            "ray_csr_mapper fixed-capacity total copy failed");
        TORCH_CHECK(
            cudaMemcpy(&overflow_host, overflow_count.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess,
            "ray_csr_mapper fixed-capacity overflow copy failed");
        stats[3] = static_cast<float>(total_host);
        stats[4] = 0.f;
        stats[5] = static_cast<float>(sort_bytes);
        stats[6] = static_cast<float>(overflow_host);
        stats[7] = static_cast<float>(capacity);

        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
        cudaEventDestroy(e2_after_d2h);
        cudaEventDestroy(e3);
        cudaEventDestroy(e4);
        cudaEventDestroy(e5);
        cudaEventDestroy(e6);
        cudaEventDestroy(e7);
        return {values_sorted, ray_start, ray_end, sorted_depth_pix, timings, stats};
    }

    size_t temp_bytes = 0;
    const auto scan_query_start = std::chrono::high_resolution_clock::now();
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, counts.data_ptr<int>(), offsets.data_ptr<int>(), n_gauss + 1, stream);
    const auto scan_query_end = std::chrono::high_resolution_clock::now();
    torch::Tensor temp = torch::empty({static_cast<long>(temp_bytes)}, gauss_means.options().dtype(torch::kUInt8));
    cub::DeviceScan::ExclusiveSum(
        temp.data_ptr(),
        temp_bytes,
        counts.data_ptr<int>(),
        offsets.data_ptr<int>(),
        n_gauss + 1,
        stream);
    cudaEventRecord(e2, stream);

    const auto d2h_start = std::chrono::high_resolution_clock::now();
    const int total = copy_scalar_d2h_ray<int>(offsets.data_ptr<int>() + n_gauss, stream);
    const auto d2h_end = std::chrono::high_resolution_clock::now();
    stats[3] = static_cast<float>(total);
    stats[4] = static_cast<float>(temp_bytes);
    if (total <= 0) {
        cudaEventSynchronize(e2);
        float count_ms = 0.f;
        float scan_ms = 0.f;
        cudaEventElapsedTime(&count_ms, e0, e1);
        cudaEventElapsedTime(&scan_ms, e1, e2);
        timings[0] = count_ms;
        timings[1] = scan_ms;
        timings[2] = std::chrono::duration<float, std::milli>(d2h_end - d2h_start).count();
        timings[9] = std::chrono::duration<float, std::milli>(scan_query_end - scan_query_start).count();
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
        cudaEventDestroy(e2_after_d2h);
        cudaEventDestroy(e3);
        cudaEventDestroy(e4);
        cudaEventDestroy(e5);
        cudaEventDestroy(e6);
        cudaEventDestroy(e7);
        return {empty_i, ray_start, ray_end, empty_f, timings, stats};
    }

    torch::Tensor keys = torch::empty({total}, long_opts);
    torch::Tensor keys_sorted = torch::empty({total}, long_opts);
    torch::Tensor values = torch::empty({total}, int_opts);
    torch::Tensor values_sorted = torch::empty({total}, int_opts);
    cudaEventRecord(e2_after_d2h, stream);
    kernel_ray_csr_emit<<<grid, block, 0, stream>>>(
        n_gauss,
        reinterpret_cast<const int2*>(ray_ranges.data_ptr<int>()),
        offsets.data_ptr<int>(),
        depth_pix.data_ptr<float>(),
        keys.data_ptr<int64_t>(),
        values.data_ptr<int>());
    cudaEventRecord(e3, stream);

    size_t sort_bytes = 0;
    const auto sort_query_start = std::chrono::high_resolution_clock::now();
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_bytes,
        keys.data_ptr<int64_t>(),
        keys_sorted.data_ptr<int64_t>(),
        values.data_ptr<int>(),
        values_sorted.data_ptr<int>(),
        total,
        0,
        64,
        stream);
    const auto sort_query_end = std::chrono::high_resolution_clock::now();
    torch::Tensor sort_temp = torch::empty({static_cast<long>(sort_bytes)}, gauss_means.options().dtype(torch::kUInt8));
    stats[5] = static_cast<float>(sort_bytes);
    cub::DeviceRadixSort::SortPairs(
        sort_temp.data_ptr(),
        sort_bytes,
        keys.data_ptr<int64_t>(),
        keys_sorted.data_ptr<int64_t>(),
        values.data_ptr<int>(),
        values_sorted.data_ptr<int>(),
        total,
        0,
        64,
        stream);
    cudaEventRecord(e4, stream);

    const int range_grid = (total + block - 1) / block;
    kernel_ray_csr_ranges<<<range_grid, block, 0, stream>>>(
        total,
        Tw,
        keys_sorted.data_ptr<int64_t>(),
        ray_start.data_ptr<int>(),
        ray_end.data_ptr<int>());
    cudaEventRecord(e5, stream);
    torch::Tensor sorted_depth_pix = torch::empty({total}, float_opts);
    kernel_ray_csr_sorted_depths<<<range_grid, block, 0, stream>>>(
        total,
        values_sorted.data_ptr<int>(),
        depth_pix.data_ptr<float>(),
        sorted_depth_pix.data_ptr<float>());
    cudaEventRecord(e6, stream);
    cudaEventRecord(e7, stream);
    cudaEventSynchronize(e7);

    float count_ms = 0.f;
    float scan_ms = 0.f;
    float emit_ms = 0.f;
    float sort_ms = 0.f;
    float ranges_ms = 0.f;
    float depths_ms = 0.f;
    float total_gpu_ms = 0.f;
    cudaEventElapsedTime(&count_ms, e0, e1);
    cudaEventElapsedTime(&scan_ms, e1, e2);
    cudaEventElapsedTime(&emit_ms, e2_after_d2h, e3);
    cudaEventElapsedTime(&sort_ms, e3, e4);
    cudaEventElapsedTime(&ranges_ms, e4, e5);
    cudaEventElapsedTime(&depths_ms, e5, e6);
    cudaEventElapsedTime(&total_gpu_ms, e0, e6);
    timings[0] = count_ms;
    timings[1] = scan_ms;
    timings[2] = std::chrono::duration<float, std::milli>(d2h_end - d2h_start).count();
    timings[3] = emit_ms;
    timings[4] = sort_ms;
    timings[5] = ranges_ms;
    timings[6] = depths_ms;
    timings[8] = total_gpu_ms;
    timings[9] = std::chrono::duration<float, std::milli>(scan_query_end - scan_query_start).count();
    timings[10] = std::chrono::duration<float, std::milli>(sort_query_end - sort_query_start).count();

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e2_after_d2h);
    cudaEventDestroy(e3);
    cudaEventDestroy(e4);
    cudaEventDestroy(e5);
    cudaEventDestroy(e6);
    cudaEventDestroy(e7);
    return {values_sorted, ray_start, ray_end, sorted_depth_pix, timings, stats};
}

__global__ void kernel_ray_tile_attenuation(
    int H,
    int W,
    int Tw,
    int Th,
    int tile_size_px,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float k_sigma,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm,
    const float* __restrict__ slice_grid_pack,
    const int* __restrict__ tile_range_start,
    const int* __restrict__ tile_range_end,
    const int* __restrict__ sorted_gauss_ids,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_attenuations,
    const float* __restrict__ pose,
    float* __restrict__ modulation) {
    const int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_tiles = Tw * Th;
    if (tile_id >= n_tiles) {
        return;
    }

    const int ty = tile_id / Tw;
    const int tx = tile_id - ty * Tw;
    const int x0 = tx * tile_size_px;
    const int y0 = ty * tile_size_px;
    const int x1 = min(W, x0 + tile_size_px);
    const int y1 = min(H, y0 + tile_size_px);
    const int cx = min(W - 1, x0 + tile_size_px / 2);
    const int cy = min(H - 1, y0 + tile_size_px / 2);

    float xi_lat = 0.f;
    float xi_ax = 0.f;
    sg_linear_pixel_to_plane_mm(slice_grid_pack, cx, cy, &xi_lat, &xi_ax);

    const float3 axis_lat = ours_normalize3(make_float3(pose[0], pose[4], pose[8]));
    const float3 axis_ax = ours_normalize3(make_float3(pose[1], pose[5], pose[9]));
    const float3 axis_ele = ours_normalize3(make_float3(pose[2], pose[6], pose[10]));
    const float3 origin = make_float3(pose[3], pose[7], pose[11]);

    float optical_depth = 0.f;
    for (int scan_ty = 0; scan_ty <= ty; ++scan_ty) {
        const int tid = scan_ty * Tw + tx;
        const int s = tile_range_start[tid];
        const int e = tile_range_end[tid];
        if (s < 0 || e <= s) {
            continue;
        }
        for (int j = s; j < e; ++j) {
            const int g = sorted_gauss_ids[j];
            float mu_lat = 0.f;
            float mu_ax = 0.f;
            float mu_ele = 0.f;
            float sigma_lat = 1.f;
            float sigma_ele = 1.f;
            gaussian_probe_stats(
                g,
                gauss_means,
                gauss_scales,
                gauss_quats,
                axis_lat,
                axis_ax,
                axis_ele,
                origin,
                &mu_lat,
                &mu_ax,
                &mu_ele,
                &sigma_lat,
                &sigma_ele);

            const int center_ty =
                static_cast<int>(floorf(((mu_ax - axial_near_mm) / axial_spacing_mm_per_pix) / tile_size_px));
            if (center_ty != scan_ty) {
                continue;
            }
            if (mu_ax > xi_ax + axial_margin_mm) {
                continue;
            }

            const float inv_lat = 1.f / fmaxf(1e-6f, sigma_lat * k_sigma);
            const float inv_ele = 1.f / fmaxf(1e-6f, sigma_ele * k_sigma);
            const float dlat = (xi_lat - mu_lat) * inv_lat;
            const float dele = mu_ele * inv_ele;
            const float quad = dlat * dlat + dele * dele;
            if (!(quad >= 0.f) || !isfinite(quad)) {
                continue;
            }

            const float weight = expf(-0.5f * quad);
            if (!isfinite(weight) || weight < weight_min) {
                continue;
            }
            const float attenuation = fmaxf(0.f, gauss_attenuations[g]);
            optical_depth += strength * attenuation * weight;
        }
    }

    float transmission = expf(-optical_depth);
    transmission = fminf(1.f, fmaxf(min_transmission, transmission));
    const float tile_modulation = 1.f - beta + beta * transmission;
    for (int yy = y0; yy < y1; ++yy) {
        for (int xx = x0; xx < x1; ++xx) {
            modulation[yy * W + xx] = tile_modulation;
        }
    }
}

__global__ void kernel_ray_tile_attenuation_backward(
    int H,
    int W,
    int Tw,
    int Th,
    int tile_size_px,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float k_sigma,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm,
    const float* __restrict__ slice_grid_pack,
    const int* __restrict__ tile_range_start,
    const int* __restrict__ tile_range_end,
    const int* __restrict__ sorted_gauss_ids,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_attenuations,
    const float* __restrict__ pose,
    const float* __restrict__ local_colors,
    const float* __restrict__ modulation,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_attenuations) {
    const int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_tiles = Tw * Th;
    if (tile_id >= n_tiles) {
        return;
    }

    const int ty = tile_id / Tw;
    const int tx = tile_id - ty * Tw;
    const int x0 = tx * tile_size_px;
    const int y0 = ty * tile_size_px;
    const int x1 = min(W, x0 + tile_size_px);
    const int y1 = min(H, y0 + tile_size_px);
    const int cx = min(W - 1, x0 + tile_size_px / 2);
    const int cy = min(H - 1, y0 + tile_size_px / 2);

    float grad_mod = 0.f;
    for (int yy = y0; yy < y1; ++yy) {
        for (int xx = x0; xx < x1; ++xx) {
            const int p = yy * W + xx;
            grad_mod += grad_output[p] * local_colors[p];
        }
    }
    if (grad_mod == 0.f || beta == 0.f || strength == 0.f) {
        return;
    }

    const float mod = modulation[cy * W + cx];
    const float unclamped_transmission = (mod - (1.f - beta)) / fmaxf(beta, 1e-12f);
    if (!(unclamped_transmission > min_transmission) || unclamped_transmission > 1.f) {
        return;
    }

    float xi_lat = 0.f;
    float xi_ax = 0.f;
    sg_linear_pixel_to_plane_mm(slice_grid_pack, cx, cy, &xi_lat, &xi_ax);

    const float3 axis_lat = ours_normalize3(make_float3(pose[0], pose[4], pose[8]));
    const float3 axis_ax = ours_normalize3(make_float3(pose[1], pose[5], pose[9]));
    const float3 axis_ele = ours_normalize3(make_float3(pose[2], pose[6], pose[10]));
    const float3 origin = make_float3(pose[3], pose[7], pose[11]);

    const float common = -grad_mod * beta * unclamped_transmission * strength;
    for (int scan_ty = 0; scan_ty <= ty; ++scan_ty) {
        const int tid = scan_ty * Tw + tx;
        const int s = tile_range_start[tid];
        const int e = tile_range_end[tid];
        if (s < 0 || e <= s) {
            continue;
        }
        for (int j = s; j < e; ++j) {
            const int g = sorted_gauss_ids[j];
            float mu_lat = 0.f;
            float mu_ax = 0.f;
            float mu_ele = 0.f;
            float sigma_lat = 1.f;
            float sigma_ele = 1.f;
            gaussian_probe_stats(
                g,
                gauss_means,
                gauss_scales,
                gauss_quats,
                axis_lat,
                axis_ax,
                axis_ele,
                origin,
                &mu_lat,
                &mu_ax,
                &mu_ele,
                &sigma_lat,
                &sigma_ele);

            const int center_ty =
                static_cast<int>(floorf(((mu_ax - axial_near_mm) / axial_spacing_mm_per_pix) / tile_size_px));
            if (center_ty != scan_ty) {
                continue;
            }
            if (mu_ax > xi_ax + axial_margin_mm) {
                continue;
            }

            const float inv_lat = 1.f / fmaxf(1e-6f, sigma_lat * k_sigma);
            const float inv_ele = 1.f / fmaxf(1e-6f, sigma_ele * k_sigma);
            const float dlat = (xi_lat - mu_lat) * inv_lat;
            const float dele = mu_ele * inv_ele;
            const float quad = dlat * dlat + dele * dele;
            if (!(quad >= 0.f) || !isfinite(quad)) {
                continue;
            }

            const float weight = expf(-0.5f * quad);
            if (!isfinite(weight) || weight < weight_min) {
                continue;
            }
            atomicAdd(grad_attenuations + g, common * weight);
        }
    }
}

}  // namespace

torch::Tensor RayTileAttenuationCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_attenuations,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    const torch::Tensor& us_focus_params,
    int H,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    int tile_size_px,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    int shader_weight_mode,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm) {
    TORCH_CHECK(gauss_means.is_cuda(), "RayTileAttenuationCUDA expects CUDA tensors.");
    TORCH_CHECK(gauss_scales.is_cuda(), "RayTileAttenuationCUDA expects CUDA tensors.");
    TORCH_CHECK(gauss_quats.is_cuda(), "RayTileAttenuationCUDA expects CUDA tensors.");
    TORCH_CHECK(gauss_attenuations.is_cuda(), "RayTileAttenuationCUDA expects CUDA tensors.");
    TORCH_CHECK(pose_probe_to_world.is_cuda(), "RayTileAttenuationCUDA expects CUDA pose.");
    TORCH_CHECK(beta >= 0.f && beta <= 1.f, "ray attenuation beta must be in [0, 1].");
    TORCH_CHECK(strength >= 0.f, "ray attenuation strength must be >= 0.");
    TORCH_CHECK(min_transmission > 0.f && min_transmission <= 1.f, "min_transmission must be in (0, 1].");
    TORCH_CHECK(weight_min >= 0.f, "weight_min must be >= 0.");

    torch::Tensor grid_pack = ensure_slice_grid_pack_ray(
        slice_grid_geometry,
        gauss_means,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm);
    torch::Tensor us_params = ensure_us_focus_params_ray(us_focus_params, gauss_means);

    auto mapped = OursTileGaussMapperCUDA(
        gauss_means,
        gauss_scales,
        gauss_quats,
        pose_probe_to_world,
        grid_pack,
        us_params,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        tile_size_px,
        elev_corridor_half_mm,
        slab_half_thickness_mm,
        k_sigma,
        false,
        shader_weight_mode);

    const torch::Tensor& sorted_gauss_ids = std::get<0>(mapped);
    const torch::Tensor& tile_range_start = std::get<1>(mapped);
    const torch::Tensor& tile_range_end = std::get<2>(mapped);
    return RayTileAttenuationFromTileCsrCUDA(
        gauss_means,
        gauss_scales,
        gauss_quats,
        gauss_attenuations,
        pose_probe_to_world,
        grid_pack,
        sorted_gauss_ids,
        tile_range_start,
        tile_range_end,
        H,
        W,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        tile_size_px,
        k_sigma,
        beta,
        strength,
        min_transmission,
        weight_min,
        axial_margin_mm);
}

torch::Tensor RayTileAttenuationFromTileCsrCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_attenuations,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    const torch::Tensor& sorted_gauss_ids,
    const torch::Tensor& tile_range_start,
    const torch::Tensor& tile_range_end,
    int H,
    int W,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    int tile_size_px,
    float k_sigma,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm) {
    TORCH_CHECK(gauss_means.is_cuda(), "RayTileAttenuationFromTileCsrCUDA expects CUDA tensors.");
    TORCH_CHECK(sorted_gauss_ids.is_cuda(), "RayTileAttenuationFromTileCsrCUDA expects CUDA CSR tensors.");
    TORCH_CHECK(beta >= 0.f && beta <= 1.f, "ray attenuation beta must be in [0, 1].");
    TORCH_CHECK(strength >= 0.f, "ray attenuation strength must be >= 0.");
    TORCH_CHECK(min_transmission > 0.f && min_transmission <= 1.f, "min_transmission must be in (0, 1].");
    TORCH_CHECK(weight_min >= 0.f, "weight_min must be >= 0.");
    TORCH_CHECK(axial_margin_mm >= 0.f, "axial_margin_mm must be >= 0.");

    const int n_gauss = static_cast<int>(gauss_means.size(0));
    const int n_pixels = H * W;
    auto float_opts = gauss_means.options().dtype(torch::kFloat32);
    torch::Tensor modulation = torch::ones({n_pixels}, float_opts);
    if (n_gauss == 0 || n_pixels <= 0 || beta == 0.f || strength == 0.f) {
        return modulation.view({H, W});
    }

    const int Tw = (W + tile_size_px - 1) / tile_size_px;
    const int Th = (H + tile_size_px - 1) / tile_size_px;
    const int n_tiles = Tw * Th;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int block = 128;
    const int grid = (n_tiles + block - 1) / block;
    kernel_ray_tile_attenuation<<<grid, block, 0, stream>>>(
        H,
        W,
        Tw,
        Th,
        tile_size_px,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        k_sigma,
        beta,
        strength,
        min_transmission,
        weight_min,
        axial_margin_mm,
        slice_grid_geometry.data_ptr<float>(),
        tile_range_start.data_ptr<int>(),
        tile_range_end.data_ptr<int>(),
        sorted_gauss_ids.data_ptr<int>(),
        gauss_means.data_ptr<float>(),
        gauss_scales.data_ptr<float>(),
        gauss_quats.data_ptr<float>(),
        gauss_attenuations.data_ptr<float>(),
        pose_probe_to_world.data_ptr<float>(),
        modulation.data_ptr<float>());
    return modulation.view({H, W});
}

torch::Tensor RayTileAttenuationBackwardFromTileCsrCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_attenuations,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    const torch::Tensor& sorted_gauss_ids,
    const torch::Tensor& tile_range_start,
    const torch::Tensor& tile_range_end,
    const torch::Tensor& local_colors,
    const torch::Tensor& modulation,
    const torch::Tensor& grad_output,
    int H,
    int W,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    int tile_size_px,
    float k_sigma,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm) {
    TORCH_CHECK(gauss_means.is_cuda(), "RayTileAttenuationBackwardFromTileCsrCUDA expects CUDA tensors.");
    TORCH_CHECK(gauss_attenuations.is_cuda(), "RayTileAttenuationBackwardFromTileCsrCUDA expects CUDA tensors.");
    TORCH_CHECK(sorted_gauss_ids.is_cuda(), "RayTileAttenuationBackwardFromTileCsrCUDA expects CUDA CSR tensors.");
    TORCH_CHECK(local_colors.is_cuda(), "RayTileAttenuationBackwardFromTileCsrCUDA expects CUDA image tensors.");
    TORCH_CHECK(beta >= 0.f && beta <= 1.f, "ray attenuation beta must be in [0, 1].");
    TORCH_CHECK(strength >= 0.f, "ray attenuation strength must be >= 0.");
    TORCH_CHECK(min_transmission > 0.f && min_transmission <= 1.f, "min_transmission must be in (0, 1].");
    TORCH_CHECK(weight_min >= 0.f, "weight_min must be >= 0.");

    const int n_gauss = static_cast<int>(gauss_means.size(0));
    const int n_pixels = H * W;
    torch::Tensor grad_attenuations = torch::zeros_like(gauss_attenuations);
    if (n_gauss == 0 || n_pixels <= 0 || beta == 0.f || strength == 0.f) {
        return grad_attenuations;
    }

    const int Tw = (W + tile_size_px - 1) / tile_size_px;
    const int Th = (H + tile_size_px - 1) / tile_size_px;
    const int n_tiles = Tw * Th;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int block = 128;
    const int grid = (n_tiles + block - 1) / block;
    torch::Tensor local_colors_f = local_colors.to(torch::kFloat32).contiguous().view({-1});
    torch::Tensor modulation_f = modulation.to(torch::kFloat32).contiguous().view({-1});
    torch::Tensor grad_output_f = grad_output.to(torch::kFloat32).contiguous().view({-1});
    kernel_ray_tile_attenuation_backward<<<grid, block, 0, stream>>>(
        H,
        W,
        Tw,
        Th,
        tile_size_px,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        k_sigma,
        beta,
        strength,
        min_transmission,
        weight_min,
        axial_margin_mm,
        slice_grid_geometry.data_ptr<float>(),
        tile_range_start.data_ptr<int>(),
        tile_range_end.data_ptr<int>(),
        sorted_gauss_ids.data_ptr<int>(),
        gauss_means.data_ptr<float>(),
        gauss_scales.data_ptr<float>(),
        gauss_quats.data_ptr<float>(),
        gauss_attenuations.data_ptr<float>(),
        pose_probe_to_world.data_ptr<float>(),
        local_colors_f.data_ptr<float>(),
        modulation_f.data_ptr<float>(),
        grad_output_f.data_ptr<float>(),
        grad_attenuations.data_ptr<float>());
    return grad_attenuations;
}

torch::Tensor RayCsrAttenuationCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_attenuations,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    int H,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    int tile_size_px,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm,
    float quadrature_window_radius_mm,
    int quadrature_points,
    int attenuation_formula) {
    auto out = RayCsrAttenuationForwardCUDA(
        gauss_means,
        gauss_scales,
        gauss_quats,
        gauss_attenuations,
        pose_probe_to_world,
        slice_grid_geometry,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        tile_size_px,
        elev_corridor_half_mm,
        slab_half_thickness_mm,
        k_sigma,
        beta,
        strength,
        min_transmission,
        weight_min,
        axial_margin_mm,
        quadrature_window_radius_mm,
        quadrature_points,
        attenuation_formula);
    return std::get<0>(out);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RayCsrAttenuationForwardCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_attenuations,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    int H,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    int tile_size_px,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm,
    float quadrature_window_radius_mm,
    int quadrature_points,
    int attenuation_formula) {
    TORCH_CHECK(gauss_means.is_cuda(), "RayCsrAttenuationCUDA expects CUDA tensors.");
    TORCH_CHECK(gauss_attenuations.is_cuda(), "RayCsrAttenuationCUDA expects CUDA tensors.");
    TORCH_CHECK(pose_probe_to_world.is_cuda(), "RayCsrAttenuationCUDA expects CUDA pose.");
    TORCH_CHECK(beta >= 0.f && beta <= 1.f, "ray attenuation beta must be in [0, 1].");
    TORCH_CHECK(strength >= 0.f, "ray attenuation strength must be >= 0.");
    TORCH_CHECK(min_transmission > 0.f && min_transmission <= 1.f, "min_transmission must be in (0, 1].");
    TORCH_CHECK(weight_min >= 0.f, "weight_min must be >= 0.");
    TORCH_CHECK(axial_margin_mm >= 0.f, "axial_margin_mm must be >= 0.");
    TORCH_CHECK(quadrature_window_radius_mm >= 0.f, "quadrature_window_radius_mm must be >= 0.");
    TORCH_CHECK(quadrature_points >= 1 && quadrature_points <= 3, "quadrature_points must be 1, 2, or 3.");

    const int n_gauss = static_cast<int>(gauss_means.size(0));
    const int n_pixels = H * W;
    auto float_opts = gauss_means.options().dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(gauss_means.device());
    torch::Tensor modulation = torch::ones({n_pixels}, float_opts);
    torch::Tensor empty_i = torch::zeros({0}, int_opts);
    torch::Tensor empty_f = torch::zeros({0}, float_opts);
    if (n_gauss == 0 || n_pixels <= 0 || beta == 0.f || strength == 0.f) {
        return {modulation.view({H, W}), empty_i, empty_i, empty_i, empty_f};
    }

    torch::Tensor grid_pack = ensure_slice_grid_pack_ray(
        slice_grid_geometry,
        gauss_means,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm);
    const int fixed_total_capacity = attenuation_formula == 1 ? ray_fixed_total_capacity_from_env() : 0;
    const int fixed_capacity_per_ray =
        (attenuation_formula == 1 && fixed_total_capacity <= 0) ? ray_fixed_capacity_per_ray_from_env() : 0;
    auto mapped = build_ray_csr(
        gauss_means,
        gauss_scales,
        gauss_quats,
        pose_probe_to_world,
        grid_pack,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        1,
        elev_corridor_half_mm,
        slab_half_thickness_mm,
        k_sigma,
        fixed_total_capacity,
        fixed_capacity_per_ray);
    const torch::Tensor& sorted_gauss_ids = std::get<0>(mapped);
    const torch::Tensor& ray_start = std::get<1>(mapped);
    const torch::Tensor& ray_end = std::get<2>(mapped);
    const torch::Tensor& sorted_depth_pix = std::get<3>(mapped);
    if (sorted_gauss_ids.numel() == 0) {
        return {modulation.view({H, W}), ray_start, ray_end, sorted_gauss_ids, sorted_depth_pix};
    }

    const int Tw = W;
    const int Th = H;
    const int n_pixels_launch = H * W;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int block = 128;
    if (attenuation_formula == 1) {
        const int grid_x = (W + block - 1) / block;
        kernel_ray_csr_attenuation_prefix_slice_depth<<<grid_x, block, 0, stream>>>(
            H,
            W,
            Tw,
            1,
            beta,
            strength,
            min_transmission,
            weight_min,
            axial_margin_mm,
            axial_spacing_mm_per_pix,
            grid_pack.data_ptr<float>(),
            ray_start.data_ptr<int>(),
            ray_end.data_ptr<int>(),
            sorted_gauss_ids.data_ptr<int>(),
            sorted_depth_pix.data_ptr<float>(),
            gauss_means.data_ptr<float>(),
            gauss_scales.data_ptr<float>(),
            gauss_attenuations.data_ptr<float>(),
            pose_probe_to_world.data_ptr<float>(),
            modulation.data_ptr<float>());
    } else {
        const int grid = (n_pixels_launch + block - 1) / block;
        kernel_ray_csr_attenuation<<<grid, block, 0, stream>>>(
            H,
            W,
            Tw,
            Th,
            1,
            axial_spacing_mm_per_pix,
            axial_near_mm,
            k_sigma,
            beta,
            strength,
            min_transmission,
            weight_min,
            axial_margin_mm,
            quadrature_window_radius_mm,
            quadrature_points,
            attenuation_formula,
            grid_pack.data_ptr<float>(),
            ray_start.data_ptr<int>(),
            ray_end.data_ptr<int>(),
            sorted_gauss_ids.data_ptr<int>(),
            sorted_depth_pix.data_ptr<float>(),
            gauss_means.data_ptr<float>(),
            gauss_scales.data_ptr<float>(),
            gauss_quats.data_ptr<float>(),
            gauss_attenuations.data_ptr<float>(),
            pose_probe_to_world.data_ptr<float>(),
            modulation.data_ptr<float>());
    }
    return {modulation.view({H, W}), ray_start, ray_end, sorted_gauss_ids, sorted_depth_pix};
}

std::tuple<torch::Tensor, torch::Tensor> RayCsrPathContributionStatsCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_attenuations,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    int H,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    int tile_size_px,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    float strength,
    float weight_min,
    float axial_margin_mm,
    float quadrature_window_radius_mm,
    int quadrature_points,
    int attenuation_formula) {
    TORCH_CHECK(gauss_means.is_cuda(), "RayCsrPathContributionStatsCUDA expects CUDA tensors.");
    TORCH_CHECK(gauss_attenuations.is_cuda(), "RayCsrPathContributionStatsCUDA expects CUDA attenuations.");
    TORCH_CHECK(pose_probe_to_world.is_cuda(), "RayCsrPathContributionStatsCUDA expects CUDA pose.");
    TORCH_CHECK(strength >= 0.f, "ray attenuation strength must be >= 0.");
    TORCH_CHECK(weight_min >= 0.f, "ray attenuation weight_min must be >= 0.");
    TORCH_CHECK(axial_margin_mm >= 0.f, "ray attenuation axial_margin_mm must be >= 0.");
    TORCH_CHECK(quadrature_window_radius_mm >= 0.f, "quadrature_window_radius_mm must be >= 0.");
    TORCH_CHECK(quadrature_points >= 1 && quadrature_points <= 3, "quadrature_points must be 1, 2, or 3.");

    const int n_gauss = static_cast<int>(gauss_means.size(0));
    auto float_opts = gauss_means.options().dtype(torch::kFloat32);
    torch::Tensor path_contribution = torch::zeros({n_gauss}, float_opts);
    torch::Tensor path_support = torch::zeros({n_gauss}, float_opts);
    if (n_gauss == 0 || H <= 0 || W <= 0 || strength == 0.f) {
        return {path_contribution, path_support};
    }

    torch::Tensor grid_pack = ensure_slice_grid_pack_ray(
        slice_grid_geometry,
        gauss_means,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm);
    const int fixed_total_capacity = attenuation_formula == 1 ? ray_fixed_total_capacity_from_env() : 0;
    const int fixed_capacity_per_ray =
        (attenuation_formula == 1 && fixed_total_capacity <= 0) ? ray_fixed_capacity_per_ray_from_env() : 0;
    auto mapped = build_ray_csr(
        gauss_means,
        gauss_scales,
        gauss_quats,
        pose_probe_to_world,
        grid_pack,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        1,
        elev_corridor_half_mm,
        slab_half_thickness_mm,
        k_sigma,
        fixed_total_capacity,
        fixed_capacity_per_ray);
    const torch::Tensor& sorted_gauss_ids = std::get<0>(mapped);
    const torch::Tensor& ray_start = std::get<1>(mapped);
    const torch::Tensor& ray_end = std::get<2>(mapped);
    const torch::Tensor& sorted_depth_pix = std::get<3>(mapped);
    if (sorted_gauss_ids.numel() == 0) {
        return {path_contribution, path_support};
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int block = 128;
    const int grid = (H * W + block - 1) / block;
    kernel_ray_csr_path_contribution_stats<<<grid, block, 0, stream>>>(
        H,
        W,
        W,
        1,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        strength,
        weight_min,
        axial_margin_mm,
        quadrature_window_radius_mm,
        quadrature_points,
        attenuation_formula,
        grid_pack.data_ptr<float>(),
        ray_start.data_ptr<int>(),
        ray_end.data_ptr<int>(),
        sorted_gauss_ids.data_ptr<int>(),
        sorted_depth_pix.data_ptr<float>(),
        gauss_means.data_ptr<float>(),
        gauss_scales.data_ptr<float>(),
        gauss_quats.data_ptr<float>(),
        gauss_attenuations.data_ptr<float>(),
        pose_probe_to_world.data_ptr<float>(),
        path_contribution.data_ptr<float>(),
        path_support.data_ptr<float>());
    return {path_contribution, path_support};
}

std::tuple<torch::Tensor, torch::Tensor> RayCsrPathContributionStatsFromCsrCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_attenuations,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    const torch::Tensor& ray_start,
    const torch::Tensor& ray_end,
    const torch::Tensor& sorted_gauss_ids,
    const torch::Tensor& sorted_depth_pix,
    int H,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float strength,
    float weight_min,
    float axial_margin_mm,
    float quadrature_window_radius_mm,
    int quadrature_points,
    int attenuation_formula) {
    TORCH_CHECK(gauss_means.is_cuda(), "RayCsrPathContributionStatsFromCsrCUDA expects CUDA tensors.");
    TORCH_CHECK(ray_start.is_cuda(), "RayCsrPathContributionStatsFromCsrCUDA expects CUDA ray CSR.");
    TORCH_CHECK(strength >= 0.f, "ray attenuation strength must be >= 0.");
    TORCH_CHECK(weight_min >= 0.f, "ray attenuation weight_min must be >= 0.");
    TORCH_CHECK(axial_margin_mm >= 0.f, "ray attenuation axial_margin_mm must be >= 0.");
    TORCH_CHECK(quadrature_window_radius_mm >= 0.f, "quadrature_window_radius_mm must be >= 0.");
    TORCH_CHECK(quadrature_points >= 1 && quadrature_points <= 3, "quadrature_points must be 1, 2, or 3.");

    const int n_gauss = static_cast<int>(gauss_means.size(0));
    auto float_opts = gauss_means.options().dtype(torch::kFloat32);
    torch::Tensor path_contribution = torch::zeros({n_gauss}, float_opts);
    torch::Tensor path_support = torch::zeros({n_gauss}, float_opts);
    if (n_gauss == 0 || H <= 0 || W <= 0 || strength == 0.f || sorted_gauss_ids.numel() == 0) {
        return {path_contribution, path_support};
    }

    torch::Tensor grid_pack = ensure_slice_grid_pack_ray(
        slice_grid_geometry,
        gauss_means,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int block = 128;
    const int grid = (H * W + block - 1) / block;
    kernel_ray_csr_path_contribution_stats<<<grid, block, 0, stream>>>(
        H,
        W,
        W,
        1,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        strength,
        weight_min,
        axial_margin_mm,
        quadrature_window_radius_mm,
        quadrature_points,
        attenuation_formula,
        grid_pack.data_ptr<float>(),
        ray_start.data_ptr<int>(),
        ray_end.data_ptr<int>(),
        sorted_gauss_ids.data_ptr<int>(),
        sorted_depth_pix.data_ptr<float>(),
        gauss_means.data_ptr<float>(),
        gauss_scales.data_ptr<float>(),
        gauss_quats.data_ptr<float>(),
        gauss_attenuations.data_ptr<float>(),
        pose_probe_to_world.data_ptr<float>(),
        path_contribution.data_ptr<float>(),
        path_support.data_ptr<float>());
    return {path_contribution, path_support};
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
RayCsrAttenuationForwardProfileCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_attenuations,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    int H,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    int tile_size_px,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm,
    float quadrature_window_radius_mm,
    int quadrature_points,
    int attenuation_formula) {
    TORCH_CHECK(gauss_means.is_cuda(), "RayCsrAttenuationForwardProfileCUDA expects CUDA tensors.");
    TORCH_CHECK(gauss_attenuations.is_cuda(), "RayCsrAttenuationForwardProfileCUDA expects CUDA tensors.");
    TORCH_CHECK(pose_probe_to_world.is_cuda(), "RayCsrAttenuationForwardProfileCUDA expects CUDA pose.");
    TORCH_CHECK(beta >= 0.f && beta <= 1.f, "ray attenuation beta must be in [0, 1].");
    TORCH_CHECK(strength >= 0.f, "ray attenuation strength must be >= 0.");
    TORCH_CHECK(min_transmission > 0.f && min_transmission <= 1.f, "min_transmission must be in (0, 1].");
    TORCH_CHECK(weight_min >= 0.f, "weight_min must be >= 0.");
    TORCH_CHECK(axial_margin_mm >= 0.f, "axial_margin_mm must be >= 0.");
    TORCH_CHECK(quadrature_window_radius_mm >= 0.f, "quadrature_window_radius_mm must be >= 0.");
    TORCH_CHECK(quadrature_points >= 1 && quadrature_points <= 3, "quadrature_points must be 1, 2, or 3.");

    const int n_gauss = static_cast<int>(gauss_means.size(0));
    const int n_pixels = H * W;
    auto float_opts = gauss_means.options().dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(gauss_means.device());
    torch::Tensor modulation = torch::ones({n_pixels}, float_opts);
    torch::Tensor empty_i = torch::zeros({0}, int_opts);
    torch::Tensor empty_f = torch::zeros({0}, float_opts);
    torch::Tensor timings = torch::zeros({11}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor stats = torch::zeros({8}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    if (n_gauss == 0 || n_pixels <= 0 || beta == 0.f || strength == 0.f) {
        return {modulation.view({H, W}), empty_i, empty_i, empty_i, empty_f, timings, stats};
    }

    torch::Tensor grid_pack = ensure_slice_grid_pack_ray(
        slice_grid_geometry,
        gauss_means,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm);
    const int fixed_total_capacity = attenuation_formula == 1 ? ray_fixed_total_capacity_from_env() : 0;
    const int fixed_capacity_per_ray =
        (attenuation_formula == 1 && fixed_total_capacity <= 0) ? ray_fixed_capacity_per_ray_from_env() : 0;
    auto mapped = build_ray_csr_profile(
        gauss_means,
        gauss_scales,
        gauss_quats,
        pose_probe_to_world,
        grid_pack,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        1,
        elev_corridor_half_mm,
        slab_half_thickness_mm,
        k_sigma,
        fixed_total_capacity,
        fixed_capacity_per_ray);
    const torch::Tensor& sorted_gauss_ids = std::get<0>(mapped);
    const torch::Tensor& ray_start = std::get<1>(mapped);
    const torch::Tensor& ray_end = std::get<2>(mapped);
    const torch::Tensor& sorted_depth_pix = std::get<3>(mapped);
    timings = std::get<4>(mapped).clone();
    stats = std::get<5>(mapped).clone();
    if (sorted_gauss_ids.numel() == 0) {
        return {modulation.view({H, W}), ray_start, ray_end, sorted_gauss_ids, sorted_depth_pix, timings, stats};
    }

    const int Tw = W;
    const int Th = H;
    const int n_pixels_launch = H * W;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int block = 128;
    cudaEvent_t em0, em1;
    cudaEventCreate(&em0);
    cudaEventCreate(&em1);
    cudaEventRecord(em0, stream);
    if (attenuation_formula == 1) {
        const int grid_x = (W + block - 1) / block;
        kernel_ray_csr_attenuation_prefix_slice_depth<<<grid_x, block, 0, stream>>>(
            H,
            W,
            Tw,
            1,
            beta,
            strength,
            min_transmission,
            weight_min,
            axial_margin_mm,
            axial_spacing_mm_per_pix,
            grid_pack.data_ptr<float>(),
            ray_start.data_ptr<int>(),
            ray_end.data_ptr<int>(),
            sorted_gauss_ids.data_ptr<int>(),
            sorted_depth_pix.data_ptr<float>(),
            gauss_means.data_ptr<float>(),
            gauss_scales.data_ptr<float>(),
            gauss_attenuations.data_ptr<float>(),
            pose_probe_to_world.data_ptr<float>(),
            modulation.data_ptr<float>());
    } else {
        const int grid = (n_pixels_launch + block - 1) / block;
        kernel_ray_csr_attenuation<<<grid, block, 0, stream>>>(
            H,
            W,
            Tw,
            Th,
            1,
            axial_spacing_mm_per_pix,
            axial_near_mm,
            k_sigma,
            beta,
            strength,
            min_transmission,
            weight_min,
            axial_margin_mm,
            quadrature_window_radius_mm,
            quadrature_points,
            attenuation_formula,
            grid_pack.data_ptr<float>(),
            ray_start.data_ptr<int>(),
            ray_end.data_ptr<int>(),
            sorted_gauss_ids.data_ptr<int>(),
            sorted_depth_pix.data_ptr<float>(),
            gauss_means.data_ptr<float>(),
            gauss_scales.data_ptr<float>(),
            gauss_quats.data_ptr<float>(),
            gauss_attenuations.data_ptr<float>(),
            pose_probe_to_world.data_ptr<float>(),
            modulation.data_ptr<float>());
    }
    cudaEventRecord(em1, stream);
    cudaEventSynchronize(em1);
    float modulation_ms = 0.f;
    cudaEventElapsedTime(&modulation_ms, em0, em1);
    timings[7] = modulation_ms;
    cudaEventDestroy(em0);
    cudaEventDestroy(em1);
    return {modulation.view({H, W}), ray_start, ray_end, sorted_gauss_ids, sorted_depth_pix, timings, stats};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RayCsrAttenuationBackwardCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_attenuations,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    const torch::Tensor& ray_start,
    const torch::Tensor& ray_end,
    const torch::Tensor& sorted_gauss_ids,
    const torch::Tensor& sorted_depth_pix,
    const torch::Tensor& local_colors,
    const torch::Tensor& modulation,
    const torch::Tensor& grad_output,
    int H,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    int tile_size_px,
    float elev_corridor_half_mm,
    float slab_half_thickness_mm,
    float k_sigma,
    float beta,
    float strength,
    float min_transmission,
    float weight_min,
    float axial_margin_mm,
    float quadrature_window_radius_mm,
    int quadrature_points,
    int attenuation_formula) {
    TORCH_CHECK(gauss_means.is_cuda(), "RayCsrAttenuationBackwardCUDA expects CUDA tensors.");
    TORCH_CHECK(gauss_attenuations.is_cuda(), "RayCsrAttenuationBackwardCUDA expects CUDA tensors.");
    TORCH_CHECK(ray_start.is_cuda(), "RayCsrAttenuationBackwardCUDA expects CUDA ray CSR tensors.");
    TORCH_CHECK(local_colors.is_cuda(), "RayCsrAttenuationBackwardCUDA expects CUDA image tensors.");
    TORCH_CHECK(beta >= 0.f && beta <= 1.f, "ray attenuation beta must be in [0, 1].");
    TORCH_CHECK(strength >= 0.f, "ray attenuation strength must be >= 0.");
    TORCH_CHECK(min_transmission > 0.f && min_transmission <= 1.f, "min_transmission must be in (0, 1].");
    TORCH_CHECK(weight_min >= 0.f, "weight_min must be >= 0.");
    TORCH_CHECK(axial_margin_mm >= 0.f, "axial_margin_mm must be >= 0.");
    TORCH_CHECK(quadrature_window_radius_mm >= 0.f, "quadrature_window_radius_mm must be >= 0.");
    TORCH_CHECK(quadrature_points >= 1 && quadrature_points <= 3, "quadrature_points must be 1, 2, or 3.");

    const int n_gauss = static_cast<int>(gauss_means.size(0));
    const int n_pixels = H * W;
    torch::Tensor grad_attenuations = torch::zeros_like(gauss_attenuations);
    torch::Tensor grad_means = torch::zeros_like(gauss_means);
    torch::Tensor grad_scales = torch::zeros_like(gauss_scales);
    torch::Tensor grad_quats = torch::zeros_like(gauss_quats);
    if (n_gauss == 0 || n_pixels <= 0 || beta == 0.f || strength == 0.f) {
        return {grad_attenuations, grad_means, grad_scales, grad_quats};
    }
    if (sorted_gauss_ids.numel() == 0) {
        return {grad_attenuations, grad_means, grad_scales, grad_quats};
    }

    torch::Tensor grid_pack = ensure_slice_grid_pack_ray(
        slice_grid_geometry,
        gauss_means,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm);
    const int Tw = W;
    const int Th = H;
    const int n_pixels_launch = H * W;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int block = 128;
    torch::Tensor local_colors_f = local_colors.to(torch::kFloat32).contiguous().view({-1});
    torch::Tensor modulation_f = modulation.to(torch::kFloat32).contiguous().view({-1});
    torch::Tensor grad_output_f = grad_output.to(torch::kFloat32).contiguous().view({-1});
    if (attenuation_formula == 1) {
        const int grid_x = (W + block - 1) / block;
        kernel_ray_csr_attenuation_backward_prefix_slice_depth<<<grid_x, block, 0, stream>>>(
            H,
            W,
            Tw,
            1,
            beta,
            strength,
            min_transmission,
            weight_min,
            axial_margin_mm,
            axial_spacing_mm_per_pix,
            grid_pack.data_ptr<float>(),
            ray_start.data_ptr<int>(),
            ray_end.data_ptr<int>(),
            sorted_gauss_ids.data_ptr<int>(),
            sorted_depth_pix.data_ptr<float>(),
            gauss_means.data_ptr<float>(),
            gauss_scales.data_ptr<float>(),
            gauss_attenuations.data_ptr<float>(),
            pose_probe_to_world.data_ptr<float>(),
            local_colors_f.data_ptr<float>(),
            modulation_f.data_ptr<float>(),
            grad_output_f.data_ptr<float>(),
            grad_attenuations.data_ptr<float>(),
            grad_means.data_ptr<float>(),
            grad_scales.data_ptr<float>());
        return {grad_attenuations, grad_means, grad_scales, grad_quats};
    }

    const int grid = (n_pixels_launch + block - 1) / block;
    kernel_ray_csr_attenuation_backward<<<grid, block, 0, stream>>>(
        H,
        W,
        Tw,
        Th,
        1,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        k_sigma,
        beta,
        strength,
        min_transmission,
        weight_min,
        axial_margin_mm,
        quadrature_window_radius_mm,
        quadrature_points,
        attenuation_formula,
        grid_pack.data_ptr<float>(),
        ray_start.data_ptr<int>(),
        ray_end.data_ptr<int>(),
        sorted_gauss_ids.data_ptr<int>(),
        sorted_depth_pix.data_ptr<float>(),
        gauss_means.data_ptr<float>(),
        gauss_scales.data_ptr<float>(),
        gauss_quats.data_ptr<float>(),
        gauss_attenuations.data_ptr<float>(),
        pose_probe_to_world.data_ptr<float>(),
        local_colors_f.data_ptr<float>(),
        modulation_f.data_ptr<float>(),
        grad_output_f.data_ptr<float>(),
        grad_attenuations.data_ptr<float>(),
        grad_means.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        grad_quats.data_ptr<float>());
    return {grad_attenuations, grad_means, grad_scales, grad_quats};
}
