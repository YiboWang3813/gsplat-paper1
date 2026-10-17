#include <cuda_runtime.h>
#include <math_constants.h>

#include "shader/forward.cuh"
#include "shader/plane_cond2d.cuh"
#include "shader/shared.cuh"
#include "shader/slice_grid_geometry.cuh"
#include "shader/us_focus_device.cuh"

namespace {

constexpr int US_FOCUS_TILE_CAND_BATCH = 64;

struct UsFocusTileCandidate {
    int g;
    float base_weight;
    float intensity;
    float mu2_lat;
    float mu2_ax;
    float inv2_00;
    float inv2_01;
    float inv2_11;
    float r_lat_mm;
    float r_ax_mm;
};

__device__ __forceinline__ bool us_focus_make_tile_candidate(
    int g,
    float k_sigma,
    const float* __restrict__ slice_grid_pack,
    const float* __restrict__ us_focus_params,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    float3 axis_lat,
    float3 axis_ax,
    float3 axis_ele,
    float3 origin,
    UsFocusTileCandidate* __restrict__ out) {
    const float* mptr = gauss_means + g * 3;
    const float* sptr = gauss_scales + g * 3;
    const float* qptr = gauss_quats + g * 4;
    us_focus_device::UsFocusFootprintOut fp;
    if (!us_focus_device::compute_us_focus_footprint_mm(
            mptr,
            sptr,
            qptr,
            axis_lat,
            axis_ax,
            axis_ele,
            origin,
            slice_grid_pack[SG_NEAR_MM],
            k_sigma,
            us_focus_params,
            &fp)) {
        return false;
    }

    const float det_eps =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_DET_EPS, 1e-6f);
    const float denom_ele = fmaxf(fp.sc22 + fp.sigma_ele * fp.sigma_ele + det_eps, 1e-12f);
    const float slice_response = expf(-0.5f * fp.mu_ele * fp.mu_ele / denom_ele);
    const float det_gamma =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_DET_GAMMA, 0.25f);
    const float norm_factor = us_focus_device::det_norm_factor(fp.det2 + det_eps, det_gamma);
    const float base_weight = gauss_opacities[g] * slice_response * norm_factor;
    if (!isfinite(base_weight)) {
        return false;
    }

    out->g = g;
    out->base_weight = base_weight;
    out->intensity = gauss_intensities[g];
    out->mu2_lat = fp.mu2_lat;
    out->mu2_ax = fp.mu2_ax;
    out->inv2_00 = fp.inv2_00;
    out->inv2_01 = fp.inv2_01;
    out->inv2_11 = fp.inv2_11;
    out->r_lat_mm = fp.r_lat_mm;
    out->r_ax_mm = fp.r_ax_mm;
    return true;
}

__device__ __forceinline__ float us_focus_eval_tile_candidate(
    const UsFocusTileCandidate& cand,
    float xi_lat,
    float xi_ax,
    float weight_min) {
    if (cand.g < 0) {
        return 0.f;
    }
    if (fabsf(xi_lat - cand.mu2_lat) > cand.r_lat_mm || fabsf(xi_ax - cand.mu2_ax) > cand.r_ax_mm) {
        return 0.f;
    }

    const float dlat = xi_lat - cand.mu2_lat;
    const float dax = xi_ax - cand.mu2_ax;
    const float qlat = cand.inv2_00 * dlat + cand.inv2_01 * dax;
    const float qax = cand.inv2_01 * dlat + cand.inv2_11 * dax;
    const float quad = dlat * qlat + dax * qax;
    if (!(quad >= 0.f) || !isfinite(quad)) {
        return 0.f;
    }

    const float weight = cand.base_weight * expf(-0.5f * quad);
    if (!isfinite(weight) || weight < weight_min) {
        return 0.f;
    }
    return cand.intensity * weight;
}

__global__ void kernel_us_focus_tile2d_additive_forward(
    int H,
    int W,
    int Tw,
    int tile_size_px,
    float k_sigma,
    const float* __restrict__ slice_grid_pack,
    const float* __restrict__ us_focus_params,
    float fixed_bg_color,
    const int* __restrict__ tile_range_start,
    const int* __restrict__ tile_range_end,
    const int* __restrict__ sorted_gauss_ids,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    const float* __restrict__ pose,
    float* __restrict__ pixel_colors) {
    (void)fixed_bg_color;
    __shared__ UsFocusTileCandidate sh_cands[US_FOCUS_TILE_CAND_BATCH];

    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int lane = local_y * blockDim.x + local_x;
    const int tx = blockIdx.x;
    const int ty = blockIdx.y;
    const int x = tx * tile_size_px + local_x;
    const int y = ty * tile_size_px + local_y;
    const bool pixel_valid = (local_x < tile_size_px) && (local_y < tile_size_px) && (x < W) && (y < H);
    const int tid = ty * Tw + tx;

    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 raw_ele = make_float3(pose[2], pose[6], pose[10]);
    const float3 axis_lat = ours_normalize3(raw_lat);
    const float3 axis_ax = ours_normalize3(raw_ax);
    const float3 axis_ele = ours_normalize3(raw_ele);
    const float3 origin = make_float3(pose[3], pose[7], pose[11]);

    const float weight_min =
        us_focus_device::us_focus_param_or_default(us_focus_params, us_focus_device::US_FOCUS_WEIGHT_MIN, 1e-6f);

    float accum = 0.f;
    float xi_lat = 0.f;
    float xi_ax = 0.f;
    if (pixel_valid) {
        sg_linear_pixel_to_plane_mm(slice_grid_pack, x, y, &xi_lat, &xi_ax);
    }
    const int s = tile_range_start[tid];
    const int e = tile_range_end[tid];
    if (s >= 0 && e > s) {
        for (int base = s; base < e; base += US_FOCUS_TILE_CAND_BATCH) {
            const int batch_count = min(US_FOCUS_TILE_CAND_BATCH, e - base);
            if (lane < batch_count) {
                UsFocusTileCandidate cand;
                cand.g = -1;
                const int g = sorted_gauss_ids[base + lane];
                if (us_focus_make_tile_candidate(
                        g,
                        k_sigma,
                        slice_grid_pack,
                        us_focus_params,
                        gauss_means,
                        gauss_scales,
                        gauss_quats,
                        gauss_opacities,
                        gauss_intensities,
                        axis_lat,
                        axis_ax,
                        axis_ele,
                        origin,
                        &cand)) {
                    sh_cands[lane] = cand;
                } else {
                    sh_cands[lane].g = -1;
                }
            }
            __syncthreads();

            if (pixel_valid) {
                #pragma unroll
                for (int k = 0; k < US_FOCUS_TILE_CAND_BATCH; ++k) {
                    if (k >= batch_count) {
                        break;
                    }
                    accum += us_focus_eval_tile_candidate(sh_cands[k], xi_lat, xi_ax, weight_min);
                }
            }
            __syncthreads();
        }
    }

    if (pixel_valid) {
        pixel_colors[y * W + x] = accum;
    }
}

__global__ void kernel_ours_tile_csr_forward(
    int n_pixels,
    int H,
    int W,
    int Tw,
    int Th,
    int tile_size_px,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float k_sigma,
    int shader_weight_mode,
    const float* __restrict__ slice_grid_pack,
    const float* __restrict__ us_focus_params,
    int compositing_mode,
    float fixed_bg_color,
    const int* __restrict__ tile_range_start,
    const int* __restrict__ tile_range_end,
    const int* __restrict__ sorted_gauss_ids,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    const float* __restrict__ pose,
    float* __restrict__ pixel_colors) {
    (void)Th;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    int y = p / W;
    int x = p - y * W;

    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 raw_ele = make_float3(pose[2], pose[6], pose[10]);
    float3 axis_lat = ours_normalize3(raw_lat);
    float3 axis_ax = ours_normalize3(raw_ax);
    float3 axis_ele = ours_normalize3(raw_ele);
    float3 origin = make_float3(pose[3], pose[7], pose[11]);

    int ts = tile_size_px;
    int tx = x / ts;
    int ty = y / ts;
    int tid = ty * Tw + tx;
    int s = tile_range_start[tid];
    int e = tile_range_end[tid];

    float accum = 0.f;
    float T = 1.f;
    float C = 0.f;
    if (s >= 0 && e > s) {
        for (int j = s; j < e; ++j) {
            int g = sorted_gauss_ids[j];
            float alpha = 0.f;
            float I = 0.f;
            if (shader_weight_mode == SHADER_WEIGHT_PLANE_COND2D) {
                ours_plane_cond2d_eval_at_pixel(
                    x,
                    y,
                    k_sigma,
                    g,
                    gauss_means,
                    gauss_scales,
                    gauss_quats,
                    gauss_opacities,
                    gauss_intensities,
                    axis_lat,
                    axis_ax,
                    axis_ele,
                    origin,
                    slice_grid_pack,
                    &alpha,
                    &I);
            } else if (shader_weight_mode == SHADER_WEIGHT_US_FOCUS) {
                us_focus_device::UsFocusEvalOut out;
                if (us_focus_device::us_focus_eval_at_pixel(
                        x,
                        y,
                        k_sigma,
                        g,
                        gauss_means,
                        gauss_scales,
                        gauss_quats,
                        gauss_opacities,
                        gauss_intensities,
                        axis_lat,
                        axis_ax,
                        axis_ele,
                        origin,
                        slice_grid_pack,
                        us_focus_params,
                        &out)) {
                    alpha = out.weight;
                    I = out.intensity;
                }
            } else {
                ours_eval_gauss_contrib_at_pixel(
                    x,
                    y,
                    W,
                    slice_grid_pack[SG_LAT_COL_OFFSET],
                    lateral_spacing_mm_per_pix,
                    axial_spacing_mm_per_pix,
                    axial_near_mm,
                    k_sigma,
                    g,
                    gauss_means,
                    gauss_scales,
                    gauss_quats,
                    gauss_opacities,
                    gauss_intensities,
                    pose,
                    axis_lat,
                    axis_ax,
                    axis_ele,
                    origin,
                    &alpha,
                    &I);
            }
            if (shader_weight_mode != SHADER_WEIGHT_US_FOCUS && alpha < OURS_ALPHA_MIN) {
                continue;
            }
            if (compositing_mode == 0) {
                accum += I * alpha;
            } else {
                C += I * alpha * T;
                T *= (1.f - alpha);
                if (T < 1e-4f) break;
            }
        }
    }
    if (compositing_mode == 0) {
        pixel_colors[p] = accum;
    } else {
        pixel_colors[p] = C + T * fixed_bg_color;
    }
}

__global__ void kernel_ours_pixel_csr_forward(
    int n_pixels,
    int H,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float k_sigma,
    int shader_weight_mode,
    const float* __restrict__ slice_grid_pack,
    const float* __restrict__ us_focus_params,
    int compositing_mode,
    float fixed_bg_color,
    const int* __restrict__ pixel_gauss_offsets,
    const int* __restrict__ pixel_gauss_counts,
    const int* __restrict__ pixel_gauss_indices,
    const float* __restrict__ gauss_means,
    const float* __restrict__ gauss_scales,
    const float* __restrict__ gauss_quats,
    const float* __restrict__ gauss_opacities,
    const float* __restrict__ gauss_intensities,
    const float* __restrict__ pose,
    float* __restrict__ pixel_colors) {
    (void)H;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    int y = p / W;
    int x = p - y * W;

    float3 raw_lat = make_float3(pose[0], pose[4], pose[8]);
    float3 raw_ax = make_float3(pose[1], pose[5], pose[9]);
    float3 raw_ele = make_float3(pose[2], pose[6], pose[10]);
    float3 axis_lat = ours_normalize3(raw_lat);
    float3 axis_ax = ours_normalize3(raw_ax);
    float3 axis_ele = ours_normalize3(raw_ele);
    float3 origin = make_float3(pose[3], pose[7], pose[11]);

    int off = pixel_gauss_offsets[p];
    int cnt = pixel_gauss_counts[p];

    float accum = 0.f;
    float T = 1.f;
    float C = 0.f;
    for (int k = 0; k < cnt; ++k) {
        int g = pixel_gauss_indices[off + k];
        float alpha = 0.f;
        float I = 0.f;
        if (shader_weight_mode == SHADER_WEIGHT_PLANE_COND2D) {
            ours_plane_cond2d_eval_at_pixel(
                x,
                y,
                k_sigma,
                g,
                gauss_means,
                gauss_scales,
                gauss_quats,
                gauss_opacities,
                gauss_intensities,
                axis_lat,
                axis_ax,
                axis_ele,
                origin,
                slice_grid_pack,
                &alpha,
                &I);
        } else if (shader_weight_mode == SHADER_WEIGHT_US_FOCUS) {
            us_focus_device::UsFocusEvalOut out;
            if (us_focus_device::us_focus_eval_at_pixel(
                    x,
                    y,
                    k_sigma,
                    g,
                    gauss_means,
                    gauss_scales,
                    gauss_quats,
                    gauss_opacities,
                    gauss_intensities,
                    axis_lat,
                    axis_ax,
                    axis_ele,
                    origin,
                    slice_grid_pack,
                    us_focus_params,
                    &out)) {
                alpha = out.weight;
                I = out.intensity;
            }
        } else {
            ours_eval_gauss_contrib_at_pixel(
                x,
                y,
                W,
                slice_grid_pack[SG_LAT_COL_OFFSET],
                lateral_spacing_mm_per_pix,
                axial_spacing_mm_per_pix,
                axial_near_mm,
                k_sigma,
                g,
                gauss_means,
                gauss_scales,
                gauss_quats,
                gauss_opacities,
                gauss_intensities,
                pose,
                axis_lat,
                axis_ax,
                axis_ele,
                origin,
                &alpha,
                &I);
        }
        if (shader_weight_mode != SHADER_WEIGHT_US_FOCUS && alpha < OURS_ALPHA_MIN) {
            continue;
        }
        if (compositing_mode == 0) {
            accum += I * alpha;
        } else {
            C += I * alpha * T;
            T *= (1.f - alpha);
            if (T < 1e-4f) break;
        }
    }
    if (compositing_mode == 0) {
        pixel_colors[p] = accum;
    } else {
        pixel_colors[p] = C + T * fixed_bg_color;
    }
}

}  // namespace

void RenderOursSliceForwardCUDA_Impl(
    int n_pixels,
    int n_gauss,
    int H,
    int W,
    int num_tiles_w,
    int num_tiles_h,
    int tile_size_px,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float k_sigma,
    int shader_weight_mode,
    const float* slice_grid_pack,
    const float* us_focus_params,
    int compositing_mode,
    float fixed_bg_color,
    const int* tile_range_start,
    const int* tile_range_end,
    const int* sorted_gauss_ids,
    const float* gauss_means,
    const float* gauss_scales,
    const float* gauss_quats,
    const float* gauss_opacities,
    const float* gauss_intensities,
    const float* pose_probe_to_world,
    float* pixel_colors,
    cudaStream_t stream) {
    (void)n_gauss;
    if (shader_weight_mode == SHADER_WEIGHT_US_FOCUS && compositing_mode == 0 && tile_size_px <= 16) {
        const dim3 block(tile_size_px, tile_size_px, 1);
        const dim3 grid(num_tiles_w, num_tiles_h, 1);
        kernel_us_focus_tile2d_additive_forward<<<grid, block, 0, stream>>>(
            H,
            W,
            num_tiles_w,
            tile_size_px,
            k_sigma,
            slice_grid_pack,
            us_focus_params,
            fixed_bg_color,
            tile_range_start,
            tile_range_end,
            sorted_gauss_ids,
            gauss_means,
            gauss_scales,
            gauss_quats,
            gauss_opacities,
            gauss_intensities,
            pose_probe_to_world,
            pixel_colors);
        return;
    }
    const int block = 128;
    const int grid = (n_pixels + block - 1) / block;
    kernel_ours_tile_csr_forward<<<grid, block, 0, stream>>>(
        n_pixels,
        H,
        W,
        num_tiles_w,
        num_tiles_h,
        tile_size_px,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        k_sigma,
        shader_weight_mode,
        slice_grid_pack,
        us_focus_params,
        compositing_mode,
        fixed_bg_color,
        tile_range_start,
        tile_range_end,
        sorted_gauss_ids,
        gauss_means,
        gauss_scales,
        gauss_quats,
        gauss_opacities,
        gauss_intensities,
        pose_probe_to_world,
        pixel_colors);
}

void RenderOursSliceForwardPixelCSR_CUDA_Impl(
    int n_pixels,
    int n_gauss,
    int H,
    int W,
    float lateral_spacing_mm_per_pix,
    float axial_spacing_mm_per_pix,
    float axial_near_mm,
    float k_sigma,
    int shader_weight_mode,
    const float* slice_grid_pack,
    const float* us_focus_params,
    int compositing_mode,
    float fixed_bg_color,
    const int* pixel_gauss_offsets,
    const int* pixel_gauss_counts,
    const int* pixel_gauss_indices,
    const float* gauss_means,
    const float* gauss_scales,
    const float* gauss_quats,
    const float* gauss_opacities,
    const float* gauss_intensities,
    const float* pose_probe_to_world,
    float* pixel_colors,
    cudaStream_t stream) {
    (void)n_gauss;
    const int block = 128;
    const int grid = (n_pixels + block - 1) / block;
    kernel_ours_pixel_csr_forward<<<grid, block, 0, stream>>>(
        n_pixels,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm,
        k_sigma,
        shader_weight_mode,
        slice_grid_pack,
        us_focus_params,
        compositing_mode,
        fixed_bg_color,
        pixel_gauss_offsets,
        pixel_gauss_counts,
        pixel_gauss_indices,
        gauss_means,
        gauss_scales,
        gauss_quats,
        gauss_opacities,
        gauss_intensities,
        pose_probe_to_world,
        pixel_colors);
}
