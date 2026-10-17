#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <torch/extension.h>

#include "mapper/core.cuh"
#include "shader/backward.cuh"
#include "shader/forward.cuh"
#include "shader/plane_cond2d.cuh"
#include "shader/slice_grid_geometry.cuh"
#include "shader/us_focus.cuh"
#include "shader/wrapper_shader.cuh"

namespace {

torch::Tensor ensure_slice_grid_pack(
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

torch::Tensor ensure_us_focus_params(const torch::Tensor& us_focus_params, const torch::Tensor& ref) {
    if (us_focus_params.numel() > 0) {
        return us_focus_params.to(torch::kFloat32).contiguous().view({-1});
    }
    auto float_opts = ref.options().dtype(torch::kFloat32);
    return torch::zeros({14}, float_opts);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ProjectedSliceForwardCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_opacities,
    const torch::Tensor& gauss_intensities,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    const torch::Tensor& us_focus_params,
    const int H,
    const int W,
    const float lateral_spacing_mm_per_pix,
    const float axial_spacing_mm_per_pix,
    const float axial_near_mm,
    const int tile_size_px,
    const float elev_corridor_half_mm,
    const float slab_half_thickness_mm,
    const float k_sigma,
    const int mapper_kind,
    const int num_pixels_lateral,
    const int num_pixels_axial,
    const bool sort_by_v,
    const int compositing_mode,
    const float fixed_bg_color,
    const int shader_weight_mode) {
    const int n_gauss = static_cast<int>(gauss_means.size(0));
    const int n_pixels = H * W;
    auto float_opts = gauss_means.options().dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(gauss_means.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor pixel_colors = torch::zeros({n_pixels}, float_opts);
    torch::Tensor empty_i = torch::zeros({0}, int_opts);

    torch::Tensor grid_pack = ensure_slice_grid_pack(
        slice_grid_geometry,
        gauss_means,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm);
    torch::Tensor us_params = ensure_us_focus_params(us_focus_params, gauss_means);
    TORCH_CHECK(
        shader_weight_mode != SHADER_WEIGHT_US_FOCUS || mapper_kind == 0,
        "us_focus currently supports tile mapper only.");
    TORCH_CHECK(
        shader_weight_mode != SHADER_WEIGHT_US_FOCUS || compositing_mode == 0,
        "us_focus currently supports additive compositing only.");

    if (n_gauss == 0 || n_pixels <= 0) {
        torch::Tensor empty_rects = torch::zeros({0, 4}, int_opts);
        return {pixel_colors.view({H, W}), empty_rects, empty_i, empty_i, empty_i};
    }

    torch::Tensor gauss_rects;
    torch::Tensor csr0, csr1, csr2;

    if (mapper_kind == 0) {
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
            sort_by_v,
            shader_weight_mode);
        csr0 = std::get<0>(mapped);
        csr1 = std::get<1>(mapped);
        csr2 = std::get<2>(mapped);
        gauss_rects = std::get<3>(mapped);

        const int Tw = (W + tile_size_px - 1) / tile_size_px;
        const int Th = (H + tile_size_px - 1) / tile_size_px;

        RenderOursSliceForwardCUDA_Impl(
            n_pixels,
            n_gauss,
            H,
            W,
            Tw,
            Th,
            tile_size_px,
            lateral_spacing_mm_per_pix,
            axial_spacing_mm_per_pix,
            axial_near_mm,
            k_sigma,
            shader_weight_mode,
            grid_pack.data_ptr<float>(),
            us_params.data_ptr<float>(),
            compositing_mode,
            fixed_bg_color,
            csr1.data_ptr<int>(),
            csr2.data_ptr<int>(),
            csr0.data_ptr<int>(),
            gauss_means.data_ptr<float>(),
            gauss_scales.data_ptr<float>(),
            gauss_quats.data_ptr<float>(),
            gauss_opacities.data_ptr<float>(),
            gauss_intensities.data_ptr<float>(),
            pose_probe_to_world.data_ptr<float>(),
            pixel_colors.data_ptr<float>(),
            stream);
    } else {
        auto mapped = OursPixelGaussMapperCUDA(
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
            elev_corridor_half_mm,
            slab_half_thickness_mm,
            k_sigma,
            num_pixels_lateral,
            num_pixels_axial,
            sort_by_v,
            shader_weight_mode);
        torch::Tensor pixel_gauss_counts = std::get<0>(mapped);
        torch::Tensor pixel_gauss_offsets = std::get<1>(mapped);
        torch::Tensor pixel_gauss_indices = std::get<2>(mapped);
        gauss_rects = std::get<3>(mapped);
        csr0 = pixel_gauss_counts;
        csr1 = pixel_gauss_offsets;
        csr2 = pixel_gauss_indices;

        RenderOursSliceForwardPixelCSR_CUDA_Impl(
            n_pixels,
            n_gauss,
            H,
            W,
            lateral_spacing_mm_per_pix,
            axial_spacing_mm_per_pix,
            axial_near_mm,
            k_sigma,
            shader_weight_mode,
            grid_pack.data_ptr<float>(),
            us_params.data_ptr<float>(),
            compositing_mode,
            fixed_bg_color,
            pixel_gauss_offsets.data_ptr<int>(),
            pixel_gauss_counts.data_ptr<int>(),
            pixel_gauss_indices.data_ptr<int>(),
            gauss_means.data_ptr<float>(),
            gauss_scales.data_ptr<float>(),
            gauss_quats.data_ptr<float>(),
            gauss_opacities.data_ptr<float>(),
            gauss_intensities.data_ptr<float>(),
            pose_probe_to_world.data_ptr<float>(),
            pixel_colors.data_ptr<float>(),
            stream);
    }

    return {pixel_colors.view({H, W}), gauss_rects, csr0, csr1, csr2};
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
ProjectedSliceBackwardCUDA(
    const torch::Tensor& gauss_means,
    const torch::Tensor& gauss_scales,
    const torch::Tensor& gauss_quats,
    const torch::Tensor& gauss_opacities,
    const torch::Tensor& gauss_intensities,
    const torch::Tensor& pose_probe_to_world,
    const torch::Tensor& slice_grid_geometry,
    const torch::Tensor& us_focus_params,
    const torch::Tensor& gauss_rects,
    const int gauss_rect_is_pixels,
    const int H,
    const int W,
    const int tile_size_px,
    const float lateral_spacing_mm_per_pix,
    const float axial_spacing_mm_per_pix,
    const float axial_near_mm,
    const float k_sigma,
    const int compositing_mode,
    const float fixed_bg_color,
    const int mapper_kind,
    const torch::Tensor& csr0,
    const torch::Tensor& csr1,
    const torch::Tensor& csr2,
    const torch::Tensor& grad_pixel_colors,
    const int shader_weight_mode,
    const bool compute_pose_grad) {
    const int n_gauss = static_cast<int>(gauss_means.size(0));
    const int n_pixels = H * W;
    auto float_opts = gauss_means.options().dtype(torch::kFloat32);

    torch::Tensor grad_gauss_means = torch::zeros_like(gauss_means);
    torch::Tensor grad_gauss_scales = torch::zeros_like(gauss_scales);
    torch::Tensor grad_gauss_quats = torch::zeros_like(gauss_quats);
    torch::Tensor grad_gauss_opacities = torch::zeros_like(gauss_opacities);
    torch::Tensor grad_gauss_intensities = torch::zeros_like(gauss_intensities);
    torch::Tensor grad_pose = torch::zeros({4, 4}, float_opts);

    torch::Tensor gc = grad_pixel_colors.to(torch::kFloat32).reshape({-1}).contiguous();

    torch::Tensor grid_pack = ensure_slice_grid_pack(
        slice_grid_geometry,
        gauss_means,
        H,
        W,
        lateral_spacing_mm_per_pix,
        axial_spacing_mm_per_pix,
        axial_near_mm);
    torch::Tensor us_params = ensure_us_focus_params(us_focus_params, gauss_means);

    const int wsp_stride = (shader_weight_mode == SHADER_WEIGHT_PLANE_COND2D)
        ? PLANE_COND2D_WSP_STRIDE
        : ((shader_weight_mode == SHADER_WEIGHT_US_FOCUS) ? 35 : 14);
    torch::Tensor workspace = torch::empty({0}, float_opts);
    if (n_gauss > 0) {
        workspace = torch::zeros({n_gauss * wsp_stride}, float_opts);
    }

    const int* p_tile_rs = nullptr;
    const int* p_tile_re = nullptr;
    const int* p_sorted = nullptr;
    const int* p_pix_cnt = nullptr;
    const int* p_pix_off = nullptr;
    const int* p_pix_idx = nullptr;
    if (mapper_kind == 0) {
        if (csr0.numel() > 0) p_sorted = csr0.data_ptr<int>();
        if (csr1.numel() > 0) p_tile_rs = csr1.data_ptr<int>();
        if (csr2.numel() > 0) p_tile_re = csr2.data_ptr<int>();
    } else {
        if (csr0.numel() > 0) p_pix_cnt = csr0.data_ptr<int>();
        if (csr1.numel() > 0) p_pix_off = csr1.data_ptr<int>();
        if (csr2.numel() > 0) p_pix_idx = csr2.data_ptr<int>();
    }

    if (shader_weight_mode == SHADER_WEIGHT_US_FOCUS) {
        us_focus_backward_cu::RenderOursSliceBackwardUsFocus_Impl(
            n_gauss,
            H,
            W,
            (W + tile_size_px - 1) / tile_size_px,
            (H + tile_size_px - 1) / tile_size_px,
            tile_size_px,
            k_sigma,
            gauss_rects.data_ptr<int>(),
            p_tile_rs,
            p_tile_re,
            p_sorted,
            gauss_means.data_ptr<float>(),
            gauss_scales.data_ptr<float>(),
            gauss_quats.data_ptr<float>(),
            gauss_opacities.data_ptr<float>(),
            gauss_intensities.data_ptr<float>(),
            pose_probe_to_world.data_ptr<float>(),
            grid_pack.data_ptr<float>(),
            us_params.data_ptr<float>(),
            gc.data_ptr<float>(),
            (workspace.numel() > 0) ? workspace.data_ptr<float>() : nullptr,
            static_cast<int>(workspace.numel()),
            compute_pose_grad,
            grad_gauss_means.data_ptr<float>(),
            grad_gauss_scales.data_ptr<float>(),
            grad_gauss_quats.data_ptr<float>(),
            grad_gauss_opacities.data_ptr<float>(),
            grad_gauss_intensities.data_ptr<float>(),
            grad_pose.data_ptr<float>(),
            at::cuda::getCurrentCUDAStream());
    } else if (shader_weight_mode == SHADER_WEIGHT_PLANE_COND2D) {
        RenderOursSliceBackwardPlaneCond2D_Impl(
            n_pixels,
            n_gauss,
            H,
            W,
            tile_size_px,
            gauss_rect_is_pixels,
            k_sigma,
            compositing_mode,
            fixed_bg_color,
            mapper_kind,
            p_tile_rs,
            p_tile_re,
            p_sorted,
            p_pix_off,
            p_pix_cnt,
            p_pix_idx,
            (n_gauss > 0) ? workspace.data_ptr<float>() : nullptr,
            (n_gauss > 0) ? static_cast<int>(workspace.numel()) : 0,
            gauss_rects.data_ptr<int>(),
            gauss_means.data_ptr<float>(),
            gauss_scales.data_ptr<float>(),
            gauss_quats.data_ptr<float>(),
            gauss_opacities.data_ptr<float>(),
            gauss_intensities.data_ptr<float>(),
            pose_probe_to_world.data_ptr<float>(),
            grid_pack.data_ptr<float>(),
            gc.data_ptr<float>(),
            grad_gauss_means.data_ptr<float>(),
            grad_gauss_scales.data_ptr<float>(),
            grad_gauss_quats.data_ptr<float>(),
            grad_gauss_opacities.data_ptr<float>(),
            grad_gauss_intensities.data_ptr<float>(),
            grad_pose.data_ptr<float>(),
            at::cuda::getCurrentCUDAStream());
    } else {
        ours_backward_cu::RenderOursSliceBackwardCUDA_Impl(
            n_pixels,
            n_gauss,
            H,
            W,
            tile_size_px,
            gauss_rect_is_pixels,
            lateral_spacing_mm_per_pix,
            axial_spacing_mm_per_pix,
            axial_near_mm,
            k_sigma,
            grid_pack.data_ptr<float>(),
            compositing_mode,
            fixed_bg_color,
            mapper_kind,
            p_tile_rs,
            p_tile_re,
            p_sorted,
            p_pix_off,
            p_pix_cnt,
            p_pix_idx,
            (compositing_mode != 0 && n_gauss > 0) ? workspace.data_ptr<float>() : nullptr,
            (compositing_mode != 0 && n_gauss > 0) ? static_cast<int>(workspace.numel()) : 0,
            gauss_rects.data_ptr<int>(),
            gauss_means.data_ptr<float>(),
            gauss_scales.data_ptr<float>(),
            gauss_quats.data_ptr<float>(),
            gauss_opacities.data_ptr<float>(),
            gauss_intensities.data_ptr<float>(),
            pose_probe_to_world.data_ptr<float>(),
            gc.data_ptr<float>(),
            grad_gauss_means.data_ptr<float>(),
            grad_gauss_scales.data_ptr<float>(),
            grad_gauss_quats.data_ptr<float>(),
            grad_gauss_opacities.data_ptr<float>(),
            grad_gauss_intensities.data_ptr<float>(),
            grad_pose.data_ptr<float>(),
            at::cuda::getCurrentCUDAStream());
    }

    return std::make_tuple(
        grad_gauss_means,
        grad_gauss_scales,
        grad_gauss_quats,
        grad_gauss_opacities,
        grad_gauss_intensities,
        grad_pose);
}
