#pragma once
#include <torch/extension.h>

// =========================================================================
// Forward Interface (PyTorch Wrapper)
// 参数顺序与 __init__.py 中 _C.shade_forward 一致
// =========================================================================
std::tuple<
    torch::Tensor, // pixel_colors (P,)
    torch::Tensor, // pixel_depths (P,)
    torch::Tensor, // pixel_accum (P,)
    torch::Tensor, // pixel_T (P,)
    torch::Tensor, // pixel_contrib (P,)
    torch::Tensor, // gauss_Rs (G, 9)
    torch::Tensor  // gauss_sigma_invs (G, 9)
>
RasterizeGaussiansForwardCUDA(
    const int W,
    const float fixed_bg_color,
    const int compositing_mode,
    // 2. 只读 - 高斯几何 (packed)
    const torch::Tensor& gauss_means_intensities,   // (G, 4) .xyz=mean, .w=intensity
    const torch::Tensor& gauss_scales_opacities,    // (G, 4) .xyz=scale, .w=opacity
    const torch::Tensor& gauss_quats,               // (G, 4)
    // 3. 只读 - 像素位置与探头
    const torch::Tensor& pixel_positions_packed,   // (P, 4) .xyz=pos, .w=pad
    const torch::Tensor& probe_origin,             // (3,)
    const torch::Tensor& probe_vec_elevational,    // (3,)
    // 4. 只读 - 像素->高斯映射
    const torch::Tensor& pixel_gauss_offsets,       // (P,)
    const torch::Tensor& pixel_gauss_counts,       // (P,)
    const torch::Tensor& pixel_gauss_indices,      // (Total_Interactions,)
    // 5. 可写 - 中间变量 (可选，空或 shape 正确时写入)
    const torch::Tensor& gauss_intermediates       // (G, 3) [contribution, occlusion, participation]
);

// =========================================================================
// Backward Interface (PyTorch Wrapper)
// 参数顺序与 __init__.py 中 _C.shade_backward 一致；仅返回 3 个梯度张量
// =========================================================================
std::tuple<
    torch::Tensor, // dL_dGaussMeansIntensities (G, 4) Packed
    torch::Tensor, // dL_dGaussQuats (G, 4)
    torch::Tensor  // dL_dGaussScalesOpacities (G, 4) Packed
>
RasterizeGaussiansBackwardCUDA(
    const int W,
    const float fixed_bg_color,
    const int compositing_mode,
    // 2. 只读 - 前向的输入变量 (packed)
    const torch::Tensor& gauss_means_intensities,
    const torch::Tensor& gauss_scales_opacities,
    const torch::Tensor& gauss_quats,
    // 2.2 像素位置与探头
    const torch::Tensor& pixel_positions_packed,
    const torch::Tensor& probe_origin,
    const torch::Tensor& probe_vec_elevational,
    // 2.3 像素->高斯映射
    const torch::Tensor& pixel_gauss_offsets,
    const torch::Tensor& pixel_gauss_counts,
    const torch::Tensor& pixel_gauss_indices,
    // 3. 只读 - 反向必须的前向中间变量
    const torch::Tensor& gauss_Rs,
    const torch::Tensor& gauss_sigma_invs,
    const torch::Tensor& pixel_accum,
    const torch::Tensor& pixel_T,
    // 4. 只读 - 上游梯度
    const torch::Tensor& dL_dPixelColors,
    const torch::Tensor& dL_dPixelDepths
);
