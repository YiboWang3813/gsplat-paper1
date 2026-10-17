import torch
import torch.nn as nn
from . import _C
from dataclasses import dataclass

def ensure_tensor(t: torch.Tensor, dtype=torch.float32):
    if t is None: return None
    if not t.is_cuda: t = t.cuda()
    if t.dtype != dtype: t = t.to(dtype=dtype)
    return t.contiguous()

# -------------------------------------------------------------------------
# 1. Configuration (Dataclass)
# -------------------------------------------------------------------------
@dataclass
class ShaderConfig:
    """着色器配置参数"""
    H: int
    W: int

# -------------------------------------------------------------------------
# 2. Autograd Function (Core Logic)
# -------------------------------------------------------------------------
class _ShadeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        # 1. 标量 / 配置
        config: ShaderConfig,
        # 2. 只读 - 高斯属性
        gauss_means: torch.Tensor, # (G, 3)
        gauss_scales: torch.Tensor, # (G, 3)
        gauss_quats: torch.Tensor, # (G, 4)
        gauss_opacities: torch.Tensor, # (G,)
        gauss_intensities: torch.Tensor, # (G,)
        # 3. 只读 - 像素位置
        pixel_positions, 
        # 4. 只读 - 像素->高斯映射
        pixel_gauss_offsets: torch.Tensor, # (P,)
        pixel_gauss_counts: torch.Tensor, # (P,)
        pixel_gauss_indices: torch.Tensor, # (Total_Interactions,)
    ):
        # 1. 调用 C++ 核心渲染
        (pixel_colors, gauss_Rs, gauss_sigma_invs) = _C.forward(
            # 1. 只读 - 高斯属性
            gauss_means, gauss_scales, gauss_quats, gauss_opacities, gauss_intensities,
            # 2. 只读 - 像素位置
            pixel_positions,
            # 3. 只读 - 像素->高斯映射
            pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
        )

        # 2. 保存上下文供反向传播使用
        ctx.config = config
        ctx.save_for_backward(
            # 1. 只读 -高斯属性
            gauss_means, gauss_scales, gauss_quats, gauss_opacities, gauss_intensities,
            # 2. 只读 - 像素位置
            pixel_positions,
            # 3. 只读 - 像素->高斯映射
            pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
            # 4. 只读 - 反向必须的前向中间变量
            gauss_Rs, gauss_sigma_invs
        )

        return pixel_colors

    @staticmethod
    def backward(ctx, dL_dPixelColors):
        # 1. 恢复上下文
        config = ctx.config
        (
            # 1. 只读 - 高斯属性
            gauss_means, gauss_scales, gauss_quats, gauss_opacities, gauss_intensities,
            # 2. 只读 - 像素位置
            pixel_positions,
            # 3. 只读 - 像素->高斯映射
            pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
            # 4. 只读 - 反向必须的前向中间变量
            gauss_Rs, gauss_sigma_invs
        ) = ctx.saved_tensors

        # 保证梯度连续
        dL_dPixelColors = dL_dPixelColors.contiguous()

        # 2. 执行 C++ Backward
        (
            dL_dGaussMeans,
            dL_dGaussScales,
            dL_dGaussQuats,
            dL_dGaussOpacities,
            dL_dGaussIntensities,
        ) = _C.backward(
            # 1. 只读 - 高斯属性
            gauss_sigma_invs, gauss_opacities, gauss_intensities,
            # 2. 只读 - 像素->高斯映射
            pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
            # 3. 只读 - 上游梯度
            dL_dPixelColors,
        )

        return (
            None, # config
            dL_dGaussMeans,
            dL_dGaussScales,
            dL_dGaussQuats,
            dL_dGaussOpacities,
            dL_dGaussIntensities,
            None, # pixel_positions
            None, # pixel_gauss_offsets
            None, # pixel_gauss_counts
            None, # pixel_gauss_indices
        )

# -------------------------------------------------------------------------
# 3. PyTorch Module (High-Level API)
# -------------------------------------------------------------------------
class Shader(nn.Module):
    def __init__(self, config: ShaderConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        # 1. 只读 - 高斯属性
        gauss_means: torch.Tensor, 
        gauss_scales: torch.Tensor, 
        gauss_quats: torch.Tensor,
        gauss_opacities: torch.Tensor,
        gauss_intensities: torch.Tensor, 
        # 2. 只读 - 像素位置
        pixel_positions: torch.Tensor, 
        # 3. 只读 - 像素->高斯映射
        pixel_gauss_offsets: torch.Tensor,
        pixel_gauss_counts: torch.Tensor,
        pixel_gauss_indices: torch.Tensor,
    ) -> torch.Tensor:

        # 1. 类型安全检查
        gauss_means = ensure_tensor(gauss_means)
        gauss_scales = ensure_tensor(gauss_scales)
        gauss_quats = ensure_tensor(gauss_quats)
        gauss_opacities = ensure_tensor(gauss_opacities)
        gauss_intensities = ensure_tensor(gauss_intensities)

        pixel_positions = ensure_tensor(pixel_positions)

        pixel_gauss_offsets = ensure_tensor(pixel_gauss_offsets, dtype=torch.int32)
        pixel_gauss_counts = ensure_tensor(pixel_gauss_counts, dtype=torch.int32)
        pixel_gauss_indices = ensure_tensor(pixel_gauss_indices, dtype=torch.int32)

        # 2. 调用核心算子
        pixel_colors = _ShadeFunction.apply(
            self.config,
            gauss_means, gauss_scales, gauss_quats, gauss_opacities, gauss_intensities,
            pixel_positions,
            pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
        )

        # 3. reshape 
        color_image = pixel_colors.view(self.config.H, self.config.W)

        return color_image
