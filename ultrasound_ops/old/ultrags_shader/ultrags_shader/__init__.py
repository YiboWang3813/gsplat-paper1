import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Union, Dict, Optional, Any

from . import _C

def ensure_tensor(t: torch.Tensor, dtype=torch.float32):
    if t is None: return None
    if not t.is_cuda: t = t.cuda()
    if t.dtype != dtype: t = t.to(dtype=dtype)
    return t.contiguous()

def pack_vec3(t: torch.Tensor):
    if t.shape[-1] == 3:
        t = F.pad(t, (0, 1), mode='constant', value=0.0)
    return t.contiguous()


# -------------------------------------------------------------------------
# 1. Configuration (Dataclass)
# -------------------------------------------------------------------------
@dataclass
class ShaderConfig:
    """着色器配置参数"""
    H: int
    W: int
    fixed_bg_color: float
    compositing_mode: str = "alpha"


def _mode_to_id(mode: str) -> int:
    m = str(mode).lower()
    if m == "alpha":
        return 0
    if m == "additive_sum":
        return 1
    if m == "additive_norm":
        return 2
    raise ValueError(f"Unsupported compositing_mode: {mode}")

# -------------------------------------------------------------------------
# 2. Autograd Function (Core Logic)
# -------------------------------------------------------------------------
class _ShadeFunction(torch.autograd.Function):
    
    @staticmethod
    def _pack_gauss_attributes(
        gauss_means: torch.Tensor, 
        gauss_scales: torch.Tensor, 
        gauss_intensities: torch.Tensor, 
        gauss_opacities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 维度对齐 (N,) -> (N, 1)
        if gauss_intensities.dim() == 1: gauss_intensities = gauss_intensities.unsqueeze(1)
        if gauss_opacities.dim() == 1: gauss_opacities = gauss_opacities.unsqueeze(1)
        
        # 拼接并确保内存连续
        gauss_means_intensities = torch.cat([gauss_means, gauss_intensities], dim=1).contiguous()
        gauss_scales_opacities = torch.cat([gauss_scales, gauss_opacities], dim=1).contiguous()
        
        return gauss_means_intensities, gauss_scales_opacities

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
        # 3. 只读 - 像素位置和探头
        pixel_positions: torch.Tensor, # (P, 3)
        probe_origin: torch.Tensor,    # (3,)
        probe_vec_elevational: torch.Tensor, # (3,)
        # 4. 只读 - 像素->高斯映射
        pixel_gauss_offsets: torch.Tensor, # (P,)
        pixel_gauss_counts: torch.Tensor, # (P,)
        pixel_gauss_indices: torch.Tensor, # (Total_Interactions,)
        # 5. 选项 / 标志位
        return_intermediates: bool
    ):
        num_gauss = gauss_means.shape[0]

        # 1. 数据准备 (Packing)
        pixel_positions_packed = pack_vec3(pixel_positions) # (P, 4)
        gauss_means_intensities, gauss_scales_opacities = _ShadeFunction._pack_gauss_attributes(
            gauss_means, gauss_scales, gauss_intensities, gauss_opacities
        ) # (G, 4) Packed
        gauss_quats = gauss_quats.contiguous() # (G, 4)
        # 确保depth需要的张量连续
        probe_origin = probe_origin.contiguous()
        probe_vec_elevational = probe_vec_elevational.contiguous()

        # 2. 分配可能会填充的Gauss中间变量
        if return_intermediates:
            # [Contribution, Occlusion, Participation]
            gauss_intermediates = torch.empty((num_gauss, 3), dtype=torch.float32, device=gauss_means.device)
        else: 
            gauss_intermediates = torch.empty(0, device=gauss_means.device)

        # 3. 执行 C++ Forward
        (
            pixel_colors, pixel_depths,
            pixel_accum, pixel_T, pixel_contrib,
            gauss_Rs, gauss_sigma_invs
        ) = _C.shade_forward(
            # 1. 标量 / 配置
            config.W, config.fixed_bg_color, _mode_to_id(config.compositing_mode),
            # 2. 只读 - 高斯几何 (packed)
            gauss_means_intensities, gauss_scales_opacities, gauss_quats,
            # 3. 只读 - 像素位置与探头
            pixel_positions_packed, probe_origin, probe_vec_elevational,
            # 4. 只读 - 像素->高斯映射
            pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
            # 5. 可写 - 中间变量
            gauss_intermediates # 这里需要把CUDA改成检查是否传入Shape正确的非空tensor
        ) 

        # 4. 保存上下文供反向传播使用
        ctx.config = config
        ctx.save_for_backward(
            # 1. 高斯属性 (packed)
            gauss_means_intensities, gauss_scales_opacities, gauss_quats,
            # 2. 像素位置与探头
            pixel_positions_packed, probe_origin, probe_vec_elevational,
            # 3. 像素->高斯映射
            pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
            # 4. 只读 - 反向必须的前向中间变量
            gauss_Rs, gauss_sigma_invs, pixel_accum, pixel_T
        )
        
        # 5. 根据return_intermediates决定是否返回打包的Pixel中间变量
        if return_intermediates:
            pixel_intermediates = torch.stack([pixel_accum, pixel_T, pixel_contrib], dim=1) # (P, 3)
        else:
            pixel_intermediates = torch.empty(0, device=pixel_positions.device)
        
        return pixel_colors, pixel_depths, pixel_intermediates, gauss_intermediates

    @staticmethod
    def backward(ctx, dL_dPixelColors, dL_dPixelDepths, dL_dPixelIntermediates, dL_dGaussIntermediates): 

        # 0. 屏蔽无效梯度
        dL_dPixelIntermediates = None
        dL_dGaussIntermediates = None
        
        # 0. 确保梯度连续
        dL_dPixelColors = dL_dPixelColors.contiguous()
        dL_dPixelDepths = dL_dPixelDepths.contiguous()
        
        # 1. 恢复上下文
        config = ctx.config
        (
            # 1. 高斯属性 (packed)
            gauss_means_intensities, gauss_scales_opacities, gauss_quats,
            # 2. 像素位置与探头
            pixel_positions_packed, probe_origin, probe_vec_elevational,
            # 3. 像素->高斯映射
            pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
            # 4. 只读 - 反向必须的前向中间变量
            gauss_Rs, gauss_sigma_invs, pixel_accum, pixel_T
        ) = ctx.saved_tensors
        
        # 2. 执行 C++ Backward
        (
            dL_dGaussMeansIntensities,   # (G, 4) Packed Gradient
            dL_dGaussQuats,   # (G, 4)
            dL_dGaussScalesOpacities, # (G, 4) Packed Gradient
        ) = _C.shade_backward(
            # 1. 标量
            config.W, config.fixed_bg_color, _mode_to_id(config.compositing_mode),
            # 2. 只读 - 前向的输入变量
            # 2.1 高斯属性 (packed)
            gauss_means_intensities, gauss_scales_opacities, gauss_quats,
            # 2.2 像素位置与探头
            pixel_positions_packed, probe_origin, probe_vec_elevational,
            # 2.3 像素->高斯映射
            pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
            # 3. 只读 - 反向必须的前向中间变量
            gauss_Rs, gauss_sigma_invs, pixel_accum, pixel_T,
            # 4. 只读 - 上游梯度
            dL_dPixelColors, dL_dPixelDepths
        )

        # 3. Unpacking Gradients (关键: 拆解 float4 梯度)
        dL_dGaussMeans = dL_dGaussMeansIntensities[:, :3]
        dL_dGaussIntensities = dL_dGaussMeansIntensities[:, 3:4].squeeze(1) # Restore to (G,)
        
        dL_dGaussScales = dL_dGaussScalesOpacities[:, :3]
        dL_dGaussOpacities = dL_dGaussScalesOpacities[:, 3:4].squeeze(1) # Restore to (G,)

        # 4. 返回梯度 (对应 forward 输入参数位置)
        return (
            None, # config
            dL_dGaussMeans,
            dL_dGaussScales,
            dL_dGaussQuats,
            dL_dGaussOpacities,
            dL_dGaussIntensities,
            None, # pixel_positions
            None, # probe_origin
            None, # probe_vec_elevational
            None, # pixel_gauss_offsets
            None, # pixel_gauss_counts
            None, # pixel_gauss_indices
            None, # return_intermediates
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
        # 2. 只读 - 像素位置和探头
        pixel_positions: torch.Tensor, 
        probe_origin: torch.Tensor,
        probe_vec_elevational: torch.Tensor,
        # 3. 只读 - 像素->高斯映射
        pixel_gauss_offsets: torch.Tensor,
        pixel_gauss_counts: torch.Tensor,
        pixel_gauss_indices: torch.Tensor,
        # 4. 选项 / 标志位
        return_intermediates: bool = False 
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        
        # 1. 类型安全检查
        gauss_means = ensure_tensor(gauss_means)
        gauss_scales = ensure_tensor(gauss_scales)
        gauss_quats = ensure_tensor(gauss_quats)
        gauss_opacities = ensure_tensor(gauss_opacities)
        gauss_intensities = ensure_tensor(gauss_intensities)

        pixel_positions = ensure_tensor(pixel_positions)
        probe_origin = ensure_tensor(probe_origin)
        probe_vec_elevational = ensure_tensor(probe_vec_elevational)

        pixel_gauss_offsets = ensure_tensor(pixel_gauss_offsets, dtype=torch.int32)
        pixel_gauss_counts = ensure_tensor(pixel_gauss_counts, dtype=torch.int32)
        pixel_gauss_indices = ensure_tensor(pixel_gauss_indices, dtype=torch.int32)

        # 2. 调用核心算子
        (pixel_colors, pixel_depths, pixel_intermediates, gauss_intermediates) = _ShadeFunction.apply(
            self.config,
            gauss_means, gauss_scales, gauss_quats, gauss_opacities, gauss_intensities,
            pixel_positions, probe_origin, probe_vec_elevational,
            pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
            return_intermediates 
        )

        # 3. reshape 
        color_image = pixel_colors.view(self.config.H, self.config.W)
        depth_image = pixel_depths.view(self.config.H, self.config.W)

        # 4. 解包中间变量
        if return_intermediates:
            intermediates = {
                "pixel_accum": pixel_intermediates[:, 0], # (P,)
                "pixel_T": pixel_intermediates[:, 1], # (P,)
                "pixel_contrib": pixel_intermediates[:, 2], # (P,)
                "gauss_contribution": gauss_intermediates[:, 0], # (N,)
                "gauss_occlusion": gauss_intermediates[:, 1], # (N,)
                "gauss_participation": gauss_intermediates[:, 2], # (N,)
            }
        else:
            intermediates = {}
        
        return color_image, depth_image, intermediates