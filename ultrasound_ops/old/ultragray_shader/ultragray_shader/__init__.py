import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Union, Dict, Optional, Any

# 假设编译后的扩展名为 _C
from . import _C

def ensure_tensor(t: torch.Tensor, dtype=torch.float32):
    if t is None: return None
    if not t.is_cuda: t = t.cuda()
    if t.dtype != dtype: t = t.to(dtype=dtype)
    return t.contiguous()

def pack_vec3(t: torch.Tensor):
    """
    将 (..., 3) 的向量填充为 (..., 4) 以满足 CUDA float4 对齐要求。
    用于 means 和 scales。
    """
    if t.shape[-1] == 3:
        t = F.pad(t, (0, 1), mode='constant', value=0.0)
    return t.contiguous()

# -------------------------------------------------------------------------
# 1. Configuration (Dataclass)
# -------------------------------------------------------------------------
@dataclass
class UltraShaderConfig:
    """
    超声着色器配置参数。
    虽然后端能从输入推导 W/H，但保留配置类有助于后续扩展（如添加 gain, dynamic range 等参数）。
    """
    # 目前主要用于 API 统一，具体尺寸由输入张量决定
    pass

# -------------------------------------------------------------------------
# 2. Autograd Function (Core Logic)
# -------------------------------------------------------------------------
class _UltraShadeFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
        # --- 几何输入 ---
        means: torch.Tensor,           # (N, 3)
        scales: torch.Tensor,          # (N, 3)
        quats: torch.Tensor,           # (N, 4)
        # --- 物理属性 ---
        attenuations: torch.Tensor,    # (N,)
        intensities: torch.Tensor,     # (N,)
        # --- 映射索引 (CSR) ---
        ray_counts: torch.Tensor,      # (W,)
        ray_offsets: torch.Tensor,     # (W,)
        ray_indices: torch.Tensor,     # (TotalHits,)
        # --- 场景数据 ---
        pixel_positions: torch.Tensor, # (W, H, 3)
        ray_origins: torch.Tensor,     # (W, 3)
        ray_dirs: torch.Tensor         # (W, 3)
    ):
        # 1. 数据准备 (Packing for float4*)
        # C++ wrapper 期望 means/scales 是 float4*，所以必须 padding
        means_packed = pack_vec3(means)   # (N, 4)
        scales_packed = pack_vec3(scales) # (N, 4)
        quats = quats.contiguous()        # (N, 4)

        # C++ wrapper 期望 pixel_pos/origins/dirs 是 float* (stride 3)，保持原样
        pixel_positions = pixel_positions.contiguous()
        ray_origins = ray_origins.contiguous()
        ray_dirs = ray_dirs.contiguous()

        # 2. 执行 C++ Forward
        final_image = _C.shade_forward(
            means_packed,
            scales_packed,
            quats,
            attenuations,
            intensities,
            ray_counts,
            ray_offsets,
            ray_indices,
            pixel_positions,
            ray_origins,
            ray_dirs
        )

        # 3. 保存上下文供反向传播使用
        # 注意：保存 packed 版本以供 backward 复用，避免重复 padding
        ctx.save_for_backward(
            means_packed, scales_packed, quats,
            attenuations, intensities,
            ray_counts, ray_offsets, ray_indices,
            pixel_positions, ray_origins, ray_dirs
        )
        
        return final_image

    @staticmethod
    def backward(ctx, grad_final_image): 
        # 1. 恢复上下文
        (
            means_packed, scales_packed, quats,
            attenuations, intensities,
            ray_counts, ray_offsets, ray_indices,
            pixel_positions, ray_origins, ray_dirs
        ) = ctx.saved_tensors
        
        grad_final_image = grad_final_image.contiguous()
        num_gauss = means_packed.shape[0]

        # 2. 申请梯度 Buffer
        # C++ backward 期望 float4* 用于 means/scales/quats
        dL_dMeans_packed = torch.zeros((num_gauss, 4), dtype=torch.float32, device=means_packed.device)
        dL_dScales_packed = torch.zeros((num_gauss, 4), dtype=torch.float32, device=means_packed.device)
        dL_dQuats = torch.zeros((num_gauss, 4), dtype=torch.float32, device=means_packed.device)
        
        # 标量梯度
        dL_dAttens = torch.zeros((num_gauss,), dtype=torch.float32, device=means_packed.device)
        dL_dIntensities = torch.zeros((num_gauss,), dtype=torch.float32, device=means_packed.device)

        # 3. 执行 C++ Backward
        _C.shade_backward(
            grad_final_image,
            means_packed,
            scales_packed,
            quats,
            attenuations,
            intensities,
            ray_counts,
            ray_offsets,
            ray_indices,
            pixel_positions,
            ray_origins,
            ray_dirs,
            # Outputs
            dL_dMeans_packed,
            dL_dScales_packed,
            dL_dQuats,
            dL_dAttens,
            dL_dIntensities
        )

        # 4. Unpacking Gradients (关键步骤)
        # 将 (N, 4) 的梯度切片回 (N, 3) 以匹配 Autograd 接口
        dL_dMeans = dL_dMeans_packed[:, :3]
        dL_dScales = dL_dScales_packed[:, :3]

        # 5. 返回梯度 (对应 forward 输入参数位置)
        # 输入参数顺序: means, scales, quats, attenuations, intensities, 
        #             ray_counts, ray_offsets, ray_indices, 
        #             pixel_positions, ray_origins, ray_dirs
        return (
            dL_dMeans, 
            dL_dScales,
            dL_dQuats, 
            dL_dAttens, 
            dL_dIntensities, 
            None, # ray_counts
            None, # ray_offsets
            None, # ray_indices
            None, # pixel_positions
            None, # ray_origins
            None  # ray_dirs
        )

# -------------------------------------------------------------------------
# 3. PyTorch Module (High-Level API)
# -------------------------------------------------------------------------
class UltraShader(nn.Module):
    def __init__(self, config: Optional[UltraShaderConfig] = None):
        super().__init__()
        self.config = config if config is not None else UltraShaderConfig()

    def forward(
        self,
        # Attributes
        means: torch.Tensor,           # (N, 3)
        scales: torch.Tensor,          # (N, 3)
        quats: torch.Tensor,           # (N, 4)
        attenuations: torch.Tensor,    # (N,)
        intensities: torch.Tensor,     # (N,)
        # CSR
        ray_counts: torch.Tensor,      # (W,)
        ray_offsets: torch.Tensor,     # (W,)
        ray_indices: torch.Tensor,     # (TotalHits,)
        # Geometry
        pixel_positions: torch.Tensor, # (W, H, 3)
        ray_origins: torch.Tensor,     # (W, 3)
        ray_dirs: torch.Tensor         # (W, 3)
    ) -> torch.Tensor:
        """
        前向 B-Mode 超声着色过程。
        
        Args:
            means: (N, 3) 高斯均值 (World Space)
            scales: (N, 3) 高斯缩放
            quats: (N, 4) 高斯旋转四元数
            attenuations: (N,) 高斯衰减系数 logits (将被 sigmoid 激活)
            intensities: (N,) 高斯回声强度
            
            ray_counts/offsets/indices: Mapper 输出的 CSR 索引结构
            
            pixel_positions: (W, H, 3) 每个像素的 3D 坐标
            ray_origins: (W, 3) 每条扫描线的起点
            ray_dirs: (W, 3) 每条扫描线的方向 (归一化)
            
        Returns:
            B-Mode Image Tensor (W, H)
        """
        
        # 1. 类型安全检查
        means = ensure_tensor(means)
        scales = ensure_tensor(scales)
        quats = ensure_tensor(quats)
        attenuations = ensure_tensor(attenuations)
        intensities = ensure_tensor(intensities)
        
        ray_counts = ensure_tensor(ray_counts, dtype=torch.int32)
        ray_offsets = ensure_tensor(ray_offsets, dtype=torch.int32)
        ray_indices = ensure_tensor(ray_indices, dtype=torch.int32)
        
        pixel_positions = ensure_tensor(pixel_positions)
        ray_origins = ensure_tensor(ray_origins)
        ray_dirs = ensure_tensor(ray_dirs)

        # 2. 调用核心算子
        final_image = _UltraShadeFunction.apply(
            means, scales, quats,
            attenuations, intensities,
            ray_counts, ray_offsets, ray_indices,
            pixel_positions, ray_origins, ray_dirs
        )
        
        return final_image