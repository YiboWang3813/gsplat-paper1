import torch
import torch.nn.functional as F
import math
from typing import List, Tuple

# 假设你的编译模块叫 _C
from . import _C

def ensure_tensor(t: torch.Tensor, dtype=torch.float32):
    if t is None: return None
    if not t.is_cuda: t = t.cuda()
    if t.dtype != dtype: t = t.to(dtype=dtype)
    return t.contiguous()

def pack_vec3(t: torch.Tensor):
    # 确保是 (N, 4) 布局以符合 float4 内存对齐，或者 C++ 端只读前3个
    # 如果 C++ 是 float4*，这里必须 padding 到 4
    if t.shape[-1] == 3:
        t = F.pad(t, (0, 1), mode='constant', value=0.0)
    return t.contiguous()

class RayMapper: 
    def __init__(self):
        pass 

    @staticmethod
    def map(
        # --- Gaussian Parameters ---
        gauss_means: torch.Tensor,     # (G, 3) [World Space, mm]
        gauss_scales: torch.Tensor,    # (G, 3) [World Space, mm]

        # --- Probe Geometry ---
        probe_origin: torch.Tensor,    # (3,)
        vec_elevational: torch.Tensor, # (3,)
        vec_lateral: torch.Tensor,     # (3,)
        vec_axial: torch.Tensor,       # (3,)

        # --- Imaging Parameters ---
        image_width: int,
        width_mm: float,
        near_mm: float,
        
        # --- Physics / Simulation Parameters ---
        k_sigma: float = 3.0,          # 3 sigma 覆盖范围
        elev_beam_width_mm: float = 2.0, # 论文中的 delta，模拟波束厚度
        elev_cutoff_margin_mm: float = 0.0 # 额外的容差，通常设为0或很小的值
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 1. Tensor 检查与转换
        gauss_means = ensure_tensor(gauss_means)
        gauss_scales = ensure_tensor(gauss_scales)
        probe_origin = ensure_tensor(probe_origin)
        vec_elevational = ensure_tensor(vec_elevational)
        vec_lateral = ensure_tensor(vec_lateral)
        vec_axial = ensure_tensor(vec_axial)

        num_gauss = gauss_means.shape[0]

        # 2. 计算 Elevational 扰动 (Perturbations)
        # 论文  提到 "cosine-weighted distribution"。
        # 论文  提到最大幅度 delta = 2mm。
        # 算法：对 Uniform(0,1) 进行反正弦变换可得 Cosine 分布
        
        # u ~ Uniform[0, 1]
        u = torch.rand(num_gauss, device=gauss_means.device, dtype=torch.float32)
        
        # 变换到 [-1, 1] 且符合余弦分布 (中间密，两边疏)
        # asin 的域是 [-1, 1]，值域是 [-pi/2, pi/2]
        # 除以 (pi/2) 归一化到 [-1, 1]
        perturbations_norm = torch.asin(2.0 * u - 1.0) / (math.pi / 2.0)
        
        # 缩放到物理宽度 [-delta, delta]
        perturbations = perturbations_norm * elev_beam_width_mm
        
        perturbations = perturbations.contiguous()

        # 3. Pack Data (padding to float4 for CUDA float4 alignment)
        gauss_means_packed = pack_vec3(gauss_means)
        gauss_scales_packed = pack_vec3(gauss_scales)

        # 4. 调用 C++ 扩展
        # 注意：这里的参数顺序必须和 C++ wrapper 中的 launch_ray_mapper 严格一致
        ray_counts, ray_offsets, ray_indices = _C.launch_ray_mapper(
            gauss_means_packed,
            gauss_scales_packed,
            perturbations,      # 新增的预计算扰动

            probe_origin,
            vec_lateral,        # 注意：确认 C++ 里的顺序是 Lat/Ele/Axial 还是什么
            vec_elevational,
            vec_axial,

            image_width,
            width_mm,
            near_mm,
            elev_cutoff_margin_mm, # 对应 C++ 的 elev_cutoff_mm
            k_sigma
        )

        return ray_counts, ray_offsets, ray_indices