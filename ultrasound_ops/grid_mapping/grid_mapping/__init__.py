import torch
from . import _C

class MappingResult:
    """封装 Grid 映射结果，作为 Scene 与 Trainer 之间的物理桥梁"""
    def __init__(
        self, 
        tile_counts, tile_offsets, tile_indices, 
        pixel_tile_ids, pixel_counts, pixel_offsets, pixel_indices
    ):
        # Tile 级索引数据
        self.tile_counts = tile_counts
        self.tile_offsets = tile_offsets
        self.tile_indices = tile_indices
        
        # Pixel 级映射数据
        self.pixel_tile_ids = pixel_tile_ids
        self.pixel_counts = pixel_counts
        self.pixel_offsets = pixel_offsets
        self.pixel_indices = pixel_indices

def build_pixel_gauss_mapping(
    grid_min: torch.Tensor,
    tile_size: torch.Tensor,
    grid_size: torch.Tensor,
    pixel_positions: torch.Tensor,
    means: torch.Tensor,
    sigma_invs: torch.Tensor
) -> MappingResult:
    """
    Grid-based Mapping 核心算子：
    利用空间哈希网格快速筛选 3x3x3 邻域内满足马氏距离阈值的高斯候选集。
    """
    # 确保所有输入均为 CUDA 连续内存
    # 注意：grid_size 通常需要 int32
    device = pixel_positions.device
    
    tensors = [grid_min, tile_size, grid_size, pixel_positions, means, sigma_invs]
    processed_tensors = []
    for t in tensors:
        if not t.is_cuda:
            t = t.to(device)
        processed_tensors.append(t.contiguous())

    # 调用编译好的 C++ 扩展
    res_list = _C.build_mapping(*processed_tensors)

    return MappingResult(*res_list)