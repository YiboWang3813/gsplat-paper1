# ours_renderer

单张 B 模式切片上的 3D 高斯 **投影渲染** CUDA 扩展，管线对齐 `ultrags_renderer` 的 **Mapper（建 CSR）+ Shader（消费 CSR）**：

- **Mapper**：`TileGaussMapper`（tile CSR：`sorted_gauss_ids` + `tile_range_start/end` + `gauss_tile_rects`）或 `PixelGaussMapper`（像素 CSR：`counts` / `offsets` / `indices` + `gauss_pixel_rect`）。
- **Shader**：`probe3d` 为探头系 3D 马氏；`plane_cond2d` 为严格切片截面
  \(\alpha=\mathrm{clamp}(o\cdot\exp(-\tfrac12(q_{\mathrm{cond}}+\mu_z^2/S_{zz})),0,1)\)（Schur 条件 \(q_{\mathrm{cond}}\)）。
  **加性**或 **alpha** 合成；alpha 反传为 per-pixel 重放 + workspace + finalize。`plane_cond2d` 的 tile/pixel mapper 以 Schur 中心 \(\mu_2\) 建覆盖矩形。

**世界坐标一律为毫米（mm）**：`gauss_means` / `gauss_scales` 与 `SliceMeta.pose` 的平移同一世界系；`pose_probe_to_world` 为 probe（齐次 mm）→ world（mm）。

**高斯四元数 (w, x, y, z)**：与 `scene` / `ultrags_renderer` 一致（例如 `quats_init: [1, 0, 0, 0]` 为单位旋转）。勿按 (x, y, z, w) 传入。

## 能力矩阵

| | additive | alpha |
|---|---|---|
| **tile** | `OursTileGaussMapperCUDA` + tile CSR shader | 同上；反传 `kernel_ours_alpha_back_tile_csr` |
| **pixel** | `OursPixelGaussMapperCUDA` + pixel CSR shader | 同上；反传 `kernel_ours_alpha_back_pixel_csr` |

Mapper 输出离散 CSR，**不对 mapper 求导**；可训练参数的高斯/位姿梯度由 shader 反传给出。

## Elevational 硬剔除（Mapper）

通过 elev 走廊半宽 `elev_corridor_half_mm`（通常取 `num_slices_elevational * delta_elevational_mm / 2`，由训练侧从数据集 meta 拼进 `OursRenderConfig`）加上可选 `slab_half_thickness_mm`，再加上椭球沿 elev 方向由 `k_sigma` 放大的半径，对 `|μ_z|` 做 AABB 剔除（见 `src/mapper/`）。

## 排序（与 alpha 一致）

`mapper_sort_by_v=True`（默认）时 radix 次级键按成像行坐标 **v 升序**（浅层先合成）。`compositing_mode="alpha"` 时必须开启（`OursRenderConfig.__post_init__` 会校验）。

## Alpha 反传说明

- 前向保存 CSR（tile 或 pixel），反传按像素逆序重放 alpha 链，再写回 `means/scales/quats/opacities/intensities/pose`。
- 单像素参与合成的高斯层数上限为 **256**（`OURS_MAX_ALPHA_LAYERS`）；更深堆叠时梯度可能不完整。

## 依赖与安装

- PyTorch + CUDA；CUB（`#include <cub/cub.cuh>`），`setup.py` 使用 `-DTHRUST_IGNORE_CUB_VERSION_CHECK`。

```bash
cd ops/ours_renderer
./rebuild.sh
```

`rebuild.sh` 校验 `_C` 导出 `ProjectedSliceForwardCUDA` / `ProjectedSliceBackwardCUDA`（名称保留以兼容现有脚本）。

## Python

- `OursRenderConfig`：`ours_renderer/types.py`（`lateral_spacing_mm_per_pix`、`axial_near_mm`、`elev_corridor_half_mm`、`k_sigma`、`mapper_kind`、`num_pixels_*`、`compositing_mode` 等）。
- `OursRenderer`：`nn.Module`，`forward(..., pose_probe_to_world)` → `(H, W)`；`autograd` 对 additive / alpha 均调用 `ProjectedSliceBackwardCUDA`。
- 训练：`methods/ours/render.build_ours_render_config` 从 `DatasetMeta` + `model.slice_render` 填配置；高斯初始化与 UltraGS 相同，走 `Scene.build_init_gaussians()` → `OursGaussContainer.init_from_scene`（世界 mm 的 `pixel_pos` + `scene.sampler`）。

## 目录（CUDA）

| 路径 | 说明 |
|------|------|
| `include/mapper/`、`src/mapper/` | Tile / Pixel CSR 构建（`core_tile.cu` / `core_pixel.cu`） |
| `include/shader/`、`src/shader/` | `forward` / `backward` / `shared` / `wrapper_shader` |
| `src/bindings.cpp` | pybind 导出 |

Host 侧对 Tensor 请使用 `.to(torch::kFloat32)`，避免在 `.cu` 中直接 `.float()` 触发 nvcc 解析问题。

## 测试

```bash
python tests/ops/ours_renderer/validate_same_effect_cpu_gpu.py
```

在 **单 tile 覆盖全图**、**pixel bucket 足够大以覆盖全图**（测试中 `num_pixels_lateral=2W`、`num_pixels_axial=2H`）且放宽 elev 剔除时，CPU 稠密参考与 GPU 在四种组合上对齐：

- tile + additive / tile + alpha  
- pixel + additive / pixel + alpha  

每种组合另做 `y.sum().backward()` 有限梯度冒烟。

历史一键 smoke 脚本已清理；需要精确复查旧 launcher 时看 git history。当前验证优先直接运行上面的测试脚本或新增 `scripts/methods/ours/<idea>/slurm/` 入口。

## 旧 checkpoint（归一化世界）

若 checkpoint 中存的是归一化坐标：`means_mm = means_norm * scene_scale + scene_center`；`pose_probe_to_world` 可由 `methods/ours/geometry.pose_probe_world_to_normalized` 的逆变换从旧 pose 恢复（该文件保留为迁移辅助）。

## 参考

`docs/literature/ours_renderer/GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction.md`
