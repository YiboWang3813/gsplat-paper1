# ultrags_renderer：代码即文档

本文档严格对应当前实现，覆盖 `ultrags_renderer` 的 Python API、C++/CUDA 绑定、Mapper 与 Shader 的数学定义和张量关系。  
目标是做到：**按文档可直接定位到代码路径并复现实例行为**。

---

## 1. 包结构与调用链（严格对齐代码）

### 1.1 目录分层

```text
ops/ultrags_renderer/
  ultrags_renderer/
    __init__.py
    types.py
    api.py
    autograd.py
  src/
    bindings.cpp
    mapper/
      wrapper_mapper.cu
      core.cu
    shader/
      wrapper_shader.cu
      forward.cu
      backward.cu
  include/
    mapper/
      wrapper_mapper.cuh
      core.cuh
    shader/
      wrapper_shader.cuh
      shared.cuh
      forward.cuh
      backward.cuh
```

### 1.2 端到端调用顺序

```text
单帧路径:
UltragsRenderer.forward
  -> _RenderFunction.forward
    -> _C.MapperForwardCUDA
    -> _C.ShaderForwardCUDA
  -> (autograd backward)
    -> _C.ShaderBackwardCUDA

Batch 路径:
UltragsBatchedRenderer.forward
  -> _BatchedRenderFunction.forward
    -> _C.MapperForwardCUDABatched
    -> _C.ShaderForwardCUDABatched
  -> (autograd backward)
    -> _C.ShaderBackwardCUDABatched
```

对应绑定导出（`src/bindings.cpp`）：

```cpp
m.def("MapperForwardCUDA", &MapperForwardCUDA, "Mapper forward (CUDA)");
m.def("MapperForwardCUDABatched", &MapperForwardCUDABatched, "Mapper forward batched (CUDA)");
m.def("ShaderForwardCUDA", &ShaderForwardCUDA, "Shader forward (CUDA)");
m.def("ShaderForwardCUDABatched", &ShaderForwardCUDABatched, "Shader forward batched (CUDA)");
m.def("ShaderBackwardCUDA", &ShaderBackwardCUDA, "Shader backward (CUDA)");
m.def("ShaderBackwardCUDABatched", &ShaderBackwardCUDABatched, "Shader backward batched (CUDA)");
```

---

## 2. 统一输入输出契约（Python 层）

### 2.1 配置与几何（`types.py`）

- `UltragsRenderConfig`
  - `H, W`
  - `lateral_spacing_mm_per_pix, axial_spacing_mm_per_pix`
  - `slab_half_thickness_mm, axial_near_mm, k_sigma`
  - `mapper_sort_by_depth`
  - `fixed_bg_color, compositing_mode`
- `SliceGeometry`
  - `origin_world`
  - `axis_lateral_world`
  - `axis_axial_world`
  - `axis_elevational_world`

配置层基础约束（`__post_init__`）：

```python
H > 0, W > 0
lateral_spacing_mm_per_pix > 0
axial_spacing_mm_per_pix > 0
slab_half_thickness_mm >= 0
k_sigma > 0
```

### 2.2 `forward` 输入关系（`autograd.py`）

输入张量约束（`_validate_input_shapes`）：

```python
gauss_means: (G, 3)
gauss_scales: (G, 3)
gauss_quats: (G, 4)
gauss_opacities: (G,)
gauss_intensities: (G,)
pixel_positions_world: (P, 3), P == H * W
probe_origin_world: (3,)
axis_lateral_world/axis_axial_world/axis_elevational_world: (3,)
```

以及轴向量有限值与非零范数检查：

```python
torch.isfinite(axis).all()
torch.linalg.norm(axis) > 1e-8
```

### 2.3 打包规则（vec3 -> vec4，属性拼接）

```python
pixel_positions_packed = pack_vec3(pixel_positions_world)  # (P,4)
gauss_means_packed = pack_vec3(gauss_means)                # (G,4)
gauss_scales_packed = pack_vec3(gauss_scales)              # (G,4)
gauss_means_intensities = cat([gauss_means, gauss_intensities], dim=1)  # (G,4)
gauss_scales_opacities = cat([gauss_scales, gauss_opacities], dim=1)     # (G,4)
```

---

## 3. MapperForwardCUDA（候选稀疏映射）

一句话：输入像素与高斯几何，输出 CSR 三件套：

```text
pixel_gauss_counts:  (P,) int32
pixel_gauss_offsets: (P,) int32
pixel_gauss_indices: (nnz,) int32
```

记号：

- 高斯数 \(G\)
- 像素数 \(P=H\times W\)
- 像素 \(p\) 的列表起点 \(b_p\)、长度 \(n_p\)、第 \(k\) 个高斯 \(g_{p,k}\)

\[
b_p=\texttt{pixel\_gauss\_offsets}[p],\quad
n_p=\texttt{pixel\_gauss\_counts}[p],\quad
g_{p,k}=\texttt{pixel\_gauss\_indices}[b_p+k]
\]

### 3.1 关键几何约定

探头坐标轴顺序固定：

\[
x=\text{lateral},\quad y=\text{axial},\quad z=\text{elevational}
\]

`projected_radii` 分量顺序固定：

```text
.x = lateral, .y = axial, .z = elevational
```

### 3.2 CUDA 流程（`src/mapper/wrapper_mapper.cu` + `src/mapper/core.cu`）

```text
Kernel1: kernel_precompute_gauss_projected_radii
Kernel2: kernel_project_gauss_and_compute_tile_rects
Scan:    cub::DeviceScan::ExclusiveSum
Kernel3: kernel_generate_tile_depth_keys
Sort:    cub::DeviceRadixSort::SortPairs
Kernel4: kernel_identify_tile_ranges
Kernel5: kernel_count_pixel_candidates
Scan:    counts -> offsets
Kernel6: kernel_fill_pixel_candidate_indices
```

#### (a) 投影半径（Manhattan bound）

\[
r(\mathbf{s},R,\mathbf{d})=\sum_{i=0}^{2}\left|s_i\cdot(\mathbf{a}_i^\top\mathbf{d})\right|
\]

\[
r_{\text{lat/ax/ele}}^{\text{proj}} = k_\sigma \cdot r(\mathbf{s},R,\mathbf{axis}_{\text{lat/ax/ele}})
\]

#### (b) slab 裁剪 + 像素投影

设 \(\mathbf{v}=\boldsymbol{\mu}-\mathbf{O}\)：

\[
d_{\text{ele}}=\mathbf{v}^\top\mathbf{a}_{\text{ele}},\quad
|d_{\text{ele}}| \le r_{\text{ele}}^{\text{proj}} + t_{\text{slab}}
\]

\[
u=\frac{\mathbf{v}^\top\mathbf{a}_{\text{lat}}}{\Delta_{\text{lat}}}+\frac{W-1}{2},\quad
v=\frac{\mathbf{v}^\top\mathbf{a}_{\text{ax}}-\text{near}}{\Delta_{\text{ax}}}
\]

#### (c) tile 内深度排序 key

\[
\text{key}=(\text{tile\_id}\ll 32)\ |\ \text{depth\_key}
\]

其中 `depth_key` 来自 `gauss_axial_pix` 的量化，`sort_by_depth=False` 时仅按 tile 分组。

---

## 4. ShaderForwardCUDA（前向着色）

输入：

```text
gauss_means_intensities: (G,4)  # xyz + intensity
gauss_scales_opacities:  (G,4)  # xyz + opacity
gauss_quats:             (G,4)
pixel_positions_packed:  (P,4)
probe_origin:            (3,)
probe_vec_elevational:   (3,)
pixel_gauss_offsets/counts/indices: CSR 映射
```

输出：

```text
pixel_colors:   (P,)
pixel_depths:   (P,)
pixel_accum:    (P,)
pixel_T:        (P,)
pixel_contrib:  (P,)
gauss_Rs:       (G,9)
gauss_sigma_invs:(G,9)
```

### 4.1 高斯预处理

\[
\mathbf{R}_g=\text{quat2rot}(\mathbf{q}_g),\qquad
\boldsymbol{\Sigma}_g^{-1}=\text{cov\_inv}(\mathbf{s}_g,\mathbf{R}_g)
\]

### 4.2 高斯权重与合成

\[
\mathbf{d}_{p,g}=\mathbf{x}_p-\boldsymbol{\mu}_g,\quad
G_{p,g}=\exp\left(-\frac12\mathbf{d}_{p,g}^\top\boldsymbol{\Sigma}_g^{-1}\mathbf{d}_{p,g}\right)
\]

\[
\alpha_{p,g}=o_g\cdot G_{p,g}
\]

`alpha` 模式：

\[
w_{p,g}=\alpha_{p,g}\,T_p,\quad
C_p\leftarrow C_p + I_g\,w_{p,g},\quad
T_p\leftarrow T_p(1-\alpha_{p,g})
\]

`additive_sum` / `additive_norm` 模式由 `compositing_mode` 控制，分别做加和或归一化加和。

### 4.3 深度累积（elevational）

\[
d_g=(\boldsymbol{\mu}_g-\mathbf{O}_{\text{probe}})^\top\mathbf{v}_{\text{ele}},\quad
D_p\leftarrow D_p + w_{p,g}\,d_g
\]

### 4.4 背景合成

\[
\tilde{C}_p = C_p + T_p\cdot c_{\text{bg}},\quad
C_p^{\text{out}}=\text{clip}(\tilde{C}_p,0,1)
\]

---

## 5. ShaderBackwardCUDA（反向传播）

反向只覆盖 Shader（Mapper 不可微，CSR 映射视为离散结构）。

对应调用：

```text
_RenderFunction.backward
  -> _C.ShaderBackwardCUDA(...)
```

梯度回传到可学习参数：

```text
dL_dGaussMeansIntensities  -> dL_dGaussMeans, dL_dGaussIntensities
dL_dGaussScalesOpacities   -> dL_dGaussScales, dL_dGaussOpacities
dL_dGaussQuats             -> dL_dGaussQuats
```

变量拆包关系（`autograd.py`）：

```python
dL_dGaussMeans = dL_dGaussMeansIntensities[:, :3]
dL_dGaussIntensities = dL_dGaussMeansIntensities[:, 3]
dL_dGaussScales = dL_dGaussScalesOpacities[:, :3]
dL_dGaussOpacities = dL_dGaussScalesOpacities[:, 3]
```

实现注意点：

- `src/shader/backward.cu` 中 `MAX_G=256`，每像素 backward 缓存步数上限 256；
- `alpha < 1/255` 与 forward 一致地跳过；
- 先 `kernel_pixel_compose_color_backward`，再 `kernel_pixel_alpha_accum_backward`，最后 `kernel_gauss_preprocess_backward`。

---

## 6. intermediates 与可观测量

当 `return_intermediates=True`：

```python
pixel_intermediates = stack([pixel_accum, pixel_T, pixel_contrib], dim=1)  # (P,3)
gauss_intermediates: (G,3)  # contribution / occlusion / participation
```

`api.py` 输出的 `intermediates` 字段：

```text
pixel_accum
pixel_T
pixel_contrib
gauss_contribution
gauss_occlusion
gauss_participation
pixel_gauss_counts
pixel_gauss_offsets
pixel_gauss_indices
num_pixel_gauss_pairs
```

---

## 7. 构建与验收

构建脚本：`rebuild.sh`

```bash
PYTHON_BIN=${PYTHON_BIN:-python}
$PYTHON_BIN -m pip uninstall -y ultrags_renderer || true
$PYTHON_BIN setup.py clean --all || true
$PYTHON_BIN -m pip install -v -e .
```

安装后 smoke check（脚本内已包含）：

```python
from ultrags_renderer import _C
hasattr(_C, "MapperForwardCUDA")
hasattr(_C, "ShaderForwardCUDA")
hasattr(_C, "ShaderBackwardCUDA")
```

---

## 8. 维护 checklist（面向后续改动）

1. **轴顺序不可漂移**：`lat/ax/ele` 对应 `.x/.y/.z`。  
2. **CSR 顺序即渲染顺序**：改 mapper 排序会直接改变 shader 合成行为。  
3. **前反向阈值保持一致**：`alpha < 1/255` 必须同步。  
4. **关注 `MAX_G` 限制**：高密度场景需要检查 backward 截断影响。  
5. **优先保持接口名稳定**：`MapperForwardCUDA / ShaderForwardCUDA / ShaderBackwardCUDA` 已在 Python 层绑定使用。

---

## 9. Batch 算子实现细节（新增）

本节对应 `shared_first_then_extend` 方案：**batch 内共享同一组高斯参数**，每个样本几何独立。

### 9.1 Batch 输入输出契约

Python 侧（`ultrags_renderer/api_batched.py`）：

```text
pixel_positions_world: (B, P, 3), P=H*W
geometry.origin_world: (B, 3)
geometry.axis_lateral_world / axis_axial_world / axis_elevational_world: (B, 3)
```

Mapper 输出在 batch 维展平：

```text
pixel_gauss_counts:  (B*P,)
pixel_gauss_offsets: (B*P,)
pixel_gauss_indices: (nnz_total,)
```

Shader 输出同样在 batch 维展平，再在 `api_batched.py` reshape 为 `(B, H, W)`。

### 9.2 Mapper batched 关键实现

对应文件：

- `src/mapper/core.cu`
- `src/mapper/wrapper_mapper.cu`

关键点：

1. 几何按 batch 读取：`vec_*[b*3:(b+1)*3]`
2. sort key 编码 `composite_tile = b * n_tiles + tile_local`，确保不同样本 tile 不串组
3. `sorted_gauss_ids` 采用 `bg = b * n_gauss + g`
4. count/fill kernel 中通过 `bg % n_gauss` 回到共享高斯索引

### 9.3 Shader batched 关键实现

对应文件：

- `src/shader/forward.cu`
- `src/shader/backward.cu`

关键点：

1. 全局像素索引 `gp` 映射到 `b = gp / pixels_per_image`
2. 每个样本使用自身 `probe_origin[b] / vec_elevational[b]`
3. 高斯参数仍共享（`g` 不带 batch 维）
4. backward 对高斯梯度使用 `atomicAdd` 跨 batch 累加

---

## 10. 已落地性能优化（batch 路径）

### 10.1 去主机往返与同步热点

位置：`src/mapper/wrapper_mapper.cu`

已优化：

1. **移除 batched offsets 的 CPU roundtrip**
   - 旧：`gauss_instance_offsets.cpu()` -> CPU 前缀加法 -> `copy_` 回 GPU
   - 新：GPU 上提取 batch totals + CUB scan + kernel 加前缀
2. **`pair_batch_offsets` 改为按需返回**
   - `MapperForwardCUDABatched(..., return_pair_batch_offsets)`
   - 仅 `return_intermediates=True` 时计算，训练默认可跳过
3. **移除 `pair_batch_offsets` 的 CPU 构造**
   - 旧：`pixel_gauss_offsets.cpu()` 后在 CPU 填
   - 新：GPU kernel 直接写 `pair_batch_offsets`

### 10.2 batched kernel 读几何的 shared-memory 缓存

位置：`src/mapper/core.cu`

已优化 kernel：

- `kernel_precompute_gauss_projected_radii_batched`
- `kernel_project_gauss_and_compute_tile_rects_batched`
- `kernel_count_pixel_candidates_batched`
- `kernel_fill_pixel_candidate_indices_batched`

收益逻辑：同 block 内 `b` 固定，几何向量放 shared 可减少重复 global load。

### 10.3 数据组装侧复用缓冲

位置：

- `scene/meta/slice_meta/collate.py`
- `scene/scene.py`

已优化：

1. `stack_slice_metas(..., buffers=...)` 支持预分配 tensor + `copy_` 填充
2. `Scene.next_train_batch` 按 `batch_size` 复用缓存，减少每 iter `torch.stack` 分配

---

## 11. 性能实验结论（当前基线）

历史实验脚本已清理；需要精确复查旧 launcher 时看 git history。

控制变量：

- `num_init = 1,000,000`
- 关闭 DC（densify/prune）
- A/B 等效图像数对齐

代表性结果（当前版本）：

```text
A(single, b=1): iter=4000, processed_images=4000, sec_per_image=0.04392
B(batch , b=4): iter=1000, processed_images=4000, sec_per_image=0.02909
Speedup(A/B by sec_per_image) = 1.5097x
```

解释：

- 总耗时不直接可比（B 每 iter 处理更多图像）
- 采用 `sec_per_image` / `img_per_sec` 作为公平口径
- 目前 batch 路径稳定在约 `1.5x` 量级提速（不同批次会有少量波动）

---

## 12. 近期算子改动：支持 per-sample 子集高斯（Local Window + Batch）

### 12.1 改动动机

原 batched 路径只支持 batch 内共享同一套 active gaussians。  
在 Local Window 训练中需要“每个样本（每个 window）使用不同高斯子集”，因此新增了 batch 内逐样本激活掩码能力。

### 12.2 新增接口与参数

- C++ 绑定签名扩展（`MapperForwardCUDABatched`）：
  - 新增参数 `active_gauss_mask_batched`（`uint8`，形状 `[B, G]`）
- Python 侧扩展：
  - `ultrags_renderer/api_batched.py`：`UltragsBatchedRenderer.forward(..., active_gauss_mask_batched=None, ...)`
  - `ultrags_renderer/autograd_batched.py`：`_BatchedRenderFunction.forward` 透传该掩码

当 `active_gauss_mask_batched` 为空时，行为与旧版完全一致（全激活）。

### 12.3 CUDA 实现位置与语义

位置：

- `src/mapper/wrapper_mapper.cu`
- `src/mapper/core.cu`

语义：

- `active_gauss_mask_batched[b, g] == 1`：样本 `b` 激活高斯 `g`
- `== 0`：该样本不考虑此高斯

实现方式：

- 在 batched mapper 的 `precompute/project` 阶段对未激活高斯直接置零/跳过，使其不会进入 tile/key/count/fill 后续流程。
- 排序与 CSR 构建仍保持原逻辑，仅候选集合被掩码收缩。

### 12.4 与 methods 层的对接

`methods/ultrags/render_integrated.py` 的 `render_batch` 已新增参数：

- `active_gauss_mask_batched`

`methods/ultrags/offline_callbacks_local_window.py` 已对接为：

1. 对 batch chunk 内样本子集求并集 `active_global_indices=union`
2. 同时构造 `[B, |union|]` 的 `active_gauss_mask_batched`
3. batched 渲染时传入二者，达到“每样本子集不同”的真实语义

### 12.5 等价性验证

新增测试：

- `tests/ops/ultrags_renderer/test_batched_equivalence.py::test_batched_with_active_mask_matches_serial_subsets`

验证内容：

- batched + per-sample mask 结果与“逐帧按各自子集单独渲染”结果一致（forward 等价）。
