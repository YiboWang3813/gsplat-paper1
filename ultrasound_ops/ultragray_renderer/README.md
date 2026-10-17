# ultragray_renderer：代码即文档

本文档严格对应当前实现，覆盖 `ultragray_renderer` 的 Python API、C++/CUDA 绑定、Mapper/Shader 计算流程与关键公式。  
目标是做到：**按文档可直接定位代码并复现实例行为**。

---

## 1. 包结构与调用链（严格对齐代码）

### 1.1 目录分层

```text
ops/ultragray_renderer/
  ultragray_renderer/
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
UltragrayRenderer.forward
  -> _UltragrayRenderFunction.forward
    -> _C.MapperForwardCUDA
    -> _C.ShaderForwardCUDA
  -> (autograd backward)
    -> _C.ShaderBackwardCUDA
```

对应绑定导出（`src/bindings.cpp`）：

```cpp
m.def("MapperForwardCUDA", &MapperForwardCUDA, "Mapper forward (CUDA)");
m.def("ShaderForwardCUDA", &ShaderForwardCUDA, "Shader forward (CUDA)");
m.def("ShaderBackwardCUDA", &ShaderBackwardCUDA, "Shader backward (CUDA)");
```

---

## 2. 输入输出契约（Python 层）

### 2.1 配置与几何（`types.py`）

- `UltragrayRenderConfig`
  - `image_width`, `image_height`
  - `width_mm`, `near_mm`
  - `elev_cutoff_margin_mm`, `k_sigma`
  - `ops_max_offset_mm`（OPS 扰动上限）
  - `ops_train_only`（方法层配置；默认只在训练 jitter 路径启用 OPS）
  - `quadrature_window_radius_mm`（closest approach 周围积分半径；`0` 表示全路径积分）
  - `elev_beam_width_mm`（旧配置兼容字段；新代码使用 `ops_max_offset_mm`）
- `ProbeGeometry`
  - `origin_world`
  - `axis_lateral_world`
  - `axis_elevational_world`
  - `axis_axial_world`

### 2.2 `forward` 输入张量（`api.py` / `autograd.py`）

```python
means:          (G, 3)
scales:         (G, 3)
quats:          (G, 4)
attenuations:   (G,)      # tau 的 raw logits
sh_coeffs:      (G, 4)    # degree-1 SH: [c0, c1, c2, c3]
pixel_positions:(W, H, 3)
ray_origins:    (W, 3)
ray_dirs:       (W, 3)
```

### 2.3 输出

- 仅输出单一 `bmode_image`（`(H, W)`）
- 不返回中间变量/调试分支/depth 分支

---

## 3. MapperForwardCUDA（射线候选映射）

一句话：输入高斯几何与探头坐标系，输出按 ray 分组的 CSR 三件套：

```text
ray_counts:  (W,) int32
ray_offsets: (W,) int32
ray_indices: (nnz,) int32   # 按 (ray, depth) 排序后的高斯索引
```

CUDA 流程（`src/mapper/wrapper_mapper.cu` + `src/mapper/core.cu`）：

```text
Kernel1: project_and_cull (depth/elev/lateral 范围筛选)
Scan:    cub::DeviceScan::ExclusiveSum
Kernel2: generate_sort_keys (RayID | Depth)
Sort:    cub::DeviceRadixSort::SortPairs
Kernel3: build_csr_structure (ray starts/ends)
```

实现约定：

- 采用保守半径 `radius_safe = max(scale_xyz) * k_sigma`
- `depth` 与 `elevation` 双重剔除后，映射到 lateral 像素范围
- 前缀和使用 `N` 长度 exclusive scan，并补写 `offsets[N] = total_hits`

---

## 4. ShaderForwardCUDA（物理前向）

核心形式与说明书一致：

\[
B(p)=T(p)\cdot E(p)
\]

### 4.1 衰减项 \(T(p)\)

- 使用有限段积分（3-point Gauss-Legendre）计算 \(\psi_i\)
- 单高斯透射：
\[
T_i=\tau_i + (1-\tau_i)\exp(-\psi_i),\quad \tau_i=\sigma(\text{attenuation\_logit}_i)
\]
- 按 ray 深度顺序累乘得到 `transmission`

### 4.2 回波项 \(E(p)\)

- 3D Mahalanobis 权重：
\[
w_i(p)=\exp\left(-\frac12(p-\mu_i)^T\Sigma_i^{-1}(p-\mu_i)\right)
\]
- 方向相关回波（degree-1 SH）：
\[
I_i(d)=c_{i,0}+c_{i,1}d_x+c_{i,2}d_y+c_{i,3}d_z
\]
- soft coverage：
\[
S=\sum_i w_i,\quad \gamma=1-\exp(-S),\quad
E=\gamma\cdot\frac{\sum_i I_i(d)w_i}{\sum_i w_i+\epsilon}
\]

### 4.3 OPS（ray 级）

- 扰动以 ray 为单位采样（cosine-weighted）
- 对每条 ray 的 origin 沿 `axis_elevational` 偏移，再参与渲染

---

## 5. ShaderBackwardCUDA（反向传播）

反向仅对 Shader 可微部分求导，Mapper 仍视为离散结构。

梯度输出：

```text
dL_dMeans:   (G,4)  # Python 侧截取 :3
dL_dScales:  (G,4)  # Python 侧截取 :3
dL_dQuats:   (G,4)
dL_dAttens:  (G,)
dL_dSH:      (G,4)
```

关键点：

- 透射路径使用与 forward 对齐的有限段 GL3 积分梯度（非无限域近似）
- 回波路径对 `sh_coeffs` 逐分量累计梯度
- 协方差梯度继续通过 `transform_grads_sigma_to_R_S` 与 `transform_grads_R_to_Quat` 回传到 `scales/quats`

---

## 6. 构建与验收

构建脚本：`rebuild.sh`

```bash
cd ops/ultragray_renderer
bash rebuild.sh
```

安装后最小检查：

```python
from ultragray_renderer import _C
hasattr(_C, "MapperForwardCUDA")
hasattr(_C, "ShaderForwardCUDA")
hasattr(_C, "ShaderBackwardCUDA")
```
