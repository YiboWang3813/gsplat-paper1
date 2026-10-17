# 椭球映射算子（ellipsoid_mapper）：数学与实现对照

本文档将 `ellipsoid_mapper` 的 **坐标约定、数学定义、CUDA kernel 流水线与张量形状** 对齐说明，并与仓库内 `ops/ellipsoid_mapper/ellipsoid_mapper/__init__.py`、`ops/ellipsoid_mapper/src/wrapper.cu`、`ops/ellipsoid_mapper/src/core.cu`、`ops/ellipsoid_mapper/include/core.cuh` **一一对应**，便于后续维护/重构时快速定位每个变量的语义。

**一句话**：给定每个像素在世界坐标系下的位置与一组 3D 高斯椭球（中心/尺度/四元数），本算子在给定 slice 的探头几何（origin + 3 轴方向）与成像采样参数下，输出稀疏映射 `pixel_gauss_counts/offsets/indices`，用于后续 shader 对每个像素只遍历其候选高斯列表。

**记号**：$G$ 为高斯个数；图像大小为 $H \times W$；$P = HW$，像素展平索引 $p\in\{0,\ldots,P-1\}$（row-major）；每像素候选高斯列表由三件套给出：

- 起始下标 $b_p$ → `pixel_gauss_offsets[p]`
- 个数 $n_p$ → `pixel_gauss_counts[p]`
- 第 $k$ 个高斯 id：$g_{p,k} = \texttt{pixel\_gauss\_indices}[b_p + k]`

---

## 1. 硬约定：坐标系、轴顺序、单位与排序语义

### 1.1 探头坐标系（Probe frame）轴顺序

本项目对 **探头坐标系**的轴定义为：

- $x$ = **lateral**
- $y$ = **axial**（超声波传播/深度方向）
- $z$ = **elevational**（slice 平面的法向）

因此，凡是用 `float3` 存储 “沿 probe 三轴的量”（例如投影半径），都采用相同分量顺序：

> `.x = lateral, .y = axial, .z = elevational`

该约定写死在 `ops/ellipsoid_mapper/include/core.cuh` 顶部注释中。

### 1.2 世界坐标输入与单位

- `pixel_positions_world` / `gauss_means_world` / `geometry.origin_world`：**WORLD 坐标系**下的位置（单位 **mm**）
- `geometry.axis_*_world`：WORLD 坐标系下的单位向量（建议单位长度、近似正交）
- `*_spacing_mm_per_pix`：mm/pix
- `axial_near_mm`：轴向起始深度（mm，且在当前数据约定中非负）

### 1.3 tile 内排序语义（非常重要）

在同一 tile 内，我们将候选高斯按 **axial 从小到大（更浅 → 更深）** 排序：

> axial 更小的椭球被认为更早被声束“遇到”（先经过）→ 排序在前。

实现上当前使用 `v_pix`（轴向像素坐标）生成 `depth_key`，并通过 radix sort 得到 tile 内顺序；详见 §3.3。

---

## 2. 参数与张量定义（LaTeX ↔ 代码）

### 2.1 高斯椭球参数（输入）

对每个高斯 $g$：

- **中心**：$\boldsymbol{\mu}_g\in\mathbb{R}^3$ → `gauss_means_world` `(G,3)`
- **尺度**：$\mathbf{s}_g=(s_{g,x},s_{g,y},s_{g,z})^\top$ → `gauss_scales` `(G,3)`
- **四元数**：$\mathbf{q}_g\in\mathbb{R}^4$ → `gauss_quats` `(G,4)`（内部用 `compute_rot_from_quat` 转旋转矩阵）

Python 侧会把 vec3 **padding** 成 `float4` 以便 CUDA 侧按 `float4*` 读取：

- `pack_vec3([x,y,z]) -> [x,y,z,0]`

对应：`ops/ellipsoid_mapper/ellipsoid_mapper/__init__.py` 中 `pack_vec3`，以及 `wrapper.cu`/`core.cu` 中 `reinterpret_cast<float4*>` + `unpack_float4`。

### 2.2 slice 几何与配置（dataclass）

Python 强制以结构化方式传入：

- `SliceGeometry`
  - `origin_world` `(3,)`
  - `axis_lateral_world` `(3,)`
  - `axis_axial_world` `(3,)`
  - `axis_elevational_world` `(3,)`
- `EllipsoidMapperConfig`
  - `H, W`
  - `lateral_spacing_mm_per_pix`
  - `axial_spacing_mm_per_pix`
  - `slab_half_thickness_mm`
  - `axial_near_mm`
  - `k_sigma`

对应：`ops/ellipsoid_mapper/ellipsoid_mapper/__init__.py`

### 2.3 输出稀疏映射（输出）

返回三件套：

- `pixel_gauss_counts` `(P,) int32`
- `pixel_gauss_offsets` `(P,) int32`（exclusive scan）
- `pixel_gauss_indices` `(nnz,) int32`，其中 \(nnz = \sum_p n_p\)

对应：`ops/ellipsoid_mapper/src/wrapper.cu` 末尾的 Count/Scan/Fill 三步。

---

## 3. CUDA 流水线（Kernel-by-kernel）

本算子整体是 **tile-based** 的两段式候选生成：

1. **每个高斯**：投影到像素平面→得到覆盖的 tile rect→展开为多个 tile 实例并排序；
2. **每个像素**：仅在自己 tile 的候选列表中做快速相交检查→生成最终稀疏映射。

### 3.1 Kernel 1：预计算投影半径（Manhattan bound）

目标：对每个高斯 $g$，计算其在三个探头轴方向上的投影“半宽”，并乘上 `k_sigma`，输出：

\[
\mathbf{r}_g^{\mathrm{proj}} = \begin{bmatrix} r_{g,\mathrm{lat}}\\ r_{g,\mathrm{ax}}\\ r_{g,\mathrm{ele}} \end{bmatrix} \quad\text{（mm，且已乘 }k_\sigma\text{）}
\]

实现：`kernel_precompute_gauss_projected_radii`（`ops/ellipsoid_mapper/src/core.cu`）

- 由四元数得旋转矩阵 $R$（`compute_rot_from_quat`）
- 用 Manhattan 上界计算某方向 `dir` 上的投影半宽：

\[
r(\mathbf{s},R,\mathbf{d})
\;=\;
\sum_{i\in\{0,1,2\}}
\left|
s_i \cdot (\mathbf{a}_i^\top \mathbf{d})
\right|,
\quad \mathbf{a}_i=\text{第 }i\text{ 列轴向量}
\]

对应函数：`compute_projected_radius`（`ops/ellipsoid_mapper/include/core.cuh`）

输出存储约定（probe xyz 顺序）：

- `projected_radii[g].x` = lateral
- `projected_radii[g].y` = axial
- `projected_radii[g].z` = elevational

### 3.2 Kernel 2：slab 裁剪 + 投影到像素平面 + tile rect

目标：对每个高斯中心 $\boldsymbol{\mu}_g$（WORLD），在 slice 的 WORLD 轴上投影得到：

设 $\mathbf{O}$ 为 `origin_world`，$\mathbf{a}_{lat},\mathbf{a}_{ax},\mathbf{a}_{ele}$ 为三个单位轴向量（WORLD）。

\[
\mathbf{v} = \boldsymbol{\mu}_g - \mathbf{O}
\]

**elevational slab 裁剪**：

\[
d_{ele} = \mathbf{v}^\top \mathbf{a}_{ele},\quad
\left|d_{ele}\right| \le r_{g,ele}^{proj} + t_{\mathrm{slab}}
\]

其中 \(t_{\mathrm{slab}}=\) `slab_half_thickness_mm`。

**投影到成像平面并离散化为像素坐标**：

\[
x_{lat} = \mathbf{v}^\top \mathbf{a}_{lat},\qquad
y_{ax} = \mathbf{v}^\top \mathbf{a}_{ax}
\]

\[
u = \frac{x_{lat}}{\Delta_{lat}} + \frac{W-1}{2},\qquad
v = \frac{y_{ax} - \mathrm{near}}{\Delta_{ax}}
\]

其中 \(\Delta_{lat}=\) `lateral_spacing_mm_per_pix`，\(\Delta_{ax}=\) `axial_spacing_mm_per_pix`。

再由投影半宽得到像素半径：

\[
r_u = \max\left(1,\frac{r_{g,lat}^{proj}}{\Delta_{lat}}\right),\qquad
r_v = \max\left(1,\frac{r_{g,ax}^{proj}}{\Delta_{ax}}\right)
\]

从而得到覆盖 tile rect（tile size = 16）：

- `min_tx = floor((u - r_u)/tile)`
- `max_tx = ceil((u + r_u)/tile)`
- `min_ty = floor((v - r_v)/tile)`
- `max_ty = ceil((v + r_v)/tile)`

并 clamp 到 `[0, grid_w)` / `[0, grid_h)`。

实现：`kernel_project_gauss_and_compute_tile_rects`（`ops/ellipsoid_mapper/src/core.cu`）

输出：

- `tiles_touched_count[g]`：该高斯覆盖 tile 数
- `gauss_tile_rects[g]`：`int4(min_tx,min_ty,max_tx,max_ty)`
- `gauss_axial_pix[g]`：上式中的 `v`（用于排序 key）

### 3.3 Kernel 3：生成 (tile_id, depth_key) 并排序

目标：把每个高斯的 tile rect 展开为多个实例：

- 每个实例对应一个 tile，携带 `gauss_id`
- key 的高 32 位为 `tile_id`，低 32 位为 `depth_key`

\[
\text{key} = (\text{tile\_id} \ll 32)\;|\;\text{depth\_key}
\]

其中 `depth_key` 基于 `gauss_axial_pix[g] = v` 构造，保证同 tile 内按 axial 从小到大排列。

实现：`kernel_generate_tile_depth_keys`（`ops/ellipsoid_mapper/src/core.cu`）+ CUB radix sort（`ops/ellipsoid_mapper/src/wrapper.cu`）

**关于 depth_key 的“offset/quantize”**：

- 当前场景 `v_pix` 预期非负，但仍保留正 offset 作为 guard，避免 cast 到 `uint32` 时 underflow（例如极端 out-of-range 或 jitter）。
- `kDepthKeyOffset` 的单位是 **pix**（因为 `d` 是 `v_pix`），这是防御性实现细节。

排序完成后，用 `kernel_identify_tile_ranges` 为每个 tile 写入：

- `tile_ranges_start[tile_id]`
- `tile_ranges_end[tile_id]`

满足该 tile 的候选实例区间为 `[start,end)`。

### 3.4 Kernel 4/5：像素在本 tile 候选列表中精筛并写出稀疏映射

对每个像素 $p$（二维网格 launch）：

1. 计算 `tile_id(p)`，取出候选区间 `[start,end)`；
2. 对每个候选高斯 id `g`，做一个 **仅在 lat/ax 平面** 的快速相交：

设像素世界坐标为 $\mathbf{x}_p$，高斯中心为 $\boldsymbol{\mu}_g$：

\[
\mathbf{v} = \boldsymbol{\mu}_g - \mathbf{x}_p
\]

\[
d_{ax} = \mathbf{v}^\top \mathbf{a}_{ax},\quad |d_{ax}| \le r_{g,ax}^{proj}
\]
\[
d_{lat} = \mathbf{v}^\top \mathbf{a}_{lat},\quad |d_{lat}| \le r_{g,lat}^{proj}
\]

满足则认为该高斯是像素的候选。

实现：

- Count：`kernel_count_pixel_candidates`
- Fill：`kernel_fill_pixel_candidate_indices`
- 检查函数：`check_intersection_optimized`（`ops/ellipsoid_mapper/include/core.cuh`）

最终由 CUB scan 将 counts → offsets，并填充 `pixel_gauss_indices`。

---

## 4. 入口、形状检查与调用链

### 4.1 Python 入口（强类型）

`EllipsoidMapper.map(config, geometry, pixel_positions_world, gauss_means_world, gauss_scales, gauss_quats)`

对应：`ops/ellipsoid_mapper/ellipsoid_mapper/__init__.py`

### 4.2 C++ 入口安全检查（强一致性）

C++ 侧要求：

\[
P = H \times W
\]

在 `launch_ellipsoid_mapper` 中有：

- `TORCH_CHECK(n_pixels == image_height * image_width, ...)`

对应：`ops/ellipsoid_mapper/src/wrapper.cu`

### 4.3 文件映射（给 agent 的索引）

| 阶段 | 代码入口 | 主要文件 |
| :--- | :--- | :--- |
| Python API / dataclass | `EllipsoidMapper.map` | `ops/ellipsoid_mapper/ellipsoid_mapper/__init__.py` |
| 张量分配、scan、sort、输出构造 | `launch_ellipsoid_mapper` | `ops/ellipsoid_mapper/src/wrapper.cu` |
| CUDA kernels | `kernel_*` | `ops/ellipsoid_mapper/src/core.cu` |
| 约定与设备函数 | `compute_projected_radius`, `check_intersection_optimized` | `ops/ellipsoid_mapper/include/core.cuh` |

---

## 5. 实现要点与常见坑（维护者 checklist）

1. **轴顺序不要再漂移**：`projected_radii` 的 `.x/.y/.z` 永远是 `lat/ax/ele`。任何修改必须同时改注释与所有使用点。  
2. **slab_half_thickness 是半厚度**：代码中 slab 厚度参数已统一命名为 `slab_half_thickness_mm`。  
3. **排序语义固定**：tile 内按 axial 从小到大。若后续 shader 假设相反顺序，需要一起调整。  
4. **这是候选生成而非精确椭球相交**：像素精筛只做 lat/ax 的盒式判定（Manhattan bound + 1D 阈值），不会严格等价于真实椭球与射线/体素相交。  
5. **性能与正确性权衡点**：tile size 固定为 16（`SLICE_TILE_SIZE`），改变它会影响候选密度与排序实例数，需要系统评估。
