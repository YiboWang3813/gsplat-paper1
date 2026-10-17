# 超声高斯着色器（ultrags_shader）：数学与实现对照

本文档给出 **前向 / 反向** 的显式 LaTeX 公式，并与仓库内 `ultrags_shader/__init__.py`、`src/forward.cu`、`src/backward.cu`、`src/wrapper.cu`、`include/shared.cuh`、`ops/common/covariance.cuh` **一一对应**，便于新 agent 在不读全量代码的情况下复现或修改算子。

**记号**：$G$ 为高斯个数；$P = H \times W$，像素展平索引 $p \in \{0,\ldots,P-1\}$；像素 $p$ 按固定顺序遍历若干高斯，第 $k$ 步对应高斯下标 $g_k$（由 `pixel_gauss_indices` 给出）。

---

## 1. 参数与张量定义（LaTeX ↔ 代码）

### 1.1 每个高斯 $g$ 的可学习量

* **中心**：$\boldsymbol{\mu}_g \in \mathbb{R}^3$ → `gauss_means` $(G,3)$  
* **各向异性尺度**：$\mathbf{s}_g = (s_{g,x}, s_{g,y}, s_{g,z})^\top$ → `gauss_scales` $(G,3)$（进入 `compute_cov3d_from_scale_rot` 构造精度矩阵）  
* **单位四元数**：$\mathbf{q}_g \in \mathbb{R}^4$（存储顺序与 CUDA 一致：张量第 0 列对应 `float4.x`，在 `compute_rot_from_quat` 中作为 $w$，即行向量为 $[w,x,y,z]$）→ `gauss_quats` $(G,4)$  
* **不透明度**：$o_g$ → `gauss_opacities` $(G,)$  
* **标量强度**：$I_g$ → `gauss_intensities` $(G,)$  

**Python 打包（连续内存，CUDA `float4`）：**

* `gauss_means_intensities[g]` = $(\mu_{g,x},\mu_{g,y},\mu_{g,z}, I_g)$  
* `gauss_scales_opacities[g]` = $(s_{g,x},s_{g,y},s_{g,z}, o_g)$  

### 1.2 像素与探头（仅前向使用，无梯度）

* **像素世界坐标**：$\mathbf{x}_p \in \mathbb{R}^3$ → `pixel_positions` $(P,3)$；`pixel_positions_packed` 为 $(x_p,y_p,z_p,0)$  
* **探头原点**：$\mathbf{O}_{\mathrm{probe}} \in \mathbb{R}^3$ → `probe_origin` $(3,)$  
* **Elevational 方向（建议单位向量）**：$\mathbf{v}_{\mathrm{ele}} \in \mathbb{R}^3$ → `probe_vec_elevational` $(3,)$  

### 1.3 配置与稀疏映射

* 图像尺寸：`ShaderConfig.H`, `W`，$P = HW$  
* 背景标量：$c_{\mathrm{bg}}$ → `fixed_bg_color`  
* 像素 $p$ 的高斯列表：  
  * 起始下标 $b_p$ → `pixel_gauss_offsets[p]`  
  * 个数 $n_p$ → `pixel_gauss_counts[p]`  
  * 第 $k$ 个高斯下标 $g_{p,k} =$ `pixel_gauss_indices[b_p + k]`  

### 1.4 前向中间量与输出（符号 ↔ 代码）

| 数学符号 | 代码 | Shape |
| :--- | :--- | :--- |
| $\mathbf{R}_g$（行主序展平） | `gauss_Rs` | $(G,9)$ |
| $\boldsymbol{\Sigma}_g^{-1}$（行主序展平） | `gauss_sigma_invs` | $(G,9)$ |
| 混合前颜色累积 $C_p^{\mathrm{pre}}$ | `pixel_accum` | $(P,)$ |
| 混合结束透射率 $T_p^{\mathrm{end}}$ | `pixel_T` | $(P,)$ |
| $\sum_k \alpha_k$（统计） | `pixel_contrib` | $(P,)$ |
| Elevational 深度图 $D_p$ | `pixel_depths` | $(P,)$ |
| 最终颜色 $C_p$（clamp 后） | `pixel_colors` | $(P,)$ |

---

## 2. 前向传播（Forward）

### 2.1 旋转与精度矩阵（每个高斯 $g$）

由四元数得旋转矩阵 $\mathbf{R}_g \in \mathbb{R}^{3\times 3}$。令对角阵 $\mathbf{D}_g = \mathrm{diag}(s_{g,x}^{-2}, s_{g,y}^{-2}, s_{g,z}^{-2})$（实现中对 $s$ 有下界），则

$$
\mathbf{M}_g = \mathbf{R}_g \mathbf{D}_g, \qquad
\boldsymbol{\Sigma}_g^{-1} = \mathbf{M}_g \mathbf{R}_g^\top.
$$

**实现对应：** `kernel_gauss_preprocess_forward` → `compute_rot_from_quat` + `compute_cov3d_from_scale_rot`，写入 `gauss_Rs`、`gauss_sigma_invs`。

---

### 2.2 马氏指数与高斯权重

对像素 $p$ 与当前步高斯中心 $\boldsymbol{\mu}$：

$$
\mathbf{d}_{p} = \mathbf{x}_p - \boldsymbol{\mu}, \qquad
e_{p} = -\frac{1}{2}\,\mathbf{d}_{p}^\top \boldsymbol{\Sigma}^{-1} \mathbf{d}_{p}, \qquad
G_{p} = \exp(e_{p}).
$$

**实现对应：** `compute_gaussian_weight`（`shared.cuh`）。

---

### 2.3 前向 Alpha 混合（按列表顺序）

对固定像素 $p$，设当前步前透射率为 $T$（初值 $T^{(0)}=1$）。对列表中下一个高斯（若 $\alpha < 1/255$ 则 **跳过该步**，不参与混合与写回）：

$$
\alpha = o\, G_{p}, \qquad w = \alpha\, T, \qquad
C_p^{\mathrm{pre}} \leftarrow C_p^{\mathrm{pre}} + I\, w, \qquad
T \leftarrow T(1-\alpha).
$$

若更新后 $T < 10^{-4}$，置 $T \leftarrow 0$ 并结束该像素循环。

**实现对应：** `kernel_pixel_alpha_accum_forward` 中 `gaussian_weight`、`alpha`、`weight`、`color_accum`、`transmittance`；`pixel_contrib` 累加 $\alpha$。

---

### 2.4 Elevational 深度（与颜色同权混合）

定义相对于探头的 elevational 标量：

$$
d = (\boldsymbol{\mu} - \mathbf{O}_{\mathrm{probe}})^\top \mathbf{v}_{\mathrm{ele}}.
$$

每步与颜色一致使用权重 $w$：

$$
D_p \leftarrow D_p + d\, w.
$$

**实现对应：** `depth_elevational`、`depth_accum`。

---

### 2.5 与背景合成与 Clamp

记混合结束为 $C_p^{\mathrm{pre}}$、$T_p^{\mathrm{end}}$：

$$
\tilde{C}_p = C_p^{\mathrm{pre}} + T_p^{\mathrm{end}}\, c_{\mathrm{bg}}, \qquad
C_p = \mathrm{clip}(\tilde{C}_p,\, 0,\, 1).
$$

**实现对应：** `kernel_pixel_compose_color_forward` → `composed_color`、`pixel_colors`。

---

## 3. 反向传播（Backward）

上游给定 $\dfrac{\partial L}{\partial C_p}$（`dL_dPixelColors`，对 clamp 后输出；实现中合成反传视同为对 $\tilde{C}_p$ 的梯度）与 $\dfrac{\partial L}{\partial D_p}$（`dL_dPixelDepths`）。中间量 `pixel_accum`、`pixel_T`、`gauss_intermediates` **不参与反传**。

### 3.1 合成步的梯度

$$
\frac{\partial L}{\partial C_p^{\mathrm{pre}}} = \frac{\partial L}{\partial \tilde{C}_p}, \qquad
\frac{\partial L}{\partial T_p^{\mathrm{end}}} = \frac{\partial L}{\partial \tilde{C}_p}\, c_{\mathrm{bg}}.
$$

**实现对应：** `kernel_pixel_compose_color_backward` → `dL_dPixelAccum`、`dL_dPixelT`。

---

### 3.2 单步混合的反向（倒序；与代码变量名对齐）

对某一保留步（未被 $\alpha < 1/255$ 丢弃），记混合前透射率为 $T_{\mathrm{before}}$，当前 $\alpha$、强度 $I$、elevational 深度 $d = (\boldsymbol{\mu}-\mathbf{O}_{\mathrm{probe}})^\top\mathbf{v}_{\mathrm{ele}}$，高斯权重 $G$（$\alpha = o\, G$）。记从后方传入的、对「该步之后透射率」的梯度为 $\dfrac{\partial L}{\partial T_{\mathrm{after}}}$（代码中 `dL_dT_running` 迭代更新）。

**对强度：**

$$
\frac{\partial L}{\partial I} \mathrel{+}= \frac{\partial L}{\partial C_p^{\mathrm{pre}}}\, \alpha\, T_{\mathrm{before}}.
$$

**对 elevational 深度标量 $d$（链到 $\boldsymbol{\mu}$）：**

$$
\frac{\partial L}{\partial d} = \frac{\partial L}{\partial D_p}\, \alpha\, T_{\mathrm{before}}, \qquad
\frac{\partial L}{\partial \boldsymbol{\mu}} \mathrel{+}= \frac{\partial L}{\partial d}\, \mathbf{v}_{\mathrm{ele}}.
$$

**对 $\alpha$ 的三路合并：**

$$
\frac{\partial L}{\partial \alpha}
= \frac{\partial L}{\partial C_p^{\mathrm{pre}}}\, I\, T_{\mathrm{before}} + \frac{\partial L}{\partial D_p}\, d\, T_{\mathrm{before}} + \frac{\partial L}{\partial T_{\mathrm{after}}}\, \bigl(-T_{\mathrm{before}}\bigr).
$$

**更新对 $T_{\mathrm{before}}$ 的梯度（供前一步使用）：**

$$
\frac{\partial L}{\partial T_{\mathrm{before}}}
= \frac{\partial L}{\partial C_p^{\mathrm{pre}}}\, I\, \alpha + \frac{\partial L}{\partial D_p}\, d\, \alpha + \frac{\partial L}{\partial T_{\mathrm{after}}}\, (1-\alpha).
$$

**对不透明度与几何权重：**

$$
\frac{\partial L}{\partial o} = \frac{\partial L}{\partial \alpha}\, G, \qquad
\frac{\partial L}{\partial G} = \frac{\partial L}{\partial \alpha}\, o.
$$

**对 $\boldsymbol{\Sigma}^{-1}$（与 rasterizer 相同的二次型链式）：**  
令 $\mathbf{d} = \mathbf{x}_p - \boldsymbol{\mu}$，$G = \exp(e)$，$e = -\frac{1}{2}\mathbf{d}^\top\boldsymbol{\Sigma}^{-1}\mathbf{d}$，则

$$
\frac{\partial e}{\partial \boldsymbol{\Sigma}^{-1}} = -\frac{1}{2}\,\mathbf{d}\mathbf{d}^\top, \qquad
\frac{\partial L}{\partial \boldsymbol{\Sigma}^{-1}} \mathrel{+}= \frac{\partial L}{\partial G}\, G \cdot \Bigl(-\frac{1}{2}\Bigr)\, \mathbf{d}\mathbf{d}^\top.
$$

**对 $\boldsymbol{\mu}$（经 $G$）：**  
$\dfrac{\partial e}{\partial \boldsymbol{\mu}} = \boldsymbol{\Sigma}^{-1}\mathbf{d}$，故

$$
\frac{\partial L}{\partial \boldsymbol{\mu}} \mathrel{+}= \frac{\partial L}{\partial G}\, G\, \bigl(-\boldsymbol{\Sigma}^{-1}\mathbf{d}\bigr)
\quad\text{（与 3.2 中 depth 项叠加）}.
$$

**实现对应：** `kernel_pixel_alpha_accum_backward` Pass 2：`dL_dColor`、`dL_dDepth_elevational`、`dL_dAlpha_*`、`dL_dT_running`、`dL_dOpacity`、`sigma_inv_grad_scale`、`atomicAdd` 至 `dL_dGaussMeansIntensities` / `dL_dGaussScalesOpacities` / `dL_dGaussSigmaInvs`。

**注意：** 实现中 Pass 1 最多缓存 `MAX_G = 256` 步；超过则 backward 仅覆盖前 256 个有效步，与 forward 可能不一致。

---

### 3.3 精度矩阵 → 尺度与旋转 → 四元数

对每个高斯 $g$，已知 $\dfrac{\partial L}{\partial \boldsymbol{\Sigma}_g^{-1}}$（展平为 9 维，内部缓冲 `dL_dGaussSigmaInvs`）：

* `transform_grads_sigma_to_R_S`：得到 $\dfrac{\partial L}{\partial \mathbf{R}_g}$、$\dfrac{\partial L}{\partial \mathbf{s}_g}$（scale 的 xyz；**不透明度梯度已在 3.2 写入 `.w`**）。  
* `transform_grads_R_to_Quat`：$\dfrac{\partial L}{\partial \mathbf{q}_g}$。

**实现对应：** `kernel_gauss_preprocess_backward`。

---

### 3.4 Python 端梯度拆包

* $\dfrac{\partial L}{\partial \boldsymbol{\mu}_g}$、`dL_dGaussIntensities` ← `dL_dGaussMeansIntensities` 前 3 列与第 4 列  
* $\dfrac{\partial L}{\partial \mathbf{s}_g}$、`dL_dGaussOpacities` ← `dL_dGaussScalesOpacities` 前 3 列与第 4 列  
* $\dfrac{\partial L}{\partial \mathbf{q}_g}$ ← `dL_dGaussQuats`  

**实现对应：** `_ShadeFunction.backward`（`__init__.py`）。

---

## 4. 调用链与文件映射（给 agent 的索引）

| 阶段 | CUDA 入口 | 主要文件 |
| :--- | :--- | :--- |
| PyTorch 打包 / autograd | `_C.shade_forward` / `_C.shade_backward` | `ultrags_shader/__init__.py` |
| 张量分配与指针 | `RasterizeGaussiansForwardCUDA` / `BackwardCUDA` | `src/wrapper.cu` |
| 前向三 kernel | `kernel_gauss_preprocess_forward` → `kernel_pixel_alpha_accum_forward` → `kernel_pixel_compose_color_forward` | `src/forward.cu` |
| 反向三 kernel | `kernel_pixel_compose_color_backward` → `kernel_pixel_alpha_accum_backward` → `kernel_gauss_preprocess_backward` | `src/backward.cu` |
| $G_p$ 与 $\boldsymbol{\Sigma}^{-1}$ 构造 | `compute_gaussian_weight`、`compute_cov3d_from_scale_rot` | `shared.cuh`、`covariance.cuh` |

---

## 5. 实现要点（与 rasterizer 文档同级摘要）

1. **顺序敏感**：列表顺序即混合顺序，改变 `pixel_gauss_indices` 即改变前向与反向。  
2. **阈值一致**：$\alpha < 1/255$ 在 forward 与 backward Pass 1 均跳过，保证同一路径。  
3. **`MAX_G`**：backward 每像素最多 256 个有效混合步；训练数据需控制每像素列表长度或预先截断至与 forward 一致。  
4. **可选统计**：`gauss_intermediates` $(G,3)$ 仅调试；`fill_gauss_intermediates` 由 Python 传入张量 shape $(G,3)$ 触发。
