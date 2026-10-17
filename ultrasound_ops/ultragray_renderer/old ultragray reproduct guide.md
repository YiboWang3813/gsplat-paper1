# Method

## 0) 逐小节映射（对应论文结构）

- `2. Basics of Gaussian Splatting` -> 本笔记 `1) Section 2: 3DGS 基础`
- `3.1 Ultrasound Physics` -> 本笔记 `2) Section 3.1: 超声物理前提`
- `3.2 Ultrasound-Adaptive Gaussian Forward Model` -> 本笔记 `3) Section 3.2: 超声自适应前向模型`
- `3.3 Initialization and Optimization Strategy` -> 本笔记 `4) Section 3.3: 初始化与优化策略`
- `3.4 Implementation` + `Appendix A` -> 本笔记 `5) Section 3.4 + Appendix A: 实现与超参数`

## 1) Section 2: 3DGS 基础（作为 UltraG-Ray 的起点）

标准 3DGS 中，每个高斯包含 $\mu_i,\Sigma_i,\alpha_i,c_i$，并使用 SH 建模视角相关外观。

$$
c_i(v)=\sum_{b=1}^{B}\beta_{i,b}Y_b(v) \tag{1}
$$

变量说明：
- $i$：第 $i$ 个高斯
- $v$：观察方向（单位向量）
- $Y_b(\cdot)$：第 $b$ 个 SH 基函数
- $\beta_{i,b}$：第 $i$ 个高斯在第 $b$ 个 SH 基上的系数
- $B$：SH 基函数数量（由 SH 阶数决定）

$$
w_i(p)=\exp\left(-\frac12 (p-u_i)^T(\Sigma_i^{2D})^{-1}(p-u_i)\right) \tag{2}
$$

变量说明：
- $p$：图像平面上的像素坐标
- $u_i$：高斯中心投影到图像平面的坐标
- $\Sigma_i^{2D}$：高斯在图像平面的 2D 协方差
- $w_i(p)$：高斯对像素 $p$ 的 footprint 权重

$$
T_i(p)=T_{i-1}(p)\left(1-\alpha_i(p)\right),\quad T_0=1 \tag{3}
$$

变量说明：
- $T_i(p)$：处理到第 $i$ 个高斯后，像素 $p$ 的剩余透射率
- $\alpha_i(p)$：第 $i$ 个高斯在像素 $p$ 的有效不透明度（通常与 $w_i$ 相关）

$$
C(p,v)=\sum_{i=1}^{N} T_{i-1}(p)\alpha_i(p)c_i(v) \tag{4}
$$

变量说明：
- $C(p,v)$：像素 $p$ 在视角 $v$ 下的渲染结果
- $N$：参与该像素合成的高斯数量

这一套是光学风格的发射-吸收模型，UltraG-Ray 在 Section 3 将其改为超声物理一致的建模。

## 2) Section 3.1: 超声物理前提

核心前提：B-mode 成像可视为“发射-传播-衰减-反射-接收”的过程。与自然图像不同，超声中衰减、阴影和视角相关散射更强，因此需要显式建模透射与回波，而不能直接照搬 RGB/opacity 渲染。

## 3) Section 3.2: 超声自适应前向模型

每个高斯参数：
- 几何：$\mu_i, \Sigma_i$
- 物理：可学习透射参数 $\tau_i$
- 回波强度：$I_i$

其中 $\Sigma_i$ 通过尺度 $s_i$ 与旋转四元数 $q_i$ 参数化，保证数值稳定和正定。

对每个像素，B-mode 强度被分解为：

$$
B(p)=T(p)\cdot E(p) \tag{5}
$$

变量说明：
- $B(p)$：像素 $p$ 的最终 B-mode 强度
- $T(p)$：传输项（沿射线累计衰减后的剩余能量）
- $E(p)$：回波项（局部后向散射/反射贡献）

### (a) 衰减建模：射线-高斯积分

将射线从世界坐标变换到高斯规范空间：

$$
o_g=S_i^{-1}R_i^T(o-\mu_i),\quad d_g=S_i^{-1}R_i^Td,\quad l_g=\|d_g\|l \tag{6}
$$

变量说明：
- $o,d$：射线起点与方向（由探头位姿定义）
- $R_i,S_i$：第 $i$ 个高斯的旋转与尺度矩阵
- $o_g,d_g$：规范空间中的射线起点与方向
- $l$：世界坐标下积分段长度

定义射线与高斯重叠积分：

$$
\psi_i=\int_{0}^{l}\exp\left(-\frac12\|o_g+t\,d_g\|^2\right)\,dt \tag{7}
$$

变量说明：
- $\psi_i$：射线与第 $i$ 个高斯重叠强度，越大表示对衰减影响越强

论文中该积分使用 3-point Gauss-Legendre 进行数值近似。单高斯透射项与累计透射：

$$
T(z)=\prod_{i=1}^{n}T_i,\quad T_i=\tau_i+(1-\tau_i)\exp(-\psi_i) \tag{8}
$$

变量说明：
- $n$：当前射线上的相关高斯数量
- $T_i$：第 $i$ 个高斯对应的透射因子
- $\tau_i$：可学习透射基值
- $T(z)$：深度 $z$ 处累计透射（乘法形式）

该式可视为 Beer-Lambert 型衰减的离散近似，并支持 shadowing。

### (b) 回波建模：3D Mahalanobis + SH

在 3D 空间计算权重（而非 2D splat）：

$$
w_i(p)=\exp\left(-\frac12 (p-\mu_i)^T\Sigma_i^{-1}(p-\mu_i)\right)
$$

回波振幅 $I_i(d)$ 用低阶 SH（degree=1）建模方向相关性。并引入 soft coverage：

$$
S(p)=\sum_{i=1}^{n}w_i(p),\quad \gamma(p)=1-\exp(-S(p))
$$

变量说明：
- $S(p)$：像素覆盖总量（总 footprint）
- $\gamma(p)$：软覆盖系数，覆盖弱时接近 0，覆盖充分时接近 1

最终回波：

$$
E(p)=(1-\gamma(p))I_{bkg}+\gamma(p)\frac{\sum_{i=1}^{n}I_i(d)w_i(p)}{S(p)+\epsilon} \tag{9}
$$

变量说明：
- $I_{bkg}$：背景强度（文中设为黑）
- $\epsilon$：数值稳定项
- 分母 $S(p)+\epsilon$：归一化，避免仅由高覆盖区域主导

## 4) Section 3.3: 初始化与优化策略

总损失：

$$
L=(1-\lambda_{ssim})L_1+\lambda_{ssim}L_{SSIM}+\lambda_{scale}\,\mathbb{E}[\exp(S)] \tag{10}
$$

其中：
- $\lambda_{ssim}=0.5$
- $\lambda_{scale}=10^{-3}$

变量说明：
- $L_1$：像素重建误差
- $L_{SSIM}$：结构相似度损失
- $\mathbb{E}[\exp(S)]$：尺度相关正则（抑制过大/无效高斯）

Refinement（ultrasound-specific densification）：
- 每 2500 iter 触发一次
- 从 1k 到 20k iter 启用
- 用累计梯度做 importance score
- 小尺度高斯做 duplicate，大尺度做 split
- prune 条件：$s < 5\times10^{-5}$ 或 $s > 5$ mm
- 高斯总数上限：$N_{max}=500k$

## 5) Section 3.4 + Appendix A: 实现与超参数（复现关键）

### 实现框架
- PyTorch + gsplat（CUDA 加速）
- 自定义 CUDA kernel：高斯筛选 + 射线-高斯方程求值

### 参数初始化（明确）
- 高斯位置：在局部场景随机初始化（无 SfM）
- 透射参数：$\tau=0.99$
- 尺度：各向同性初值 $0.5$ mm
- SH：从 0 阶开始（仅 DC）

### 训练设置
- 优化器：Adam
- 训练总步数：30k（文中称 epochs，但 refinement 语义按 iter）
- Batch size：8

### 分参数学习率
- means：$1\times10^{-4}$
- scales + quaternions：$5\times10^{-3}$
- transmittance：$5\times10^{-4}$
- SH DC：$5\times10^{-3}$
- SH 非 DC：$1\times10^{-5}$

### 学习率调度与 SH 调度
- 全部学习率平滑衰减到初值的 10%
- SH 在 1000 iter 后由 0 阶提升到 1 阶

### OPS（Out-of-plane sampling）
- 对每条 ray 的起点加离面扰动（cosine-weighted）
- 最大离面幅值：$\delta=2$ mm
- 作用：近似 elevational beam width，缓解高斯向单切片塌缩

# Experiment

## 1) Section 4.1 数据与采集设置（逐小节映射）

两套 pose-annotated B-mode 数据：
- in-silico spine phantom
- ex-vivo porcine muscle phantom

采集硬件/流程：
- Siemens 12L3 探头 + Siemens Acuson Juniper
- KUKA LBR iiwa 14 R820 机械臂
- ImFusion 记录图像与位姿
- 先做基于几何与机械臂末端偏移的粗标定，再用 BOBYQA 精化

数据规模（主文版本）：
- Porcine：约 $5\times5\times5$ cm$^3$，3 条 sweep（$-3^\circ,0^\circ,+3^\circ$），另有更大离面角用于附录泛化实验
- Spine：约 $13\times5\times9$ cm$^3$，多条 sweep（含不同倾角），并用额外倾角做评估

## 2) Section 4.2 对比方法与指标（逐小节映射）

对比对象：
- Ultra-NeRF
- ImplicitVol（终版中加入）
- 传统体素重建重切片 baseline（median/max compounding，终版中加入）

指标：
- PSNR
- MS-SSIM
- GMS
- GMSD（越低越好）

作者强调：仅 PSNR/MS-SSIM 可能偏向平滑结果，因此加入梯度相关指标评估细节保真。

## 3) Section 4.3 主要结果与结论（逐小节映射）

- UltraG-Ray 在两套数据上总体最优或并列最优，尤其在 MS-SSIM/GMS/GMSD 上体现了更好的纹理与结构恢复。
- 与 w/o Transmittance 的消融相比，完整模型在 shadowing 与结构一致性上明显更好，证明了将 $T(p)$ 与 $E(p)$ 分离建模的必要性。
- OPS 与 SH 模块均有贡献；去掉任一模块，至少一个数据集性能下降。
- 推理速度高（文中报告 spine 可达约 680 fps，porcine 约 95 fps），并且显存占用较低。

## 4) 对你原笔记的校正与补充

- 你原文里的 “2500 次触发、1k 到 20k、500k 上限、delta=2mm、batch=8、loss 权重与学习率” 均与论文一致，可保留。
- “epoch 还是 iter” 的疑问：论文文字混用较多，但 refinement 逻辑实质是按 iteration 间隔执行。
- “第一次 refinement 数量骤降再回升” 与附录优化曲线解释一致：初始随机高斯先被 prune，后续通过 split/duplicate 回补细节。