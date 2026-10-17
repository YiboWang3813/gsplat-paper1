#include "core.cuh"

// ==========================================
// Kernel 1: Geometry Projection & Culling
// ==========================================
// 负责：
// 1. 将高斯从世界坐标系转换到探头(Probe)坐标系。
// 2. 应用外部传入的随机扰动(Perturbation)模拟切片厚度。
// 3. 计算高斯在图像 Lateral 轴上的覆盖范围(Range)。
__global__ void kernel_project_and_cull(
    const int n_gauss,
    const int image_width,
    const float pixel_spacing_mm, // 物理宽度 / 图像宽度
    const float near_clip_mm,     // 近平面裁剪距离
    const float elev_cutoff_mm,   // 切片厚度阈值
    const float k_sigma,          // 高斯包围盒倍率 (通常为3.0)
    
    // Data Inputs
    const float4* __restrict__ means_world,     // [N] 高斯中心 (World)
    const float4* __restrict__ scales_world,    // [N] 高斯缩放
    const float* __restrict__ elev_perturbations, // [N] 预计算的 Elevation 轴随机扰动

    // Probe Geometry (Ptrs to avoid constant memory limits)
    const float* __restrict__ probe_origin,
    const float* __restrict__ axis_lateral,
    const float* __restrict__ axis_elevational,
    const float* __restrict__ axis_axial,

    // Outputs
    int* num_rays_covered, // [N] 该高斯覆盖了多少条 Ray (用于 Scan)
    int2* ray_minmax,       // [N] {min_ray_idx, max_ray_idx}
    float* depth_probe       // [N] 高斯在探头坐标系下的深度 Z
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gauss) return;

    // --- 1. Load Data & Geometry ---
    float3 origin   = make_float3(probe_origin[0], probe_origin[1], probe_origin[2]);
    float3 axis_u   = make_float3(axis_lateral[0], axis_lateral[1], axis_lateral[2]);
    float3 axis_v   = make_float3(axis_elevational[0], axis_elevational[1], axis_elevational[2]);
    float3 axis_w   = make_float3(axis_axial[0], axis_axial[1], axis_axial[2]);

    float3 p_world = xyz(means_world[idx]);
    float3 s_world = xyz(scales_world[idx]);

    // --- 2. World -> Probe Transformation ---
    float3 diff = p_world - origin;
    float3 p_local;
    p_local.x = dot_prod(diff, axis_u); // Lateral (Image X)
    p_local.y = dot_prod(diff, axis_v); // Elevational (Slice Thickness)
    p_local.z = dot_prod(diff, axis_w); // Axial (Depth)

    // --- 3. Conservative Radius ---
    // 使用最大轴缩放作为保守包围球半径
    float max_scale = fmaxf(s_world.x, fmaxf(s_world.y, s_world.z));
    float radius_safe = max_scale * k_sigma;

    // --- 4. Culling Logic ---

    // [A] Depth Culling (Axial)
    // 即使中心在 near_clip 前面，只要球体有一部分进入视野即可，所以用 z + r 判断
    if (p_local.z + radius_safe < near_clip_mm) {
        num_rays_covered[idx] = 0;
        return;
    }

    // [B] Elevation Culling (Slice Thickness)
    // 关键修改：直接读取 Python 传来的扰动值
    float perturb_offset = elev_perturbations[idx]; 
    float dist_ele = fabsf(p_local.y - perturb_offset);
    
    // 如果高斯球距离"扰动后的切片平面"过远，则剔除
    if (dist_ele > (radius_safe + elev_cutoff_mm)) {
        num_rays_covered[idx] = 0;
        return;
    }

    // --- 5. Compute Ray Range (Lateral) ---
    // 将物理坐标映射到像素索引: [-W/2, W/2] -> [0, W]
    float center_pixel = (p_local.x / pixel_spacing_mm) + ((float)image_width * 0.5f);
    float radius_pixel = radius_safe / pixel_spacing_mm;

    int min_ray = (int)floorf(center_pixel - radius_pixel);
    int max_ray = (int)ceilf(center_pixel + radius_pixel);

    // Clamp to image boundaries
    min_ray = max(0, min_ray);
    max_ray = min(image_width - 1, max_ray);

    if (min_ray > max_ray) {
        num_rays_covered[idx] = 0;
        return;
    }

    // --- 6. Write Output ---
    num_rays_covered[idx] = (max_ray - min_ray + 1);
    ray_minmax[idx] = make_int2(min_ray, max_ray);
    depth_probe[idx] = p_local.z;
}

// ==========================================
// Kernel 2: Generate Sort Keys
// ==========================================
// 负责：生成 (Ray_ID, Depth) 的 Key-Value 对，用于后续排序。
__global__ void kernel_generate_sort_keys(
    const int n_gauss,
    const int* __restrict__ csr_offsets, // Scan 后的偏移量
    const int2* __restrict__ ray_minmax,
    const float* __restrict__ depths,
    
    uint64_t* __restrict__ sort_keys,    // Output Keys: [RayID (32b) | Depth (32b)]
    int* __restrict__ sort_values        // Output Values: Gaussian Index
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gauss) return;

    int start_offset = csr_offsets[idx];
    int count = csr_offsets[idx+1] - start_offset;
    
    if (count <= 0) return;

    int2 range = ray_minmax[idx];
    float d = depths[idx];
    
    // 深度量化：保证非负且单调，用于基数排序
    // 假设深度精度保留到 0.001mm
    float d_clamped = fmaxf(0.0f, d); 
    uint32_t depth_quantized = (uint32_t)(d_clamped * 1000.0f); 

    int writer_head = start_offset;
    for (int r = range.x; r <= range.y; ++r) {
        // Construct 64-bit Key
        // High 32 bits: Ray Index (保证按 Scanline 分组)
        // Low  32 bits: Depth (保证同 Scanline 内按深度排序)
        uint64_t key = ((uint64_t)r << 32) | depth_quantized;
        
        sort_keys[writer_head] = key;
        sort_values[writer_head] = idx; // 记录原始高斯 ID
        
        writer_head++;
    }
}

// ==========================================
// Kernel 3: Build CSR Metadata
// ==========================================
// 负责：分析排序后的 Keys，找出每条 Ray 在数组中的 Start/End 索引。
__global__ void kernel_build_csr_structure(
    const int n_total_hits,
    const uint64_t* __restrict__ sorted_keys,
    int* __restrict__ ray_start_indices,
    int* __restrict__ ray_end_indices
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total_hits) return;

    uint64_t key = sorted_keys[idx];
    int cur_ray = (int)(key >> 32);

    // 检查边界变化
    bool is_segment_start = (idx == 0);
    if (idx > 0) {
        int prev_ray = (int)(sorted_keys[idx - 1] >> 32);
        if (cur_ray != prev_ray) {
            is_segment_start = true;
            ray_end_indices[prev_ray] = idx; // 记录上一段的结束点
        }
    }

    if (is_segment_start) {
        ray_start_indices[cur_ray] = idx;
    }

    // 处理最后一个元素的结束点
    if (idx == n_total_hits - 1) {
        ray_end_indices[cur_ray] = n_total_hits;
    }
}