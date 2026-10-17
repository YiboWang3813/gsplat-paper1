#include "covariance.cuh" // 包含了你提供的 rotate/scale math
#include "shared.cuh"     // 包含了 compute_gaussian_weight
#include "load_utils.cuh"
#include "math_utils.cuh"


// =========================================================
// Main Kernel: Render B-Mode
// =========================================================
__global__ void render_bmode_kernel(
    // Grid: (ImageWidth, ImageHeight)
    const int W, const int H,
    
    // CSR Inputs (from Mapper)
    const int* __restrict__ ray_counts,   // (W)
    const int* __restrict__ ray_offsets,  // (W)
    const int* __restrict__ ray_indices,  // (TotalHits) Sorted by depth

    // Geometry Inputs
    const float4* __restrict__ means,     // (G, 3)
    const float4* __restrict__ scales,    // (G, 3)
    const float4* __restrict__ quats,     // (G, 4)
    
    // Physics Parameters (Learnable)
    const float* __restrict__ attenuations, // (G) Range [0, 1] usually passed through sigmoid
    const float* __restrict__ intensities,  // (G) Echo strength

    // Scene Inputs
    const float* __restrict__ pixel_positions, // (W, H, 3) Flat
    const float* __restrict__ ray_origins,     // (W, 3)
    const float* __restrict__ ray_dirs,        // (W, 3) Normalized

    // Output
    float* __restrict__ output_image           // (W, H)
) {
    // 1. Identify Pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Width (Ray Index)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Height (Depth Index)

    if (x >= W || y >= H) return;

    // 2. Get Ray Data
    int count = ray_counts[x];
    if (count == 0) {
        output_image[y * W + x] = 0.0f; // No gaussians
        return;
    }
    
    int offset = ray_offsets[x];
    
    // Get Pixel 3D Position
    // pixel_positions is (W, H, 3) => index (x * H + y) * 3
    const float* p_ptr = pixel_positions + (x * H + y) * 3;
    float3 p = make_float3(p_ptr[0], p_ptr[1], p_ptr[2]);

    float3 ray_o = make_float3(ray_origins[x*3+0], ray_origins[x*3+1], ray_origins[x*3+2]);
    float3 ray_d = make_float3(ray_dirs[x*3+0], ray_dirs[x*3+1], ray_dirs[x*3+2]);

    // Calculate projection of current pixel on the ray (t_pixel)
    // We assume the pixel lies roughly on the ray. t_pixel = dot(p - o, d)
    float t_pixel = dot_prod(p - ray_o, ray_d);

    // 3. Accumulators
    float transmission = 1.0f;
    float echo_numerator = 0.0f;
    float echo_denominator = 0.0f; // For soft coverage normalization
    float soft_coverage_accum = 0.0f;

    // 4. Loop over Gaussians on this Ray
    for (int k = 0; k < count; ++k) {
        int idx = ray_indices[offset + k];

        // Load Gaussian Data
        float3 mu = xyz(means[idx]);
        float3 s = xyz(scales[idx]);
        float4 q = quats[idx];
        
        // Compute Rotation & Covariance
        float R[9];
        compute_rot_from_quat(q, R);
        
        float SigmaInv[9];
        compute_cov3d_from_scale_rot(s, R, SigmaInv);

        // --- Physics A: Attenuation (Transmittance) ---
        // Logic: Only Gaussians *in front* of the pixel contribute to attenuation.
        // We use the projected center to determine order.
        float t_gauss = dot_prod(mu - ray_o, ray_d);
        
        // Eq. 8: T_i = tau + (1 - tau) * exp(-psi)
        // Note: attenuations[] input should already be sigmoided in Python or here.
        // Let's assume input is raw logits, so we apply sigmoid.
        float raw_tau = attenuations[idx];
        float tau = 1.0f / (1.0f + __expf(-raw_tau)); 

        if (t_gauss < t_pixel) {
            float psi = compute_ray_gaussian_integral(ray_o, ray_d, mu, s, R);
            // Beer-Lambert discrete approximation
            float T_i = tau + (1.0f - tau) * __expf(-psi);
            transmission *= T_i;
        }

        // --- Physics B: Echo (Reflection) ---
        // Eq. 2 & 9: Mahalanobis weight at point P
        float w = compute_gaussian_weight(p, mu, SigmaInv);
        
        // Accumulate Echo
        // Assuming intensities[] is raw, might need activation (e.g., exp or softplus)
        // Paper says "I_i(d)", let's assume intensities is the pre-computed value
        float I = intensities[idx]; 
        
        echo_numerator += I * w;
        echo_denominator += w;
        soft_coverage_accum += w;
    }

    // 5. Final Composition
    // Eq. 9: Soft coverage gamma = 1 - exp(-S(p))
    float gamma = 1.0f - __expf(-soft_coverage_accum);
    
    // Avoid div by zero
    float epsilon = 1e-6f;
    float echo_signal = (echo_numerator) / (echo_denominator + epsilon);
    
    // Combine with background (assuming I_bkg = 0 for now)
    float final_echo = gamma * echo_signal;

    // Eq. 5: B(p) = T(p) * E(p)
    float b_mode_val = transmission * final_echo;

    output_image[y * W + x] = b_mode_val;
}