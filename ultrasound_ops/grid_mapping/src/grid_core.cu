#include "grid_core.cuh"


__global__ void kernel_count_gauss_per_tile(
    int n_gauss,
    const float* __restrict__ means,
    const float* __restrict__ grid_min,
    const float* __restrict__ tile_size,
    const int*   __restrict__ grid_size,
    int* __restrict__ tile_counts
)
{
    int g = threadIdx.x + blockDim.x * blockIdx.x;
    if (g >= n_gauss) return;

    float mx = means[3 * g + 0];
    float my = means[3 * g + 1];
    float mz = means[3 * g + 2];

    float grid_min_x = grid_min[0];
    float grid_min_y = grid_min[1];
    float grid_min_z = grid_min[2];

    float tile_size_x = tile_size[0];
    float tile_size_y = tile_size[1];
    float tile_size_z = tile_size[2];

    int grid_size_x = grid_size[0];
    int grid_size_y = grid_size[1];
    int grid_size_z = grid_size[2];

    int tx = compute_tile(mx, grid_min_x, tile_size_x, grid_size_x);
    int ty = compute_tile(my, grid_min_y, tile_size_y, grid_size_y);
    int tz = compute_tile(mz, grid_min_z, tile_size_z, grid_size_z);

    if (tx < 0 || tx >= grid_size_x ||
        ty < 0 || ty >= grid_size_y ||
        tz < 0 || tz >= grid_size_z) return;

    int tile_id = tx + ty * grid_size_x + tz * grid_size_x * grid_size_y;
    atomicAdd(&tile_counts[tile_id], 1);
}

__global__ void kernel_fill_tile_indices(
    int n_gauss,
    const float* __restrict__ means,
    const float* __restrict__ grid_min,
    const float* __restrict__ tile_size,
    const int*   __restrict__ grid_size,
    const int* __restrict__ tile_offsets,
    int* __restrict__ tile_counters,
    int* __restrict__ tile_indices
)
{
    int g = threadIdx.x + blockDim.x * blockIdx.x;
    if (g >= n_gauss) return;

    float mx = means[3 * g + 0];
    float my = means[3 * g + 1];
    float mz = means[3 * g + 2];

    float grid_min_x = grid_min[0];
    float grid_min_y = grid_min[1];
    float grid_min_z = grid_min[2];

    float tile_size_x = tile_size[0];
    float tile_size_y = tile_size[1];
    float tile_size_z = tile_size[2];

    int grid_size_x = grid_size[0];
    int grid_size_y = grid_size[1];
    int grid_size_z = grid_size[2];

    int tx = compute_tile(mx, grid_min_x, tile_size_x, grid_size_x);
    int ty = compute_tile(my, grid_min_y, tile_size_y, grid_size_y);
    int tz = compute_tile(mz, grid_min_z, tile_size_z, grid_size_z);

    if (tx < 0 || tx >= grid_size_x ||
        ty < 0 || ty >= grid_size_y ||
        tz < 0 || tz >= grid_size_z) return;

    int tile_id = tx + ty * grid_size_x + tz * grid_size_x * grid_size_y;

    int pos = atomicAdd(&tile_counters[tile_id], 1);
    tile_indices[tile_offsets[tile_id] + pos] = g;
}

void count_gauss_per_tile(
    int n_gauss,
    const float* __restrict__ means,
    const float* __restrict__ grid_min,
    const float* __restrict__ tile_size,
    const int*   __restrict__ grid_size,
    int* __restrict__ tile_counts
)
{
    dim3 block(256);
    dim3 grid((n_gauss + block.x - 1) / block.x);
    kernel_count_gauss_per_tile<<<grid, block>>>(n_gauss, means, grid_min, tile_size, grid_size, tile_counts);
}

void fill_tile_indices(
    int n_gauss,
    const float* __restrict__ means,
    const float* __restrict__ grid_min,
    const float* __restrict__ tile_size,
    const int*   __restrict__ grid_size,
    const int* __restrict__ tile_offsets,
    int* __restrict__ tile_counters,
    int* __restrict__ tile_indices
)
{
    dim3 block(256);
    dim3 grid((n_gauss + block.x - 1) / block.x);
    kernel_fill_tile_indices<<<grid, block>>>(
        n_gauss, means, grid_min, tile_size, grid_size,
        tile_offsets, tile_counters, tile_indices
    );
}


__global__ void kernel_count_gauss_per_pixel(
    int n_pixels,
    const float* __restrict__ pixel_positions,
    const float* __restrict__ means,
    const float* __restrict__ sigma_invs,
    const float* __restrict__ grid_min,
    const float* __restrict__ tile_size,
    const int*   __restrict__ grid_size,
    const int* __restrict__ tile_counts,
    const int* __restrict__ tile_offsets,
    const int* __restrict__ tile_indices,
    int* __restrict__ pixel_tile_ids,
    int* __restrict__ pixel_counts
)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;

    const float grid_min_x = grid_min[0];
    const float grid_min_y = grid_min[1];
    const float grid_min_z = grid_min[2];

    const float tile_size_x = tile_size[0];
    const float tile_size_y = tile_size[1];
    const float tile_size_z = tile_size[2];

    const int grid_size_x = grid_size[0];
    const int grid_size_y = grid_size[1];
    const int grid_size_z = grid_size[2];

    const float px = pixel_positions[3 * p + 0];
    const float py = pixel_positions[3 * p + 1];
    const float pz = pixel_positions[3 * p + 2];

    const int tx = compute_tile(px, grid_min_x, tile_size_x, grid_size_x);
    const int ty = compute_tile(py, grid_min_y, tile_size_y, grid_size_y);
    const int tz = compute_tile(pz, grid_min_z, tile_size_z, grid_size_z);

    if (tx < 0 || tx >= grid_size_x ||
        ty < 0 || ty >= grid_size_y ||
        tz < 0 || tz >= grid_size_z)
    {
        pixel_tile_ids[p] = -1;
        pixel_counts[p]   = 0;
        return;
    }

    const int base_tile_id = tx + ty * grid_size_x + tz * grid_size_x * grid_size_y;
    pixel_tile_ids[p] = base_tile_id;

    int total = 0;

    #pragma unroll
    for (int dz = -ND; dz <= ND; ++dz)
    #pragma unroll
    for (int dy = -ND; dy <= ND; ++dy)
    #pragma unroll
    for (int dx = -ND; dx <= ND; ++dx)
    {
        int ntx = tx + dx;
        int nty = ty + dy;
        int ntz = tz + dz;

        if (ntx < 0 || ntx >= grid_size_x ||
            nty < 0 || nty >= grid_size_y ||
            ntz < 0 || ntz >= grid_size_z) continue;

        int tile_id = ntx + nty * grid_size_x + ntz * grid_size_x * grid_size_y;

        int count  = tile_counts[tile_id];
        int offset = tile_offsets[tile_id];

        #pragma unroll
        for (int i = 0; i < count; ++i)
        {
            int g = tile_indices[offset + i];

            float dxg = px - means[3 * g + 0];
            float dyg = py - means[3 * g + 1];
            float dzg = pz - means[3 * g + 2];

            const float* S = &sigma_invs[9 * g];

            float qx = S[0] * dxg + S[1] * dyg + S[2] * dzg;
            float qy = S[3] * dxg + S[4] * dyg + S[5] * dzg;
            float qz = S[6] * dxg + S[7] * dyg + S[8] * dzg;

            float e = -0.5f * (dxg * qx + dyg * qy + dzg * qz);
            if (e < E_CAND) continue;

            total++;
        }
    }

    pixel_counts[p] = total;
}

__global__ void kernel_fill_pixel_indices(
    int n_pixels,
    const float* __restrict__ pixel_positions,
    const float* __restrict__ means,
    const float* __restrict__ sigma_invs,
    const float* __restrict__ grid_min,
    const float* __restrict__ tile_size,
    const int*   __restrict__ grid_size,
    const int* __restrict__ tile_counts,
    const int* __restrict__ tile_offsets,
    const int* __restrict__ tile_indices,
    const int* __restrict__ pixel_offsets,
    int* __restrict__ pixel_indices
)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;

    const float grid_min_x = grid_min[0];
    const float grid_min_y = grid_min[1];
    const float grid_min_z = grid_min[2];

    const float tile_size_x = tile_size[0];
    const float tile_size_y = tile_size[1];
    const float tile_size_z = tile_size[2];

    const int grid_size_x = grid_size[0];
    const int grid_size_y = grid_size[1];
    const int grid_size_z = grid_size[2];

    const float px = pixel_positions[3 * p + 0];
    const float py = pixel_positions[3 * p + 1];
    const float pz = pixel_positions[3 * p + 2];

    const int tx = compute_tile(px, grid_min_x, tile_size_x, grid_size_x);
    const int ty = compute_tile(py, grid_min_y, tile_size_y, grid_size_y);
    const int tz = compute_tile(pz, grid_min_z, tile_size_z, grid_size_z);

    if (tx < 0 || tx >= grid_size_x ||
        ty < 0 || ty >= grid_size_y ||
        tz < 0 || tz >= grid_size_z) return;

    int write_ptr = pixel_offsets[p];

    #pragma unroll
    for (int dz = -ND; dz <= ND; ++dz)
    #pragma unroll
    for (int dy = -ND; dy <= ND; ++dy)
    #pragma unroll
    for (int dx = -ND; dx <= ND; ++dx)
    {
        int ntx = tx + dx;
        int nty = ty + dy;
        int ntz = tz + dz;

        if (ntx < 0 || ntx >= grid_size_x ||
            nty < 0 || nty >= grid_size_y ||
            ntz < 0 || ntz >= grid_size_z) continue;

        int tile_id = ntx + nty * grid_size_x + ntz * grid_size_x * grid_size_y;

        int count  = tile_counts[tile_id];
        int offset = tile_offsets[tile_id];

        #pragma unroll
        for (int i = 0; i < count; ++i)
        {
            int g = tile_indices[offset + i];

            float dxg = px - means[3 * g + 0];
            float dyg = py - means[3 * g + 1];
            float dzg = pz - means[3 * g + 2];

            const float* S = &sigma_invs[9 * g];

            float qx = S[0] * dxg + S[1] * dyg + S[2] * dzg;
            float qy = S[3] * dxg + S[4] * dyg + S[5] * dzg;
            float qz = S[6] * dxg + S[7] * dyg + S[8] * dzg;

            float e = -0.5f * (dxg * qx + dyg * qy + dzg * qz);
            if (e < E_CAND) continue;

            pixel_indices[write_ptr++] = g;
        }
    }
}

void count_gauss_per_pixel(
    int n_pixels,
    const float* __restrict__ pixel_positions,
    const float* __restrict__ means,
    const float* __restrict__ sigma_invs,
    const float* __restrict__ grid_min,
    const float* __restrict__ tile_size,
    const int*   __restrict__ grid_size,
    const int* __restrict__ tile_counts,
    const int* __restrict__ tile_offsets,
    const int* __restrict__ tile_indices,
    int* __restrict__ pixel_tile_ids,
    int* __restrict__ pixel_counts
)
{
    dim3 block(256);
    dim3 grid((n_pixels + block.x - 1) / block.x);
    kernel_count_gauss_per_pixel<<<grid, block>>>(
        n_pixels, pixel_positions, means, sigma_invs,
        grid_min, tile_size, grid_size,
        tile_counts, tile_offsets, tile_indices,
        pixel_tile_ids, pixel_counts
    );
}

void fill_pixel_indices(
    int n_pixels,
    const float* __restrict__ pixel_positions,
    const float* __restrict__ means,
    const float* __restrict__ sigma_invs,
    const float* __restrict__ grid_min,
    const float* __restrict__ tile_size,
    const int*   __restrict__ grid_size,
    const int* __restrict__ tile_counts,
    const int* __restrict__ tile_offsets,
    const int* __restrict__ tile_indices,
    const int* __restrict__ pixel_offsets,
    int* __restrict__ pixel_indices
)
{
    dim3 block(256);
    dim3 grid((n_pixels + block.x - 1) / block.x);
    kernel_fill_pixel_indices<<<grid, block>>>(
        n_pixels, pixel_positions, means, sigma_invs,
        grid_min, tile_size, grid_size,
        tile_counts, tile_offsets, tile_indices,
        pixel_offsets, pixel_indices
    );
}
