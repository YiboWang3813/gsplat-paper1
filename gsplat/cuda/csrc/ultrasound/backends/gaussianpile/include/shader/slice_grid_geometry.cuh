#pragma once

/** Device-readable pack built from ``DatasetMeta`` / ``LinearSliceMeta`` (see ``grid_geometry.py``). */
#define SLICE_GRID_PACK_LEN 16

#define SG_PROBE_TYPE 0
#define SG_H 1
#define SG_W 2
#define SG_DELTA_LAT_MM 3
#define SG_DELTA_AX_MM 4
#define SG_NEAR_MM 5
#define SG_LAT_COL_OFFSET 6
#define SG_CENTER_LATERAL 7
/** Convex extension (future): radius_mm, fov_deg, depth step stored here. */
#define SG_RADIUS_MM 8
#define SG_FOV_DEG 9

#define SLICE_GRID_PROBE_LINEAR 0
#define SLICE_GRID_PROBE_CONVEX 1

/** Shader weight modes (must match Python / autograd). */
#define SHADER_WEIGHT_PROBE3D 0
#define SHADER_WEIGHT_PLANE_COND2D 1
#define SHADER_WEIGHT_US_FOCUS 2

__device__ __forceinline__ int sg_probe_type(const float* pack) {
    return static_cast<int>(pack[SG_PROBE_TYPE] + 0.5f);
}

/**
 * Map image indices (col=x, row=y) to probe-plane mm coordinates aligned with ``LinearSliceMeta``:
 *   local_x = (col - lat_col_offset) * delta_lat   (lateral)
 *   local_y = row * delta_ax + near_mm             (axial)
 * elev = 0 on the slice plane.
 */
__device__ __forceinline__ void sg_linear_pixel_to_plane_mm(
    const float* pack,
    int col,
    int row,
    float* xi_lat_mm,
    float* xi_ax_mm) {
    const float w_off = pack[SG_LAT_COL_OFFSET];
    *xi_lat_mm = (static_cast<float>(col) - w_off) * pack[SG_DELTA_LAT_MM];
    *xi_ax_mm = static_cast<float>(row) * pack[SG_DELTA_AX_MM] + pack[SG_NEAR_MM];
}
