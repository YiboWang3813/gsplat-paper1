"""Build CUDA ``slice_grid_geometry`` pack aligned with ``LinearSliceMeta`` / ``DatasetMeta``."""

from __future__ import annotations

import torch

# Must match ``include/shader/slice_grid_geometry.cuh``.
SLICE_GRID_PACK_LEN = 16
SG_PROBE_TYPE = 0
SG_H = 1
SG_W = 2
SG_DELTA_LAT_MM = 3
SG_DELTA_AX_MM = 4
SG_NEAR_MM = 5
SG_LAT_COL_OFFSET = 6
SG_CENTER_LATERAL = 7
SG_RADIUS_MM = 8
SG_FOV_DEG = 9

SLICE_GRID_PROBE_LINEAR = 0
SLICE_GRID_PROBE_CONVEX = 1

SHADER_WEIGHT_PROBE3D = 0
SHADER_WEIGHT_PLANE_COND2D = 1
SHADER_WEIGHT_US_FOCUS = 2


def shader_weight_mode_id(mode: str) -> int:
    if mode == "probe3d":
        return SHADER_WEIGHT_PROBE3D
    if mode == "plane_cond2d":
        return SHADER_WEIGHT_PLANE_COND2D
    if mode == "us_focus":
        return SHADER_WEIGHT_US_FOCUS
    raise ValueError(f"shader_weight_mode must be 'probe3d', 'plane_cond2d', or 'us_focus', got {mode!r}.")


def build_slice_grid_pack(dataset_meta, device=None):
    """Pack slice pixel geometry for shader (linear array; convex reserved)."""
    if device is None:
        device = torch.device("cpu")
    probe_type = getattr(dataset_meta, "probe_type", "linear")
    if probe_type not in ("linear", None):
        if str(probe_type).lower() == "convex":
            raise NotImplementedError(
                "plane_cond2d / slice_grid_pack for convex probes is not implemented yet; use probe3d."
            )

    H = int(dataset_meta.H)
    W = int(dataset_meta.W)
    delta_lat = float(dataset_meta.delta_lateral_mm)
    delta_ax = float(dataset_meta.delta_axial_mm)
    near_mm = float(dataset_meta.near_mm)
    center_lateral = bool(getattr(dataset_meta, "center_lateral", True))
    w_offset = (W - 1) * 0.5 if center_lateral else 0.0

    pack = torch.zeros((SLICE_GRID_PACK_LEN,), dtype=torch.float32, device=device)
    pack[SG_PROBE_TYPE] = float(SLICE_GRID_PROBE_LINEAR)
    pack[SG_H] = float(H)
    pack[SG_W] = float(W)
    pack[SG_DELTA_LAT_MM] = delta_lat
    pack[SG_DELTA_AX_MM] = delta_ax
    pack[SG_NEAR_MM] = near_mm
    pack[SG_LAT_COL_OFFSET] = float(w_offset)
    pack[SG_CENTER_LATERAL] = 1.0 if center_lateral else 0.0
    pack[SG_RADIUS_MM] = float(getattr(dataset_meta, "radius_mm", 0.0))
    pack[SG_FOV_DEG] = float(getattr(dataset_meta, "fov_deg", 0.0))
    return pack
