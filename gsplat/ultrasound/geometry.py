"""Physical acquisition geometry shared by native ultrasound rasterizers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DatasetGeometry:
    H: int
    W: int
    delta_lateral_mm: float
    delta_axial_mm: float
    delta_elevational_mm: float
    near_mm: float = 0.0
    center_lateral: bool = True
    probe_type: str = "linear"

    @property
    def width_mm(self):
        return self.W * self.delta_lateral_mm

    @property
    def far_mm(self):
        return self.near_mm + self.H * self.delta_axial_mm

    @classmethod
    def from_metadata(cls, g):
        return cls(
            g["height"],
            g["width"],
            g["lateral_spacing_mm"],
            g["axial_spacing_mm"],
            g["elevational_spacing_mm"],
            g["near_mm"],
            g["center_lateral"],
        )


class PhysicalSlice:
    def __init__(self, geometry, pose, image=None, frame_id=0):
        self.dataset_meta, self.pose, self.frame_id = geometry, pose, frame_id
        self.image = (
            image
            if image is not None
            else torch.zeros(geometry.H, geometry.W, device=pose.device)
        )
        self.probe_origin = pose[:3, 3]
        axes = torch.nn.functional.normalize(pose[:3, :3], dim=0)
        self.vec_lateral, self.vec_axial, self.vec_elevational = axes.unbind(-1)
        y, x = torch.meshgrid(
            torch.arange(geometry.H, device=pose.device, dtype=pose.dtype),
            torch.arange(geometry.W, device=pose.device, dtype=pose.dtype),
            indexing="ij",
        )
        x = (
            x - ((geometry.W - 1) / 2 if geometry.center_lateral else 0)
        ) * geometry.delta_lateral_mm
        y = y * geometry.delta_axial_mm + geometry.near_mm
        local = torch.stack([x, y, torch.zeros_like(x)], -1)
        self.pixel_pos = (local @ pose[:3, :3].T + pose[:3, 3]).reshape(-1, 3)
        origin = (
            torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], -1)
            @ pose[:3, :3].T
            + pose[:3, 3]
        )
        self.grid_ray_origins = origin
        delta = self.pixel_pos.reshape(geometry.H, geometry.W, 3) - origin
        self.grid_t_vals = delta.norm(dim=-1)
        self.grid_ray_dirs = torch.where(
            (self.grid_t_vals < 1e-6)[..., None],
            self.vec_axial,
            delta / self.grid_t_vals[..., None].clamp_min(1e-8),
        )
        self.ray_origins, self.ray_dirs = (
            origin[0].contiguous(),
            self.grid_ray_dirs[0].contiguous(),
        )
