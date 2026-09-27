"""Benchmark-only bridge to PhysGaussian's vendored 3DGS rasterizer.

The production PhysGaussian renderer passes live particle positions and
precomputed deformed covariances to ``diff_gaussian_rasterization``.  C3 uses
the same path for its procedural Gaussian asset; this module only constructs
the fixed look-at camera required by the benchmark.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "gaussian-splatting"))

from diff_gaussian_rasterization import (  # noqa: E402
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def _projection(znear: float, zfar: float, fov_x: float,
                fov_y: float) -> torch.Tensor:
    tan_y = math.tan(fov_y * 0.5)
    tan_x = math.tan(fov_x * 0.5)
    out = torch.zeros(4, 4, dtype=torch.float32)
    out[0, 0] = 1.0 / tan_x
    out[1, 1] = 1.0 / tan_y
    out[3, 2] = 1.0
    out[2, 2] = zfar / (zfar - znear)
    out[2, 3] = -(zfar * znear) / (zfar - znear)
    return out


def _world_to_view(eye: np.ndarray, target: np.ndarray,
                   up: np.ndarray) -> torch.Tensor:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    camera_up = np.cross(right, forward)
    camera_up /= np.linalg.norm(camera_up) + 1e-12
    rotation = np.stack([right, -camera_up, forward], axis=0)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.astype(np.float32)
    transform[:3, 3] = (-rotation @ eye).astype(np.float32)
    return torch.from_numpy(transform)


def _covariance_upper(covariance: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            covariance[:, 0, 0],
            covariance[:, 0, 1],
            covariance[:, 0, 2],
            covariance[:, 1, 1],
            covariance[:, 1, 2],
            covariance[:, 2, 2],
        ],
        dim=1,
    ).contiguous()


class PhysGaussianRenderer:
    """Render positions and precomputed covariances through PhysGaussian."""

    def __init__(self, opacity: np.ndarray, sh: np.ndarray,
                 background=(1.0, 1.0, 1.0), device="cuda"):
        self.device = device
        self.opacity = torch.as_tensor(
            opacity, dtype=torch.float32, device=device).contiguous()
        self.sh = torch.as_tensor(
            sh, dtype=torch.float32, device=device).contiguous()
        self.background = torch.tensor(
            background, dtype=torch.float32, device=device)

    @torch.no_grad()
    def render(self, positions: np.ndarray, covariances: np.ndarray,
               eye: np.ndarray, target: np.ndarray, up: np.ndarray,
               fov_y_degrees: float = 45.0, width: int = 512,
               height: int = 512) -> np.ndarray:
        aspect = width / float(height)
        fov_y = math.radians(fov_y_degrees)
        fov_x = 2.0 * math.atan(math.tan(fov_y * 0.5) * aspect)
        world_view = _world_to_view(eye, target, up).to(self.device)
        projection = _projection(0.01, 100.0, fov_x, fov_y).to(self.device)
        view_t = world_view.transpose(0, 1).contiguous()
        projection_t = projection.transpose(0, 1).contiguous()
        full_projection = (
            view_t.unsqueeze(0).bmm(projection_t.unsqueeze(0))
        ).squeeze(0).contiguous()
        camera_position = torch.linalg.inv(view_t)[3, :3].contiguous()

        settings = GaussianRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=math.tan(fov_x * 0.5),
            tanfovy=math.tan(fov_y * 0.5),
            bg=self.background,
            scale_modifier=1.0,
            viewmatrix=view_t,
            projmatrix=full_projection,
            sh_degree=3,
            campos=camera_position,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)
        position = torch.as_tensor(
            positions, dtype=torch.float32, device=self.device).contiguous()
        covariance = torch.as_tensor(
            covariances, dtype=torch.float32,
            device=self.device).contiguous()
        image, _ = rasterizer(
            means3D=position,
            means2D=torch.zeros_like(position),
            shs=self.sh,
            colors_precomp=None,
            opacities=self.opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=_covariance_upper(covariance),
        )
        rgb = image.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        return (rgb * 255.0).astype(np.uint8)
