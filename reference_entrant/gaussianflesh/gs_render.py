"""
gs_render.py
Thin wrapper around diff_gaussian_rasterization for our MPM substrate.

We feed positions + cov3D_precomp (6-element upper triangle of Σ) + opacity + SH
directly to the rasterizer.  No scale/rotation eigendecomposition needed  -  Σ is
already the live deformed covariance from the MPM solver.

Public API:
    GSRenderer(opacities, shs, bg=(1,1,1), sh_degree=3)
    GSRenderer.render(positions, covs, eye, target, up, fov_y_deg, width, height) -> (H, W, 3) uint8

Conventions match PhysGaussian / 3DGS upstream: camera looks along its local +Z
axis (COLMAP-style).  We build R, T from a Pythonic look-at the same way.
"""
from __future__ import annotations
import math
import numpy as np
import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def _build_proj_matrix(znear: float, zfar: float, fov_x: float, fov_y: float) -> torch.Tensor:
    tanHalfFovY = math.tan(fov_y * 0.5)
    tanHalfFovX = math.tan(fov_x * 0.5)
    top    =  tanHalfFovY * znear
    right  =  tanHalfFovX * znear
    bottom = -top
    left   = -right
    P = torch.zeros(4, 4, dtype=torch.float32)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def _look_at_world2view(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> torch.Tensor:
    """COLMAP-convention world→view matrix from a look-at triple.
    Camera looks along its local +Z axis."""
    eye    = np.asarray(eye,    dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up     = np.asarray(up,     dtype=np.float64)
    fwd    = target - eye
    fwd   /= np.linalg.norm(fwd) + 1e-12
    right  = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-12
    up_c   = np.cross(right, fwd)
    up_c  /= np.linalg.norm(up_c) + 1e-12
    # Camera basis in world coords: x=right, y=-up (image y-down), z=fwd
    R_w2c = np.stack([right, -up_c, fwd], axis=0)   # rows
    t_w2c = -R_w2c @ eye
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R_w2c.astype(np.float32)
    Rt[:3,  3] = t_w2c.astype(np.float32)
    return torch.from_numpy(Rt)


def _cov3x3_to_upper(covs: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) → (N, 6) upper-triangle (xx, xy, xz, yy, yz, zz)  -  rasterizer convention."""
    return torch.stack([
        covs[:, 0, 0], covs[:, 0, 1], covs[:, 0, 2],
        covs[:, 1, 1], covs[:, 1, 2], covs[:, 2, 2],
    ], dim=1).contiguous()


class GSRenderer:
    def __init__(self, opacities: np.ndarray, shs: np.ndarray,
                 bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 sh_degree: int = 3,
                 device: str = "cuda"):
        self.device = device
        self.sh_degree = sh_degree
        self.bg = torch.tensor(bg, dtype=torch.float32, device=device)
        # opacities: (N, 1), shs: (N, 16, 3)
        self.opacities = torch.from_numpy(opacities.astype(np.float32)).to(device)
        self.shs       = torch.from_numpy(shs.astype(np.float32)).to(device).contiguous()
        self.N         = self.opacities.shape[0]

    @torch.no_grad()
    def render(self, positions: np.ndarray, covs: np.ndarray,
               eye: np.ndarray, target: np.ndarray, up: np.ndarray,
               fov_y_deg: float, width: int, height: int,
               znear: float = 0.01, zfar: float = 100.0) -> np.ndarray:
        aspect = width / float(height)
        fov_y  = math.radians(fov_y_deg)
        fov_x  = 2.0 * math.atan(math.tan(fov_y * 0.5) * aspect)

        world_view = _look_at_world2view(eye, target, up).to(self.device)
        proj_only  = _build_proj_matrix(znear, zfar, fov_x, fov_y).to(self.device)
        # PhysGaussian transposes both before passing to the CUDA layer.
        wv_T   = world_view.transpose(0, 1).contiguous()
        proj_T = proj_only.transpose(0, 1).contiguous()
        full_proj = (wv_T.unsqueeze(0).bmm(proj_T.unsqueeze(0))).squeeze(0).contiguous()
        cam_pos = torch.linalg.inv(wv_T)[3, :3].contiguous()

        settings = GaussianRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=math.tan(fov_x * 0.5),
            tanfovy=math.tan(fov_y * 0.5),
            bg=self.bg,
            scale_modifier=1.0,
            viewmatrix=wv_T,
            projmatrix=full_proj,
            sh_degree=self.sh_degree,
            campos=cam_pos,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)

        # Accept either numpy arrays (CPU bounce) or torch tensors (GPU-direct).
        if isinstance(positions, torch.Tensor):
            pos_t = positions.to(self.device, dtype=torch.float32).contiguous()
        else:
            pos_t = torch.from_numpy(positions.astype(np.float32)).to(self.device).contiguous()
        if isinstance(covs, torch.Tensor):
            cov_t = covs.to(self.device, dtype=torch.float32)
        else:
            cov_t = torch.from_numpy(covs.astype(np.float32)).to(self.device)
        cov6   = _cov3x3_to_upper(cov_t)
        # Screen-space placeholder for gradient (we don't actually backprop)
        means2D = torch.zeros_like(pos_t, requires_grad=False)

        image, _radii = rasterizer(
            means3D       = pos_t,
            means2D       = means2D,
            shs           = self.shs,
            colors_precomp= None,
            opacities     = self.opacities,
            scales        = None,
            rotations     = None,
            cov3D_precomp = cov6,
        )
        # image: (3, H, W) float in [0, 1]
        rgb = image.clamp(0.0, 1.0).permute(1, 2, 0).contiguous().cpu().numpy()
        return (rgb * 255.0).astype(np.uint8)


if __name__ == "__main__":
    # Smoke test: render 1000 random Gaussians as a sanity check.
    N = 1000
    rng = np.random.default_rng(0)
    pos = rng.normal(0, 0.3, (N, 3)).astype(np.float32)
    # Tiny isotropic covariances
    covs = np.zeros((N, 3, 3), dtype=np.float32)
    s = 0.01
    covs[:, 0, 0] = s * s; covs[:, 1, 1] = s * s; covs[:, 2, 2] = s * s
    opac = np.full((N, 1), 0.9, dtype=np.float32)
    shs  = np.zeros((N, 16, 3), dtype=np.float32)
    # DC term controls base color; SH DC = (color - 0.5) / 0.28209 roughly,
    # so 0.5 gives mid-grey, 2.0 gives white-ish
    shs[:, 0, :] = rng.uniform(-1, 1, (N, 3))

    renderer = GSRenderer(opac, shs, bg=(0.0, 0.0, 0.0))
    img = renderer.render(
        positions=pos, covs=covs,
        eye=np.array([0, 0, 3]), target=np.array([0, 0, 0]), up=np.array([0, 1, 0]),
        fov_y_deg=60.0, width=512, height=512,
    )
    print("rendered image shape:", img.shape, "dtype:", img.dtype, "mean:", img.mean())
