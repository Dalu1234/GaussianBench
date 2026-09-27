"""Shared helpers for scorer unit tests.

Scorer correctness is tested independently of any simulator: every test
constructs a synthetic trajectory with a KNOWN outcome and asserts the scorer
reproduces it. The benchmark is worthless if the scorers themselves are buggy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

SCENES_DIR = REPO / "scenes"


@pytest.fixture
def scenes_dir() -> Path:
    return SCENES_DIR


def load_scene_json(name: str) -> dict:
    with open(SCENES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def write_submission(out_dir: Path,
                     positions: np.ndarray,
                     times: np.ndarray,
                     velocities: np.ndarray | None = None,
                     masses: np.ndarray | None = None,
                     phase: np.ndarray | None = None,
                     deformation_gradients: np.ndarray | None = None,
                     meta: dict | None = None) -> Path:
    """Write a trajectory.parquet + meta.json submission directory.

    positions: (F, N, 3), times: (F,), velocities: (F, N, 3) or None,
    masses: (N,) or None, phase: (F, N) or None.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    F, N, _ = positions.shape
    frame = np.repeat(np.arange(F), N)
    pid = np.tile(np.arange(N), F)
    data = {
        "frame": frame,
        "time": np.repeat(times, N),
        "particle_id": pid,
        "x": positions[:, :, 0].ravel(),
        "y": positions[:, :, 1].ravel(),
        "z": positions[:, :, 2].ravel(),
    }
    cols = []
    if velocities is not None:
        data["vx"] = velocities[:, :, 0].ravel()
        data["vy"] = velocities[:, :, 1].ravel()
        data["vz"] = velocities[:, :, 2].ravel()
        cols += ["vx", "vy", "vz"]
    if masses is not None:
        data["mass"] = np.tile(masses, F)
        cols += ["mass"]
    if phase is not None:
        data["phase"] = phase.ravel().astype(np.int8)
        cols += ["phase"]
    if deformation_gradients is not None:
        if deformation_gradients.shape != (F, N, 3, 3):
            raise ValueError("deformation_gradients must have shape (F,N,3,3)")
        for i in range(3):
            for j in range(3):
                key = f"F{i}{j}"
                data[key] = deformation_gradients[:, :, i, j].ravel()
                cols.append(key)
    pd.DataFrame(data).to_parquet(out_dir / "trajectory.parquet", index=False)

    meta_out = {
        "system_name": "synthetic-test",
        "system_version": "0",
        "spec_version": "1.0.0",
        "columns_present": cols,
        "wall_clock_seconds": 0.0,
        "hardware": "none",
        "notes": "synthetic trajectory constructed by the scorer unit tests",
    }
    if meta:
        meta_out.update(meta)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    return out_dir


def write_covariances(out_dir: Path, frames: np.ndarray,
                      covs: np.ndarray) -> None:
    """Write covariances.parquet. frames: (F,) ints, covs: (F, N, 3, 3)."""
    F, N = covs.shape[0], covs.shape[1]
    rows = {
        "frame": np.repeat(np.asarray(frames, dtype=int), N),
        "particle_id": np.tile(np.arange(N), F),
        "sxx": covs[:, :, 0, 0].ravel(),
        "sxy": covs[:, :, 0, 1].ravel(),
        "sxz": covs[:, :, 0, 2].ravel(),
        "syy": covs[:, :, 1, 1].ravel(),
        "syz": covs[:, :, 1, 2].ravel(),
        "szz": covs[:, :, 2, 2].ravel(),
    }
    pd.DataFrame(rows).to_parquet(out_dir / "covariances.parquet", index=False)


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rigid_trajectory(x0: np.ndarray, v0: np.ndarray, omega_z: float,
                     times: np.ndarray):
    """Exact rigid motion: translation v0 plus rotation omega_z about the
    center of mass (equal masses assumed for the center). Returns
    (positions (F, N, 3), velocities (F, N, 3)). Conserves p, L, and KE
    exactly in the continuum sense.
    """
    x_cm0 = x0.mean(axis=0)
    r0 = x0 - x_cm0
    F = len(times)
    N = len(x0)
    pos = np.empty((F, N, 3))
    vel = np.empty((F, N, 3))
    omega = np.array([0.0, 0.0, omega_z])
    for k, t in enumerate(times):
        R = rotation_z(omega_z * t)
        r = r0 @ R.T
        pos[k] = x_cm0 + v0 * t + r
        vel[k] = v0 + np.cross(omega, r)
    return pos, vel
