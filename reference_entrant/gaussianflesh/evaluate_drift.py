"""
evaluate_drift.py
Per-entity drift logging for the TL-APIC vs UL-MLS-MPM comparison.

Wraps the existing measure_drift function and writes one CSV row per
(entity, frame) with the columns required by the CVPR 2026 4DV workshop
camera-ready comparison.
"""
import csv
import os
import time
from datetime import datetime
from typing import List, Optional

import numpy as np


CSV_COLUMNS = [
    "solver_type", "entity_id", "frame", "wall_clock_ms",
    "drift_mean", "drift_max",
    "Uyy_spread", "sigma_min_mean", "n_clamped",
    "J_mean", "J_max", "J_min",
]


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def csv_path(csv_dir: str, run_tag: Optional[str] = None) -> str:
    """Build a timestamped CSV path under csv_dir/.  Use one CSV per run
    (rows for both entities go in the same file  -  easier to plot together)."""
    _ensure_dir(csv_dir)
    if run_tag is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(csv_dir, f"drift_{run_tag}.csv")


class DriftLogger:
    """Writes per-(entity, frame) rows to a single CSV.  Open it in append
    mode so a long run can be inspected mid-way."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(CSV_COLUMNS)
        self._fh.flush()

    def log(
        self,
        solver_type: str,
        entity_id: int,
        frame: int,
        wall_clock_ms: float,
        drift_mean: float,
        drift_max: float,
        Uyy_spread: float,
        sigma_min_mean: float,
        n_clamped: int,
        J_mean: float,
        J_max: float,
        J_min: float,
    ) -> None:
        self._writer.writerow([
            solver_type, entity_id, frame, f"{wall_clock_ms:.3f}",
            f"{drift_mean:.6f}", f"{drift_max:.6f}",
            f"{Uyy_spread:.6f}", f"{sigma_min_mean:.6f}", n_clamped,
            f"{J_mean:.6f}", f"{J_max:.6f}", f"{J_min:.6f}",
        ])
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def per_entity_diagnostics(
    Us: np.ndarray,
    Fs: np.ndarray,
    sigma_mins: np.ndarray,
    start: int,
    end: int,
    sigma_clamp_bound: float = 0.2,
) -> dict:
    """Compute the per-entity scalar diagnostics that go into the CSV.

    Args:
        Us:         (N, 3, 3) symmetric stretch tensor per particle (from polar decomp)
        Fs:         (N, 3, 3) deformation gradient per particle
        sigma_mins: (N,)      min singular value of F per particle (already diagnosed on GPU)
        start, end: particle index range for this entity
        sigma_clamp_bound: matches MPM_STRESS_CLAMP / SVD floor in the kernels
                           (default 0.2 matching _pk1_corotated)
    """
    Us_e = Us[start:end]
    Fs_e = Fs[start:end]
    sm_e = sigma_mins[start:end]

    Uyy = Us_e[:, 1, 1]
    Uyy_spread = float(Uyy.max() - Uyy.min()) if Uyy.size else 0.0

    J = np.linalg.det(Fs_e)
    n_clamped = int(np.sum(sm_e <= sigma_clamp_bound + 1e-6))

    return {
        "Uyy_spread":     Uyy_spread,
        "sigma_min_mean": float(sm_e.mean())  if sm_e.size else 0.0,
        "n_clamped":      n_clamped,
        "J_mean":         float(J.mean()),
        "J_max":          float(J.max()),
        "J_min":          float(J.min()),
    }
