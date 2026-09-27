"""C3 appearance round trip (test_class: capability).

Compares the submitter's own renders of frame 0 and the settled final frame
after an elastic squash-and-release. Elastic deformation is fully
recoverable, so the two images must match; residual difference measures
appearance damage caused by the physics coupling. See the scene description
for why the submitter's renderer is used deliberately.

Metrics: PSNR and SSIM, implemented here on numpy + scipy only (no skimage,
per the dependency constraints). SSIM follows Wang, Bovik, Sheikh, Simoncelli,
"Image Quality Assessment: From Error Visibility to Structural Similarity",
IEEE Transactions on Image Processing 13(4), 2004: gaussian window
sigma = 1.5, K1 = 0.01, K2 = 0.03, dynamic range 255, computed on luminance.

Anti-gaming check: before touching the images, the scorer verifies from the
trajectory that the body really was compressed to the commanded fraction of
its rest height at some point during the run. A static submission with two
identical renders is INVALID, not PASS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from . import common


def _luminance(img: np.ndarray) -> np.ndarray:
    """(H, W, C) uint8-ish to float luminance. Grayscale passes through."""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])


def psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB. Infinite for identical images."""
    mse = float(np.mean((np.asarray(img1, float) - np.asarray(img2, float)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 255.0,
         sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03) -> float:
    """Mean SSIM on luminance, gaussian-windowed (Wang et al. 2004)."""
    x = _luminance(img1)
    y = _luminance(img2)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu_x = gaussian_filter(x, sigma)
    mu_y = gaussian_filter(y, sigma)
    sxx = gaussian_filter(x * x, sigma) - mu_x * mu_x
    syy = gaussian_filter(y * y, sigma) - mu_y * mu_y
    sxy = gaussian_filter(x * y, sigma) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    traj = common.load_trajectory(results_dir)
    rest = common.load_scene_positions(scene, scenes_dir)
    common.check_initial_condition(traj, rest)
    common.check_output_fps(traj, scene)

    crit = scene["pass_criteria"]

    # Anti-gaming: the squash must actually have happened.
    y = traj.positions[:, :, 1]
    h0 = float(y[0].max() - y[0].min())
    h_min = float((y.max(axis=1) - y.min(axis=1)).min())
    if h_min / h0 > crit["squash_verify_height_frac_max"]:
        return common.verdict_from_scene(
            scene, common.STATUS_INVALID,
            measured={"min_height_frac": h_min / h0},
            tolerance={"required_min_compression":
                       crit["squash_verify_height_frac_max"]},
            notes=["The trajectory never reached the commanded compression "
                   f"(minimum height fraction {h_min / h0:.3f}, required at "
                   f"most {crit['squash_verify_height_frac_max']}). The "
                   "squash did not run as specified, so the round-trip "
                   "comparison is meaningless."])

    renders = results_dir / "renders"
    before_p = renders / "before.png"
    after_p = renders / "after.png"
    cam_p = renders / "camera.json"
    for p in (before_p, after_p, cam_p):
        if not p.exists():
            raise common.SubmissionError(
                f"Missing {p.name} in {renders}. C3 requires renders/"
                "before.png, renders/after.png, and camera.json "
                "(SPEC.md section 2.4).")
    with open(cam_p, "r", encoding="utf-8") as f:
        json.load(f)  # must at least be valid JSON

    before = np.asarray(Image.open(before_p).convert("RGB"))
    after = np.asarray(Image.open(after_p).convert("RGB"))
    if before.shape != after.shape:
        raise common.SubmissionError(
            f"before.png {before.shape} and after.png {after.shape} have "
            "different shapes; both renders must use the identical camera "
            "and resolution.")

    psnr_db = psnr(before, after)
    ssim_val = ssim(before, after)

    ok = psnr_db > crit["psnr_min_db"] and ssim_val > crit["ssim_min"]
    return common.verdict_from_scene(
        scene,
        common.STATUS_PASS if ok else common.STATUS_FAIL,
        measured={"psnr_db": min(psnr_db, 100.0),
                  "ssim": ssim_val,
                  "min_height_frac_during_run": h_min / h0},
        reference={"psnr_db": "inf (identical renders)", "ssim": 1.0},
        tolerance={"psnr_min_db": crit["psnr_min_db"],
                   "ssim_min": crit["ssim_min"]},
        notes=(["PSNR reported as 100 dB (images identical, "
                "true value infinite)."] if np.isinf(psnr_db) else []))
