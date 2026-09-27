"""Unit tests for the C3 appearance round trip scorer.

Known outcomes:
  - Identical images must give infinite PSNR and SSIM 1, and PASS.
  - A shifted copy of the image must FAIL.
  - A trajectory that never squashed must return INVALID even with
    identical (trivially matching) renders.
"""

import json

import numpy as np
from PIL import Image

from conftest import load_scene_json, write_submission
from scoring import common
from scoring import score_c3_roundtrip as c3

SCENE = load_scene_json("c3_roundtrip_v1.json")
FPS = SCENE["simulation"]["output_fps"]
DUR = SCENE["simulation"]["duration_s"]


def _texture(seed=3, size=96):
    """Smooth random texture so SSIM has structure to compare."""
    rng = np.random.default_rng(seed)
    img = rng.random((size, size, 3))
    from scipy.ndimage import gaussian_filter
    for c in range(3):
        img[..., c] = gaussian_filter(img[..., c], 3.0)
    img -= img.min()
    img /= img.max()
    return (img * 255).astype(np.uint8)


def _squash_trajectory(scenes_dir, compress=True):
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    if compress:
        # Squash y to 78 percent at t in [1.0, 1.5], recover by t = 2.5.
        target = SCENE["boundary_conditions"]["squash_plate"]["target_height_frac"]
        squish = np.interp(t, [0.0, 1.0, 1.5, 2.5, DUR],
                           [1.0, target - 0.02, target - 0.02, 1.0, 1.0])
        y0 = rest[:, 1].min()
        pos[:, :, 1] = y0 + (rest[None, :, 1] - y0) * squish[:, None]
    return pos, t


def _write_renders(sub, before, after):
    renders = sub / "renders"
    renders.mkdir(exist_ok=True)
    Image.fromarray(before).save(renders / "before.png")
    Image.fromarray(after).save(renders / "after.png")
    with open(renders / "camera.json", "w", encoding="utf-8") as f:
        json.dump({"position": [0, 0.1, 0.5], "look_at": [0, 0.08, 0],
                   "up": [0, 1, 0], "fov_deg": 45,
                   "width": before.shape[1], "height": before.shape[0]}, f)


def test_metrics_identical_images():
    img = _texture()
    assert np.isinf(c3.psnr(img, img))
    assert c3.ssim(img, img) == 1.0


def test_metrics_shifted_image_degrade():
    img = _texture()
    shifted = np.roll(img, 5, axis=1)
    assert c3.psnr(img, shifted) < 35.0
    assert c3.ssim(img, shifted) < 0.97


def test_identical_renders_pass(tmp_path, scenes_dir):
    pos, t = _squash_trajectory(scenes_dir)
    sub = write_submission(tmp_path / "sub", pos, t)
    img = _texture()
    _write_renders(sub, img, img.copy())
    v = c3.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.measured["ssim"] == 1.0


def test_shifted_render_fails(tmp_path, scenes_dir):
    pos, t = _squash_trajectory(scenes_dir)
    sub = write_submission(tmp_path / "sub", pos, t)
    img = _texture()
    _write_renders(sub, img, np.roll(img, 5, axis=1))
    v = c3.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL


def test_no_squash_is_invalid(tmp_path, scenes_dir):
    pos, t = _squash_trajectory(scenes_dir, compress=False)
    sub = write_submission(tmp_path / "sub", pos, t)
    img = _texture()
    _write_renders(sub, img, img.copy())
    v = c3.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_INVALID
