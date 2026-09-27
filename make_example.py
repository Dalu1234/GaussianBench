"""Generate the bundled synthetic example submission in results/example.

This is NOT a simulator. It writes analytically constructed trajectories that
exercise the full scoring pipeline so a new user can run:

    python make_example.py
    python run_all.py --results-dir results/example

and see a complete report. Each generated submission is honest about what it
is (see the meta.json notes): closed-form motions, not simulation output.

The two thermal scenes (a4_meltfront_v1 and c2_blobhealth_melt_v1) are
deliberately NOT generated, so the report demonstrates the NA outcome for a
system without a capability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from conftest import (rigid_trajectory, write_covariances,  # noqa: E402
                      write_submission)
from reference import beams  # noqa: E402

SCENES = REPO / "scenes"
OUT = REPO / "results" / "example"
META = {"system_name": "example-analytic",
        "notes": "Closed-form motion, bundled example, not a simulator."}


def _scene(name: str) -> dict:
    with open(SCENES / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _times(scene: dict) -> np.ndarray:
    fps = scene["simulation"]["output_fps"]
    dur = scene["simulation"]["duration_s"]
    return np.arange(int(round(dur * fps)) + 1) / fps


def _rest(scene: dict) -> np.ndarray:
    return np.load(SCENES / scene["geometry"]["particle_positions_file"])


def example_a1a():
    scene = _scene("a1a_conservation_inplace_v1.json")
    x0, t = _rest(scene), _times(scene)
    v0 = np.array(scene["initial_conditions"]["linear_velocity_m_s"])
    wz = scene["initial_conditions"]["angular_velocity_rad_s"][2]
    pos, vel = rigid_trajectory(x0, v0, wz, t)
    write_submission(OUT / scene["scene_id"], pos, t, velocities=vel, meta=META)
    return scene["scene_id"]


def example_a1b():
    scene = _scene("a1b_conservation_drifting_v1.json")
    x0, t = _rest(scene), _times(scene)
    v0 = np.array(scene["initial_conditions"]["linear_velocity_m_s"])
    wz = scene["initial_conditions"]["angular_velocity_rad_s"][2]
    pos, vel = rigid_trajectory(x0, v0, wz, t)
    # A1b requires the domain-handling declaration: meta.json notes becomes
    # an object, free text moves to notes.text (SPEC.md section 2.2.1).
    meta = dict(META)
    meta["notes"] = {"domain_handling": "mesh_free",
                     "text": "Closed-form rigid motion, bundled example, "
                             "not a simulator; no computational domain "
                             "exists, so mesh_free is the honest category."}
    write_submission(OUT / scene["scene_id"], pos, t, velocities=vel,
                     meta=meta)
    return scene["scene_id"]


def example_a2():
    scene = _scene("a2_beam_frequency_v1.json")
    rest, t = _rest(scene), _times(scene)
    L, w, h = scene["geometry"]["size_m"]
    m = scene["material"]
    f_ref = beams.cantilever_f1(m["youngs_modulus_pa"], m["density_kg_m3"],
                                L, w, h)
    s = (rest[:, 0] - rest[:, 0].min()) / L
    shape = beams.mode1_shape(s) / beams.mode1_shape(np.array([1.0]))[0]
    osc = 0.005 * np.sin(2 * np.pi * f_ref * t)[:, None] * shape[None, :]
    osc *= np.exp(-0.05 * t)[:, None]
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    pos[:, :, 1] += osc
    write_submission(OUT / scene["scene_id"], pos, t, meta=META)
    return scene["scene_id"]


def example_a3():
    scene = _scene("a3_wavespeed_v1.json")
    rest, t = _rest(scene), _times(scene)
    bc = scene["boundary_conditions"]["pulse"]
    m = scene["material"]
    c = float(np.sqrt(m["youngs_modulus_pa"] / m["density_kg_m3"]))
    x = rest[:, 0] - rest[:, 0].min()
    tau = t[:, None] - x[None, :] / c
    W = bc["width_s"]
    u = np.where((tau >= 0) & (tau <= W),
                 bc["amplitude_m"] * np.sin(np.pi * np.clip(tau, 0, W) / W),
                 0.0)
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    pos[:, :, 0] += u
    write_submission(OUT / scene["scene_id"], pos, t, meta=META)
    return scene["scene_id"]


def example_a5():
    scene = _scene("a5_stiffness_v1.json")
    rest, t = _rest(scene), _times(scene)
    bc = scene["boundary_conditions"]["prescribed_face"]
    nu = scene["material"]["poisson_ratio"]
    ramp = np.clip(t / bc["ramp_end_s"], 0.0, 1.0)
    x0 = rest[:, 0]
    eps_ax = bc["displacement_m"][0] / (x0.max() - x0.min())
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    pos[:, :, 0] += ramp[:, None] * eps_ax * (x0 - x0.min())[None, :]
    for axis in (1, 2):
        c0 = rest[:, axis] - rest[:, axis].mean()
        pos[:, :, axis] += ramp[:, None] * (-nu * eps_ax) * c0[None, :]
    write_submission(OUT / scene["scene_id"], pos, t, meta=META)
    return scene["scene_id"]


def example_c1():
    scene = _scene("c1_covariance_v1.json")
    rest, t = _rest(scene), _times(scene)
    F_target = np.asarray(scene["boundary_conditions"]["all_faces"]["F_target"])
    ramp = t / t[-1]
    pos = np.empty((len(t), len(rest), 3))
    native_F = np.empty((len(t), len(rest), 3, 3))
    for k, r in enumerate(ramp):
        F = np.eye(3) + r * (F_target - np.eye(3))
        pos[k] = rest @ F.T
        native_F[k] = F
    sub = write_submission(
        OUT / scene["scene_id"], pos, t,
        deformation_gradients=native_F, meta=META,
    )
    rng = np.random.default_rng(11)
    A = rng.normal(0.0, 1.0, (len(rest), 3, 3))
    cov0 = np.einsum("nij,nkj->nik", A, A) * 1e-6 + np.eye(3)[None] * 1e-6
    cov1 = np.einsum("ij,njk,lk->nil", F_target, cov0, F_target)
    write_covariances(sub, [0, len(t) - 1], np.stack([cov0, cov1]))
    return scene["scene_id"]


def example_c2_shear():
    scene = _scene("c2_blobhealth_shear_v1.json")
    rest, t = _rest(scene), _times(scene)
    gpeak = scene["boundary_conditions"]["shear_drive"]["gamma_peak"]
    gamma = gpeak * (1.0 - np.abs(2.0 * t / t[-1] - 1.0))
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    pos[:, :, 0] += gamma[:, None] * rest[None, :, 1]
    covs = np.empty((len(t), len(rest), 3, 3))
    native_F = np.empty((len(t), len(rest), 3, 3))
    for k, g in enumerate(gamma):
        F = np.eye(3)
        F[0, 1] = g
        native_F[k] = F
        covs[k] = (F @ (1e-6 * np.eye(3)) @ F.T)[None, :, :]
    sub = write_submission(
        OUT / scene["scene_id"], pos, t,
        deformation_gradients=native_F, meta=META,
    )
    write_covariances(sub, np.arange(len(t)), covs)
    return scene["scene_id"]


def example_c3():
    scene = _scene("c3_roundtrip_v1.json")
    rest, t = _rest(scene), _times(scene)
    target = scene["boundary_conditions"]["squash_plate"]["target_height_frac"]
    dur = t[-1]
    squish = np.interp(t, [0.0, 1.0, 1.5, 2.5, dur],
                       [1.0, target - 0.02, target - 0.02, 1.0, 1.0])
    y0 = rest[:, 1].min()
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    pos[:, :, 1] = y0 + (rest[None, :, 1] - y0) * squish[:, None]
    sub = write_submission(OUT / scene["scene_id"], pos, t, meta=META)

    rng = np.random.default_rng(5)
    img = rng.random((96, 96, 3))
    for c in range(3):
        img[..., c] = gaussian_filter(img[..., c], 3.0)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    renders = sub / "renders"
    renders.mkdir(exist_ok=True)
    Image.fromarray(img).save(renders / "before.png")
    Image.fromarray(img).save(renders / "after.png")
    with open(renders / "camera.json", "w", encoding="utf-8") as f:
        json.dump({"position": [0, 0.1, 0.5], "look_at": [0, 0.08, 0],
                   "up": [0, 1, 0], "fov_deg": 45, "width": 96, "height": 96},
                  f)
    return scene["scene_id"]


def example_c4():
    scene = _scene("c4_captured_frequency_v1.json")
    from scoring.score_c4_captured import reference_frequency
    rest, t = _rest(scene), _times(scene)
    f_ref = reference_frequency(scene, rest)
    L = rest[:, 0].max() - rest[:, 0].min()
    s = (rest[:, 0] - rest[:, 0].min()) / L
    shape = beams.mode1_shape(s) / beams.mode1_shape(np.array([1.0]))[0]
    osc = 0.004 * np.sin(2 * np.pi * f_ref * t)[:, None] * shape[None, :]
    osc *= np.exp(-0.05 * t)[:, None]
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    pos[:, :, 1] += osc
    write_submission(OUT / scene["scene_id"], pos, t, meta=META)
    return scene["scene_id"]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (example_a1a, example_a1b, example_a2, example_a3, example_a5,
               example_c1, example_c2_shear, example_c3, example_c4):
        sid = fn()
        print(f"example: {sid}")
    print("\nSkipped (demonstrates NA): a4_meltfront_v1, c2_blobhealth_melt_v1")
    print("Now run: python run_all.py --results-dir results/example")
