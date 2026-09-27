"""Unit tests for the variant generator.

Known outcomes:
  - Scales must land in the documented ranges and be deterministic per seed.
  - The variant's reference frequency must recompute from the scaled
    material block: a synthetic trajectory oscillating at the VARIANT's
    Euler-Bernoulli frequency must PASS when scored against the variant
    scene, proving nobody can tune to the public instance.
"""

import json
import sys
from pathlib import Path

import numpy as np

from conftest import REPO, write_submission
from scoring import common
from scoring import score_a2_frequency as a2
from reference import beams

sys.path.insert(0, str(REPO / "variants"))
from generate import generate  # noqa: E402


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_scales_in_range_and_deterministic(tmp_path):
    p1 = generate("a2_beam_frequency_v1", 42, tmp_path / "v1")
    p2 = generate("a2_beam_frequency_v1", 42, tmp_path / "v2")
    s1, s2 = _load(p1), _load(p2)
    scales = s1["provenance"]["variant_scales"]
    assert 0.5 <= scales["youngs_modulus"] <= 2.0
    assert 0.8 <= scales["density"] <= 1.25
    assert 0.8 <= scales["length"] <= 1.2
    # deterministic: same seed, same variant
    assert s1["material"] == s2["material"]
    assert s1["geometry"]["size_m"] == s2["geometry"]["size_m"]


def test_different_seed_differs(tmp_path):
    s1 = _load(generate("a2_beam_frequency_v1", 1, tmp_path / "a"))
    s2 = _load(generate("a2_beam_frequency_v1", 2, tmp_path / "b"))
    assert s1["material"]["youngs_modulus_pa"] != s2["material"]["youngs_modulus_pa"]


def test_variant_reference_recomputes(tmp_path):
    out = tmp_path / "gen"
    scene = _load(generate("a2_beam_frequency_v1", 7, out))
    rest = np.load(out / scene["geometry"]["particle_positions_file"])

    # The variant's own reference frequency, from its scaled blocks.
    L, w, h = scene["geometry"]["size_m"]
    m = scene["material"]
    f_ref = beams.cantilever_f1(m["youngs_modulus_pa"], m["density_kg_m3"],
                                L, w, h)

    fps = scene["simulation"]["output_fps"]
    dur = scene["simulation"]["duration_s"]
    t = np.arange(int(round(dur * fps)) + 1) / fps
    s = (rest[:, 0] - rest[:, 0].min()) / L
    shape = beams.mode1_shape(s) / beams.mode1_shape(np.array([1.0]))[0]
    osc = 0.005 * np.sin(2 * np.pi * f_ref * t)[:, None] * shape[None, :]
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    pos[:, :, 1] += osc

    sub = write_submission(tmp_path / "sub", pos, t)
    v = a2.score(scene, sub, out)   # scenes_dir is the variant output dir
    assert v.status == common.STATUS_PASS
    assert v.measured["rel_error"] < 0.02
