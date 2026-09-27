"""Unit tests for the m1 material validation scorer.

Known outcomes, all synthetic:
  - A submission equal to the measured (Yeoh) curve passes at ~machine zero.
  - A submission at half the measured stress fails.
  - A submission that only reaches small stretch (too few points in range)
    is rejected with a helpful message.
  - A missing stress_strain file raises a SubmissionError (-> NA at runner).
"""

import csv

import numpy as np
import pytest

from conftest import load_scene_json
from scoring import common
from scoring import score_m1_material as m1
from reference import yeoh_rubber

SCENE = load_scene_json("m1_material_validation_v1.json")


def _write(tmp_path, stretches, stresses):
    d = tmp_path / "sub"
    d.mkdir()
    with open(d / "stress_strain.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stretch", "cauchy_stress_pa"])
        for s, sig in zip(stretches, stresses):
            w.writerow([s, sig])
    return d


def test_exact_measured_curve_passes(tmp_path, scenes_dir):
    lam = np.linspace(1.05, 2.5, 20)
    sig = yeoh_rubber.uniaxial_cauchy_stress(lam)
    sub = _write(tmp_path, lam, sig)
    v = m1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.measured["median_rel_error"] < 1e-9


def test_too_soft_curve_fails(tmp_path, scenes_dir):
    lam = np.linspace(1.05, 2.5, 20)
    sig = 0.5 * yeoh_rubber.uniaxial_cauchy_stress(lam)   # 50% too soft
    sub = _write(tmp_path, lam, sig)
    v = m1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL
    assert v.measured["median_rel_error"] > 0.4


def test_within_tolerance_passes(tmp_path, scenes_dir):
    # a curve 10% high everywhere: median error 0.10 < 0.15 tolerance -> PASS
    lam = np.linspace(1.05, 2.5, 20)
    sig = 1.10 * yeoh_rubber.uniaxial_cauchy_stress(lam)
    sub = _write(tmp_path, lam, sig)
    v = m1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS


def test_insufficient_range_rejected(tmp_path, scenes_dir):
    lam = np.array([1.01, 1.02, 1.03])           # only tiny strains
    sig = yeoh_rubber.uniaxial_cauchy_stress(lam)
    sub = _write(tmp_path, lam, sig)
    with pytest.raises(common.SubmissionError, match="stretch range"):
        m1.score(SCENE, sub, scenes_dir)


def test_missing_file_rejected(tmp_path, scenes_dir):
    d = tmp_path / "sub"
    d.mkdir()
    with pytest.raises(common.SubmissionError, match="stress_strain"):
        m1.score(SCENE, d, scenes_dir)
