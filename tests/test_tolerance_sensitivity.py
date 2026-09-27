from analysis.tolerance_sensitivity import scaled_scene


def test_scales_error_budgets_but_not_regime_gates():
    scene = {"pass_criteria": {
        "rel_tolerance": 0.1,
        "prescribed_face_tolerance_rel": 0.05,
        "direction_cosine_min": 0.98,
        "min_eigenvalue_ratio": 1e-3,
        "psnr_min_db": 35.0,
    }}
    strict = scaled_scene(scene, 0.5)["pass_criteria"]
    assert strict["rel_tolerance"] == 0.05
    assert strict["prescribed_face_tolerance_rel"] == 0.05
    assert strict["direction_cosine_min"] == 0.99
    assert strict["min_eigenvalue_ratio"] == 2e-3
    assert strict["psnr_min_db"] > 35.0
