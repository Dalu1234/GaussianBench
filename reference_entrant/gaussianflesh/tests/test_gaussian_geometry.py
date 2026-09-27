import numpy as np

from gaussian_geometry import (
    _pad_or_truncate,
    _sphere_interior_check,
    fill_shape_grid_based,
    sample_cuboid_volume,
    sample_sphere_volume,
)


def test_sample_sphere_volume_bounds_and_determinism():
    a = sample_sphere_volume(32, r=1.5, seed=7)
    b = sample_sphere_volume(32, r=1.5, seed=7)

    assert a.shape == (32, 3)
    assert np.all(np.linalg.norm(a, axis=1) <= 1.5)
    np.testing.assert_array_equal(a, b)


def test_sample_cuboid_volume_bounds():
    half_extents = np.array([1.0, 2.0, 0.5])
    pts = sample_cuboid_volume(24, half_extents, seed=3)

    assert pts.shape == (24, 3)
    assert np.all(np.abs(pts) <= half_extents)


def test_pad_or_truncate_reaches_target_count():
    pos = np.arange(12, dtype=np.float32).reshape(4, 3)
    volumes = np.arange(4, dtype=np.float32)
    covs = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)

    p_short, v_short, c_short = _pad_or_truncate(pos, volumes, covs, 2)
    assert p_short.shape == (2, 3)
    assert v_short.shape == (2,)
    assert c_short.shape == (2, 3, 3)

    p_long, v_long, c_long = _pad_or_truncate(pos, volumes, covs, 6, rng_seed=5)
    assert p_long.shape == (6, 3)
    assert v_long.shape == (6,)
    assert c_long.shape == (6, 3, 3)
    np.testing.assert_array_equal(p_long[:4], pos)


def test_fill_shape_grid_based_outputs_consistent_arrays():
    pos, volumes, covs, dx, n_actual = fill_shape_grid_based(
        _sphere_interior_check(1.0),
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        target_n=32,
        ppc=4,
        seed=11,
    )

    assert pos.shape[0] == volumes.shape[0] == covs.shape[0] == n_actual
    assert pos.shape[1:] == (3,)
    assert covs.shape[1:] == (3, 3)
    assert dx > 0.0
    assert np.all(volumes > 0.0)