"""boundary_distance, boundary_prob, and the oriented-signal line kernel."""

import numpy as np

from fbp.core.surfaces import (
    N_ORIENT,
    boundary_distance,
    boundary_prob,
    line_kernel,
    oriented_signal,
)


def test_boundary_distance_zero_at_boundary_and_grows_inward() -> None:
    seg = np.zeros((10, 10), dtype=np.uint8)
    seg[2:8, 2:8] = 1
    seg[4:6, 4:6] = 2  # a boundary patch inside the field
    dist = boundary_distance(seg)
    assert dist.dtype == np.float32
    assert dist[4, 4] == 0.0  # boundary pixels are the EDT source
    assert dist[2, 2] > 0.0  # field pixels away from any boundary are positive


def test_boundary_prob_is_one_minus_p_boundary() -> None:
    prob = np.zeros((3, 4, 4), dtype=np.float32)
    prob[2] = 0.3
    out = boundary_prob(prob)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 0.7)


def test_line_kernel_orientations() -> None:
    horiz = line_kernel(0.0, 5)
    vert = line_kernel(np.pi / 2, 5)
    assert horiz[2].sum() > 0.99 and vert[:, 2].sum() > 0.99
    assert abs(horiz.sum() - 1.0) < 1e-6


def test_oriented_signal_shape_and_range() -> None:
    pb = np.random.default_rng(0).uniform(0, 1, (12, 12)).astype(np.float32)
    sig = oriented_signal(pb)
    assert sig.shape == (N_ORIENT, 12, 12)
    assert sig.min() >= 0.0 and sig.max() <= 1.0
