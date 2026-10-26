"""ExtentBoundaryThreshold: extent and boundary thresholds, as in DECODE."""

import numpy as np

from fbp import ExtentBoundaryThreshold


def two_field_scene() -> tuple[np.ndarray, np.ndarray]:
    """Two fields separated by a confident boundary line, plus a faint internal line."""
    H, W = 40, 60
    prob = np.zeros((3, H, W), dtype=np.float32)
    prob[0] = 1.0
    field = np.zeros((H, W), dtype=bool)
    field[4:36, 4:56] = True
    prob[1][field] = 0.9
    prob[0][field] = 0.05
    prob[2][field] = 0.05
    prob[2][4:36, 30] = 0.9  # strong boundary column
    prob[1][4:36, 30] = 0.05
    prob[2][20, 4:30] = 0.4  # weak internal line
    prob[1][20, 4:30] = 0.55
    seg = prob.argmax(0).astype(np.uint8)
    return seg, prob


def test_thr_splits_on_the_strong_boundary() -> None:
    seg, prob = two_field_scene()
    out = ExtentBoundaryThreshold(
        min_field_probability=0.5, max_boundary_probability=0.5
    )(seg, prob)
    assert out.max() == 2  # boundary column with p_bnd .9 splits the field


def test_thr_shape_and_dtype() -> None:
    seg, prob = two_field_scene()
    out = ExtentBoundaryThreshold(
        min_field_probability=0.5, max_boundary_probability=0.5
    )(seg, prob)
    assert out.dtype == np.int32 and out.shape == seg.shape
