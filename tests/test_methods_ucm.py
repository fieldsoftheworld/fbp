"""UltrametricContours: over-segment, then merge weak borders."""

import numpy as np

from fbp import UltrametricContours


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


def test_ucm_threshold_monotone() -> None:
    seg, prob = two_field_scene()
    low = UltrametricContours(foreground="not_background", merge_threshold=0.2)(
        seg, prob
    )
    high = UltrametricContours(foreground="not_background", merge_threshold=0.95)(
        seg, prob
    )
    assert low.max() >= high.max() >= 1
    assert low.max() >= 2  # the strong boundary survives a low threshold


def test_ucm_dispatch_shape_and_dtype() -> None:
    seg, prob = two_field_scene()
    out = UltrametricContours(foreground="not_background", merge_threshold=0.3)(
        seg, prob
    )
    assert out.shape == seg.shape and out.max() >= 1
