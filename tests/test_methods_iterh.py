"""``iterh-a{a}``: iterative marker depth, re-segmenting large regions."""

import numpy as np

from fbp.methods import run


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


def test_iterh_resegments_large_regions_more_than_small_area_cap() -> None:
    seg, prob = two_field_scene()
    coarse = run("nbg-iterh-a100000", seg, prob)  # a_max huge: never re-segments
    fine = run("nbg-iterh-a100", seg, prob)  # a_max tiny: re-segments everything
    assert fine.max() >= coarse.max() >= 1


def test_iterh_dispatch_shape_and_dtype() -> None:
    seg, prob = two_field_scene()
    out = run("arg-iterh-a100", seg, prob)
    assert out.dtype == np.int32 and out.shape == seg.shape
