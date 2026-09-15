"""Prior post-processing baselines run and behave monotonically in their thresholds."""

import numpy as np

from boundaryvote.priors import iterh, swst, thr, ucm
from boundaryvote.variants import foreground, run_variant


def two_field_scene() -> tuple[np.ndarray, np.ndarray]:
    """Two fields separated by a confident boundary line, plus a faint internal line."""
    H, W = 40, 60
    prob = np.zeros((3, H, W), dtype=np.float32)
    prob[0] = 1.0
    field = np.zeros((H, W), bool)
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
    fg = foreground(seg, prob, "nbg")
    low = ucm(fg, seg, prob, 0.2)
    high = ucm(fg, seg, prob, 0.95)
    assert low.max() >= high.max() >= 1
    assert low.max() >= 2  # the strong boundary survives a low threshold


def test_thr_components() -> None:
    _seg, prob = two_field_scene()
    lab = thr(prob, 0.5, 0.5)
    assert lab.max() == 2  # boundary column with p_bnd .9 splits the field


def test_iterh_and_swst_run() -> None:
    seg, prob = two_field_scene()
    fg = foreground(seg, prob, "nbg")
    a = iterh(fg, seg, (8.0, 4.0, 2.0, 1.0), a_max=100)
    b = iterh(fg, seg, (8.0, 4.0, 2.0, 1.0), a_max=10**9)
    assert a.max() >= b.max() >= 1
    s = swst(fg, seg, n=20, m=5, t=0.5)
    assert s.dtype == np.int32 and s.max() >= 1


def test_run_variant_dispatch() -> None:
    seg, prob = two_field_scene()
    for name in (
        "nbg-ucm-k0.3",
        "nbg-swst-n20-m5-t0.5",
        "nbg-iterh-a100",
        "arg-thr-te0.5-tb0.5",
    ):
        out = run_variant(name, seg, prob)
        assert out.shape == seg.shape and out.max() >= 1, name
