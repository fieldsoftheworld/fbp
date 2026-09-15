"""The oriented-watershed baseline runs, merges below its level, and dispatches."""

import numpy as np
from test_priors import two_field_scene

from boundaryvote.priors_recent import (
    N_ORIENT,
    border_weights,
    line_kernel,
    orientation_bin,
    owt,
)
from boundaryvote.variants import foreground, run_variant


def test_line_kernel_orientations() -> None:
    horiz = line_kernel(0.0, 5)
    vert = line_kernel(np.pi / 2, 5)
    assert horiz[2].sum() > 0.99 and vert[:, 2].sum() > 0.99
    assert abs(horiz.sum() - 1.0) < 1e-6


def test_orientation_bin_axes() -> None:
    r = np.full(10, 3)
    c = np.arange(10)
    assert orientation_bin(r, c) == 0  # horizontal border
    assert orientation_bin(c, r) == N_ORIENT // 2  # vertical border
    assert orientation_bin(r[:2], c[:2]) is None


def test_border_weights_follow_boundary_probability() -> None:
    seg, prob = two_field_scene()
    fg = foreground(seg, prob, "nbg")
    col = np.arange(fg.shape[1])[None, :]
    row = np.arange(fg.shape[0])[:, None]
    lab = np.zeros(fg.shape, dtype=np.int32)
    lab[fg] = 1
    lab[fg & (col > 30)] = 2
    lab[fg & (row > 20) & (col < 30)] = 3
    lab[fg & (col == 30)] = 0  # arc on the strong boundary column
    lab[fg & (row == 20) & (col < 30)] = 0  # arc on the weak internal row
    from boundaryvote.priors_recent import oriented_signal

    w = border_weights(lab, fg, oriented_signal(prob[2]))
    assert w[(1, 2)][0] > 0.6 > w[(1, 3)][0]  # strong column beats the weak row


def test_owt_level_monotone_and_dispatch() -> None:
    seg, prob = two_field_scene()
    fg = foreground(seg, prob, "nbg")
    low = owt(fg, seg, prob, 0.1)
    high = owt(fg, seg, prob, 0.95)
    assert low.max() >= high.max() >= 1
    assert low.max() >= 2  # the strong boundary survives a low level
    out = run_variant("nbg-owt-l0.5-h0.2+A9", seg, prob, px_m2=1.0)
    assert out.dtype == np.int32 and out.max() >= 1
    zero = run_variant("nbg-owt-l0.5-h0", seg, prob)
    assert zero.max() >= 2  # every regional minimum seeds; the strong column stays


def test_line_statistics_dispatch() -> None:
    seg, prob = two_field_scene()
    outs = [
        run_variant(name, seg, prob)
        for name in ("nbg-sup-h2-t0.5", "nbg-asup-h2-t0.5", "nbg-dsup-h2-t0.5")
    ]
    assert all(o.max() >= 2 for o in outs)  # the p_bnd .9 column splits under each
