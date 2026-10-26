"""``owt-l{l}-h{h}``: oriented watershed and hierarchical merging."""

import numpy as np

from fbp.core.postprocess import foreground
from fbp.core.surfaces import N_ORIENT, oriented_signal
from fbp.methods import run
from fbp.methods.owt import border_weights, orientation_bin


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
    w = border_weights(lab, fg, oriented_signal(prob[2]))
    assert w[(1, 2)][0] > 0.6 > w[(1, 3)][0]  # strong column beats the weak row


def test_owt_level_monotone_and_dispatch() -> None:
    seg, prob = two_field_scene()
    low = run("nbg-owt-l0.1-h0.05", seg, prob)
    high = run("nbg-owt-l0.95-h0.05", seg, prob)
    assert low.max() >= high.max() >= 1
    assert low.max() >= 2  # the strong boundary survives a low level
    out = run("nbg-owt-l0.5-h0.2+A9", seg, prob, px_m2=1.0)
    assert out.dtype == np.int32 and out.max() >= 1


def test_owt_zero_depth_seeds_every_minimum() -> None:
    seg, prob = two_field_scene()
    zero = run("nbg-owt-l0.5-h0", seg, prob)
    assert zero.max() >= 2  # every regional minimum seeds; the strong column stays


def test_owt_regions_come_from_a_lines_free_watershed() -> None:
    """With no merging, the output is the lines-free watershed's regions."""
    from scipy.ndimage import gaussian_filter

    from fbp.core.regions import lines_from_labels, relabel_dense, watershed_regions

    rng = np.random.default_rng(0)
    pb = gaussian_filter(rng.random((64, 64)), 1.5).astype(np.float32)
    pb = (pb - pb.min()) / (pb.max() - pb.min())
    prob = np.stack([np.zeros_like(pb), 1 - pb, pb])
    seg = np.ones((64, 64), dtype=np.uint8)
    fg = seg != 0
    surf = oriented_signal(pb).max(axis=0).astype(np.float32)
    over = watershed_regions(surf, fg, 0.0, lines=False, mode="min").labels
    expected = relabel_dense(lines_from_labels(over + fg, fg))
    got = run("nbg-owt-l0-h0", seg, prob)
    assert np.array_equal(got > 0, expected > 0)
    pairs = np.unique(np.stack([got[got > 0], expected[got > 0]]), axis=1)
    assert len(np.unique(pairs[0])) == len(np.unique(pairs[1])) == pairs.shape[1]
