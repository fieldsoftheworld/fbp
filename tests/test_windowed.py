"""``run_windowed``: windowed BoundaryVote without seam duplicates."""

import numpy as np
import pytest

from fbp import run
from fbp.windowed import run_windowed

STRATEGY = "nbg-pb-h0.01-t0.3+A900"


def fields(
    h: int, w: int, boxes: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Field blocks (r0, r1, c0, c1) with a 1-px boundary ring each, on background."""
    seg = np.zeros((h, w), np.uint8)
    prob = np.zeros((3, h, w), np.float32)
    prob[0] = 1.0
    for r0, r1, c0, c1 in boxes:
        seg[r0:r1, c0:c1] = 2
        seg[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1] = 1
    prob[:, seg == 1] = np.array([0.05, 0.9, 0.05])[:, None]
    prob[:, seg == 2] = np.array([0.05, 0.15, 0.8])[:, None]
    return seg, prob


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True when both label arrays group the foreground pixels identically."""
    if not np.array_equal(a > 0, b > 0):
        return False
    m = a > 0
    pairs = np.unique(np.stack([a[m], b[m]]), axis=1)
    return len(np.unique(pairs[0])) == pairs.shape[1] == len(np.unique(pairs[1]))


def test_small_fields_match_whole_image() -> None:
    boxes = [
        (r, r + 14, c, c + 14) for r in range(2, 120, 18) for c in range(2, 120, 18)
    ]
    seg, prob = fields(128, 128, boxes)
    whole = run(STRATEGY, seg, prob, px_m2=9.0)
    win = run_windowed(STRATEGY, seg, prob, px_m2=9.0, core=48, halo=16)
    assert same_partition(whole, win)


@pytest.mark.parametrize(("merge_frac", "expected"), [(0.1, 1), (2.0, 2)])
def test_field_across_seam(merge_frac: float, expected: int) -> None:
    # 80 px tall field over the core boundary at row 64; halo 16 < half the field,
    # so each window sees a truncated copy whose centroid falls in its own core
    seg, prob = fields(128, 64, [(20, 100, 10, 50)])
    whole = run(STRATEGY, seg, prob, px_m2=9.0)
    win = run_windowed(
        STRATEGY, seg, prob, px_m2=9.0, core=64, halo=16, merge_frac=merge_frac
    )
    assert int(whole.max()) == 1
    assert int(win.max()) == expected
    if expected == 1:
        assert same_partition(whole, win)


def test_labels_are_dense_int32() -> None:
    seg, prob = fields(96, 96, [(4, 30, 4, 30), (40, 90, 40, 90)])
    win = run_windowed(STRATEGY, seg, prob, px_m2=9.0, core=32, halo=8)
    assert win.dtype == np.int32
    assert np.array_equal(np.unique(win), np.arange(int(win.max()) + 1))


def test_whole_copy_and_truncated_copy_merge() -> None:
    # The lower window (rows 40-128) sees the whole field and owns it (centroid row
    # ~84); the upper window (rows 0-88) sees only its top, whose centroid (~57) is in
    # the upper core. Only the truncated copy touches a window edge.
    mask = np.zeros((128, 64), bool)
    mask[42:52, 2:42] = True
    mask[52:90, 2:10] = True
    mask[90:110, 2:62] = True
    seg = np.where(mask, 1, 0).astype(np.uint8)
    ring = mask & ~np.pad(mask, 1)[2:, 1:-1] | mask & ~np.pad(mask, 1)[:-2, 1:-1]
    ring |= mask & ~np.pad(mask, 1)[1:-1, 2:] | mask & ~np.pad(mask, 1)[1:-1, :-2]
    seg[ring] = 2
    prob = np.zeros((3, *seg.shape), np.float32)
    prob[0] = 1.0
    prob[:, seg == 1] = np.array([0.05, 0.9, 0.05])[:, None]
    prob[:, seg == 2] = np.array([0.05, 0.15, 0.8])[:, None]
    whole = run(STRATEGY, seg, prob, px_m2=9.0)
    win = run_windowed(STRATEGY, seg, prob, px_m2=9.0, core=64, halo=24)
    assert int(whole.max()) == 1
    assert same_partition(whole, win)
