"""Foreground rules and the fixed-order post-split cleanup steps."""

import numpy as np
import pytest

from fbp.core.postprocess import (
    apply_post_steps,
    drop_small,
    fill_holes,
    foreground,
    merge_small,
)


def two_blocks() -> tuple[np.ndarray, np.ndarray]:
    H, W = 24, 48
    seg = np.zeros((H, W), dtype=np.uint8)
    prob = np.zeros((3, H, W), dtype=np.float32)
    prob[0] = 1.0
    seg[2:22, 2:22] = 1
    seg[2:22, 26:46] = 1
    prob[1][seg == 1] = 0.9
    prob[0][seg == 1] = 0.05
    prob[2][seg == 1] = 0.05
    return seg, prob


def test_foreground_modes() -> None:
    seg, prob = two_blocks()
    seg[10:12, 22:26] = 2  # a boundary bridge, only "nbg" counts it as foreground
    prob[2][10:12, 22:26] = 0.9
    prob[1][10:12, 22:26] = 0.05
    assert foreground(seg, prob, "nbg").sum() > foreground(seg, prob, "arg").sum()
    assert foreground(seg, prob, "pf50").sum() == (prob[1] > 0.5).sum()
    otsu = foreground(seg, prob, "otsu")
    assert otsu.dtype == bool


def test_foreground_unknown_mode_raises() -> None:
    seg, prob = two_blocks()
    with pytest.raises(ValueError):
        foreground(seg, prob, "bogus")


def test_drop_small_removes_and_relabels_densely() -> None:
    inst = np.zeros((10, 10), dtype=np.int32)
    inst[1:5, 1:5] = 1  # 16 px
    inst[7:9, 7:9] = 2  # 4 px
    out = drop_small(inst, min_area=10)
    assert out.max() == 1
    assert set(np.unique(out)) == {0, 1}


def test_fill_holes_and_merge_small() -> None:
    inst = np.zeros((20, 20), dtype=np.int32)
    inst[2:12, 2:12] = 1
    inst[6:8, 6:8] = 0  # a 4-px hole inside parcel 1
    inst[15:17, 15:17] = 2  # a 4-px sliver, not adjacent to parcel 1
    inst[14, 2] = 3  # a 1-px parcel adjacent to nothing

    assert (fill_holes(inst, 0) == inst).all()  # off
    filled = fill_holes(inst, 10)
    assert (filled[6:8, 6:8] == 1).all() and filled.max() == inst.max()
    # a hole touching the image border is background, not a hole
    edge = np.zeros((10, 10), dtype=np.int32)
    edge[3:10, 3:10] = 1
    assert (fill_holes(edge, -1) == edge).all()

    touching = np.zeros((10, 20), dtype=np.int32)
    touching[2:8, 2:10] = 1
    touching[2:4, 11:13] = 2  # small, separated by a one-pixel gap at col 10
    merged = merge_small(touching, 10)
    assert merged.max() == 1
    assert (merged > 0).sum() >= (touching > 0).sum()
    assert ((touching > 0) & (merged == 0)).sum() == 0
    assert drop_small(touching, 10).max() == 1
    assert (drop_small(touching, 10) > 0).sum() < (touching > 0).sum()


def test_merge_small_leaves_isolated_small_parcel_alone() -> None:
    inst = np.zeros((20, 20), dtype=np.int32)
    inst[1:3, 1:3] = 1  # small, far from anything
    inst[15:19, 15:19] = 2  # big, out of merge reach
    out = merge_small(inst, 100)
    assert (out[1:3, 1:3] > 0).all()
    assert (out[1:3, 1:3] != out[15, 15]).all()


def test_apply_post_steps_order_is_fill_then_merge_then_drop() -> None:
    inst = np.zeros((20, 20), dtype=np.int32)
    inst[2:12, 2:12] = 1
    inst[6:8, 6:8] = 0  # hole to fill
    inst[15:16, 15:16] = (
        2  # tiny parcel, dropped since it has no neighbour to merge into
    )
    out = apply_post_steps(inst, min_area=5, hole_area=10, merge_area=0)
    assert (out[6:8, 6:8] > 0).all()
    assert out.max() == 1
