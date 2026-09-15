"""`run_variant`'s strategy-string parser and the post-step modifiers it wires up."""

import numpy as np
import pytest

from boundaryvote.variants import (
    components,
    drop_small,
    fill_holes,
    foreground,
    merge_small,
    run_variant,
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


def test_area_modifier_converts_ground_area_with_px_m2() -> None:
    """`+A900` at px_m2=9 drops parcels under 100 px."""
    seg, prob = two_blocks()
    fg = foreground(seg, prob, "nbg")
    wired = run_variant("nbg-none+A900", seg, prob, px_m2=9.0)
    np.testing.assert_array_equal(wired, drop_small(components(fg), 100))
    # both blocks are 400 px, so a 3600 m^2 (400 px) floor keeps them and
    # 3609 m^2 (401 px) removes them
    assert run_variant("nbg-none+A3600", seg, prob, px_m2=9.0).max() == 2
    assert run_variant("nbg-none+A3609", seg, prob, px_m2=9.0).max() == 0


def test_area_modifier_without_pixel_area_raises() -> None:
    seg, prob = two_blocks()
    with pytest.raises(ValueError, match="pixel area"):
        run_variant("nbg-none+A900", seg, prob)


def test_unknown_modifier_raises() -> None:
    seg, prob = two_blocks()
    with pytest.raises(ValueError, match="unknown modifier"):
        run_variant("nbg-none+z5", seg, prob)


def test_hole_fill_modifier_matches_direct_call() -> None:
    seg, prob = two_blocks()
    fg = foreground(seg, prob, "nbg")
    wired = run_variant("nbg-none+f5", seg, prob)
    direct = fill_holes(components(fg), 5)
    np.testing.assert_array_equal(wired, direct)


def test_merge_modifier_matches_direct_call() -> None:
    seg, prob = two_blocks()
    fg = foreground(seg, prob, "nbg")
    wired = run_variant("nbg-none+m5", seg, prob)
    direct = merge_small(components(fg), 5)
    np.testing.assert_array_equal(wired, direct)


def test_fill_holes_noop_when_no_background() -> None:
    full = np.ones((5, 5), dtype=np.int32)
    np.testing.assert_array_equal(fill_holes(full, -1), full)


def test_merge_small_noop_when_nothing_below_threshold() -> None:
    inst = np.zeros((10, 10), dtype=np.int32)
    inst[1:5, 1:5] = 1
    np.testing.assert_array_equal(merge_small(inst, 1), inst)


def test_merge_small_leaves_isolated_small_parcel_alone() -> None:
    """A small parcel with no neighbour within reach has nothing to merge into."""
    inst = np.zeros((20, 20), dtype=np.int32)
    inst[1:3, 1:3] = 1  # small, far from anything
    inst[15:19, 15:19] = 2  # big, out of merge reach
    out = merge_small(inst, 100)
    assert (out[1:3, 1:3] > 0).all()
    assert (out[1:3, 1:3] != out[15, 15]).all()
