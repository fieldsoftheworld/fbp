"""`run_variant`'s strategy-string parser and the post-step modifiers it wires up."""

import numpy as np
import pytest

from boundaryvote.variants import (
    components,
    drop_small_unsupported,
    fill_holes,
    foreground,
    merge_small,
    run_variant,
    widen_gaps,
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


def test_area_modifier_matches_pixel_modifier_via_px_m2() -> None:
    """`+A900` at px_m2=9 must equal `+a100` (900 / 9 = 100 px)."""
    seg, prob = two_blocks()
    a = run_variant("nbg-none+a100", seg, prob)
    big_a = run_variant("nbg-none+A900", seg, prob, px_m2=9.0)
    np.testing.assert_array_equal(a, big_a)


def test_area_modifier_without_pixel_area_raises() -> None:
    seg, prob = two_blocks()
    with pytest.raises(ValueError, match="pixel area"):
        run_variant("nbg-none+A900", seg, prob)


def test_unknown_modifier_raises() -> None:
    seg, prob = two_blocks()
    with pytest.raises(ValueError, match="unknown modifier"):
        run_variant("nbg-none+z5", seg, prob)


def test_msup_without_markers_raises() -> None:
    seg, prob = two_blocks()
    with pytest.raises(ValueError, match="instance marker image"):
        run_variant("nbg-msup-t0.5", seg, prob)


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


def test_keep_tau_modifier_matches_direct_call() -> None:
    seg, prob = two_blocks()
    fg = foreground(seg, prob, "nbg")
    wired = run_variant("nbg-none+a300+K50", seg, prob)
    direct = drop_small_unsupported(components(fg), 300, seg, 0.5)
    np.testing.assert_array_equal(wired, direct)


def test_drop_small_unsupported_spares_predicted_outline() -> None:
    """A small parcel is spared only when >= keep_tau of its border is boundary."""
    inst = np.zeros((10, 10), dtype=np.int32)
    inst[1:9, 1:6] = 1  # big parcel
    inst[3:5, 7:9] = 2  # small, isolated parcel below the area threshold

    seg_supported = np.zeros((10, 10), dtype=np.uint8)
    seg_supported[3:5, 6] = 2  # boundary right next to half of parcel 2's border
    spared = drop_small_unsupported(inst, min_area=10, seg=seg_supported, keep_tau=0.5)
    assert (spared == 2).any()

    seg_unsupported = np.zeros((10, 10), dtype=np.uint8)
    dropped = drop_small_unsupported(
        inst, min_area=10, seg=seg_unsupported, keep_tau=0.5
    )
    assert not (dropped == 2).any()


def test_drop_small_unsupported_noop_below_min_area_or_with_no_small_parcels() -> None:
    inst = np.zeros((10, 10), dtype=np.int32)
    inst[1:9, 1:6] = 1
    seg = np.zeros((10, 10), dtype=np.uint8)
    assert (drop_small_unsupported(inst, 0, seg, 0.5) == inst).all()
    assert (drop_small_unsupported(inst, 10, seg, 0.5) == inst).all()  # no small ids


def test_widen_gaps_noop_when_foreground_has_no_gap() -> None:
    solid = np.ones((10, 10), dtype=np.int32)
    fg = solid > 0
    out = widen_gaps(solid, fg, 3)
    np.testing.assert_array_equal(out, solid)


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
