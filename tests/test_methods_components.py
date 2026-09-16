"""``none`` and ``close{r}``, plus the ``+A``/``+f``/``+m`` post-step modifiers."""

import numpy as np
import pytest

from fbp.core.postprocess import drop_small, fill_holes, foreground, merge_small
from fbp.core.regions import components
from fbp.methods import run


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


def test_none_is_components_of_the_foreground() -> None:
    seg, prob = two_blocks()
    out = run("nbg-none", seg, prob)
    np.testing.assert_array_equal(out, components(foreground(seg, prob, "nbg")))


def test_close_bridges_a_one_pixel_gap() -> None:
    seg, prob = two_blocks()
    seg[10:14, 24] = 2  # a thin boundary seam between the two blocks
    seg[10:14, 22:24] = 1
    seg[10:14, 25:26] = 1
    prob[1][seg == 1] = 0.9
    out = run("arg-close1", seg, prob)
    assert out.max() == 2


def test_area_modifier_converts_ground_area_with_px_m2() -> None:
    """``+A900`` at px_m2=9 drops parcels under 100 px."""
    seg, prob = two_blocks()
    fg = foreground(seg, prob, "nbg")
    wired = run("nbg-none+A900", seg, prob, px_m2=9.0)
    np.testing.assert_array_equal(wired, drop_small(components(fg), 100))
    # both blocks are 400 px, so a 3600 m^2 (400 px) floor keeps them and
    # 3609 m^2 (401 px) removes them
    assert run("nbg-none+A3600", seg, prob, px_m2=9.0).max() == 2
    assert run("nbg-none+A3609", seg, prob, px_m2=9.0).max() == 0


def test_area_modifier_without_pixel_area_raises() -> None:
    seg, prob = two_blocks()
    with pytest.raises(ValueError, match="pixel area"):
        run("nbg-none+A900", seg, prob)


def test_hole_fill_modifier_matches_direct_call() -> None:
    seg, prob = two_blocks()
    fg = foreground(seg, prob, "nbg")
    wired = run("nbg-none+f5", seg, prob)
    direct = fill_holes(components(fg), 5)
    np.testing.assert_array_equal(wired, direct)


def test_merge_modifier_matches_direct_call() -> None:
    seg, prob = two_blocks()
    fg = foreground(seg, prob, "nbg")
    wired = run("nbg-none+m5", seg, prob)
    direct = merge_small(components(fg), 5)
    np.testing.assert_array_equal(wired, direct)
