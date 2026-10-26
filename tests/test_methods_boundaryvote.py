"""BoundaryVote: watershed proposals kept or merged by a boundary vote."""

import itertools
from pathlib import Path
from typing import get_args

import numpy as np
import pytest

from fbp import BoundaryVote
from fbp.core.postprocess import foreground
from fbp.methods.boundaryvote import LineScore, Surface

CACHE = Path(__file__).resolve().parents[1] / "out" / "cache" / "planet_b3" / "test"
DENSE10 = (
    "belgium",
    "cambodia",
    "croatia",
    "germany",
    "latvia",
    "lithuania",
    "portugal",
    "slovenia",
    "south_africa",
    "sweden",
)


def two_fields(gap_boundary: bool, boundary_pb: float) -> tuple[np.ndarray, np.ndarray]:
    """Two 20x20 blocks; a boundary column or a thin neck joins them."""
    H, W = 24, 48
    seg = np.zeros((H, W), dtype=np.uint8)
    prob = np.zeros((3, H, W), dtype=np.float32)
    prob[0] = 1.0
    seg[2:22, 2:22] = 1
    seg[2:22, 26:46] = 1
    prob[1][seg == 1] = 0.9
    prob[0][seg == 1] = 0.05
    prob[2][seg == 1] = 0.05
    seg[10:14, 22:26] = 1
    prob[1][10:14, 22:26] = 0.9
    prob[0][10:14, 22:26] = 0.05
    prob[2][10:14, 22:26] = 0.05
    if gap_boundary:
        col = 24
        seg[10:14, col] = 2
        seg[12, col] = 1  # one-pixel gap in the boundary line
        prob[2][10:14, col] = boundary_pb
        prob[1][10:14, col] = 1.0 - boundary_pb - 0.05
    return seg, prob


def touching_pairs(inst: np.ndarray) -> int:
    """Number of 4-adjacent pixel pairs carrying two different non-zero labels."""
    a, b = inst[:, :-1], inst[:, 1:]
    n = int(((a != b) & (a > 0) & (b > 0)).sum())
    a, b = inst[:-1, :], inst[1:, :]
    return n + int(((a != b) & (a > 0) & (b > 0)).sum())


def test_unsupported_neck_is_merged() -> None:
    seg, prob = two_fields(gap_boundary=False, boundary_pb=0.0)
    inst = BoundaryVote(
        surface="boundary_distance", seed_depth=1, min_line_support=0.5
    )(seg, prob)
    assert inst.max() == 1  # the geometric neck split is not supported -> one field


def test_supported_gap_is_kept() -> None:
    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    inst = BoundaryVote(
        foreground="not_background",
        surface="boundary_distance",
        seed_depth=1,
        min_line_support=0.5,
    )(seg, prob)
    assert inst.max() == 2
    assert (inst[10:14, 22:26] == 0).any()


@pytest.mark.parametrize("surface", get_args(Surface))
@pytest.mark.parametrize("line_score", get_args(LineScore))
def test_every_surface_and_line_score_runs(
    surface: Surface, line_score: LineScore
) -> None:
    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    method = BoundaryVote(
        foreground="not_background",
        surface=surface,
        seed_depth=2,
        line_score=line_score,
        min_line_support=0.5,
    )
    out = method(seg, prob)
    assert out.dtype == np.int32 and out.shape == seg.shape and out.min() >= 0


def test_kept_splits_never_touch_on_real_patches() -> None:
    checked = 0
    for country in DENSE10:
        src = CACHE / country
        if not src.is_dir():
            pytest.skip("planet_b3 cache missing")
        for npz in itertools.islice(sorted(src.glob("*.npz")), 4):
            with np.load(npz) as d:
                seg, prob = d["seg"].astype(np.uint8), d["prob"].astype(np.float32)
            for support in (0.7, 0.9):
                method = BoundaryVote(
                    foreground="not_background",
                    surface="boundary_distance",
                    seed_depth=2,
                    min_line_support=support,
                )
                inst = method(seg, prob)
                assert touching_pairs(inst) == 0, f"{country}/{npz.stem} {method}"
            checked += 1
    assert checked >= 20


def test_min_line_and_min_area_and_band() -> None:
    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    base = BoundaryVote(
        foreground="not_background",
        surface="boundary_distance",
        seed_depth=1,
        min_line_support=0.5,
    )(seg, prob)
    assert base.max() == 2
    # a minimum line length longer than the seam merges the two fields
    long_line = BoundaryVote(
        foreground="not_background",
        surface="boundary_distance",
        seed_depth=1,
        min_line_support=0.5,
        min_line_length=50,
    )(seg, prob)
    assert long_line.max() == 1
    # the tolerance band can only raise support, never lower it
    banded = BoundaryVote(
        foreground="not_background",
        surface="boundary_distance",
        seed_depth=1,
        min_line_support=0.5,
        support_radius=1,
    )(seg, prob)
    assert banded.max() >= base.max()


def test_probability_share_falls_back_to_fraction_with_a_support_radius() -> None:
    """With a support radius, the probability-share vote is the fraction vote."""
    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    soft = BoundaryVote(
        foreground="not_background",
        surface="boundary_distance",
        seed_depth=1,
        line_score="probability_share",
        min_line_support=0.5,
        support_radius=1,
    )(seg, prob)
    frac = BoundaryVote(
        foreground="not_background",
        surface="boundary_distance",
        seed_depth=1,
        min_line_support=0.5,
        support_radius=1,
    )(seg, prob)
    np.testing.assert_array_equal(soft, frac)


def test_supported_split_falls_back_when_no_markers_found() -> None:
    """A seed depth deeper than the distance surface leaves no seeds."""
    seg, prob = two_fields(gap_boundary=False, boundary_pb=0.0)
    inst = BoundaryVote(
        foreground="not_background",
        surface="boundary_distance",
        seed_depth=100,
        min_line_support=0.5,
    )(seg, prob)
    assert inst.max() == 1


def test_empty_patch() -> None:
    seg = np.zeros((8, 8), dtype=np.uint8)
    prob = np.zeros((3, 8, 8), dtype=np.float32)
    prob[0] = 1.0
    for method in (
        BoundaryVote(surface="boundary_distance", seed_depth=2, min_line_support=0.7),
        BoundaryVote(seed_depth=0.01, min_line_support=0.3),
    ):
        assert method(seg, prob).max() == 0


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
    prob[2][4:36, 30] = 0.9
    prob[1][4:36, 30] = 0.05
    prob[2][20, 4:30] = 0.4
    prob[1][20, 4:30] = 0.55
    seg = prob.argmax(0).astype(np.uint8)
    return seg, prob


def test_oriented_surface_proposals() -> None:
    """The vote on oriented-signal proposals keeps the strong boundary, h=0 included."""
    seg, prob = two_field_scene()
    for method in (
        BoundaryVote(
            foreground="not_background",
            surface="oriented_boundary",
            seed_depth=0.05,
            min_line_support=0.7,
        ),
        BoundaryVote(
            foreground="not_background",
            surface="oriented_boundary",
            seed_depth=0,
            min_line_support=0.7,
        ),
        BoundaryVote(
            foreground="not_background",
            seed_depth=0.1,
            min_line_support=0.7,
            min_area_m2=9,
        ),
    ):
        out = method(seg, prob, pixel_area_m2=1.0)
        assert out.dtype == np.int32 and out.max() >= 2


def test_not_background_includes_the_boundary_gap() -> None:
    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    assert (
        foreground(seg, prob, "not_background").sum()
        > foreground(seg, prob, "argmax").sum()
    )
