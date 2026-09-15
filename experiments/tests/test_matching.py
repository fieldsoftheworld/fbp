"""STRtree-accelerated matching must equal FTP's brute-force greedy matcher."""

import numpy as np
import pytest
import shapely.geometry
from numpy.random._generator import Generator

from boundaryvote import ftp_compat as ftp
from boundaryvote.matching import THRESHOLDS, diagnostics, greedy, iou_matrix, match
from boundaryvote.strategies import elbow, nms_pool, roundtrip, vote


def random_boxes(
    rng: Generator, n: int, size: int = 100
) -> list[shapely.geometry.Polygon]:
    out = []
    for _ in range(n):
        x, y = rng.uniform(0, size, 2)
        w, h = rng.uniform(3, 20, 2)
        out.append(shapely.geometry.box(x, y, x + w, y + h))
    return out


def test_match_equals_ftp() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        gt = random_boxes(rng, int(rng.integers(0, 30)))
        pred = random_boxes(rng, int(rng.integers(0, 30)))
        ours = match(gt, pred)
        ref = ftp.match_shapes(gt, pred, list(THRESHOLDS))
        for t in THRESHOLDS:
            assert (ours.tp[t], ours.fp[t], ours.fn[t]) == ref["per_t"][t]
        assert [(i, j) for i, j, _ in ours.pairs_50] == [
            (i, j) for i, j, _ in ref["matched_pairs_low"]
        ]


def test_diagnostics_split_merge() -> None:
    big = shapely.geometry.box(0, 0, 10, 10)
    left = shapely.geometry.box(0, 0, 5, 10)
    right = shapely.geometry.box(5, 0, 10, 10)
    d = diagnostics([big], [left, right])
    assert d == {"gt_split": 1, "pred_merge": 0, "gt_missed": 1, "pred_spurious": 0}
    d = diagnostics([left, right], [big])
    assert d == {"gt_split": 0, "pred_merge": 1, "gt_missed": 0, "pred_spurious": 0}
    far = shapely.geometry.box(50, 50, 55, 55)
    d = diagnostics([left], [far])
    assert d["gt_missed"] == 1 and d["pred_spurious"] == 1


def test_vote_majority_and_ties() -> None:
    shape = (20, 20)
    a = [shapely.geometry.box(0, 0, 10, 10)]
    b = [shapely.geometry.box(0, 0, 10, 10), shapely.geometry.box(12, 12, 18, 18)]
    # K=2, tau=0.5 -> k_min=1: union (ties to foreground)
    out = vote([a, b], shape)
    assert len(out) == 2
    # K=3 -> k_min=2: only the box present twice survives
    out = vote([a, b, []], shape)
    assert len(out) == 1 and out[0].area == 100


def test_roundtrip_merges_touching_polygons() -> None:
    shape = (20, 20)
    left = shapely.geometry.box(0, 0, 5, 10)
    right = shapely.geometry.box(5, 0, 10, 10)
    assert len(roundtrip([left, right], shape)) == 1


def test_nms_and_elbow_smoke() -> None:
    a = shapely.geometry.box(0, 0, 10, 10)
    b = shapely.geometry.box(0, 0, 10, 9)  # IoU 0.9 with a
    c = shapely.geometry.box(20, 20, 25, 25)
    out = nms_pool([[a, c], [b], [a]])
    assert len(out) == 2
    assert elbow([[a], [a, c], [a, c]]) == [a, c]


def test_iou_matrix_empty_gt_or_pred() -> None:
    box = shapely.geometry.box(0, 0, 10, 10)
    assert iou_matrix([], [box]).shape == (0, 1)
    assert iou_matrix([box], []).shape == (1, 0)


def test_iou_matrix_touching_boxes_have_zero_iou() -> None:
    """Boxes that only share an edge intersect but have zero overlap area."""
    left = shapely.geometry.box(0, 0, 5, 5)
    right = shapely.geometry.box(5, 0, 10, 5)
    assert iou_matrix([left], [right])[0, 0] == 0.0


def test_iou_exactly_0p5_is_not_matched_at_threshold_0p5() -> None:
    """Matching uses strict `>`, so an exact IoU-0.5 tie is a miss at t=0.5."""
    pred = shapely.geometry.box(0, 0, 10, 10)  # area 100
    gt = shapely.geometry.box(0, 0, 20, 10)  # area 200, fully contains pred
    ious = iou_matrix([gt], [pred])
    assert ious[0, 0] == pytest.approx(0.5)
    m = match([gt], [pred], thresholds=(0.5,))
    assert m.tp[0.5] == 0 and m.fp[0.5] == 1 and m.fn[0.5] == 1


def test_greedy_tie_break_favours_earlier_gt() -> None:
    """Two GT boxes tie at IoU 0.5 with the same prediction; greedy is GT-order."""
    pred = shapely.geometry.box(0, 0, 10, 10)  # area 100
    gt_a = shapely.geometry.box(0, 0, 20, 10)  # area 200, iou 0.5 with pred
    gt_b = shapely.geometry.box(0, 0, 10, 20)  # area 200, iou 0.5 with pred
    ious = iou_matrix([gt_a, gt_b], [pred])
    pairs = greedy(ious, 0.3)
    assert pairs == [(0, 0, pytest.approx(0.5))]  # gt_a wins, gt_b's candidate is taken


def test_diagnostics_empty_gt_or_pred() -> None:
    box = shapely.geometry.box(0, 0, 10, 10)
    assert diagnostics([], [box]) == {
        "gt_split": 0,
        "pred_merge": 0,
        "gt_missed": 0,
        "pred_spurious": 1,
    }
    assert diagnostics([box], []) == {
        "gt_split": 0,
        "pred_merge": 0,
        "gt_missed": 1,
        "pred_spurious": 0,
    }
