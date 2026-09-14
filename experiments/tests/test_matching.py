"""STRtree-accelerated matching must equal FTP's brute-force greedy matcher."""

import numpy as np
import shapely.geometry
from fieldvote import ftp_compat as ftp
from fieldvote.matching import THRESHOLDS, diagnostics, match
from fieldvote.strategies import elbow, nms_pool, roundtrip, vote


def random_boxes(rng, n, size=100):
    out = []
    for _ in range(n):
        x, y = rng.uniform(0, size, 2)
        w, h = rng.uniform(3, 20, 2)
        out.append(shapely.geometry.box(x, y, x + w, y + h))
    return out


def test_match_equals_ftp():
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


def test_diagnostics_split_merge():
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


def test_vote_majority_and_ties():
    shape = (20, 20)
    a = [shapely.geometry.box(0, 0, 10, 10)]
    b = [shapely.geometry.box(0, 0, 10, 10), shapely.geometry.box(12, 12, 18, 18)]
    # K=2, tau=0.5 -> k_min=1: union (ties to foreground)
    out = vote([a, b], shape)
    assert len(out) == 2
    # K=3 -> k_min=2: only the box present twice survives
    out = vote([a, b, []], shape)
    assert len(out) == 1 and out[0].area == 100


def test_roundtrip_merges_touching_polygons():
    shape = (20, 20)
    left = shapely.geometry.box(0, 0, 5, 10)
    right = shapely.geometry.box(5, 0, 10, 10)
    assert len(roundtrip([left, right], shape)) == 1


def test_nms_and_elbow_smoke():
    a = shapely.geometry.box(0, 0, 10, 10)
    b = shapely.geometry.box(0, 0, 10, 9)  # IoU 0.9 with a
    c = shapely.geometry.box(20, 20, 25, 25)
    out = nms_pool([[a, c], [b], [a]])
    assert len(out) == 2
    assert elbow([[a], [a, c], [a, c]]) == [a, c]
