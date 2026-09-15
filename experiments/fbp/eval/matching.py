"""Per-patch matching and object-level diagnostics.

``match`` reproduces FTP's ``_match_shapes`` (greedy, GT order, first prediction
with IoU > t wins) but prefilters pairs with an STRtree, so the result is
identical and much faster on dense patches. ``tests/test_eval_matching.py``
checks equality against FTP's implementation.

``diagnostics`` counts the outcomes PQ hides: reference parcels split across
several predictions, predictions merging several parcels, missed parcels, and
spurious predictions, using a 50% area-coverage rule.
"""

from dataclasses import dataclass

import numpy as np
import shapely

from .ftp_compat import AP_IOU_THRESHOLDS

THRESHOLDS: tuple[float, ...] = tuple(AP_IOU_THRESHOLDS)
COVER = 0.5


def iou_matrix(gt: list, pred: list) -> np.ndarray:
    """Pairwise IoU of every (gt, pred) polygon, prefiltered with an STRtree."""
    n_gt, n_pred = len(gt), len(pred)
    ious = np.zeros((n_gt, n_pred), dtype=np.float32)
    if n_gt == 0 or n_pred == 0:
        return ious
    tree = shapely.STRtree(pred)
    for i, g in enumerate(gt):
        for j in tree.query(g, predicate="intersects"):
            p = pred[j]
            inter = g.intersection(p).area
            if inter <= 0:
                continue
            union = g.area + p.area - inter
            if union <= 0:
                continue
            ious[i, j] = inter / union
    return ious


def greedy(ious: np.ndarray, t: float) -> list[tuple[int, int, float]]:
    """FTP's greedy rule on a precomputed IoU matrix."""
    n_gt, _n_pred = ious.shape
    matched_j: set[int] = set()
    pairs = []
    for i in range(n_gt):
        row = ious[i]
        for j in np.flatnonzero(row > t):
            if j in matched_j:
                continue
            matched_j.add(int(j))
            pairs.append((i, int(j), float(row[j])))
            break
    return pairs


@dataclass
class MatchResult:
    """Per-threshold TP/FP/FN counts, plus the pairs matched at the first threshold."""

    n_gt: int
    n_pred: int
    tp: dict[float, int]
    fp: dict[float, int]
    fn: dict[float, int]
    pairs_50: list[tuple[int, int, float]]
    ious: np.ndarray


def match(
    gt: list, pred: list, thresholds: tuple[float, ...] = THRESHOLDS
) -> MatchResult:
    """Greedy IoU matching at each threshold; ``pairs_50`` is the first threshold's."""
    ious = iou_matrix(gt, pred)
    tp, fp, fn = {}, {}, {}
    pairs_50: list[tuple[int, int, float]] = []
    for t in thresholds:
        pairs = greedy(ious, t)
        tp[t] = len(pairs)
        fp[t] = len(pred) - len(pairs)
        fn[t] = len(gt) - len(pairs)
        if t == thresholds[0]:
            pairs_50 = pairs
    return MatchResult(len(gt), len(pred), tp, fp, fn, pairs_50, ious)


def diagnostics(gt: list, pred: list) -> dict[str, int]:
    """Split / merge / miss / spurious counts under a 50% coverage rule.

    * a prediction *claims* a reference parcel if it covers > COVER of that
      parcel's area
    * a reference parcel *absorbs* a prediction if > COVER of the prediction
      lies inside it
    * split:    reference parcel absorbing >= 2 predictions
    * merge:    prediction claiming >= 2 reference parcels
    * missed:   reference parcel with no prediction covering > COVER of it
    * spurious: prediction with < COVER of its area on any reference parcel
    """
    n_gt, n_pred = len(gt), len(pred)
    out = {"gt_split": 0, "pred_merge": 0, "gt_missed": 0, "pred_spurious": 0}
    if n_gt == 0:
        out["pred_spurious"] = n_pred
        return out
    if n_pred == 0:
        out["gt_missed"] = n_gt
        return out
    tree = shapely.STRtree(pred)
    gt_area = np.array([g.area for g in gt])
    pred_area = np.array([p.area for p in pred])
    inter = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, g in enumerate(gt):
        for j in tree.query(g, predicate="intersects"):
            inter[i, j] = g.intersection(pred[j]).area
    frac_of_gt = inter / np.maximum(gt_area[:, None], 1e-9)
    frac_of_pred = inter / np.maximum(pred_area[None, :], 1e-9)
    out["gt_split"] = int(((frac_of_pred > COVER).sum(axis=1) >= 2).sum())
    out["pred_merge"] = int(((frac_of_gt > COVER).sum(axis=0) >= 2).sum())
    out["gt_missed"] = int((frac_of_gt.max(axis=1) <= COVER).sum())
    out["pred_spurious"] = int(
        (inter.sum(axis=0) / np.maximum(pred_area, 1e-9) < COVER).sum()
    )
    return out
