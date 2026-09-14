"""Ways of turning the candidate depth sweep into one polygon set per patch.

All functions take polygon lists in the eval-grid pixel frame and return the
same. ``vote`` is the per-pixel majority over the depth sweep; ``roundtrip`` is
the single-depth control that applies the same rasterize->polygonize step to one
depth; ``nms_pool`` and ``elbow`` are the alternatives ported from the legacy
``multi_hmin_ensemble.py`` so the comparison stays like-for-like.
"""

import math

import numpy as np
import rasterio.features
import shapely
import shapely.geometry

from . import ftp_compat as ftp


def rasterize(polys: list, shape: tuple[int, int]) -> np.ndarray:
    if not polys:
        return np.zeros(shape, dtype=np.uint8)
    return rasterio.features.rasterize(
        ((p, 1) for p in polys), out_shape=shape, fill=0, dtype="uint8"
    )


def polygonize(mask: np.ndarray) -> list:
    """Four-connected foreground components, identical to FTP's ``_extract_shapes``."""
    return ftp.extract_shapes(mask)


def roundtrip(polys: list, shape: tuple[int, int]) -> list:
    return polygonize(rasterize(polys, shape))


def vote_map(candidates: list[list], shape: tuple[int, int]) -> np.ndarray:
    """Per-pixel foreground vote count over K candidate polygon sets."""
    votes = np.zeros(shape, dtype=np.int32)
    for polys in candidates:
        if polys:
            votes += rasterize(polys, shape)
    return votes


def vote(candidates: list[list], shape: tuple[int, int], tau: float = 0.5) -> list:
    """Keep pixels with >= ceil(K*tau) foreground votes, then polygonize.

    Matches legacy ``strategy_c_pixel_vote``: ``k_min = max(1, ceil(K * tau))``,
    so even K at tau=0.5 assigns ties to foreground; a candidate with no
    polygons contributes an all-zero mask.
    """
    k = len(candidates)
    k_min = max(1, math.ceil(k * tau))
    votes = vote_map(candidates, shape)
    return polygonize((votes >= k_min).astype(np.uint8))


def iou(a, b) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def nms_pool(candidates: list[list], iou_thresh: float = 0.5) -> list:
    """Legacy strategy A: pool every candidate polygon, rank by how many other
    candidates contain an IoU>=thresh match, then greedy NMS at the same IoU."""
    pool = [p for polys in candidates for p in polys if not p.is_empty]
    if not pool:
        return []
    tree = shapely.STRtree(pool)
    support = np.zeros(len(pool), dtype=np.int32)
    for i, p in enumerate(pool):
        for j in tree.query(p, predicate="intersects"):
            if j != i and iou(p, pool[j]) >= iou_thresh:
                support[i] += 1
    order = np.argsort(-support, kind="stable")
    keep: list[int] = []
    suppressed = np.zeros(len(pool), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        for j in tree.query(pool[i], predicate="intersects"):
            if j != i and not suppressed[j] and iou(pool[i], pool[j]) >= iou_thresh:
                suppressed[j] = True
    return [pool[i] for i in keep]


def elbow(candidates: list[list]) -> list:
    """Legacy strategy D: pick the depth whose relative change in polygon count
    to the next depth is smallest (candidates ordered by increasing depth)."""
    counts = [len(p) for p in candidates]
    if len(counts) == 1:
        return candidates[0]
    best_i, best_score = 0, float("inf")
    for i in range(len(counts) - 1):
        if counts[i] == 0:
            continue
        score = abs(counts[i + 1] - counts[i]) / max(counts[i], 1)
        if score < best_score:
            best_i, best_score = i, score
    return candidates[best_i]
