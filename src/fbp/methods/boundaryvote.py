"""BoundaryVote: watershed the foreground, then keep the lines the boundary supports.

The watershed proposes splits. Each line between two regions is kept when enough of
its pixels look like boundary, and otherwise the two regions merge.
"""

from dataclasses import dataclass
from typing import Literal, override

import numpy as np
from scipy.ndimage import binary_dilation

from fbp.core.merge import merge_by_lines
from fbp.core.regions import pairs_at_lines, watershed_regions
from fbp.core.surfaces import boundary_distance, boundary_prob, oriented_signal
from fbp.methods.base import Method

Surface = Literal["boundary_probability", "boundary_distance", "oriented_boundary"]
LineScore = Literal["fraction", "probability_share", "mean_boundary", "margin"]


@dataclass(frozen=True, kw_only=True)
class BoundaryVote(Method):
    """Watershed the foreground and keep the lines whose pixels vote for boundary.

    Attributes:
        surface: What the watershed floods. ``"boundary_probability"`` is
            1 - p(boundary), ``"boundary_distance"`` is the distance to the nearest
            predicted boundary pixel, and ``"oriented_boundary"`` is 1 - the oriented
            boundary signal of Zhu et al. (2024).
        seed_depth: Minimum depth of a seed maximum, in the surface's units
            (probability, or pixels for ``"boundary_distance"``). 0 seeds every
            regional maximum.
        line_score: How each line pixel votes. ``"fraction"`` counts pixels where
            p(boundary) >= p(field). ``"probability_share"`` averages
            p(boundary) / (p(boundary) + p(field)), ``"mean_boundary"`` averages
            p(boundary), and ``"margin"`` averages (p(boundary) - p(field) + 1) / 2.
        min_line_support: Keep a line when its mean vote is at least this.
        min_line_length: Merge across lines shorter than this many pixels.
        support_radius: A line pixel also votes for boundary when any pixel within
            this many pixels does. With ``"probability_share"`` this switches the vote
            to ``"fraction"``.
    """

    surface: Surface = "boundary_probability"
    seed_depth: float
    line_score: LineScore = "fraction"
    min_line_support: float
    min_line_length: int = 0
    support_radius: int = 0

    @override
    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        context: dict | None,
    ) -> np.ndarray:
        del context
        surf = _surface(self.surface, fg_mask, seg, prob)
        regions = watershed_regions(
            surf, fg_mask, self.seed_depth, lines=True, backend=self.backend
        )
        if regions.fallback:
            return regions.labels
        inst, lines = regions.labels, regions.lines
        if not lines.any():
            return inst
        r, c, pairs = pairs_at_lines(inst, lines)
        valid = pairs[:, 0] > 0
        support = _line_support(prob, r, c, self.line_score, self.support_radius)
        keys = pairs[valid, 0].astype(np.int64) * (inst.max() + 1) + pairs[valid, 1]
        uniq, inv = np.unique(keys, return_inverse=True)
        n_px = np.bincount(inv, minlength=len(uniq))
        n_sup = np.bincount(inv, weights=support[valid], minlength=len(uniq))
        keep_line = (n_sup / n_px >= self.min_line_support) & (
            n_px >= self.min_line_length
        )
        return merge_by_lines(inst, r, c, pairs, keep_line, uniq)


def _surface(
    surface: Surface, fg_mask: np.ndarray, seg: np.ndarray, prob: np.ndarray
) -> np.ndarray:
    if surface == "boundary_probability":
        return boundary_prob(prob) * fg_mask
    if surface == "boundary_distance":
        return boundary_distance(seg) * fg_mask
    if surface == "oriented_boundary":
        return (1.0 - oriented_signal(prob[2].astype(np.float32)).max(axis=0)) * fg_mask
    raise ValueError(f"unknown surface {surface!r}")


def _line_support(
    prob: np.ndarray, r: np.ndarray, c: np.ndarray, score: LineScore, radius: int
) -> np.ndarray:
    """Each line pixel's vote for boundary, in [0, 1]."""
    is_bnd = prob[2] >= prob[1]
    if radius > 0:
        is_bnd = binary_dilation(
            is_bnd, structure=np.ones((2 * radius + 1,) * 2, dtype=bool)
        )
    pb, pf = prob[2][r, c].astype(np.float64), prob[1][r, c].astype(np.float64)
    if score == "fraction" or (score == "probability_share" and radius > 0):
        return is_bnd[r, c].astype(np.float64)
    if score == "probability_share":
        return np.divide(pb, pb + pf, out=np.full(pb.shape, 0.5), where=(pb + pf) > 0)
    if score == "mean_boundary":
        return pb
    if score == "margin":
        return (pb - pf + 1.0) / 2.0
    raise ValueError(f"unknown line score {score!r}")
