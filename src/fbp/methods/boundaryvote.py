"""``pb|sup|osup|ssup|asup|dsup-h{h}-t{tau}``: watershed, then vote on its lines.

Watershed the chosen surface at depth ``h``, then keep or merge each line by a
vote of the pixels along it: a line survives only if at least ``tau`` of its
pixels favour the boundary class, otherwise the two regions it separates
merge. The surface and the line's support statistic both depend on ``kind``:

* ``pb``   1 - p_boundary, seeded at its own regional maxima (BoundaryVote as
           shipped).
* ``pbg``  as ``pb`` but the surface is 1 - max(p_boundary, |grad p_field|) and a
           line pixel also counts as supported where the field probability has an
           edge: cuts at rims the boundary head missed (centre pivots).
* ``sup``  the EDT-of-boundary surface, ``h`` in pixels (the distance-map form).
* ``osup`` 1 - the oriented boundary signal of Zhu et al. (max over orientation).
* ``ssup``/``asup``/``dsup``  the distance form (as ``sup``) with another line
           statistic: mean probability share, mean p_boundary, mean margin.

``h = 0`` seeds every regional maximum. ``+l{L}`` requires a kept line to be
at least ``L`` px long; ``+b{R}`` lets a line pixel count as supported when
any pixel within ``R`` favours boundary.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.ndimage import binary_dilation

from fbp.core.merge import merge_by_lines
from fbp.core.regions import pairs_at_lines, watershed_regions
from fbp.core.surfaces import (
    boundary_distance,
    boundary_or_edge,
    boundary_prob,
    field_edge,
    oriented_signal,
)
from fbp.methods.base import Method
from fbp.methods.registry import register

# kind -> the watershed surface the split proposals come from (default "edt")
SURFACE = {"pb": "pb", "pbg": "pbg", "osup": "owt"}
# kind -> how a watershed line's support is scored (default "frac")
LINE_SCORE = {
    "sup": "frac",
    "pb": "frac",
    "pbg": "fracg",
    "osup": "frac",
    "ssup": "soft",
    "asup": "mean",
    "dsup": "diff",
}
KINDS = ("pb", "pbg", "sup", "osup", "ssup", "asup", "dsup")
EDGE_TAU = 0.5  # field-edge strength at which a line pixel counts as supported (pbg)


def _surface(
    kind: str, fg_mask: np.ndarray, seg: np.ndarray, prob: np.ndarray
) -> np.ndarray:
    """The surface whose regional maxima seed the over-segmentation."""
    name = SURFACE.get(kind, "edt")
    if name == "edt":
        return boundary_distance(seg) * fg_mask
    if name == "pb":
        return boundary_prob(prob) * fg_mask
    if name == "pbg":
        return boundary_or_edge(prob) * fg_mask
    return (1.0 - oriented_signal(prob[2].astype(np.float32)).max(axis=0)) * fg_mask


def _line_support(
    prob: np.ndarray, r: np.ndarray, c: np.ndarray, score: str, band: int
) -> np.ndarray:
    """Per line pixel, its vote in [0, 1] under the chosen statistic."""
    is_bnd = prob[2] >= prob[1]
    if band > 0:
        # a line can sit a pixel off the predicted boundary; count a pixel as
        # supported when any pixel within `band` of it favours boundary
        is_bnd = binary_dilation(
            is_bnd, structure=np.ones((2 * band + 1, 2 * band + 1), dtype=bool)
        )
    pb, pf = prob[2][r, c].astype(np.float64), prob[1][r, c].astype(np.float64)
    if score == "fracg":
        return (is_bnd[r, c] | (field_edge(prob)[r, c] >= EDGE_TAU)).astype(np.float64)
    if score == "frac" or (score == "soft" and band > 0):
        # the band is a spatial rule, so the soft score falls back to the hard test
        return is_bnd[r, c].astype(np.float64)
    if score == "soft":
        return np.divide(pb, pb + pf, out=np.full(pb.shape, 0.5), where=(pb + pf) > 0)
    if score == "mean":
        return pb
    return (pb - pf + 1.0) / 2.0  # score == "diff"


@register(*KINDS)
@dataclass(frozen=True)
class BoundaryVoteMethod(Method):
    """Watershed at depth ``h``, keeping a line only when its vote clears ``tau``."""

    kind: str = "sup"
    h: float = 0.0
    tau: float = 0.0
    min_line: int = 0  # +l
    band: int = 0  # +b
    rule: ClassVar[str] = "sup"

    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        ctx: dict | None = None,
    ) -> np.ndarray:
        """Watershed at depth ``h``, then keep or merge lines by their vote."""
        del ctx
        surf = _surface(self.kind, fg_mask, seg, prob)
        regions = watershed_regions(
            surf, fg_mask, self.h, lines=True, backend=self.backend
        )
        if regions.fallback:
            return regions.labels
        inst, lines = regions.labels, regions.lines
        if not lines.any():
            return inst
        r, c, pairs = pairs_at_lines(inst, lines)
        valid = pairs[:, 0] > 0
        score = LINE_SCORE.get(self.kind, "frac")
        support = _line_support(prob, r, c, score, self.band)
        keys = pairs[valid, 0].astype(np.int64) * (inst.max() + 1) + pairs[valid, 1]
        uniq, inv = np.unique(keys, return_inverse=True)
        n_px = np.bincount(inv, minlength=len(uniq))
        n_sup = np.bincount(inv, weights=support[valid], minlength=len(uniq))
        keep_line = (n_sup / n_px >= self.tau) & (n_px >= self.min_line)
        return merge_by_lines(inst, r, c, pairs, keep_line, uniq)

    def _rule_id(self) -> str:
        return f"{self.kind}-h{self.h:g}-t{self.tau:g}"

    def _extra_mods(self) -> list[tuple[str, int]]:
        out = []
        if self.min_line:
            out.append(("l", self.min_line))
        if self.band:
            out.append(("b", self.band))
        return out

    @classmethod
    def from_parts(
        cls,
        fg: str,
        rule: str,
        params: list,
        mods: dict[str, int],
    ) -> "BoundaryVoteMethod":
        """Build from a parsed ``<fg>-{kind}-h{h}-t{tau}`` id."""
        h_s, t_s = params
        return cls(
            **cls._common_kwargs(fg, mods),
            kind=rule,
            h=float(h_s),
            tau=float(t_s),
            min_line=mods.get("l", 0),
            band=mods.get("b", 0),
        )
