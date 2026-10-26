"""``owt-l{l}-h{h}``: oriented watershed and hierarchical merging (Zhu et al. 2024).

An oriented boundary signal is the mean of p_boundary along a short line at
each of eight orientations; its maximum over orientations is watershed-
segmented inside the foreground, seeded at its regional minima. The arc
between two regions is weighted by the signal at the arc's own orientation
along the arc's pixels, and regions merge greedily below one level ``lvl``.
Higra is not installed here, so the hierarchy cut is skimage's greedy merge.

Two watersheds share the same seeds: the lines=False one gives the regions
the RAG merges, the lined one gives the arcs the oriented weights are read on.

Seeding on ``surf``'s minima goes through ``watershed_regions(..., mode="min")``
rather than negating ``surf`` into a maxima search: h_maxima/h_minima scale
their floating-point tolerance by each pixel's own magnitude, so negating
after adding an offset (to keep the region outside ``fg`` from reading as a
false maximum) changes that tolerance and does flip which of two nearly-tied
pixels seeds first on real data. ``mode="min"`` avoids the offset entirely.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from fbp.core.merge import merge_hierarchy
from fbp.core.regions import pairs_at_lines, watershed_regions
from fbp.core.surfaces import N_ORIENT, oriented_signal
from fbp.methods.base import Method
from fbp.methods.registry import register

H_MIN = 0.05  # default depth of the regional minima seeding the over-segmentation


def orientation_bin(r: np.ndarray, c: np.ndarray) -> int | None:
    """Principal direction of a pixel set as an index into ``N_ORIENT`` angles."""
    if len(r) < 3:
        return None
    x, y = c - c.mean(), -(r - r.mean())
    cov = np.cov(np.stack([x, y]))
    _, vec = np.linalg.eigh(cov)
    vx, vy = vec[:, -1]
    theta = np.arctan2(vy, vx) % np.pi
    return int(round(theta / (np.pi / N_ORIENT))) % N_ORIENT


def border_pixels(
    lined: np.ndarray, fg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows, cols, and sorted region pairs of the watershed-line pixels."""
    r, c, pairs = pairs_at_lines(lined, fg & (lined == 0))
    keep = pairs[:, 0] > 0
    return r[keep], c[keep], pairs[keep]


def border_weights(
    lined: np.ndarray, fg: np.ndarray, sig: np.ndarray
) -> dict[tuple[int, int], tuple[float, int]]:
    """Per region pair: mean oriented signal along their arc and the arc's length."""
    r, c, pairs = border_pixels(lined, fg)
    if len(r) == 0:
        return {}
    keys = pairs[:, 0].astype(np.int64) * (lined.max() + 1) + pairs[:, 1]
    order = np.argsort(keys, kind="stable")
    keys, r, c = keys[order], r[order], c[order]
    bounds = np.flatnonzero(np.diff(keys)) + 1
    out = {}
    unoriented = sig.max(axis=0)
    for rr, cc, k in zip(
        np.split(r, bounds), np.split(c, bounds), keys[np.r_[0, bounds]], strict=True
    ):
        b = orientation_bin(rr, cc)
        vals = unoriented[rr, cc] if b is None else sig[b][rr, cc]
        a, bb = divmod(int(k), lined.max() + 1)
        out[(a, bb)] = (float(vals.mean()), len(rr))
    return out


@register("owt")
@dataclass(frozen=True)
class OWTMethod(Method):
    """Oriented watershed seeded at ``h``-minima, merged greedily below ``lvl``."""

    lvl: float = 0.0
    h: float = H_MIN
    rule: ClassVar[str] = "owt"

    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        ctx: dict | None = None,
    ) -> np.ndarray:
        """Oriented watershed at ``h``, greedily merged below ``lvl``."""
        del seg, ctx  # the method reads the probability, not the argmax
        if not fg_mask.any():
            return np.zeros(fg_mask.shape, dtype=np.int32)
        sig = oriented_signal(prob[2].astype(np.float32))
        surf = sig.max(axis=0).astype(np.float32)
        regions = watershed_regions(surf, fg_mask, self.h, lines=False, mode="min")
        if regions.fallback:
            return regions.labels
        over = regions.labels
        if over.max() <= 1:
            return over
        lined = watershed_regions(surf, fg_mask, self.h, lines=True, mode="min").labels
        weights = border_weights(lined, fg_mask, sig)
        return merge_hierarchy(over, surf, fg_mask, self.lvl, weights)

    def _rule_id(self) -> str:
        return f"owt-l{self.lvl:g}-h{self.h:g}"

    @classmethod
    def from_parts(
        cls,
        fg: str,
        rule: str,
        params: list,
        mods: dict[str, int],
    ) -> "OWTMethod":
        """Build from a parsed ``<fg>-owt-l{l}[-h{h}]`` id; ``h`` defaults to 0.05."""
        del rule
        kv = dict(params)
        return cls(
            **cls._common_kwargs(fg, mods),
            lvl=float(kv["l"]),
            h=float(kv.get("h", H_MIN)),
        )
