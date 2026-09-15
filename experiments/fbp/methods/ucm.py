"""``ucm-k{k}``: over-segment then merge below k, the ultrametric-contour-map recipe.

Persello et al. (2019) apply this mechanism (Arbelaez et al. 2011; Maninis et
al. 2018) to fields: over-segment the foreground with the FTP pipeline's
distance watershed at a small depth, weight each region adjacency by the mean
boundary probability along the shared border, and greedily merge the weakest
adjacency while it is below ``k``.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from fbp.core.merge import merge_hierarchy
from fbp.core.regions import watershed_regions
from fbp.core.surfaces import boundary_distance
from fbp.methods.base import Method
from fbp.methods.registry import register

H_OVERSEGMENT = 0.5  # depth of the over-segmenting watershed


@register("ucm")
@dataclass(frozen=True)
class UCMMethod(Method):
    """Distance-watershed over-segmentation, merged below ``k`` on p_boundary."""

    k: float = 0.0
    rule: ClassVar[str] = "ucm"

    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        ctx: dict | None = None,
    ) -> np.ndarray:
        """Over-segment at a small depth, then merge below ``k``."""
        del ctx
        regions = watershed_regions(
            boundary_distance(seg), fg_mask, H_OVERSEGMENT, lines=False
        )
        if regions.fallback:
            return regions.labels
        return merge_hierarchy(regions.labels, prob[2], fg_mask, self.k)

    def _rule_id(self) -> str:
        return f"ucm-k{self.k:g}"

    @classmethod
    def from_parts(
        cls,
        fg: str,
        rule: str,
        params: list,
        mods: dict[str, int],
    ) -> "UCMMethod":
        """Build from a parsed ``<fg>-ucm-k{k}`` id."""
        del rule
        kv = dict(params)
        return cls(**cls._common_kwargs(fg, mods), k=float(kv["k"]))
