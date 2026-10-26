"""Ultrametric contour maps: over-segment, then merge regions with weak borders.

Persello et al. (2019) apply this mechanism (Arbelaez et al. 2011; Maninis et
al. 2018) to fields: over-segment the foreground with the FTP pipeline's
distance watershed at a small depth, weight each region adjacency by the mean
boundary probability along the shared border, and greedily merge the weakest
adjacency while it is below ``merge_threshold``.
"""

from dataclasses import dataclass
from typing import override

import numpy as np

from fbp.core.merge import merge_hierarchy
from fbp.core.regions import watershed_regions
from fbp.core.surfaces import boundary_distance
from fbp.methods.base import Method

H_OVERSEGMENT = 0.5  # depth of the over-segmenting watershed


@dataclass(frozen=True, kw_only=True)
class UltrametricContours(Method):
    """Over-segment with a shallow distance watershed, then merge weak borders.

    Attributes:
        merge_threshold: Merge two regions while the mean boundary probability on
            their shared border is under this.
    """

    merge_threshold: float

    @override
    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        context: dict | None,
    ) -> np.ndarray:
        del context
        regions = watershed_regions(
            boundary_distance(seg), fg_mask, H_OVERSEGMENT, lines=False
        )
        if regions.fallback:
            return regions.labels
        return merge_hierarchy(regions.labels, prob[2], fg_mask, self.merge_threshold)
