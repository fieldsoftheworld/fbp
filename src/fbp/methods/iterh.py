"""Iterative watershed: re-split regions that stay too large at shrinking depths.

Koyuncu et al. (2016): watershed at the largest depth in ``HS``; any region
larger than ``max_region_px`` pixels is re-segmented with the next smaller depth,
recursively. The re-segmentation only ever needs the region's own pixels, so
it runs on that region's padded bounding box rather than the full image.
"""

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy.ndimage import find_objects

from fbp.core.regions import relabel_dense, watershed_regions
from fbp.core.surfaces import boundary_distance
from fbp.methods.base import Method

HS = (8.0, 4.0, 2.0, 1.0)


def _watershed(dist: np.ndarray, mask: np.ndarray, h: float) -> np.ndarray:
    return watershed_regions(dist, mask, h, lines=True).labels


def _resegment(dist: np.ndarray, region: np.ndarray, h: float) -> np.ndarray:
    """``_watershed`` restricted to ``region``'s own padded bounding box."""
    (bbox,) = find_objects(region.astype(np.int32))
    rs, cs = bbox
    r0, r1 = max(rs.start - 1, 0), min(rs.stop + 1, region.shape[0])
    c0, c1 = max(cs.start - 1, 0), min(cs.stop + 1, region.shape[1])
    part = _watershed(dist[r0:r1, c0:c1], region[r0:r1, c0:c1], h)
    full = np.zeros(region.shape, dtype=np.int32)
    full[r0:r1, c0:c1] = part
    return full


@dataclass(frozen=True, kw_only=True)
class IterativeWatershed(Method):
    """Watershed at depth 8, then re-split regions larger than ``max_region_px``.

    Each re-split uses the next smaller depth in ``HS`` (4, 2, then 1 pixel).

    Attributes:
        max_region_px: Regions larger than this many pixels are re-split.
    """

    max_region_px: int

    @override
    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        context: dict | None,
    ) -> np.ndarray:
        del prob, context
        dist = boundary_distance(seg) * fg_mask
        lab = _watershed(dist, fg_mask, HS[0])
        for h in HS[1:]:
            ids, counts = np.unique(lab[lab > 0], return_counts=True)
            nxt = lab.copy()
            cur = int(lab.max())
            for label_id, area in zip(ids, counts, strict=True):
                if area <= self.max_region_px:
                    continue
                part = _resegment(dist, lab == label_id, h)
                if part.max() <= 1:
                    continue
                nxt[lab == label_id] = 0
                nxt[part > 0] = part[part > 0] + cur
                cur += int(part.max())
            lab = nxt
        return relabel_dense(lab)
