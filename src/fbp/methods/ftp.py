"""The FTP pipeline: watershed the field class, then connected components of the union.

The watershed draws no lines, so taking components of ``labels > 0`` merges every
watershed region back together. This reproduces FTP's polygon count exactly (see
``ftp_pipeline`` and its equivalence tests) and is not a different split from
``Components``.
"""

from dataclasses import dataclass
from typing import override

import numpy as np

from fbp.core.regions import components, watershed_regions
from fbp.core.surfaces import boundary_distance
from fbp.methods.base import Method


@dataclass(frozen=True, kw_only=True)
class FTPWatershed(Method):
    """FTP's pipeline, which always splits the field class and ignores ``foreground``.

    Attributes:
        seed_depth: Minimum depth, in pixels of boundary distance, of a watershed seed.
    """

    seed_depth: float

    @override
    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        context: dict | None,
    ) -> np.ndarray:
        del fg_mask, prob, context
        regions = watershed_regions(
            boundary_distance(seg), seg == 1, self.seed_depth, lines=False
        )
        return components(regions.labels > 0)


def ftp_pipeline(seg: np.ndarray, seed_depth: float = 2.0) -> np.ndarray:
    """The FTP pipeline as a function of the class map alone."""
    return FTPWatershed(seed_depth=seed_depth)(
        seg, np.zeros((3, *seg.shape), np.float32)
    )
