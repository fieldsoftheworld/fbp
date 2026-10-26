"""Extent and boundary thresholds, as in DECODE."""

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy.ndimage import label as ndi_label

from fbp.methods.base import Method


@dataclass(frozen=True, kw_only=True)
class ExtentBoundaryThreshold(Method):
    """Keep pixels that are likely field and unlikely boundary, then take components.

    This ignores ``foreground``.

    Attributes:
        min_field_probability: Keep pixels whose field probability is at least this.
        max_boundary_probability: Keep pixels whose boundary probability is under this.
    """

    min_field_probability: float
    max_boundary_probability: float

    @override
    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        context: dict | None,
    ) -> np.ndarray:
        del fg_mask, seg, context
        mask = (prob[1] >= self.min_field_probability) & (
            prob[2] < self.max_boundary_probability
        )
        out, _ = ndi_label(mask)
        return out.astype(np.int32, copy=False)
