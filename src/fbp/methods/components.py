"""Connected components of the foreground, optionally after closing the boundary."""

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy.ndimage import binary_closing
from scipy.ndimage import label as ndi_label

from fbp.core.regions import components
from fbp.methods.base import Method


@dataclass(frozen=True, kw_only=True)
class Components(Method):
    """Each connected component of the foreground is one field."""

    @override
    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        context: dict | None,
    ) -> np.ndarray:
        del seg, prob, context
        return components(fg_mask)


@dataclass(frozen=True, kw_only=True)
class ClosedComponents(Method):
    """Close the boundary class, then take connected components of the rest.

    Attributes:
        closing_radius: Half-width in pixels of the square used to close the
            boundary class.
    """

    closing_radius: int

    @override
    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        context: dict | None,
    ) -> np.ndarray:
        del prob, context
        size = 2 * self.closing_radius + 1
        closed = binary_closing(seg == 2, structure=np.ones((size, size), dtype=bool))
        out, _ = ndi_label(fg_mask & ~closed)
        return out.astype(np.int32, copy=False)
