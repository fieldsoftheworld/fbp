"""Gaussian-mixture extent and boundary thresholds (Amin et al. 2025).

The extent and boundary probabilities are thresholded at values fitted per
region (``fbp.eval.gmm_fit``, agent B's port of the old ``gmm_thresholds``),
the boundary mask is thinned to one pixel, subtracted from the extent mask,
and the remainder labelled by connected components. The fit itself is an
eval-time concern; this method only consumes the two already-fitted numbers.
"""

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy.ndimage import label as ndi_label
from skimage.morphology import thin

from fbp.methods.base import Method


def gmm_split(prob: np.ndarray, te: float, tb: float) -> np.ndarray:
    """Extent at least ``te`` minus the thinned boundary mask at least ``tb``."""
    extent = (1.0 - prob[0]) >= te
    boundary = thin(prob[2] >= tb)
    out, _ = ndi_label(extent & ~boundary)
    return out.astype(np.int32, copy=False)


@dataclass(frozen=True, kw_only=True)
class GaussianMixture(Method):
    """Extent minus the thinned boundary, at thresholds fitted per region.

    Pass the fitted thresholds as ``context={"gmm": (extent, boundary)}``.
    """

    @override
    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        context: dict | None,
    ) -> np.ndarray:
        del fg_mask, seg
        if not context or "gmm" not in context:
            raise ValueError(
                "GaussianMixture needs context={'gmm': (extent, boundary)}"
            )
        te, tb = context["gmm"]
        return gmm_split(prob, te, tb)
