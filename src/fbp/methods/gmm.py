"""``gmm``: Gaussian-mixture extent/boundary thresholds (Amin et al. 2025).

The extent and boundary probabilities are thresholded at values fitted per
region (``fbp.eval.gmm_fit``, agent B's port of the old ``gmm_thresholds``),
the boundary mask is thinned to one pixel, subtracted from the extent mask,
and the remainder labelled by connected components. The fit itself is an
eval-time concern; this method only consumes the two already-fitted numbers.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.ndimage import label as ndi_label
from skimage.morphology import thin

from fbp.methods.base import Method
from fbp.methods.registry import register


def gmm_split(prob: np.ndarray, te: float, tb: float) -> np.ndarray:
    """Extent at least ``te`` minus the thinned boundary mask at least ``tb``."""
    extent = (1.0 - prob[0]) >= te
    boundary = thin(prob[2] >= tb)
    out, _ = ndi_label(extent & ~boundary)
    return out.astype(np.int32, copy=False)


@register("gmm")
@dataclass(frozen=True)
class GMMMethod(Method):
    """Extent/boundary split at the region's fitted (te, tb)."""

    rule: ClassVar[str] = "gmm"
    needs_ctx: ClassVar[str | None] = "gmm"

    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        ctx: dict | None = None,
    ) -> np.ndarray:
        """Extent minus thinned boundary, at the fitted (te, tb) in ``ctx``."""
        del fg_mask, seg
        if not ctx or "gmm" not in ctx:
            raise ValueError(f"{self.id!r} needs the region's fitted (te, tb)")
        te, tb = ctx["gmm"]
        return gmm_split(prob, te, tb)

    def _rule_id(self) -> str:
        return "gmm"

    @classmethod
    def from_parts(
        cls,
        fg: str,
        rule: str,
        params: list,
        mods: dict[str, int],
    ) -> "GMMMethod":
        """Build from a parsed ``<fg>-gmm`` id; no rule params."""
        del rule, params
        return cls(**cls._common_kwargs(fg, mods))
