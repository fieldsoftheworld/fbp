"""Options every method shares: the foreground rule and the steps after the split."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fbp.core.postprocess import apply_post_steps, drop_low_field, foreground
from fbp.core.regions import relabel_dense

Foreground = Literal["argmax", "not_background", "otsu"] | float


@dataclass(frozen=True, kw_only=True)
class Method:
    """Turn a (background, field, boundary) prediction into field instances.

    Attributes:
        foreground: Pixels to split: ``"argmax"`` (field wins the argmax),
            ``"not_background"`` (field or boundary wins), ``"otsu"`` (Otsu threshold
            on the field probability), or a number, which keeps pixels whose field
            probability is above it.
        min_mean_field_probability: Drop parcels whose mean field probability is
            under this. 0 keeps every parcel.
        refill_field_probability: Split the pixels whose field probability is above
            this the same way, and add each resulting parcel that the first split
            covers less than half of. 0 turns it off.
        refill_min_uncovered: Skip that second split when less than this fraction of
            its foreground is uncovered.
        fill_holes_px: Fill holes smaller than this many pixels. -1 fills every hole.
        merge_small_px: Merge parcels smaller than this many pixels into the
            neighbour they share the longest border with.
        min_area_m2: Drop parcels smaller than this many square metres.
        fast: Round the probabilities to the 1/255 grid of uint8 rasters and run
            h-maxima and the watershed on integer levels. Needs ``ftw-fbp[fast]``.
    """

    foreground: Foreground = "argmax"
    min_mean_field_probability: float = 0.0
    refill_field_probability: float = 0.0
    refill_min_uncovered: float = 0.0
    fill_holes_px: int = 0
    merge_small_px: int = 0
    min_area_m2: float = 0.0
    fast: bool = False

    @property
    def backend(self) -> str:
        """``"fast"`` when ``fast`` is set, otherwise ``"skimage"``."""
        return "fast" if self.fast else "skimage"

    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        context: dict | None,
    ) -> np.ndarray:
        """Instance labels for ``fg_mask``. Each method implements this."""
        raise NotImplementedError

    def __call__(
        self,
        seg: np.ndarray,
        prob: np.ndarray,
        pixel_area_m2: float | None = None,
        context: dict | None = None,
    ) -> np.ndarray:
        """Instance labels (int32, 0 = background) for ``seg`` and ``prob``.

        ``seg`` is the (H, W) class map and ``prob`` the (3, H, W) background, field,
        and boundary probabilities.
        """
        if self.fast:
            from fbp.core.fast_morph import quantize_prob  # noqa: PLC0415  (optional [fast] extra)

            prob = quantize_prob(prob)
        inst = self.split(foreground(seg, prob, self.foreground), seg, prob, context)
        if self.min_mean_field_probability:
            inst = drop_low_field(inst, prob[1], self.min_mean_field_probability)
        if self.refill_field_probability:
            inst = self._refill(inst, seg, prob, context)
        min_area = 0
        if self.min_area_m2:
            if pixel_area_m2 is None:
                raise ValueError("min_area_m2 needs pixel_area_m2")
            min_area = max(1, round(self.min_area_m2 / pixel_area_m2))
        return apply_post_steps(inst, min_area, self.fill_holes_px, self.merge_small_px)

    def _refill(
        self, inst: np.ndarray, seg: np.ndarray, prob: np.ndarray, context: dict | None
    ) -> np.ndarray:
        """Add parcels from the looser foreground that ``inst`` covers under half of.

        This recovers fields whose field probability rarely wins the argmax, which the
        first split leaves empty or in fragments. Parcels the first split already
        covers stay as they are.
        """
        loose_fg = foreground(seg, prob, self.refill_field_probability)
        uncovered = np.count_nonzero(loose_fg & (inst == 0))
        if uncovered < self.refill_min_uncovered * np.count_nonzero(loose_fg):
            return inst
        loose = self.split(loose_fg, seg, prob, context)
        if self.min_mean_field_probability:
            loose = drop_low_field(loose, prob[1], self.min_mean_field_probability)
        area = np.bincount(loose.ravel())
        covered = np.bincount(loose[inst > 0], minlength=len(area))
        adopt = covered < 0.5 * area
        adopt[0] = False
        if not adopt.any():
            return inst
        take = adopt[loose]
        out = inst.copy()
        out[take] = loose[take] + int(inst.max())
        return relabel_dense(out)
