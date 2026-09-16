"""``swst-n{n}-m{m}-t{t}``: stochastic watershed (Angulo & Jeulin 2007).

Repeat ``m`` times: draw ``n`` uniform random markers inside the foreground,
run the watershed of the FTP pipeline's distance surface with lines, and
accumulate the lines. The contour probability is the line frequency; pixels
at or above ``t`` are cut and the rest of the foreground is labelled by
connected components.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.ndimage import label as ndi_label
from skimage.segmentation import watershed

from fbp.core.surfaces import boundary_distance
from fbp.methods.base import Method
from fbp.methods.registry import register


@register("swst")
@dataclass(frozen=True)
class SWSTMethod(Method):
    """Random-marker watershed line frequency, cut at ``t``."""

    n: int = 0
    m: int = 0
    t: float = 0.0
    seed: int = 0
    rule: ClassVar[str] = "swst"

    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        ctx: dict | None = None,
    ) -> np.ndarray:
        """Cut where random-marker watershed lines agree at least ``t``."""
        del prob, ctx
        if not fg_mask.any():
            return np.zeros(fg_mask.shape, dtype=np.int32)
        dist = boundary_distance(seg) * fg_mask
        rng = np.random.default_rng(self.seed)
        r, c = np.nonzero(fg_mask)
        freq = np.zeros(fg_mask.shape, dtype=np.float32)
        for _ in range(self.m):
            idx = rng.choice(len(r), size=min(self.n, len(r)), replace=False)
            markers = np.zeros(fg_mask.shape, dtype=np.int32)
            markers[r[idx], c[idx]] = np.arange(1, len(idx) + 1)
            lab = watershed(-dist, markers=markers, mask=fg_mask, watershed_line=True)
            freq += fg_mask & (lab == 0)
        freq /= self.m
        out, _ = ndi_label(fg_mask & (freq < self.t))
        return out.astype(np.int32, copy=False)

    def _rule_id(self) -> str:
        return f"swst-n{self.n}-m{self.m}-t{self.t:g}"

    @classmethod
    def from_parts(
        cls,
        fg: str,
        rule: str,
        params: list,
        mods: dict[str, int],
    ) -> "SWSTMethod":
        """Build from a parsed ``<fg>-swst-n{n}-m{m}-t{t}`` id."""
        del rule
        kv = dict(params)
        return cls(
            **cls._common_kwargs(fg, mods),
            n=int(kv["n"]),
            m=int(kv["m"]),
            t=float(kv["t"]),
        )
