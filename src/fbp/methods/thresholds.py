"""``thr-te{te}-tb{tb}``: extent/boundary thresholding, as in DECODE."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.ndimage import label as ndi_label

from fbp.methods.base import Method
from fbp.methods.registry import register


@register("thr")
@dataclass(frozen=True)
class ThresholdMethod(Method):
    """``p_field >= te`` and ``p_boundary < tb``, then connected components."""

    te: float = 0.0
    tb: float = 0.0
    rule: ClassVar[str] = "thr"

    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        ctx: dict | None = None,
    ) -> np.ndarray:
        """Extent/boundary threshold, then connected components."""
        del fg_mask, seg, ctx
        mask = (prob[1] >= self.te) & (prob[2] < self.tb)
        out, _ = ndi_label(mask)
        return out.astype(np.int32, copy=False)

    def _rule_id(self) -> str:
        return f"thr-te{self.te:g}-tb{self.tb:g}"

    @classmethod
    def from_parts(
        cls,
        fg: str,
        rule: str,
        params: list,
        mods: dict[str, int],
    ) -> "ThresholdMethod":
        """Build from a parsed ``<fg>-thr-te{te}-tb{tb}`` id."""
        del rule
        kv = dict(params)
        return cls(
            **cls._common_kwargs(fg, mods), te=float(kv["te"]), tb=float(kv["tb"])
        )
