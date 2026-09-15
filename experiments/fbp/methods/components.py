"""``none``: connected components. ``close{r}``: close the boundary class first."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.ndimage import binary_closing
from scipy.ndimage import label as ndi_label

from fbp.core.regions import components
from fbp.methods.base import Method
from fbp.methods.registry import register


@register("none")
@dataclass(frozen=True)
class ComponentsMethod(Method):
    """Connected components of the foreground, no split."""

    rule: ClassVar[str] = "none"

    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        ctx: dict | None = None,
    ) -> np.ndarray:
        """Connected components of ``fg_mask``."""
        del seg, prob, ctx
        return components(fg_mask)

    def _rule_id(self) -> str:
        return "none"

    @classmethod
    def from_parts(
        cls,
        fg: str,
        rule: str,
        params: list,
        mods: dict[str, int],
    ) -> "ComponentsMethod":
        """Build from a parsed ``<fg>-none`` id; no rule params."""
        del rule, params
        return cls(**cls._common_kwargs(fg, mods))


@register("close")
@dataclass(frozen=True)
class CloseMethod(Method):
    """Close the boundary class with a (2r+1)^2 square, then take components."""

    r: int = 0
    rule: ClassVar[str] = "close"

    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        ctx: dict | None = None,
    ) -> np.ndarray:
        """Close the boundary class, then components of what remains."""
        del prob, ctx
        closed = binary_closing(
            seg == 2, structure=np.ones((2 * self.r + 1, 2 * self.r + 1), dtype=bool)
        )
        out, _ = ndi_label(fg_mask & ~closed)
        return out.astype(np.int32, copy=False)

    def _rule_id(self) -> str:
        return f"close{self.r}"

    @classmethod
    def from_parts(
        cls,
        fg: str,
        rule: str,
        params: list,
        mods: dict[str, int],
    ) -> "CloseMethod":
        """Build from a parsed ``<fg>-close{r}`` id."""
        del rule
        (r_s,) = params
        return cls(**cls._common_kwargs(fg, mods), r=int(r_s))
