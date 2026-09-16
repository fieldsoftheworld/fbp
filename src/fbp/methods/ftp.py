"""``ws{h}``: the FTP pipeline, watershed at depth h then components of the union.

Field instances never touch (the watershed has no lines), so unioning them
back with ``inst > 0`` and re-labelling by connected components collapses
every watershed decision away again; this reproduces the FTP pipeline's own
polygon count exactly (see ``methods.ftp_pipeline`` and the equivalence
tests), not a meaningfully different split from ``none``.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from fbp.core.regions import components, watershed_regions
from fbp.core.surfaces import boundary_distance
from fbp.methods.base import Method
from fbp.methods.registry import register


@register("ws")
@dataclass(frozen=True)
class FTPMethod(Method):
    """Watershed of the field class at depth ``h``, FTP's own pipeline."""

    h: float = 0.0
    rule: ClassVar[str] = "ws"

    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        ctx: dict | None = None,
    ) -> np.ndarray:
        """Watershed the field class, then components of the union of its instances."""
        del fg_mask, prob, ctx  # FTP always splits the field class, not the fg rule
        regions = watershed_regions(
            boundary_distance(seg), seg == 1, self.h, lines=False
        )
        return components(regions.labels > 0)

    def _rule_id(self) -> str:
        return f"ws{self.h:g}"

    @classmethod
    def from_parts(
        cls,
        fg: str,
        rule: str,
        params: list,
        mods: dict[str, int],
    ) -> "FTPMethod":
        """Build from a parsed ``<fg>-ws{h}`` id."""
        del rule
        (h_s,) = params
        return cls(**cls._common_kwargs(fg, mods), h=float(h_s))
