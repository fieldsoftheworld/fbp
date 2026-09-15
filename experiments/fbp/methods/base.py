"""The common shape every strategy id maps to: a foreground rule plus a split."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from fbp.core.postprocess import apply_post_steps, foreground


@dataclass(frozen=True)
class Method:
    """A parsed strategy id: a foreground rule, a split rule, and post-steps."""

    fg: str = "arg"  # foreground rule
    min_m2: float = 0.0  # +A, minimum parcel area in square metres
    fill_px: int = 0  # +f, fill holes under this many pixels (-1 = all)
    merge_px: int = 0  # +m, merge parcels under this many pixels into a neighbour

    rule: ClassVar[str] = ""  # registry keyword(s)
    needs_ctx: ClassVar[str | None] = None  # e.g. "gmm" when split() needs ctx[key]

    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        ctx: dict | None,
    ) -> np.ndarray:
        """The split rule this strategy implements. Subclasses override this."""
        raise NotImplementedError

    def __call__(
        self,
        seg: np.ndarray,
        prob: np.ndarray,
        px_m2: float | None = None,
        ctx: dict | None = None,
    ) -> np.ndarray:
        """Foreground, split, then the fixed post-step order."""
        fg_mask = foreground(seg, prob, self.fg)
        inst = self.split(fg_mask, seg, prob, ctx)
        min_area = 0
        if self.min_m2:
            if px_m2 is None:
                raise ValueError(f"{self.id!r} needs the pixel area to use +A")
            min_area = max(1, round(self.min_m2 / px_m2))
        return apply_post_steps(inst, min_area, self.fill_px, self.merge_px)

    def _rule_id(self) -> str:
        """The ``rule[-params]`` part of the id, before the foreground and mods."""
        raise NotImplementedError

    def _extra_mods(self) -> list[tuple[str, int]]:
        """Rule-specific modifiers rendered before the generic ones (e.g. +l, +b)."""
        return []

    @property
    def id(self) -> str:
        """The legacy strategy id string this method was parsed from."""
        parts = [f"{self.fg}-{self._rule_id()}"]
        mods = [*self._extra_mods()]
        if self.fill_px:
            mods.append(("f", self.fill_px))
        if self.merge_px:
            mods.append(("m", self.merge_px))
        if self.min_m2:
            mods.append(("A", int(self.min_m2)))
        parts.extend(f"+{key}{val}" for key, val in mods)
        return "".join(parts)

    @classmethod
    def from_parts(
        cls, fg: str, rule: str, params: list, mods: dict[str, int]
    ) -> "Method":
        """Build an instance from the parsed ``fg``, ``rule``, params and mods.

        ``params`` is positional strings for the ``h``/``t``-style rules, or
        ``(key, value)`` pairs for the ``kv``-style prior rules.
        """
        raise NotImplementedError

    @classmethod
    def _common_kwargs(cls, fg: str, mods: dict[str, int]) -> dict:
        """The base-class fields every subclass's ``from_parts`` shares."""
        return {
            "fg": fg,
            "min_m2": float(mods.get("A", 0)),
            "fill_px": mods.get("f", 0),
            "merge_px": mods.get("m", 0),
        }
