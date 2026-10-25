"""The common shape every strategy id maps to: a foreground rule plus a split."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from fbp.core.postprocess import apply_post_steps, drop_low_field, foreground
from fbp.core.regions import relabel_dense


@dataclass(frozen=True)
class Method:
    """A parsed strategy id: a foreground rule, a split rule, and post-steps."""

    fg: str = "arg"  # foreground rule
    min_m2: float = 0.0  # +A, minimum parcel area in square metres
    fill_px: int = 0  # +f, fill holes under this many pixels (-1 = all)
    merge_px: int = 0  # +m, merge parcels under this many pixels into a neighbour
    region_pf: int = 0  # +R, drop parcels whose mean field probability is under R/100
    quant: int = 0  # +q1, softmax on the 1/255 grid and the fast morphology backend
    fill_pf: int = 0  # +F, adopt pf{F} parcels where the primary covers < half of them
    fill_gate: int = 0  # +G, skip fill when < G% of the loose mask is uncovered

    rule: ClassVar[str] = ""  # registry keyword(s)
    needs_ctx: ClassVar[str | None] = None  # e.g. "gmm" when split() needs ctx[key]

    @property
    def backend(self) -> str:
        """Morphology backend for ``watershed_regions``: ``fast`` under ``+q1``."""
        return "fast" if self.quant else "skimage"

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
        """Foreground, split, region cut, then the fixed post-step order."""
        if self.quant:
            from fbp.core.fast_morph import quantize_prob  # noqa: PLC0415  (optional [fast] extra)

            prob = quantize_prob(prob)
        fg_mask = foreground(seg, prob, self.fg)
        inst = self.split(fg_mask, seg, prob, ctx)
        if self.region_pf:
            inst = drop_low_field(inst, prob[1], self.region_pf / 100.0)
        if self.fill_pf:
            inst = self._fill(inst, seg, prob, ctx)
        min_area = 0
        if self.min_m2:
            if px_m2 is None:
                raise ValueError(f"{self.id!r} needs the pixel area to use +A")
            min_area = max(1, round(self.min_m2 / px_m2))
        return apply_post_steps(inst, min_area, self.fill_px, self.merge_px)

    def _fill(
        self, inst: np.ndarray, seg: np.ndarray, prob: np.ndarray, ctx: dict | None
    ) -> np.ndarray:
        """Let ``pf{F}`` parcels replace what the primary foreground only fragmented.

        A loose pass (same split rule, same ``+R`` cut) proposes parcels. Each one
        is adopted only where the primary result covers less than half of its
        footprint, i.e. where the argmax foreground produced nothing or only
        fragments (a centre pivot at field probability 0.4 with a few confident
        patches inside). Where the primary already explains the area it is kept
        untouched, so ordinary fields are not re-merged by the looser mask.
        """
        fg_loose = foreground(seg, prob, f"pf{self.fill_pf}")
        if self.fill_gate:
            # adoption needs loose parcels that are >= 50% uncovered; where almost all
            # of the loose mask already sits inside a parcel (most windows) the second
            # watershed cannot adopt anything worth its cost, so skip it
            uncovered = np.count_nonzero(fg_loose & (inst == 0))
            if uncovered < self.fill_gate / 100.0 * np.count_nonzero(fg_loose):
                return inst
        loose = self.split(fg_loose, seg, prob, ctx)
        if self.region_pf:
            loose = drop_low_field(loose, prob[1], self.region_pf / 100.0)
        if loose.max() == 0:
            return inst
        n = int(loose.max()) + 1
        area = np.bincount(loose.ravel(), minlength=n)
        covered = np.bincount(loose[inst > 0].ravel(), minlength=n)
        adopt = covered < 0.5 * area
        adopt[0] = False
        if not adopt.any():
            return inst
        take = adopt[loose]
        out = inst.copy()
        out[take] = loose[take] + int(inst.max())
        return relabel_dense(out)

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
        if self.region_pf:
            mods.append(("R", self.region_pf))
        if self.fill_pf:
            mods.append(("F", self.fill_pf))
        if self.fill_gate:
            mods.append(("G", self.fill_gate))
        if self.min_m2:
            mods.append(("A", int(self.min_m2)))
        if self.quant:
            mods.append(("q", self.quant))
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
            "region_pf": mods.get("R", 0),
            "quant": mods.get("q", 0),
            "fill_pf": mods.get("F", 0),
            "fill_gate": mods.get("G", 0),
        }
