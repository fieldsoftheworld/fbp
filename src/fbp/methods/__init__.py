"""Strategy ids, parsed into ``Method`` objects and run against (seg, prob).

A strategy id is ``<fg>-<rule>[-<param>...][+<mod>...]`` with ``fg`` one of
``arg``/``nbg``/``pf50``/``otsu`` and ``rule`` one of ``none``, ``close{r}``,
``ws{h}``, ``pb|sup|osup|ssup|asup|dsup-h{h}-t{tau}``, ``thr-te{te}-tb{tb}``,
``ucm-k{k}``, ``iterh-a{a}``, ``swst-n{n}-m{m}-t{t}``, ``owt-l{l}-h{h}``, or
``gmm``. ``+A{m2}`` drops parcels under that many square metres (needs
``px_m2``), ``+l{L}``/``+b{R}`` are BoundaryVote-only line modifiers,
``+f{A}`` fills holes under ``A`` px (``-1`` fills every hole), and ``+m{A}``
merges parcels under ``A`` px into their longest-bordering neighbour.
"""

import numpy as np

from fbp.methods import (  # noqa: F401  (import for the registration side effect)
    boundaryvote,
    components,
    ftp,
    gmm,
    iterh,
    owt,
    swst,
    thresholds,
    ucm,
)
from fbp.methods.base import Method
from fbp.methods.registry import REGISTRY

FG_MODES = ("arg", "nbg", "pf50", "otsu")


def _parse_mods(mod_strs: list[str], spec: str) -> dict[str, int]:
    """``{key: int(value)}`` from the ``+`` suffixes of a strategy id."""
    mods: dict[str, int] = {}
    for mod in mod_strs:
        key, val = mod[0], int(mod[1:])
        if key not in "lbfmA":
            raise ValueError(f"unknown modifier {mod!r} in {spec!r}")
        mods[key] = val
    return mods


def _parse_kv(rest: str) -> list[tuple[str, str]]:
    """The ``kind``'s ``key-value`` args, keyed by their first 1 (or 2) chars.

    ``te``/``tb`` keep two letters so they don't collide with ``t``.
    """
    _kind, *args = rest.split("-")
    out = []
    for a in args:
        n = 2 if a[:2] in ("te", "tb") else 1
        out.append((a[:n], a[n:]))
    return out


def parse(spec: str) -> Method:
    """Parse a legacy strategy id into its ``Method``."""
    name, *mod_strs = spec.split("+")
    mods = _parse_mods(mod_strs, spec)
    fg, _, rest = name.partition("-")
    if fg not in FG_MODES:
        raise ValueError(f"unknown foreground {fg!r} in {spec!r}")
    if rest == "none":
        return REGISTRY["none"].from_parts(fg, "none", [], mods)
    if rest == "gmm":
        return REGISTRY["gmm"].from_parts(fg, "gmm", [], mods)
    if rest.startswith("ws"):
        return REGISTRY["ws"].from_parts(fg, "ws", [rest[2:]], mods)
    if rest.startswith("close"):
        return REGISTRY["close"].from_parts(fg, "close", [rest[5:]], mods)
    kind = rest.split("-")[0]
    if kind in ("ucm", "swst", "iterh", "thr", "owt"):
        return REGISTRY[kind].from_parts(fg, kind, _parse_kv(rest), mods)
    if kind in boundaryvote.KINDS:
        _, h_s, t_s = rest.split("-")
        return REGISTRY[kind].from_parts(fg, kind, [h_s[1:], t_s[1:]], mods)
    raise ValueError(f"unknown rule {rest!r} in {spec!r}")


def run(
    spec: str,
    seg: np.ndarray,
    prob: np.ndarray,
    px_m2: float | None = None,
    ctx: dict | None = None,
) -> np.ndarray:
    """``parse(spec)(seg, prob, px_m2, ctx)``."""
    return parse(spec)(seg, prob, px_m2, ctx)


def ftp_pipeline(seg: np.ndarray, h: float = 2.0) -> np.ndarray:
    """The FTP pipeline as a method call. ``arg-ws{h}`` never reads ``prob``."""
    dummy_prob = np.zeros((3, *seg.shape), dtype=np.float32)
    return run(f"arg-ws{h:g}", seg, dummy_prob)
