"""Post-processing variants built from (seg, prob) only.

Every variant returns an int32 instance image on the eval grid; instances are
separated by 1-px background lines where a split is kept, so per-instance
polygons still agree after they are rasterized and extracted again.

Foreground rules (``FG``):
* ``arg``   field argmax (FTP default)
* ``nbg``   argmax != background, i.e. field or boundary
* ``pf50``  p_field > 0.5
* ``otsu``  p_field > Otsu threshold of the patch (self-normalising control)

Split rules:
* ``none``            connected components of the foreground
* ``ws{h}``           the FTP pipeline: watershed at depth ``h`` on the field
                      class, then connected components of the union of its instances
* ``pb-h{h}-t{tau}``  BoundaryVote: watershed of 1 - p_boundary seeded at its
                      regional maxima deeper than ``h`` (probability units), with
                      watershed lines; a line between two regions is kept only if
                      at least ``tau`` of its pixels have p_boundary >= p_field,
                      otherwise the regions merge
* ``sup-h{h}-t{tau}`` the same vote on the EDT-of-boundary surface with ``h`` in
                      pixels (the distance-map form)
* ``osup-h{h}-t{tau}`` the same on 1 - the oriented boundary signal of Zhu et al.
                      (``priors_recent.oriented_signal``)
* ``ssup``/``asup``/``dsup``  the distance form with another line statistic
                      (``LINE_SCORE``): mean probability share, mean p_boundary,
                      mean margin
* ``close{r}``        boundary class closed with a (2r+1)^2 square and subtracted
                      from the foreground before taking components (classic gap closing)
* ``ucm``, ``swst``, ``iterh``, ``thr``, ``owt``, ``gmm``  prior methods, see
                      ``priors`` and ``priors_recent``

``h = 0`` seeds every regional maximum. Modifiers are appended with ``+``:
``+l{L}`` requires a kept line to be at least ``L`` px long, ``+b{R}`` lets a
line pixel count as supported when any pixel within ``R`` favours boundary,
``+f{A}`` fills holes under ``A`` px (``-1`` fills all), ``+m{A}`` merges parcels
under ``A`` px into the neighbour they share the longest border with, and
``+A{m2}`` drops parcels under ``m2`` square metres, converted with the patch's
own pixel area so the threshold means the same on either sensor.
"""

from collections.abc import Iterable
from typing import Literal, overload

import numpy as np
from numpy import ndarray
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    distance_transform_edt,
    label,
)
from skimage.filters import threshold_otsu
from skimage.morphology import h_maxima, local_maxima
from skimage.segmentation import watershed

FG = ("arg", "nbg", "pf50", "otsu")
# Rule kind -> how a watershed line's support is scored (see supported_split).
LINE_SCORE = {
    "sup": "frac",
    "pb": "frac",
    "osup": "frac",
    "ssup": "soft",
    "asup": "mean",
    "dsup": "diff",
}
# Rule kind -> watershed surface the split proposals come from.
SURFACE = {"pb": "pb", "osup": "owt"}
PRIORS = ("ucm", "swst", "iterh", "thr", "owt")


def foreground(seg: np.ndarray, prob: np.ndarray, mode: str) -> np.ndarray:
    """Foreground mask for one of the ``FG`` rules."""
    if mode == "arg":
        return seg == 1
    if mode == "nbg":
        return seg != 0
    if mode == "pf50":
        return prob[1] > 0.5
    if mode == "otsu":
        pf = prob[1]
        if float(pf.max() - pf.min()) < 1e-3:
            return seg == 1
        return pf > threshold_otsu(pf)
    raise ValueError(mode)


def pairs_at_lines(
    inst: np.ndarray, lines: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per line pixel: (rows, cols, (n_px, 2) sorted label pairs).

    The pair is the two largest distinct positive neighbour labels; pixels
    touching < 2 labels get (0, 0).
    """
    p = np.pad(inst, 1)
    r, c = np.nonzero(lines)
    nb = np.stack([p[r, c + 1], p[r + 2, c + 1], p[r + 1, c], p[r + 1, c + 2]], axis=1)
    nb.sort(axis=1)
    # distinct positive labels per row: keep values that differ from the previous one
    first = nb[:, :1]
    distinct = np.concatenate(
        [first, np.where(nb[:, 1:] != nb[:, :-1], nb[:, 1:], 0)], axis=1
    )
    distinct.sort(axis=1)
    top2 = distinct[:, -2:]  # two largest distinct (0 if fewer than two)
    top2[top2[:, 0] == 0] = 0
    return r, c, top2


class UnionFind:
    """Disjoint-set over labels 0..n, path-compressed, lower label wins."""

    def __init__(self, n: int) -> None:
        """Start with every label its own group."""
        self.parent = np.arange(n + 1)

    def find(self, a: int) -> int:
        """Group representative of ``a``."""
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        """Merge the groups of ``a`` and ``b``."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def split_surface(
    fg: np.ndarray, seg: np.ndarray, prob: np.ndarray, surface: str
) -> np.ndarray:
    """The surface whose regional maxima seed the over-segmentation."""
    if surface == "edt":
        return distance_transform_edt(seg != 2).astype(np.float32) * fg
    if surface == "pb":
        return (1.0 - prob[2]).astype(np.float32) * fg
    if surface == "owt":
        # Zhu et al.'s oriented boundary signal as the proposal surface
        from .priors_recent import oriented_signal

        return (1.0 - oriented_signal(prob[2].astype(np.float32)).max(axis=0)) * fg
    raise ValueError(surface)


def line_support(
    prob: np.ndarray, r: np.ndarray, c: np.ndarray, score: str, band: int
) -> np.ndarray:
    """Per line pixel, its vote in [0, 1] under the chosen statistic."""
    is_bnd = prob[2] >= prob[1]
    if band > 0:
        # a line can sit a pixel off the predicted boundary; count a pixel as
        # supported when any pixel within `band` of it favours boundary
        is_bnd = binary_dilation(
            is_bnd, structure=np.ones((2 * band + 1, 2 * band + 1), dtype=bool)
        )
    pb, pf = prob[2][r, c].astype(np.float64), prob[1][r, c].astype(np.float64)
    if score == "frac" or (score == "soft" and band > 0):
        # the band is a spatial rule, so the soft score falls back to the hard test
        return is_bnd[r, c].astype(np.float64)
    if score == "soft":
        return np.divide(pb, pb + pf, out=np.full(pb.shape, 0.5), where=(pb + pf) > 0)
    if score == "mean":
        return pb
    if score == "diff":
        return (pb - pf + 1.0) / 2.0
    raise ValueError(score)


@overload
def supported_split(
    fg: np.ndarray,
    seg: np.ndarray,
    prob: np.ndarray,
    h: float,
    tau: float,
    surface: str = ...,
    return_lines: Literal[False] = ...,
    min_line: int = ...,
    band: int = ...,
    score: str = ...,
) -> np.ndarray: ...


@overload
def supported_split(
    fg: np.ndarray,
    seg: np.ndarray,
    prob: np.ndarray,
    h: float,
    tau: float,
    surface: str = ...,
    *,
    return_lines: Literal[True],
    min_line: int = ...,
    band: int = ...,
    score: str = ...,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def supported_split(
    fg: np.ndarray,
    seg: np.ndarray,
    prob: np.ndarray,
    h: float,
    tau: float,
    surface: str = "edt",
    return_lines: bool = False,
    min_line: int = 0,
    band: int = 0,
    score: str = "frac",
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boundary-supported split: watershed, then keep or merge lines by vote.

    With ``return_lines`` also return ``(lines_all, lines_kept)`` boolean masks
    of the watershed lines before and after the vote (figures).

    ``score`` picks how a line's support is measured: ``"frac"`` is the share
    of its pixels where the boundary class outscores the field class, and
    ``"soft"`` averages ``p_bnd / (p_bnd + p_field)`` instead, so a pixel that
    barely favours boundary counts for less than one that favours it strongly.
    Both land in [0, 1] and share the 0.5 neutral point. ``"mean"`` is the mean
    ``p_bnd`` along the line, so ``tau`` becomes an absolute threshold as in
    hierarchical merging, and ``"diff"`` is the mean of ``(p_bnd - p_field + 1) / 2``,
    so ``tau = 0.5`` keeps a line whose mean margin favours boundary.
    """
    if not fg.any():
        z = np.zeros(fg.shape, dtype=np.int32)
        return (z, np.zeros_like(fg), np.zeros_like(fg)) if return_lines else z
    surf = split_surface(fg, seg, prob, surface)
    # h = 0 seeds every regional maximum, the end point of a depth grid
    peaks = h_maxima(surf, h=h) if h > 0 else local_maxima(surf) & fg
    markers, n = label(peaks)
    if n == 0:
        out, _ = label(fg)
        out = out.astype(np.int32)
        return (out, np.zeros_like(fg), np.zeros_like(fg)) if return_lines else out
    inst = watershed(-surf, markers=markers, mask=fg, watershed_line=True).astype(
        np.int32
    )
    lines = fg & (inst == 0)
    if not lines.any():
        return (inst, lines, lines) if return_lines else inst
    r, c, pairs = pairs_at_lines(inst, lines)
    valid = pairs[:, 0] > 0
    support = line_support(prob, r, c, score, band)
    keys = pairs[valid, 0].astype(np.int64) * (inst.max() + 1) + pairs[valid, 1]
    uniq, inv = np.unique(keys, return_inverse=True)
    n_px = np.bincount(inv, minlength=len(uniq))
    n_sup = np.bincount(inv, weights=support[valid], minlength=len(uniq))
    keep_line = (n_sup / n_px >= tau) & (n_px >= min_line)
    uf = UnionFind(int(inst.max()))
    base = inst.max() + 1
    for k, keep in zip(uniq, keep_line, strict=True):
        if not keep:
            uf.union(int(k // base), int(k % base))
    lut = np.array([uf.find(i) for i in range(int(inst.max()) + 1)], dtype=np.int32)
    merged = lut[inst]
    # A line pixel is filled only when every region it touches was merged into one.
    # Two neighbouring line pixels can pass that test with different labels, so
    # candidates are resolved against each other before anything is written; a
    # kept split therefore always stays a gap and two parcels are never adjacent.
    p = np.pad(inst, 1)
    nbm = lut[
        np.stack([p[r, c + 1], p[r + 2, c + 1], p[r + 1, c], p[r + 1, c + 2]], axis=1)
    ]
    mx = nbm.max(axis=1)
    cand = np.where(
        valid & (mx > 0) & ((nbm == mx[:, None]) | (nbm == 0)).all(axis=1), mx, 0
    )
    cand_img = np.zeros_like(merged)
    cand_img[r, c] = cand
    q = np.pad(cand_img, 1)
    nbc = np.stack([q[r, c + 1], q[r + 2, c + 1], q[r + 1, c], q[r + 1, c + 2]], axis=1)
    conflict = ((nbc != 0) & (nbc != cand[:, None])).any(axis=1)
    keep = (cand > 0) & ~conflict
    merged[r[keep], c[keep]] = cand[keep]
    # One label per connected parcel: a merge that stays disconnected, because a
    # junction pixel was not filled, is two parcels in the polygons either way.
    dense, _ = label(merged > 0)
    dense = dense.astype(np.int32)
    if return_lines:
        return dense, lines, fg & (dense == 0)
    return dense


def gap_close(fg: np.ndarray, seg: np.ndarray, r: int) -> np.ndarray:
    """Close the boundary class with a (2r+1)^2 square, then take components."""
    closed = binary_closing(
        seg == 2, structure=np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
    )
    out, _ = label(fg & ~closed)
    return out.astype(np.int32)


def components(fg: np.ndarray) -> np.ndarray:
    """Connected components of the foreground."""
    out, _ = label(fg)
    return out.astype(np.int32)


def apply_post_steps(
    inst: ndarray, min_area: int, hole_area: int, merge_area: int
) -> ndarray:
    """Post-steps in a fixed order: fill holes, merge slivers, drop what is left."""
    return drop_small(merge_small(fill_holes(inst, hole_area), merge_area), min_area)


def drop_small(inst: np.ndarray, min_area: int) -> np.ndarray:
    """Remove parcels below ``min_area`` pixels and relabel."""
    if min_area <= 0:
        return inst
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    small = ids[counts < min_area]
    if not len(small):
        return inst
    out = inst.copy()
    out[np.isin(out, small)] = 0
    dense, _ = label(out > 0)
    return dense.astype(np.int32)


def fill_holes(inst: np.ndarray, max_area: int) -> np.ndarray:
    """Assign background holes inside a parcel to that parcel.

    ``max_area < 0`` fills every hole; otherwise only holes under ``max_area``
    pixels. A hole is a background component that touches exactly one parcel and
    not the image border, so the gap at a kept split is never filled.
    """
    if max_area == 0:
        return inst
    bg, n = label(inst == 0)
    if n == 0:
        return inst
    border = set(np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]])))
    out = inst.copy()
    p = np.pad(inst, 1)
    for i, size in zip(*np.unique(bg[bg > 0], return_counts=True), strict=True):
        if i in border or (max_area > 0 and size >= max_area):
            continue
        m = bg == i
        r, c = np.nonzero(m)
        nb = np.concatenate(
            [p[r, c + 1], p[r + 2, c + 1], p[r + 1, c], p[r + 1, c + 2]]
        )
        owners = np.unique(nb[nb > 0])
        if len(owners) == 1:
            out[m] = owners[0]
    return out


def merge_small(inst: np.ndarray, min_area: int) -> np.ndarray:
    """Merge small parcels into their longest-shared-border neighbour.

    ``min_area`` px, instead of deleting them. Parcels are separated by a
    one-pixel gap, so the neighbourhood is taken two pixels out and the gap
    between the pair is filled; otherwise the merged parcel would fall apart
    again when labels are rebuilt. A parcel with no neighbour within that
    distance is left alone.
    """
    if min_area <= 0:
        return inst
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    order = [i for _, i in sorted(zip(counts, ids, strict=True)) if _ < min_area]
    if not order:
        return inst
    out = inst.copy()
    cross = np.ones((5, 5), dtype=bool)
    for i in order:
        m = out == i
        if not m.any():
            continue
        near = binary_dilation(m, structure=cross) & ~m
        vals, freq = np.unique(out[near & (out > 0)], return_counts=True)
        if not len(vals):
            continue
        best = int(vals[freq.argmax()])
        # bridge the gap so the pair stays one connected parcel
        gap = (
            binary_dilation(m, structure=np.ones((3, 3), dtype=bool))
            & binary_dilation(out == best, structure=np.ones((3, 3), dtype=bool))
            & (out == 0)
        )
        out[gap] = best
        out[m] = best
    dense, _ = label(out > 0)
    return dense.astype(np.int32)


def parse_modifiers(
    mods: list[str], name: str, px_m2: float | None
) -> tuple[int, int, int, int, int]:
    """``(min_line, band, min_area, hole_area, merge_area)`` from ``+`` suffixes."""
    min_line, band, min_area, hole_area, merge_area = 0, 0, 0, 0, 0
    for mod in mods:
        key, val = mod[0], int(mod[1:])
        if key == "l":
            min_line = val
        elif key == "b":
            band = val
        elif key == "f":
            hole_area = val
        elif key == "m":
            merge_area = val
        elif key == "A":
            # ground area in m^2, so one setting means the same parcel size on
            # grids with different pixel sizes
            if px_m2 is None:
                raise ValueError(f"{name!r} needs the pixel area to use +A")
            min_area = max(1, int(round(val / px_m2)))
        else:
            raise ValueError(f"unknown modifier {mod!r} in {name!r}")
    return min_line, band, min_area, hole_area, merge_area


def run_variant(
    name: str,
    seg: np.ndarray,
    prob: np.ndarray,
    px_m2: float | None = None,
    gmm_thr: tuple[float, float] | None = None,
) -> np.ndarray:
    """Dispatch a variant name to its split rule (see the module docstring).

    ``gmm_thr`` is the region's fitted extent and boundary thresholds, needed
    only by ``nbg-gmm``.
    """
    name, *mods = name.split("+")
    min_line, band, min_area, hole_area, merge_area = parse_modifiers(mods, name, px_m2)
    fg_mode, _, rest = name.partition("-")
    fg = foreground(seg, prob, fg_mode)
    if rest == "none":
        inst = components(fg)
    elif rest == "gmm":
        # Amin et al.: thresholds fitted per region arrive from outside
        if gmm_thr is None:
            raise ValueError(f"{name!r} needs the region's fitted (te, tb)")
        from .priors_recent import gmm_split

        inst = gmm_split(prob, *gmm_thr)
    elif rest.startswith("ws"):
        # the FTP pipeline: watershed at depth h, then components of the union
        from .ftp_compat import watershed_instances

        dist = distance_transform_edt(seg != 2).astype(np.float32)
        ws = watershed_instances(seg, dist, h_min=float(rest[2:]), field_class=1)
        comp, _ = label(ws > 0)
        inst = comp.astype(np.int32)
    elif rest.startswith("close"):
        inst = gap_close(fg, seg, int(rest[5:]))
    elif rest.split("-")[0] in PRIORS:
        from .priors import run_prior

        inst = run_prior(rest, fg, seg, prob)
    else:
        kind, h_s, t_s = rest.split("-")
        inst = supported_split(
            fg,
            seg,
            prob,
            float(h_s[1:]),
            float(t_s[1:]),
            surface=SURFACE.get(kind, "edt"),
            min_line=min_line,
            band=band,
            score=LINE_SCORE.get(kind, "frac"),
        )
    return apply_post_steps(inst, min_area, hole_area, merge_area)


def run_variants(
    seg: np.ndarray,
    prob: np.ndarray,
    names: Iterable[str],
    px_m2: float | None = None,
    gmm_thr: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """``run_variant`` for every name in ``names``."""
    return {n: run_variant(n, seg, prob, px_m2, gmm_thr) for n in names}
