"""Model-free post-processing variants built from (seg, prob) only.

Every variant returns an int32 instance image on the eval grid; instances are
separated by 1-px background lines where a split is kept, so per-instance
polygons and their rasterize->polygonize round trip agree.

Foreground rules (``FG``):
* ``arg``   field argmax (FTP default)
* ``nbg``   argmax != background, i.e. field or boundary
* ``pf50``  p_field > 0.5
* ``otsu``  p_field > Otsu threshold of the patch (self-normalising control)

Split rules:
* ``none``            connected components of the foreground
* ``ws{h}``           the FTP pipeline: watershed at depth ``h`` on the field
                      class, then connected components of the union of its instances
* ``sup-h{h}-t{tau}`` EDT-of-boundary watershed with h-maxima markers at depth ``h``
                      and watershed lines; a line between two regions is kept only if
                      at least ``tau`` of its pixels have p_boundary >= p_field
                      ("boundary-supported split"); otherwise the regions merge
* ``pb-h{h}-t{tau}``  same, but the watershed surface is 1 - p_boundary and ``h`` is
                      in probability units
* ``close{r}``        boundary class closed with a (2r+1)^2 square and subtracted
                      from the foreground before taking components (classic gap closing)

Modifiers are appended with ``+``: ``+l{L}`` requires a kept line to be at least
``L`` px long, ``+b{R}`` lets a line pixel count as supported when any pixel
within ``R`` favours boundary, ``+a{A}`` drops parcels under ``A`` px, and
``+w{k}`` widens kept lines to ``k`` px for scoring on a coarser grid, and
``+A{m2}`` drops parcels under ``m2`` square metres, converted with the patch's
own pixel area so the threshold means the same on either sensor.
``+f{A}`` fills holes under ``A`` px (``-1`` fills all) and ``+m{A}`` merges parcels
under ``A`` px into the neighbour they share the longest border with.
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
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

FG = ("arg", "nbg", "pf50", "otsu")


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


def link_weak_lines(
    strong: ndarray,
    eligible: ndarray,
    inv: ndarray,
    rr: ndarray,
    cc: ndarray,
    shape: tuple[int, int],
) -> ndarray:
    """Hysteresis over the watershed lines.

    ``strong`` lines are kept outright. An ``eligible`` line is kept when a chain
    of eligible lines connects it to a strong one, which is decided by grouping
    eligible lines that touch in the 8-neighbourhood and keeping every group that
    holds at least one strong line.
    """
    if not eligible.any() or strong.all():
        return strong
    # paint each line's id where its pixels are, restricted to eligible lines
    ids = np.full(shape, -1, dtype=np.int64)
    ok = eligible[inv]
    ids[rr[ok], cc[ok]] = inv[ok]
    uf = UnionFind(int(inv.max()) + 1)
    pad = np.pad(ids, 1, constant_values=-1)
    here = ids[rr[ok], cc[ok]]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):  # 8-neighbourhood, once each
        nb = pad[rr[ok] + 1 + dr, cc[ok] + 1 + dc]
        both = nb >= 0
        for a, b in zip(here[both], nb[both], strict=True):
            if a != b:
                uf.union(int(a), int(b))
    roots = np.array([uf.find(i) for i in range(len(eligible))], dtype=np.int64)
    seeded = set(roots[strong & eligible].tolist())
    linked = (
        eligible & np.isin(roots, list(seeded)) if seeded else np.zeros_like(strong)
    )
    return strong | linked


@overload
def supported_split(
    fg: np.ndarray,
    seg: np.ndarray,
    prob: np.ndarray,
    h: float,
    tau: float,
    surface: str = ...,
    markers: np.ndarray | None = ...,
    return_lines: Literal[False] = ...,
    min_line: int = ...,
    band: int = ...,
    tau_lo: float | None = ...,
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
    markers: np.ndarray | None = ...,
    *,
    return_lines: Literal[True],
    min_line: int = ...,
    band: int = ...,
    tau_lo: float | None = ...,
    score: str = ...,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def supported_split(
    fg: np.ndarray,
    seg: np.ndarray,
    prob: np.ndarray,
    h: float,
    tau: float,
    surface: str = "edt",
    markers: np.ndarray | None = None,
    return_lines: bool = False,
    min_line: int = 0,
    band: int = 0,
    tau_lo: float | None = None,
    score: str = "frac",
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boundary-supported split: watershed, then keep or merge lines by support.

    ``markers`` (int label image) replaces the h-maxima seeds when given, e.g.
    an instance model's own partition; ``h`` is then ignored. With
    ``return_lines`` also return ``(lines_all, lines_kept)`` boolean masks of the
    watershed lines before and after the support test (figures).

    ``score`` picks how a line's support is measured: ``"frac"`` is the fraction
    of its pixels where the boundary class outscores the field class, and
    ``"soft"`` averages ``p_bnd / (p_bnd + p_field)`` instead, so a pixel that
    barely favours boundary counts for less than one that favours it strongly.
    Both land in [0, 1] and share the 0.5 neutral point.

    ``tau_lo`` turns the single threshold into hysteresis, as in edge linking: a
    line at or above ``tau`` is kept outright, and a line at or above ``tau_lo``
    is kept when it connects to one that was. Field boundaries form connected
    networks, so a weak segment between two strong ones is usually real.
    """
    if not fg.any():
        z = np.zeros(fg.shape, dtype=np.int32)
        return (z, np.zeros_like(fg), np.zeros_like(fg)) if return_lines else z
    boundary = seg == 2
    if surface == "edt":
        surf = distance_transform_edt(~boundary).astype(np.float32) * fg
    elif surface == "pb":
        surf = (1.0 - prob[2]).astype(np.float32) * fg
    else:
        raise ValueError(surface)
    if markers is None:
        markers, n = label(h_maxima(surf, h=h))
    else:
        # drop contested (boundary-class) pixels from the seeds so adjacent
        # instances leave room for a watershed line the support test can judge
        _, markers = np.unique(markers * (fg & ~boundary), return_inverse=True)
        markers = markers.reshape(fg.shape).astype(np.int32)
        n = int(markers.max())
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
    is_bnd = prob[2] >= prob[1]
    if band > 0:
        # a line can sit a pixel off the predicted boundary; count a pixel as
        # supported when any pixel within `band` of it favours boundary
        is_bnd = binary_dilation(
            is_bnd, structure=np.ones((2 * band + 1, 2 * band + 1), dtype=bool)
        )
    if score == "frac":
        support = is_bnd[r, c].astype(np.float64)
    elif score == "soft":
        pb, pf = prob[2][r, c].astype(np.float64), prob[1][r, c].astype(np.float64)
        support = np.divide(
            pb, pb + pf, out=np.full(pb.shape, 0.5), where=(pb + pf) > 0
        )
        if band > 0:  # the band is a spatial rule; fall back to the hard test
            support = is_bnd[r, c].astype(np.float64)
    else:
        raise ValueError(score)
    keys = pairs[valid, 0].astype(np.int64) * (inst.max() + 1) + pairs[valid, 1]
    uniq, inv = np.unique(keys, return_inverse=True)
    n_px = np.bincount(inv, minlength=len(uniq))
    n_sup = np.bincount(inv, weights=support[valid], minlength=len(uniq))
    s_line = n_sup / n_px
    keep_line = (s_line >= tau) & (n_px >= min_line)
    if tau_lo is not None and tau_lo < tau:
        keep_line = link_weak_lines(
            keep_line,
            (s_line >= tau_lo) & (n_px >= min_line),
            inv,
            r[valid],
            c[valid],
            fg.shape,
        )
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


def default_variants() -> list[str]:
    """The variant names scored by default (see ``candidates.py --variants``)."""
    v = [f"{f}-none" for f in FG]
    for f in ("arg", "nbg"):
        for h in (1.0, 2.0, 4.0):
            v.append(f"{f}-sup-h{h:g}-t0.5")
        for t in (0.3, 0.7):
            v.append(f"{f}-sup-h2-t{t:g}")
        for h in (0.1, 0.2):
            v.append(f"{f}-pb-h{h:g}-t0.5")
        for r in (1, 2):
            v.append(f"{f}-close{r}")
    # second round: push the support threshold and depth for the promising family
    for t in (0.8, 0.9, 1.0):
        v.append(f"nbg-sup-h2-t{t:g}")
    for h in (1.0, 4.0, 8.0):
        v.append(f"nbg-sup-h{h:g}-t0.7")
    v.append("arg-sup-h4-t0.7")
    return v


def widen_gaps(inst: np.ndarray, fg: np.ndarray, width_px: int) -> np.ndarray:
    """Grow the kept split lines (foreground pixels labelled 0) to ``width_px``.

    Gaps must span at least one sensor ground-sample distance to survive FTP's
    native-GSD scoring (``_cap_pred_to_gsd``): 1 px on the 3 m Planet grid, 2 px on
    the 5 m/px upsampled S2 grid scored at 10 m. Only pixels adjacent to an
    existing gap are cleared, so region exteriors are untouched.
    """
    if width_px <= 1:
        return inst
    gaps = fg & (inst == 0)
    if not gaps.any():
        return inst
    grown = binary_dilation(
        gaps, structure=np.ones((3, 3), dtype=bool), iterations=width_px - 1
    )
    out = inst.copy()
    out[grown & fg] = 0
    return out


def apply_post_steps(
    inst: ndarray,
    min_area: int,
    hole_area: int,
    merge_area: int,
    seg: ndarray | None = None,
    keep_tau: float | None = None,
) -> ndarray:
    """Post-steps in a fixed order: fill holes, merge slivers, drop what is left.

    With ``keep_tau`` the last step spares small parcels whose outline the
    network predicts as boundary (see ``drop_small_unsupported``).
    """
    out = merge_small(fill_holes(inst, hole_area), merge_area)
    if keep_tau is not None and seg is not None:
        return drop_small_unsupported(out, min_area, seg, keep_tau)
    return drop_small(out, min_area)


def drop_small_unsupported(
    inst: np.ndarray, min_area: int, seg: np.ndarray, keep_tau: float
) -> np.ndarray:
    """Drop parcels under ``min_area`` px unless their own outline is predicted.

    The flat minimum-size step removes every small polygon, including the real
    smallholder fields that are the point of the method. A parcel whose outline
    the network itself calls boundary along at least ``keep_tau`` of its border
    is kept regardless of size; the rest are dropped as before.
    """
    if min_area <= 0:
        return inst
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    small = ids[counts < min_area]
    if not len(small):
        return inst
    is_bnd = seg == 2
    p = np.pad(inst, 1)
    r, c = np.nonzero(np.isin(inst, small))
    # a parcel pixel is on its border when a 4-neighbour holds a different label
    nb = np.stack([p[r, c + 1], p[r + 2, c + 1], p[r + 1, c], p[r + 1, c + 2]], axis=1)
    on_border = (nb != inst[r, c][:, None]).any(axis=1)
    rb, cb, lab = r[on_border], c[on_border], inst[r, c][on_border]
    if not len(lab):
        return drop_small(inst, min_area)
    # supported when the border pixel itself, or a 4-neighbour of it, is boundary
    q = np.pad(is_bnd, 1)
    sup = (
        is_bnd[rb, cb]
        | q[rb, cb + 1]
        | q[rb + 2, cb + 1]
        | q[rb + 1, cb]
        | q[rb + 1, cb + 2]
    )
    order = np.unique(lab)
    idx = np.searchsorted(order, lab)
    frac = np.bincount(idx, weights=sup, minlength=len(order)) / np.bincount(
        idx, minlength=len(order)
    )
    doomed = order[frac < keep_tau]
    if not len(doomed):
        return inst
    out = inst.copy()
    out[np.isin(out, doomed)] = 0
    dense, _ = label(out > 0)
    return dense.astype(np.int32)


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


def run_variant(
    name: str,
    seg: np.ndarray,
    prob: np.ndarray,
    markers: np.ndarray | None = None,
    px_m2: float | None = None,
) -> np.ndarray:
    """Dispatch a variant name to its split rule.

    ``<fg>-none`` | ``<fg>-close{r}`` | ``<fg>-{sup|pb}-h{h}-t{tau}`` |
    ``<fg>-msup-t{tau}`` (markers = supplied instance partition).

    Optional modifiers are appended with ``+``: ``+l{L}`` minimum kept-line
    length, ``+b{R}`` support tolerance band, ``+a{A}`` minimum parcel area,
    ``+w{k}`` kept-line width, ``+H{t}`` hysteresis low bar (hundredths, so
    ``+H45`` is 0.45), and ``+K{t}`` the outline-support bar that spares a small
    parcel from the minimum-size step. Rule kind ``ssup`` scores a line by the
    average probability share instead of a pixel count. They never collide with
    a rule's own parameters.
    """
    width_px, min_line, band, min_area, hole_area, merge_area = 1, 0, 0, 0, 0, 0
    tau_lo, keep_tau = None, None
    name, *mods = name.split("+")
    for mod in mods:
        key, val = mod[0], int(mod[1:])
        if key == "w":
            width_px = val
        elif key == "a":
            min_area = val
        elif key == "b":
            band = val
        elif key == "l":
            min_line = val
        elif key == "f":
            hole_area = val
        elif key == "m":
            merge_area = val
        elif key == "H":
            tau_lo = val / 100.0
        elif key == "K":
            keep_tau = val / 100.0
        elif key == "A":
            # ground area in m^2, so one setting means the same parcel size on
            # grids with different pixel sizes
            if px_m2 is None:
                raise ValueError(f"{name!r} needs the pixel area to use +A")
            min_area = max(1, int(round(val / px_m2)))
        else:
            raise ValueError(f"unknown modifier {mod!r} in {name!r}")
    fg_mode, _, rest = name.partition("-")
    fg = foreground(seg, prob, fg_mode)
    if rest == "none":
        return apply_post_steps(
            components(fg), min_area, hole_area, merge_area, seg, keep_tau
        )
    if rest.startswith("ws"):
        # the FTP pipeline: watershed at depth h, then components of the union
        from .ftp_compat import watershed_instances

        dist = distance_transform_edt(seg != 2).astype(np.float32)
        inst = watershed_instances(seg, dist, h_min=float(rest[2:]), field_class=1)
        comp, _ = label(inst > 0)
        return apply_post_steps(
            comp.astype(np.int32), min_area, hole_area, merge_area, seg, keep_tau
        )
    if rest.startswith("close"):
        return apply_post_steps(
            gap_close(fg, seg, int(rest[5:])),
            min_area,
            hole_area,
            merge_area,
            seg,
            keep_tau,
        )
    if rest.split("-")[0] in ("ucm", "swst", "iterh", "thr"):
        from .priors import run_prior

        inst = run_prior(rest, fg, seg, prob)
        return apply_post_steps(
            widen_gaps(inst, fg, width_px),
            min_area,
            hole_area,
            merge_area,
            seg,
            keep_tau,
        )
    if rest.startswith("msup-t"):
        if markers is None:
            raise ValueError(f"{name} needs an instance marker image")
        inst = supported_split(fg, seg, prob, 0.0, float(rest[6:]), markers=markers)
        return apply_post_steps(
            widen_gaps(inst, fg, width_px),
            min_area,
            hole_area,
            merge_area,
            seg,
            keep_tau,
        )
    kind, h_s, t_s = rest.split("-")
    h = float(h_s[1:])
    tau = float(t_s[1:])
    inst = supported_split(
        fg,
        seg,
        prob,
        h,
        tau,
        surface="pb" if kind == "pb" else "edt",
        min_line=min_line,
        band=band,
        tau_lo=tau_lo,
        score="soft" if kind == "ssup" else "frac",
    )
    return apply_post_steps(
        widen_gaps(inst, fg, width_px), min_area, hole_area, merge_area, seg, keep_tau
    )


def run_variants(
    seg: np.ndarray,
    prob: np.ndarray,
    names: Iterable[str],
    markers: np.ndarray | None = None,
    px_m2: float | None = None,
) -> dict[str, np.ndarray]:
    """``run_variant`` for every name in ``names``."""
    return {n: run_variant(n, seg, prob, markers, px_m2) for n in names}
