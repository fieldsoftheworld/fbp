"""Foreground rules and the fixed-order cleanup steps applied after a split."""

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as ndi_label
from skimage.filters import threshold_otsu


def foreground(seg: np.ndarray, prob: np.ndarray, mode: str) -> np.ndarray:
    """Foreground mask for one of the ``arg``/``nbg``/``pf{NN}``/``otsu`` rules.

    ``pf{NN}`` is field probability above NN/100 (``pf50`` = 0.5); a low value such
    as ``pf15`` is meant to be paired with the ``+R`` region cut, so parcels whose
    field probability never wins the argmax (centre pivots at 0.3-0.5) still form.
    """
    if mode == "arg":
        return seg == 1
    if mode == "nbg":
        return seg != 0
    if mode.startswith("pf") and mode[2:].isdigit():
        return prob[1] > int(mode[2:]) / 100.0
    if mode == "otsu":
        pf = prob[1]
        if float(pf.max() - pf.min()) < 1e-3:
            return seg == 1
        return pf > threshold_otsu(pf)
    raise ValueError(mode)


def drop_low_field(inst: np.ndarray, pf: np.ndarray, min_mean: float) -> np.ndarray:
    """Drop parcels whose mean field probability is below ``min_mean`` (the ``+R`` cut).

    A region-level decision: a whole parcel is kept or dropped on its average, so a
    loose pixel foreground (``pf15``) can grow parcels that the argmax would break up,
    and the ones that are mostly not field are removed afterwards.
    """
    if min_mean <= 0 or inst.max() == 0:
        return inst
    n = int(inst.max()) + 1
    cnt = np.bincount(inst.ravel(), minlength=n)
    tot = np.bincount(inst.ravel(), weights=pf.ravel().astype(np.float64), minlength=n)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = tot[1:] / np.maximum(cnt[1:], 1) >= min_mean
    if keep[1:].all():
        return inst
    return np.where(keep[inst], inst, 0).astype(np.int32, copy=False)


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
    dense, _ = ndi_label(out > 0)
    return dense.astype(np.int32, copy=False)


def fill_holes(inst: np.ndarray, max_area: int) -> np.ndarray:
    """Assign background holes inside a parcel to that parcel.

    ``max_area < 0`` fills every hole; otherwise only holes under ``max_area``
    pixels. A hole is a background component that touches exactly one parcel and
    not the image border, so the gap at a kept split is never filled.
    """
    if max_area == 0:
        return inst
    bg, n = ndi_label(inst == 0)
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
    dense, _ = ndi_label(out > 0)
    return dense.astype(np.int32, copy=False)


def apply_post_steps(
    inst: np.ndarray, min_area: int, hole_area: int, merge_area: int
) -> np.ndarray:
    """Post-steps in a fixed order: fill holes, merge slivers, drop what is left."""
    return drop_small(merge_small(fill_holes(inst, hole_area), merge_area), min_area)
