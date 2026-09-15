"""Foreground rules and the fixed-order cleanup steps applied after a split."""

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as ndi_label
from skimage.filters import threshold_otsu


def foreground(seg: np.ndarray, prob: np.ndarray, mode: str) -> np.ndarray:
    """Foreground mask for one of the ``arg``/``nbg``/``pf50``/``otsu`` rules."""
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
