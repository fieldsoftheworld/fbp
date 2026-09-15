"""Watershed regions, connected components, and the label bookkeeping they share."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as ndi_label
from skimage.morphology import h_maxima, h_minima, local_maxima, local_minima
from skimage.segmentation import watershed


@dataclass(frozen=True)
class Regions:
    """One watershed's output plus the marker bookkeeping around it."""

    labels: np.ndarray  # int32, 0 outside fg and on lines (when lines=True)
    lines: np.ndarray  # bool, watershed lines inside fg
    markers: np.ndarray  # int32 marker image
    n_markers: int
    fallback: bool  # True when no marker was found and labels = components(fg)


def components(fg: np.ndarray) -> np.ndarray:
    """Connected components of the foreground."""
    out, _ = ndi_label(fg)
    return out.astype(np.int32, copy=False)


def relabel_dense(lab: np.ndarray) -> np.ndarray:
    """Labels remapped to a dense 0..n range, preserving background."""
    _, dense = np.unique(lab, return_inverse=True)
    dense = dense.reshape(lab.shape).astype(np.int32, copy=False)
    if lab.min() != 0:
        dense += 1
    return dense


def lines_from_labels(lab: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """1-px lines between different labels so adjacent regions never touch."""
    p = np.pad(lab, 1)
    c = p[1:-1, 1:-1]
    diff = ((p[:-2, 1:-1] != c) & (p[:-2, 1:-1] > 0)) | (
        (p[1:-1, :-2] != c) & (p[1:-1, :-2] > 0)
    )
    out = lab.copy()
    out[diff & fg & (c > 0)] = 0
    return out


def neighbours4(padded: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Values at the 4-neighbourhood of ``(r, c)`` in a 1-padded image."""
    return np.stack(
        [
            padded[r, c + 1],
            padded[r + 2, c + 1],
            padded[r + 1, c],
            padded[r + 1, c + 2],
        ],
        axis=1,
    )


def pairs_at_lines(
    inst: np.ndarray, lines: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per line pixel: (rows, cols, (n_px, 2) sorted label pairs).

    The pair is the two largest distinct positive neighbour labels; pixels
    touching < 2 labels get (0, 0).
    """
    p = np.pad(inst, 1)
    r, c = np.nonzero(lines)
    nb = neighbours4(p, r, c)
    nb.sort(axis=1)
    first = nb[:, :1]
    distinct = np.concatenate(
        [first, np.where(nb[:, 1:] != nb[:, :-1], nb[:, 1:], 0)], axis=1
    )
    distinct.sort(axis=1)
    top2 = distinct[:, -2:]  # two largest distinct (0 if fewer than two)
    top2[top2[:, 0] == 0] = 0
    return r, c, top2


def watershed_regions(
    surface: np.ndarray,
    fg: np.ndarray,
    h: float,
    *,
    lines: bool = True,
    mode: str = "max",
) -> Regions:
    """Watershed inside ``fg``, seeded at ``surface``'s own extrema.

    ``mode="max"`` (every other caller) seeds on maxima: ``h > 0`` seeds
    ``h``-maxima at least ``h`` deep, ``h == 0`` seeds every regional maximum,
    and outside ``fg`` is masked to 0 for the seed search (harmless for the
    non-negative distance/probability surfaces every ``max`` caller uses,
    since 0 is already their own floor).

    ``mode="min"`` (owt only) seeds on minima instead, matching
    ``skimage.morphology.h_minima``/``local_minima`` directly. It masks
    outside ``fg`` by *substituting* ``surface.max() + 1`` there rather than
    multiplying by ``fg``: h_maxima/h_minima's floating-point tolerance scales
    with each pixel's own magnitude, so zeroing every masked-out pixel would
    change the in-``fg`` pixels' tolerance too when negated into a maxima
    search, and does occasionally flip which of two nearly-tied real pixels
    counts as the seed on real patches (see the owt port report).

    Falls back to plain connected components of ``fg`` when no marker
    survives.
    """
    z = np.zeros(fg.shape, dtype=np.int32)
    if not fg.any():
        return Regions(z, np.zeros(fg.shape, dtype=bool), z, 0, True)
    if mode == "max":
        seed = surface * fg
        peaks = (h_maxima(seed, h=h) & fg) if h > 0 else (local_maxima(seed) & fg)
        flood_surface = -surface
    elif mode == "min":
        seed = np.where(fg, surface, surface.max() + 1.0).astype(surface.dtype)
        peaks = (h_minima(seed, h=h) & fg) if h > 0 else (local_minima(seed) & fg)
        flood_surface = surface
    else:
        raise ValueError(mode)
    markers, n = ndi_label(peaks)
    markers = markers.astype(np.int32, copy=False)
    if n == 0:
        return Regions(components(fg), np.zeros(fg.shape, dtype=bool), markers, 0, True)
    labels = watershed(flood_surface, markers=markers, mask=fg, watershed_line=lines)
    labels = labels.astype(np.int32, copy=False)
    line_mask = (fg & (labels == 0)) if lines else np.zeros(fg.shape, dtype=bool)
    return Regions(labels, line_mask, markers, int(n), False)
