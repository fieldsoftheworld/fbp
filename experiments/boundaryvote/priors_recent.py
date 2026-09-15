"""Recent field post-processing methods, re-implemented on (seg, prob).

* ``owt``  Oriented watershed and hierarchical merging, the way Zhu et al.
           (2024, Int. J. Digital Earth) turn a network's boundary probability
           into fields. An oriented boundary signal is the mean of ``p_bnd``
           along a short line at each of eight orientations; its maximum over
           orientations is watershed-segmented inside the foreground. The
           arc between two regions is weighted by the signal at the arc's own
           orientation along the arc's pixels, and regions merge greedily, re-weighting
           merged borders by pixel count, while the weakest border stays below
           one global level ``lvl``. Higra is not installed here, so the
           hierarchy cut is skimage's greedy merge at that level.
"""

import numpy as np
from scipy.ndimage import convolve, label
from skimage import graph
from skimage.morphology import h_minima
from skimage.segmentation import watershed

from .priors import lines_from_labels, merge_boundary, relabel_dense, weight_boundary
from .variants import pairs_at_lines

N_ORIENT = 8
LINE_PX = 5  # length of the oriented averaging line, in pixels
H_MIN = 0.05  # default depth of the regional minima that seed the over-segmentation


def line_kernel(theta: float, length: int) -> np.ndarray:
    """Normalised line of ``length`` px through the kernel centre at ``theta``."""
    k = np.zeros((length, length), dtype=np.float32)
    c = length // 2
    for t in np.linspace(-c, c, 4 * length + 1):
        r = int(round(c - t * np.sin(theta)))
        q = int(round(c + t * np.cos(theta)))
        k[r, q] = 1.0
    return k / k.sum()


def oriented_signal(pb: np.ndarray, length: int = LINE_PX) -> np.ndarray:
    """``(N_ORIENT, H, W)`` mean boundary probability along a line per orientation."""
    thetas = np.arange(N_ORIENT) * np.pi / N_ORIENT
    return np.stack(
        [convolve(pb, line_kernel(t, length), mode="nearest") for t in thetas]
    )


def border_pixels(
    lined: np.ndarray, fg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows, cols, and sorted region pairs of the watershed-line pixels.

    ``lined`` is the watershed with ``watershed_line=True``, so its zero pixels
    inside ``fg`` are the arcs; each arc pixel is keyed by the two regions it
    separates, and pixels touching fewer than two regions are dropped.
    """
    r, c, pairs = pairs_at_lines(lined, fg & (lined == 0))
    keep = pairs[:, 0] > 0
    return r[keep], c[keep], pairs[keep]


def orientation_bin(r: np.ndarray, c: np.ndarray) -> int | None:
    """Principal direction of a pixel set as an index into ``N_ORIENT`` angles."""
    if len(r) < 3:
        return None
    x, y = c - c.mean(), -(r - r.mean())
    cov = np.cov(np.stack([x, y]))
    _, vec = np.linalg.eigh(cov)
    vx, vy = vec[:, -1]
    theta = np.arctan2(vy, vx) % np.pi
    return int(round(theta / (np.pi / N_ORIENT))) % N_ORIENT


def border_weights(
    lined: np.ndarray, fg: np.ndarray, sig: np.ndarray
) -> dict[tuple[int, int], tuple[float, int]]:
    """Per region pair: mean oriented signal along their arc and the arc's length."""
    r, c, pairs = border_pixels(lined, fg)
    if len(r) == 0:
        return {}
    lab = lined
    keys = pairs[:, 0].astype(np.int64) * (lab.max() + 1) + pairs[:, 1]
    order = np.argsort(keys, kind="stable")
    keys, r, c = keys[order], r[order], c[order]
    bounds = np.flatnonzero(np.diff(keys)) + 1
    out = {}
    unoriented = sig.max(axis=0)
    for rr, cc, k in zip(
        np.split(r, bounds), np.split(c, bounds), keys[np.r_[0, bounds]], strict=True
    ):
        b = orientation_bin(rr, cc)
        vals = unoriented[rr, cc] if b is None else sig[b][rr, cc]
        a, bb = divmod(int(k), lab.max() + 1)
        out[(a, bb)] = (float(vals.mean()), len(rr))
    return out


def owt(
    fg: np.ndarray, seg: np.ndarray, prob: np.ndarray, lvl: float, h: float = H_MIN
) -> np.ndarray:
    """Oriented watershed seeded at ``h``-minima, then greedy merging below ``lvl``."""
    del seg  # the method reads the probability, not the argmax
    if not fg.any():
        return np.zeros(fg.shape, dtype=np.int32)
    sig = oriented_signal(prob[2].astype(np.float32))
    surf = sig.max(axis=0)
    surf = np.where(fg, surf, surf.max() + 1.0).astype(np.float32)
    markers, n = label(h_minima(surf, h=h) & fg)
    if n == 0:
        out, _ = label(fg)
        return out.astype(np.int32)
    over = watershed(surf, markers=markers, mask=fg).astype(np.int32)
    if over.max() <= 1:
        return over
    # The same markers with lines give the arcs the oriented weights are read on.
    lined = watershed(surf, markers=markers, mask=fg, watershed_line=True).astype(
        np.int32
    )
    rag = graph.rag_boundary(over, surf.astype(np.float64), connectivity=1)
    if 0 in rag:
        rag.remove_node(0)
    for (a, b), (w, n_px) in border_weights(lined, fg, sig).items():
        if rag.has_edge(a, b):
            rag[a][b]["weight"] = w
            rag[a][b]["count"] = n_px
    merged = graph.merge_hierarchical(
        over,
        rag,
        thresh=lvl,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_boundary,
        weight_func=weight_boundary,
    ).astype(np.int32)
    merged[~fg] = 0
    merged[fg] += 1  # merge_hierarchical may hand out label 0
    return relabel_dense(lines_from_labels(merged, fg))
