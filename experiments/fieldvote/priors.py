"""Prior label-free post-processing methods, re-implemented on (seg, prob).

All operate on the same foreground ``F`` and the same class probabilities as
our method so the comparison isolates the split/merge rule. Every function
returns an int32 label image.

* ``ucm``   Hierarchical merging on a boundary-strength map, the mechanism of
            ultrametric contour maps~(Arbelaez et al. 2011; Maninis et al. 2018)
            as applied to fields by Persello et al. (2019). Over-segment ``F``
            with the released distance watershed at a small depth (h = 0.5),
            weight each region adjacency by the mean boundary probability along
            the shared border, and greedily merge the weakest adjacency while
            it is below ``k``, re-weighting merged borders by pixel count.
* ``swst``  Stochastic watershed~(Angulo & Jeulin 2007). Repeat ``m`` times:
            draw ``n`` uniform random markers inside ``F``, run the watershed
            of the released distance surface with lines, and accumulate the
            lines. The contour probability is the line frequency; pixels with
            frequency ``>= t`` are cut and the rest of ``F`` is labelled by
            connected components.
* ``iterh`` Iterative marker depth~(Koyuncu et al. 2016). Watershed at the
            largest depth in ``hs``; any region larger than ``a_max`` pixels is
            re-segmented with the next smaller depth, recursively.
* ``thr``   Extent/boundary thresholding as in DECODE~(Waldner et al. 2020):
            ``p_field >= te`` and ``p_bnd < tb``, then connected components.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from skimage import graph
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


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


def relabel_dense(lab: np.ndarray) -> np.ndarray:
    _, dense = np.unique(lab, return_inverse=True)
    dense = dense.reshape(lab.shape).astype(np.int32)
    if lab.min() != 0:
        dense += 1
    return dense


def weight_boundary(g, src, dst, n):
    """Count-weighted mean boundary probability after merging src into dst."""
    default = {"weight": 0.0, "count": 0}
    cs, cd = g[src].get(n, default)["count"], g[dst].get(n, default)["count"]
    ws, wd = g[src].get(n, default)["weight"], g[dst].get(n, default)["weight"]
    count = cs + cd
    return {"count": count, "weight": (cs * ws + cd * wd) / count}


def merge_boundary(g, src, dst):
    return None


def ucm(fg: np.ndarray, seg: np.ndarray, prob: np.ndarray, k: float) -> np.ndarray:
    if not fg.any():
        return np.zeros(fg.shape, dtype=np.int32)
    dist = distance_transform_edt(seg != 2).astype(np.float32) * fg
    markers, n = label(h_maxima(dist, h=0.5) & fg)
    if n == 0:
        out, _ = label(fg)
        return out.astype(np.int32)
    over = watershed(-dist, markers=markers, mask=fg).astype(np.int32)
    if over.max() <= 1:
        return over
    pb = prob[2].astype(np.float64)
    rag = graph.rag_boundary(over, pb, connectivity=1)
    if 0 in rag:
        rag.remove_node(0)  # background is not a region to merge into
    merged = graph.merge_hierarchical(
        over,
        rag,
        thresh=k,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_boundary,
        weight_func=weight_boundary,
    ).astype(np.int32)
    merged[~fg] = 0
    merged[fg] += 1  # merge_hierarchical may hand out label 0
    return relabel_dense(lines_from_labels(merged, fg))


def swst(
    fg: np.ndarray, seg: np.ndarray, n: int, m: int, t: float, seed: int = 0
) -> np.ndarray:
    if not fg.any():
        return np.zeros(fg.shape, dtype=np.int32)
    dist = distance_transform_edt(seg != 2).astype(np.float32) * fg
    rng = np.random.default_rng(seed)
    r, c = np.nonzero(fg)
    freq = np.zeros(fg.shape, dtype=np.float32)
    for _ in range(m):
        idx = rng.choice(len(r), size=min(n, len(r)), replace=False)
        markers = np.zeros(fg.shape, dtype=np.int32)
        markers[r[idx], c[idx]] = np.arange(1, len(idx) + 1)
        lab = watershed(-dist, markers=markers, mask=fg, watershed_line=True)
        freq += fg & (lab == 0)
    freq /= m
    out, _ = label(fg & (freq < t))
    return out.astype(np.int32)


def iterh(
    fg: np.ndarray, seg: np.ndarray, hs: tuple[float, ...], a_max: int
) -> np.ndarray:
    if not fg.any():
        return np.zeros(fg.shape, dtype=np.int32)
    dist = distance_transform_edt(seg != 2).astype(np.float32) * fg

    def ws(mask: np.ndarray, h: float) -> np.ndarray:
        surf = dist * mask
        markers, n = label(h_maxima(surf, h=h) & mask)
        if n == 0:
            out, _ = label(mask)
            return out.astype(np.int32)
        return watershed(-surf, markers=markers, mask=mask, watershed_line=True).astype(
            np.int32
        )

    lab = ws(fg, hs[0])
    for h in hs[1:]:
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        nxt = lab.copy()
        cur = int(lab.max())
        for i, a in zip(ids, counts, strict=True):
            if a <= a_max:
                continue
            sub = lab == i
            part = ws(sub, h)
            if part.max() <= 1:
                continue
            nxt[sub] = 0
            nxt[part > 0] = part[part > 0] + cur
            cur += int(part.max())
        lab = nxt
    return relabel_dense(lab)


def thr(prob: np.ndarray, te: float, tb: float) -> np.ndarray:
    mask = (prob[1] >= te) & (prob[2] < tb)
    out, _ = label(mask)
    return out.astype(np.int32)


def run_prior(
    name: str, fg: np.ndarray, seg: np.ndarray, prob: np.ndarray
) -> np.ndarray:
    """``ucm-k0.3`` | ``swst-n40-m30-t0.5`` | ``iterh-a20000`` | ``thr-te0.5-tb0.3``."""
    kind, *args = name.split("-")
    kv = {
        a[:1] if a[:2] not in ("te", "tb") else a[:2]: a[1:]
        if a[:2] not in ("te", "tb")
        else a[2:]
        for a in args
    }
    if kind == "ucm":
        return ucm(fg, seg, prob, float(kv["k"]))
    if kind == "swst":
        return swst(fg, seg, int(kv["n"]), int(kv["m"]), float(kv["t"]))
    if kind == "iterh":
        return iterh(fg, seg, (8.0, 4.0, 2.0, 1.0), int(kv["a"]))
    if kind == "thr":
        return thr(prob, float(kv["te"]), float(kv["tb"]))
    raise ValueError(name)
