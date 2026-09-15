"""Union-find line merging and RAG-based hierarchical region merging."""

import numpy as np
from scipy.ndimage import label as ndi_label
from skimage import graph

from fbp.core.regions import lines_from_labels, neighbours4, relabel_dense


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


def weight_boundary(g: graph.RAG, src: int, dst: int, n: int) -> dict[str, int | float]:
    """Count-weighted mean boundary probability after merging src into dst."""
    default = {"weight": 0.0, "count": 0}
    cs, cd = g[src].get(n, default)["count"], g[dst].get(n, default)["count"]
    ws, wd = g[src].get(n, default)["weight"], g[dst].get(n, default)["weight"]
    count = cs + cd
    return {"count": count, "weight": (cs * ws + cd * wd) / count}


def merge_boundary(_g: graph.RAG, _src: int, _dst: int) -> None:
    """No-op merge callback: ``weight_boundary`` carries all the state we keep."""


def merge_by_lines(
    inst: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    pairs: np.ndarray,
    keep: np.ndarray,
    uniq: np.ndarray,
) -> np.ndarray:
    """Union unsupported line pairs and fill the junctions that fully merged.

    ``keep`` is a boolean per entry of ``uniq`` (the sorted, unique
    ``a * base + b`` pair keys of ``pairs``): the union-find merges every pair
    whose key is *not* kept, then a line pixel is filled only when every
    region it touches ended up in the same merged group, so a kept split
    always stays a gap and two parcels are never adjacent.
    """
    valid = pairs[:, 0] > 0
    base = int(inst.max()) + 1
    uf = UnionFind(int(inst.max()))
    for k, k_ok in zip(uniq, keep, strict=True):
        if not k_ok:
            uf.union(int(k // base), int(k % base))
    lut = np.array([uf.find(i) for i in range(int(inst.max()) + 1)], dtype=np.int32)
    merged = lut[inst]
    p = np.pad(inst, 1)
    nbm = lut[neighbours4(p, r, c)]
    mx = nbm.max(axis=1)
    cand = np.where(
        valid & (mx > 0) & ((nbm == mx[:, None]) | (nbm == 0)).all(axis=1), mx, 0
    )
    cand_img = np.zeros_like(merged)
    cand_img[r, c] = cand
    q = np.pad(cand_img, 1)
    nbc = neighbours4(q, r, c)
    conflict = ((nbc != 0) & (nbc != cand[:, None])).any(axis=1)
    keep_px = (cand > 0) & ~conflict
    merged[r[keep_px], c[keep_px]] = cand[keep_px]
    dense, _ = ndi_label(merged > 0)
    return dense.astype(np.int32, copy=False)


def merge_hierarchy(
    over: np.ndarray,
    edge_surface: np.ndarray,
    fg: np.ndarray,
    level: float,
    weights: dict[tuple[int, int], tuple[float, int]] | None = None,
) -> np.ndarray:
    """RAG-boundary hierarchical merge shared by ``ucm`` and ``owt``.

    Builds ``rag_boundary(over, edge_surface)`` with the background node
    dropped, optionally overrides specific edge weights (owt's oriented
    signal), then greedily merges below ``level`` and relabels densely.
    """
    if over.max() <= 1:
        return over.astype(np.int32, copy=False)
    rag = graph.rag_boundary(over, edge_surface.astype(np.float64), connectivity=1)
    if 0 in rag:
        rag.remove_node(0)  # background is not a region to merge into
    if weights:
        for (a, b), (w, n_px) in weights.items():
            if rag.has_edge(a, b):
                rag[a][b]["weight"] = w
                rag[a][b]["count"] = n_px
    merged = graph.merge_hierarchical(
        over,
        rag,
        thresh=level,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_boundary,
        weight_func=weight_boundary,
    ).astype(np.int32, copy=False)
    merged[~fg] = 0
    merged[fg] += 1  # merge_hierarchical may hand out label 0
    return relabel_dense(lines_from_labels(merged, fg))
