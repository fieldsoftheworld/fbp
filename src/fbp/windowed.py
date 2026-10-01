"""Run a method over a large raster in overlapping windows without seam artifacts.

The raster is cut into ``core`` x ``core`` windows, each processed with ``halo``
pixels of context on every side. An instance is kept by the window whose core holds
its pixel centroid, so small instances are never duplicated. A field wider than
about two halos that crosses a core boundary is seen truncated by both windows, and
each copy's centroid can land in its own core. Such copies are merged into one
instance when they overlap by at least ``merge_frac`` of the smaller copy and at
least one of them touches its window edge. Instances from one window never overlap;
remaining conflicts between windows go to the window whose core holds the pixel.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from fbp.methods import run


@dataclass(frozen=True)
class _Window:
    """Core ``[r0, r1) x [c0, c1)`` inside window ``[wr0, wr1) x [wc0, wc1)``."""

    r0: int
    r1: int
    c0: int
    c1: int
    wr0: int
    wr1: int
    wc0: int
    wc1: int


def _spans(size: int, core: int, halo: int) -> Iterator[tuple[int, int, int, int]]:
    """(core start, core stop, window start, window stop) along one axis."""
    for c0 in range(0, size, core):
        c1 = min(size, c0 + core)
        yield c0, c1, max(0, c0 - halo), min(size, c1 + halo)


class _Instances:
    """Union-find over global instance ids with pixel counts (id 0 is background)."""

    def __init__(self) -> None:
        self.parent = [0]
        self.area = [0]
        self.touch = [False]

    def add(self, touch: bool) -> int:
        self.parent.append(len(self.parent))
        self.area.append(0)
        self.touch.append(touch)
        return len(self.parent) - 1

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> int:
        a, b = sorted((self.find(a), self.find(b)))
        if a != b:
            self.parent[b] = a
            self.area[a] += self.area[b]
            self.touch[a] = self.touch[a] or self.touch[b]
        return a

    def roots(self, ids: np.ndarray) -> np.ndarray:
        """Root of every id in ``ids`` (any shape)."""
        uniq, inv = np.unique(ids, return_inverse=True)
        return np.array([self.find(int(g)) for g in uniq], np.int64)[inv].reshape(
            ids.shape
        )


def run_windowed(
    strategy: str,
    segmentation: np.ndarray,
    probabilities: np.ndarray,
    *,
    px_m2: float,
    core: int = 8192,
    halo: int = 512,
    merge_frac: float = 0.1,
) -> np.ndarray:
    """Run ``strategy`` over overlapping windows; dense int32 labels, 0 = background.

    Arguments match :func:`fbp.run`. ``core`` is the window stride, ``halo`` the
    context added on each side, and ``merge_frac`` the overlap (fraction of the
    smaller copy) at which two copies of a field from different windows become one
    instance.
    """
    h, w = segmentation.shape
    out = np.zeros((h, w), np.int64)
    inst = _Instances()
    for r0, r1, wr0, wr1 in _spans(h, core, halo):
        for c0, c1, wc0, wc1 in _spans(w, core, halo):
            win = _Window(r0, r1, c0, c1, wr0, wr1, wc0, wc1)
            rows, cols = slice(wr0, wr1), slice(wc0, wc1)
            lab = run(
                strategy,
                segmentation[rows, cols],
                probabilities[:, rows, cols],
                px_m2=px_m2,
            )
            if lab.any():
                _place(out, inst, lab, win, merge_frac)
    _, dense = np.unique(inst.roots(out), return_inverse=True)
    return dense.reshape(h, w).astype(np.int32)


def _owned(
    lab: np.ndarray, win: _Window, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per local label: owned by this window, touches a window edge, pixel count."""
    n = int(lab.max()) + 1
    m = lab > 0
    idx = lab[m]
    rr, cc = np.nonzero(m)
    cnt = np.bincount(idx, minlength=n)
    safe = np.maximum(cnt, 1)
    cy = np.bincount(idx, rr, n) / safe + win.wr0
    cx = np.bincount(idx, cc, n) / safe + win.wc0
    own = (cnt > 0) & (cy >= win.r0) & (cy < win.r1) & (cx >= win.c0) & (cx < win.c1)
    own[0] = False
    edge = np.zeros(lab.shape, bool)  # window edges that are not raster edges
    edge[0, :] |= win.wr0 > 0
    edge[-1, :] |= win.wr1 < shape[0]
    edge[:, 0] |= win.wc0 > 0
    edge[:, -1] |= win.wc1 < shape[1]
    touch = np.bincount(lab[edge & m], minlength=n) > 0
    return own, touch, cnt


def _match(
    inst: _Instances,
    new: np.ndarray,
    old: np.ndarray,
    stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    merge_frac: float,
) -> np.ndarray:
    """Global id for each owned local label: a merged existing instance or a new one."""
    own, touch, cnt = stats
    gid = np.zeros(len(own), np.int64)
    hit = old > 0
    if hit.any():
        pairs = np.stack([new[hit], inst.roots(old[hit])])
        uniq, overlap = np.unique(pairs, axis=1, return_counts=True)
        for (loc, g), ov in zip(uniq.T.tolist(), overlap.tolist(), strict=True):
            root = inst.find(g)
            smaller = min(int(cnt[loc]), inst.area[root])
            if ov >= merge_frac * smaller and (touch[loc] or inst.touch[root]):
                gid[loc] = inst.union(int(gid[loc]), root) if gid[loc] else root
    for loc in np.nonzero(own)[0]:
        if gid[loc] == 0:
            gid[loc] = inst.add(bool(touch[loc]))
        elif touch[loc]:
            inst.touch[inst.find(int(gid[loc]))] = True
    return gid


def _place(
    out: np.ndarray, inst: _Instances, lab: np.ndarray, win: _Window, merge_frac: float
) -> None:
    """Merge one window's owned instances into ``out``."""
    stats = _owned(lab, win, out.shape)
    own = stats[0]
    if not own.any():
        return
    view = out[win.wr0 : win.wr1, win.wc0 : win.wc1]
    sel = own[lab]
    gid = _match(inst, lab[sel], view[sel], stats, merge_frac)
    target = inst.roots(gid)[lab]
    cur = inst.roots(view)
    in_core = np.zeros(lab.shape, bool)
    in_core[
        win.r0 - win.wr0 : win.r1 - win.wr0, win.c0 - win.wc0 : win.c1 - win.wc0
    ] = True
    write = sel & ((cur == 0) | (cur == target) | in_core)
    changed = write & (cur != target)
    for g, k in zip(
        *np.unique(cur[changed & (cur > 0)], return_counts=True), strict=True
    ):
        inst.area[int(g)] -= int(k)
    for g, k in zip(*np.unique(target[changed], return_counts=True), strict=True):
        inst.area[int(g)] += int(k)
    view[write] = target[write]
