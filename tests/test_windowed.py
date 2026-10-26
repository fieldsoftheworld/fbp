"""``run_windowed``: windowed BoundaryVote without seam duplicates."""

from pathlib import Path

import numpy as np
import pytest

from fbp import BoundaryVote
from fbp.methods.base import Method
from fbp.windowed import _Instances, _place, _spans, _Window, run_windowed

METHOD = BoundaryVote(
    foreground="not_background", seed_depth=0.01, min_line_support=0.3, min_area_m2=900
)


def fields(
    h: int, w: int, boxes: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Field blocks (r0, r1, c0, c1) with a 1-px boundary ring each, on background."""
    seg = np.zeros((h, w), np.uint8)
    prob = np.zeros((3, h, w), np.float32)
    prob[0] = 1.0
    for r0, r1, c0, c1 in boxes:
        seg[r0:r1, c0:c1] = 2
        seg[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1] = 1
    prob[:, seg == 1] = np.array([0.05, 0.9, 0.05])[:, None]
    prob[:, seg == 2] = np.array([0.05, 0.15, 0.8])[:, None]
    return seg, prob


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True when both label arrays group the foreground pixels identically."""
    if not np.array_equal(a > 0, b > 0):
        return False
    m = a > 0
    pairs = np.unique(np.stack([a[m], b[m]]), axis=1)
    return len(np.unique(pairs[0])) == pairs.shape[1] == len(np.unique(pairs[1]))


def test_small_fields_match_whole_image() -> None:
    boxes = [
        (r, r + 14, c, c + 14) for r in range(2, 120, 18) for c in range(2, 120, 18)
    ]
    seg, prob = fields(128, 128, boxes)
    whole = METHOD(seg, prob, pixel_area_m2=9.0)
    win = run_windowed(METHOD, seg, prob, pixel_area_m2=9.0, core=48, halo=16)
    assert same_partition(whole, win)


@pytest.mark.parametrize(("merge_frac", "expected"), [(0.1, 1), (2.0, 2)])
def test_field_across_seam(merge_frac: float, expected: int) -> None:
    # 80 px tall field over the core boundary at row 64; halo 16 < half the field,
    # so each window sees a truncated copy whose centroid falls in its own core
    seg, prob = fields(128, 64, [(20, 100, 10, 50)])
    whole = METHOD(seg, prob, pixel_area_m2=9.0)
    win = run_windowed(
        METHOD, seg, prob, pixel_area_m2=9.0, core=64, halo=16, merge_frac=merge_frac
    )
    assert int(whole.max()) == 1
    assert int(win.max()) == expected
    if expected == 1:
        assert same_partition(whole, win)


def test_labels_are_dense_int32() -> None:
    seg, prob = fields(96, 96, [(4, 30, 4, 30), (40, 90, 40, 90)])
    win = run_windowed(METHOD, seg, prob, pixel_area_m2=9.0, core=32, halo=8)
    assert win.dtype == np.int32
    assert np.array_equal(np.unique(win), np.arange(int(win.max()) + 1))


def test_whole_copy_and_truncated_copy_merge() -> None:
    # The lower window (rows 40-128) sees the whole field and owns it (centroid row
    # ~84); the upper window (rows 0-88) sees only its top, whose centroid (~57) is in
    # the upper core. Only the truncated copy touches a window edge.
    mask = np.zeros((128, 64), bool)
    mask[42:52, 2:42] = True
    mask[52:90, 2:10] = True
    mask[90:110, 2:62] = True
    seg = np.where(mask, 1, 0).astype(np.uint8)
    ring = mask & ~np.pad(mask, 1)[2:, 1:-1] | mask & ~np.pad(mask, 1)[:-2, 1:-1]
    ring |= mask & ~np.pad(mask, 1)[1:-1, 2:] | mask & ~np.pad(mask, 1)[1:-1, :-2]
    seg[ring] = 2
    prob = np.zeros((3, *seg.shape), np.float32)
    prob[0] = 1.0
    prob[:, seg == 1] = np.array([0.05, 0.9, 0.05])[:, None]
    prob[:, seg == 2] = np.array([0.05, 0.15, 0.8])[:, None]
    whole = METHOD(seg, prob, pixel_area_m2=9.0)
    win = run_windowed(METHOD, seg, prob, pixel_area_m2=9.0, core=64, halo=24)
    assert int(whole.max()) == 1
    assert same_partition(whole, win)


def reference(
    method: Method, seg: np.ndarray, prob: np.ndarray, core: int, halo: int
) -> np.ndarray:
    """The 0.1.0 ``run_windowed``: int64 labels and one tile-wide np.unique relabel."""
    h, w = seg.shape
    out = np.zeros((h, w), np.int64)
    inst = _Instances()
    for r0, r1, wr0, wr1 in _spans(h, core, halo):
        for c0, c1, wc0, wc1 in _spans(w, core, halo):
            win = _Window(r0, r1, c0, c1, wr0, wr1, wc0, wc1)
            lab = method(
                seg[wr0:wr1, wc0:wc1], prob[:, wr0:wr1, wc0:wc1], pixel_area_m2=9.0
            )
            if lab.any():
                _place(out, inst, lab, win, 0.1)
    _, dense = np.unique(inst.roots(out), return_inverse=True)
    return dense.reshape(h, w).astype(np.int32)


def random_fields(seed: int, size: int = 192) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint random fields, some wider than the halo, many across window seams."""
    rng = np.random.default_rng(seed)
    taken = np.zeros((size, size), bool)
    boxes = []
    for _ in range(80):
        hh, ww = rng.integers(8, 70, 2)
        r0, c0 = rng.integers(0, size - hh), rng.integers(0, size - ww)
        if not taken[max(r0 - 2, 0) : r0 + hh + 2, max(c0 - 2, 0) : c0 + ww + 2].any():
            taken[r0 : r0 + hh, c0 : c0 + ww] = True
            boxes.append((int(r0), int(r0 + hh), int(c0), int(c0 + ww)))
    return fields(size, size, boxes)


@pytest.mark.parametrize("seed", range(6))
def test_labels_equal_the_0_1_0_implementation(seed: int) -> None:
    seg, prob = random_fields(seed)
    new = run_windowed(METHOD, seg, prob, pixel_area_m2=9.0, core=48, halo=16)
    assert new.dtype == np.int32
    assert np.array_equal(new, reference(METHOD, seg, prob, core=48, halo=16))


class Lazy:
    """Array stand-in that records the largest slice read."""

    def __init__(self, arr: np.ndarray) -> None:
        self.arr, self.shape, self.largest = arr, arr.shape, 0

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        """Slice the array and record the size."""
        part = self.arr[key]
        self.largest = max(self.largest, part.size)
        return part


def test_lazy_inputs_are_read_one_window_at_a_time() -> None:
    seg, prob = random_fields(1)
    lseg, lprob = Lazy(seg), Lazy(prob)
    lazy = run_windowed(METHOD, lseg, lprob, pixel_area_m2=9.0, core=48, halo=16)
    eager = run_windowed(METHOD, seg, prob, pixel_area_m2=9.0, core=48, halo=16)
    assert np.array_equal(lazy, eager)
    assert lseg.largest <= (48 + 2 * 16) ** 2
    assert lprob.largest <= 3 * (48 + 2 * 16) ** 2


def test_out_receives_the_labels(tmp_path: Path) -> None:
    seg, prob = random_fields(2)
    expected = run_windowed(METHOD, seg, prob, pixel_area_m2=9.0, core=48, halo=16)
    out = np.memmap(tmp_path / "labels", np.int32, "w+", shape=seg.shape)
    out[:] = 7  # stale contents are cleared
    got = run_windowed(METHOD, seg, prob, pixel_area_m2=9.0, core=48, halo=16, out=out)
    assert got is out
    assert np.array_equal(out, expected)
    with pytest.raises(ValueError, match="int32"):
        wrong = np.zeros(seg.shape, np.int64)
        run_windowed(METHOD, seg, prob, pixel_area_m2=9.0, core=48, halo=16, out=wrong)
