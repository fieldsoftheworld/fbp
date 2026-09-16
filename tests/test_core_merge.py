"""Union-find line merging and RAG-based hierarchical region merging."""

import numpy as np

from fbp.core.merge import UnionFind, merge_by_lines, merge_hierarchy
from fbp.core.regions import pairs_at_lines


def test_union_find_merges_and_lower_label_wins() -> None:
    uf = UnionFind(3)
    uf.union(2, 1)
    uf.union(1, 3)
    assert uf.find(1) == uf.find(2) == uf.find(3) == 1


def two_regions_with_a_line() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inst = np.zeros((5, 5), dtype=np.int32)
    inst[1:4, 1:2] = 1
    inst[1:4, 3:4] = 2
    lines = np.zeros((5, 5), dtype=bool)
    lines[1:4, 2] = True
    return inst, lines, inst.copy()


def test_merge_by_lines_keeps_a_supported_line_as_a_gap() -> None:
    inst, lines, _ = two_regions_with_a_line()
    r, c, pairs = pairs_at_lines(inst, lines)
    keys = pairs[:, 0].astype(np.int64) * (inst.max() + 1) + pairs[:, 1]
    uniq = np.unique(keys)
    keep = np.ones(len(uniq), dtype=bool)  # keep every line
    out = merge_by_lines(inst, r, c, pairs, keep, uniq)
    assert out.max() == 2
    assert (out[1:4, 2] == 0).all()  # the gap survives


def test_merge_by_lines_merges_and_fills_an_unsupported_line() -> None:
    inst, lines, _ = two_regions_with_a_line()
    r, c, pairs = pairs_at_lines(inst, lines)
    keys = pairs[:, 0].astype(np.int64) * (inst.max() + 1) + pairs[:, 1]
    uniq = np.unique(keys)
    keep = np.zeros(len(uniq), dtype=bool)  # drop every line
    out = merge_by_lines(inst, r, c, pairs, keep, uniq)
    assert out.max() == 1  # one merged parcel
    assert (out[1:4, 1:4] > 0).all()  # the seam is filled back in


def test_merge_hierarchy_single_region_is_a_no_op() -> None:
    over = np.ones((4, 4), dtype=np.int32)
    fg = over > 0
    out = merge_hierarchy(over, np.zeros((4, 4)), fg, level=0.5)
    np.testing.assert_array_equal(out, over)


def test_merge_hierarchy_merges_below_the_level() -> None:
    over = np.zeros((6, 6), dtype=np.int32)
    over[:, :3] = 1
    over[:, 3:] = 2
    fg = over > 0
    weak_edge = np.full((6, 6), 0.1)  # weak boundary everywhere: merges freely
    merged = merge_hierarchy(over, weak_edge, fg, level=0.5)
    strong_edge = np.full((6, 6), 0.9)  # strong boundary: nothing merges
    kept = merge_hierarchy(over, strong_edge, fg, level=0.5)
    assert merged.max() < kept.max()
    assert kept.max() == 2
