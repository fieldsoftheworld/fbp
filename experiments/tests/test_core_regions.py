"""Watershed regions, components, and the label/neighbour bookkeeping they share."""

import numpy as np

from fbp.core.regions import (
    components,
    lines_from_labels,
    neighbours4,
    pairs_at_lines,
    relabel_dense,
    watershed_regions,
)


def two_fields() -> tuple[np.ndarray, np.ndarray]:
    """Two 20x20 blocks joined by a thin neck, as a plain foreground mask."""
    fg = np.zeros((24, 48), dtype=bool)
    fg[2:22, 2:22] = True
    fg[2:22, 26:46] = True
    fg[10:14, 22:26] = True  # bridge
    return fg, np.ones((24, 48), dtype=np.float32)


def test_components_and_relabel_dense() -> None:
    fg, _ = two_fields()
    fg[10:14, 22:26] = False  # cut the bridge: two separate blocks
    lab = components(fg)
    assert lab.dtype == np.int32
    assert lab.max() == 2
    dense = relabel_dense(lab * 5)  # arbitrary label values, should compress to 0..2
    assert set(np.unique(dense)) == {0, 1, 2}


def test_watershed_regions_splits_two_fields() -> None:
    fg, surf = two_fields()
    surf = np.zeros_like(surf)
    surf[2:22, 2:22] = 1.0
    surf[2:22, 26:46] = 1.0
    regions = watershed_regions(surf, fg, h=0.5, lines=True)
    assert not regions.fallback
    assert regions.n_markers == 2
    assert regions.labels.max() == 2
    assert regions.lines.any()  # a watershed line separates the two markers


def test_watershed_regions_no_lines_when_disabled() -> None:
    fg, surf = two_fields()
    surf = np.zeros_like(surf)
    surf[2:22, 2:22] = 1.0
    surf[2:22, 26:46] = 1.0
    regions = watershed_regions(surf, fg, h=0.5, lines=False)
    assert not regions.lines.any()
    assert (regions.labels > 0).sum() == fg.sum()  # no gaps: every fg pixel labelled


def test_watershed_regions_empty_foreground() -> None:
    fg = np.zeros((8, 8), dtype=bool)
    surf = np.zeros((8, 8), dtype=np.float32)
    regions = watershed_regions(surf, fg, h=1.0)
    assert regions.fallback
    assert regions.labels.max() == 0
    assert not regions.lines.any()


def test_watershed_regions_falls_back_when_no_markers_found() -> None:
    """``h`` bigger than the surface's range leaves no h-maxima -> components."""
    fg, _ = two_fields()
    surf = np.full(fg.shape, 0.1, dtype=np.float32)
    regions = watershed_regions(surf, fg, h=100.0)
    assert regions.fallback
    assert regions.n_markers == 0
    np.testing.assert_array_equal(regions.labels, components(fg))


def test_lines_from_labels_erases_the_shared_border() -> None:
    lab = np.zeros((6, 6), dtype=np.int32)
    lab[1:5, 1:3] = 1
    lab[1:5, 3:5] = 2
    fg = lab > 0
    out = lines_from_labels(lab, fg)
    assert (out[:, 2] == 0).all() or (out[:, 3] == 0).all()  # the seam is erased
    assert out.max() == 2


def test_neighbours4_reads_the_four_orthogonal_neighbours() -> None:
    img = np.arange(9).reshape(3, 3)
    p = np.pad(img, 1)
    nb = neighbours4(p, np.array([1]), np.array([1]))
    # (1,1)'s neighbours: up=img[0,1]=1, down=img[2,1]=7, left=img[1,0]=3, right=5
    np.testing.assert_array_equal(nb, [[1, 7, 3, 5]])


def test_pairs_at_lines_reports_the_two_labels_a_line_pixel_touches() -> None:
    inst = np.zeros((5, 5), dtype=np.int32)
    inst[1:4, 1:2] = 1
    inst[1:4, 3:4] = 2
    lines = np.zeros((5, 5), dtype=bool)
    lines[1:4, 2] = True
    r, _c, pairs = pairs_at_lines(inst, lines)
    assert len(r) == 3
    assert (pairs == [1, 2]).all()
