"""Boundary-supported splitting keeps supported lines and merges unsupported ones."""

import numpy as np
import pytest

from boundaryvote.variants import (
    drop_small,
    foreground,
    gap_close,
    run_variant,
    run_variants,
    supported_split,
)


def two_fields(gap_boundary: bool, boundary_pb: float) -> tuple[np.ndarray, np.ndarray]:
    """Two 20x20 blocks; a boundary column or a thin neck joins them."""
    H, W = 24, 48
    seg = np.zeros((H, W), dtype=np.uint8)
    prob = np.zeros((3, H, W), dtype=np.float32)
    prob[0] = 1.0
    seg[2:22, 2:22] = 1
    seg[2:22, 26:46] = 1
    prob[1][seg == 1] = 0.9
    prob[0][seg == 1] = 0.05
    prob[2][seg == 1] = 0.05
    # bridge between the blocks through rows 10..13
    seg[10:14, 22:26] = 1
    prob[1][10:14, 22:26] = 0.9
    prob[0][10:14, 22:26] = 0.05
    prob[2][10:14, 22:26] = 0.05
    if gap_boundary:
        # the model says boundary on most of a line down the bridge column 24
        col = 24
        seg[10:14, col] = 2
        seg[12, col] = 1  # one-pixel gap in the boundary line
        prob[2][10:14, col] = boundary_pb
        prob[1][10:14, col] = 1.0 - boundary_pb - 0.05
    return seg, prob


def test_unsupported_neck_is_merged() -> None:
    seg, prob = two_fields(gap_boundary=False, boundary_pb=0.0)
    fg = foreground(seg, prob, "arg")
    # return_lines defaults to False, so this is always an array; ty cannot see
    # that from the union return type, since the function has no @overload split
    # on return_lines, so the result is cast rather than narrowed.
    inst = supported_split(fg, seg, prob, h=1.0, tau=0.5)
    assert inst.max() == 1  # the geometric neck split is not supported -> one field


def test_supported_gap_is_kept() -> None:
    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    fg = foreground(seg, prob, "nbg")
    inst = supported_split(fg, seg, prob, h=1.0, tau=0.5)
    assert inst.max() == 2
    # instances are separated by a gap of zeros somewhere on the bridge
    assert (inst[10:14, 22:26] == 0).any()


def test_foreground_modes_and_close() -> None:
    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    assert foreground(seg, prob, "nbg").sum() > foreground(seg, prob, "arg").sum()
    assert foreground(seg, prob, "pf50").sum() == (prob[1] > 0.5).sum()
    inst = gap_close(foreground(seg, prob, "arg"), seg, 1)
    assert inst.max() == 2


def test_every_foreground_and_rule_runs() -> None:
    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    names = [
        *(f"{f}-none" for f in ("arg", "nbg", "pf50", "otsu")),
        "arg-ws2",
        "arg-close1",
        "nbg-sup-h2-t0.7",
        "nbg-pb-h0.01-t0.3",
        "nbg-osup-h0-t0.7",
        "nbg-ssup-h2-t0.6",
        "nbg-asup-h2-t0.5",
        "nbg-dsup-h2-t0.5",
    ]
    out = run_variants(seg, prob, names)
    assert len(out) == len(names)
    for v in out.values():
        assert v.dtype == np.int32 and v.shape == seg.shape and v.min() >= 0


def touching_pairs(inst: np.ndarray) -> int:
    """Number of 4-adjacent pixel pairs carrying two different non-zero labels."""
    a, b = inst[:, :-1], inst[:, 1:]
    n = int(((a != b) & (a > 0) & (b > 0)).sum())
    a, b = inst[:-1, :], inst[1:, :]
    return n + int(((a != b) & (a > 0) & (b > 0)).sum())


def test_kept_splits_are_gaps_on_real_patch() -> None:
    from boundaryvote.paths import cache_dir

    src = cache_dir("smoke_b3/test") / "portugal"
    npzs = sorted(src.glob("*.npz"))
    if not npzs:
        import pytest

        pytest.skip("smoke cache missing")
    for npz in npzs:
        with np.load(npz) as d:
            seg, prob = (
                d["seg"] if "seg" in d else d["prob"].astype(np.float32).argmax(0),
                d["prob"].astype(np.float32),
            )
        for name in ("nbg-sup-h2-t0.7", "arg-sup-h2-t0.5", "nbg-sup-h1-t0.3"):
            inst = run_variants(seg.astype(np.uint8), prob, [name])[name]
            assert touching_pairs(inst) == 0, name


def test_empty_patch() -> None:
    seg = np.zeros((8, 8), dtype=np.uint8)
    prob = np.zeros((3, 8, 8), dtype=np.float32)
    prob[0] = 1.0
    names = ("arg-none", "nbg-none", "arg-ws2", "nbg-sup-h2-t0.7", "nbg-pb-h0.01-t0.3")
    for name, v in run_variants(seg, prob, names).items():
        assert v.max() == 0, name


def test_no_adjacent_labels_on_many_real_patches() -> None:
    """The kept-split invariant on a spread of real patches, not just one country."""
    import itertools

    from boundaryvote.candidates import instance_polygons
    from boundaryvote.paths import DENSE10, cache_dir
    from boundaryvote.strategies import polygonize, rasterize

    checked = 0
    for country in DENSE10:
        src = cache_dir("planet_b3/test") / country
        for npz in itertools.islice(sorted(src.glob("*.npz")), 4):
            with np.load(npz) as d:
                seg, prob = d["seg"].astype(np.uint8), d["prob"].astype(np.float32)
            for name in ("nbg-sup-h2-t0.7", "nbg-sup-h2-t0.9"):
                inst = run_variants(seg, prob, [name])[name]
                assert touching_pairs(inst) == 0, f"{country}/{npz.stem} {name}"
                n_poly = len(instance_polygons(inst))
                n_rt = len(
                    polygonize(
                        rasterize([g for _, g in instance_polygons(inst)], seg.shape)
                    )
                )
                assert inst.max() == n_poly == n_rt, f"{country}/{npz.stem} {name}"
            checked += 1
    assert checked >= 20


def test_min_line_and_min_area_and_band() -> None:
    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    base = run_variants(seg, prob, ["nbg-sup-h1-t0.5"])["nbg-sup-h1-t0.5"]
    assert base.max() == 2
    # a minimum line length longer than the seam merges the two fields
    long_line = run_variants(seg, prob, ["nbg-sup-h1-t0.5+l50"])["nbg-sup-h1-t0.5+l50"]
    assert long_line.max() == 1
    # the tolerance band can only raise support, never lower it
    banded = run_variants(seg, prob, ["nbg-sup-h1-t0.5+b1"])["nbg-sup-h1-t0.5+b1"]
    assert banded.max() >= base.max()
    # area filter removes parcels below the threshold and relabels densely
    sizes = np.bincount(base.ravel())[1:]
    filtered = drop_small(base, int(sizes.max()))
    assert filtered.max() == 1
    assert set(np.unique(filtered)) <= {0, 1}


def test_fill_holes_and_merge_small() -> None:
    from boundaryvote.variants import fill_holes, merge_small

    inst = np.zeros((20, 20), dtype=np.int32)
    inst[2:12, 2:12] = 1
    inst[6:8, 6:8] = 0  # a 4-px hole inside parcel 1
    inst[15:17, 15:17] = 2  # a 4-px sliver, not adjacent to parcel 1
    inst[14, 2] = 3  # a 1-px parcel adjacent to nothing

    assert (fill_holes(inst, 0) == inst).all()  # off
    filled = fill_holes(inst, 10)
    assert (filled[6:8, 6:8] == 1).all() and filled.max() == inst.max()
    # a hole touching the image border is background, not a hole
    edge = np.zeros((10, 10), dtype=np.int32)
    edge[3:10, 3:10] = 1
    assert (fill_holes(edge, -1) == edge).all()

    # merging keeps the pixels; dropping removes them
    touching = np.zeros((10, 20), dtype=np.int32)
    touching[2:8, 2:10] = 1
    touching[2:4, 11:13] = 2  # small, separated by a one-pixel gap at col 10
    merged = merge_small(touching, 10)
    # one parcel, keeping every original pixel plus the gap it had to bridge
    assert merged.max() == 1
    assert (merged > 0).sum() >= (touching > 0).sum()
    assert ((touching > 0) & (merged == 0)).sum() == 0
    assert drop_small(touching, 10).max() == 1
    assert (drop_small(touching, 10) > 0).sum() < (touching > 0).sum()


def test_ws_rule_matches_ftp_pipeline() -> None:
    """`arg-ws2` must reproduce the FTP pipeline's watershed-then-components step."""
    from scipy.ndimage import distance_transform_edt, label

    from boundaryvote import ftp_compat as ftp

    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    got = run_variants(seg, prob, ["arg-ws2"])["arg-ws2"]
    dist = distance_transform_edt(seg != 2).astype(np.float32)
    want, _ = label(ftp.watershed_instances(seg, dist, h_min=2.0) > 0)
    assert (got > 0).sum() == (want > 0).sum() and got.max() == want.max()


def test_foreground_unknown_mode_raises() -> None:
    seg, prob = two_fields(gap_boundary=False, boundary_pb=0.0)
    with pytest.raises(ValueError):
        foreground(seg, prob, "bogus")


def test_supported_split_unknown_surface_raises() -> None:
    seg, prob = two_fields(gap_boundary=False, boundary_pb=0.0)
    fg = foreground(seg, prob, "nbg")
    with pytest.raises(ValueError):
        supported_split(fg, seg, prob, h=1.0, tau=0.5, surface="bogus")


def test_supported_split_unknown_score_raises() -> None:
    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    fg = foreground(seg, prob, "nbg")
    with pytest.raises(ValueError):
        supported_split(fg, seg, prob, h=1.0, tau=0.5, score="bogus")


def test_supported_split_empty_foreground_returns_zeros() -> None:
    seg = np.zeros((8, 8), dtype=np.uint8)
    prob = np.zeros((3, 8, 8), dtype=np.float32)
    prob[0] = 1.0
    fg = foreground(seg, prob, "nbg")
    out = supported_split(fg, seg, prob, h=1.0, tau=0.5, return_lines=True)
    inst, lines, kept = out
    assert inst.max() == 0 and not lines.any() and not kept.any()


def test_supported_split_falls_back_when_no_markers_found() -> None:
    """`h` bigger than the EDT surface leaves no h-maxima -> one component."""
    seg, prob = two_fields(gap_boundary=False, boundary_pb=0.0)
    fg = foreground(seg, prob, "nbg")
    inst = supported_split(fg, seg, prob, h=100.0, tau=0.5)
    assert inst.max() == 1


def test_supported_split_no_lines_short_circuits() -> None:
    """A single solid block has one h-maxima marker, so no watershed line at all."""
    seg = np.zeros((10, 10), dtype=np.uint8)
    prob = np.zeros((3, 10, 10), dtype=np.float32)
    prob[0] = 1.0
    seg[2:8, 2:8] = 1
    prob[1][seg == 1] = 0.9
    prob[0][seg == 1] = 0.05
    prob[2][seg == 1] = 0.05
    fg = foreground(seg, prob, "nbg")
    inst, lines, kept = supported_split(
        fg, seg, prob, h=1.0, tau=0.5, return_lines=True
    )
    assert inst.max() == 1 and not lines.any() and not kept.any()


def test_supported_split_return_lines_reports_kept_gap() -> None:
    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    fg = foreground(seg, prob, "nbg")
    inst, lines_all, lines_kept = supported_split(
        fg, seg, prob, h=1.0, tau=0.5, return_lines=True
    )
    assert inst.max() == 2
    assert lines_all.any() and lines_kept.any()
    subset = np.logical_and(lines_kept, np.logical_not(lines_all))
    assert subset.sum() == 0  # kept lines are a subset of all lines


def test_soft_score_falls_back_to_hard_test_under_band() -> None:
    """With `band>0` the soft score is documented to fall back to the hard rule."""
    seg, prob = two_fields(gap_boundary=True, boundary_pb=0.8)
    fg = foreground(seg, prob, "nbg")
    soft = supported_split(fg, seg, prob, h=1.0, tau=0.5, score="soft", band=1)
    frac = supported_split(fg, seg, prob, h=1.0, tau=0.5, score="frac", band=1)
    np.testing.assert_array_equal(soft, frac)


def test_oriented_surface_proposals() -> None:
    """The vote on oriented-signal proposals keeps the strong boundary, h=0 included."""
    from test_priors import two_field_scene

    seg, prob = two_field_scene()
    for name in ("nbg-osup-h0.05-t0.7", "nbg-osup-h0-t0.7", "nbg-pb-h0.1-t0.7+A9"):
        out = run_variant(name, seg, prob, px_m2=1.0)
        assert out.dtype == np.int32 and out.max() >= 2
