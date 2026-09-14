"""Area-bin boundaries, per-patch tally arithmetic, and strategy-set wiring."""

import shapely.geometry

from boundaryvote.paths import H_SWEEP, h_tag
from boundaryvote.run_eval import AREA_EDGES, area_bin_of, build_strategies, tally


def test_area_bin_boundaries() -> None:
    """edges=(0.5, 2.0): small < 0.5, medium is [0.5, 2.0), large >= 2.0."""
    assert AREA_EDGES == (0.5, 2.0)
    assert area_bin_of(0.0) == "small"
    assert area_bin_of(0.49) == "small"
    assert area_bin_of(0.5) == "medium"
    assert area_bin_of(1.99) == "medium"
    assert area_bin_of(2.0) == "large"
    assert area_bin_of(100.0) == "large"


def test_tally_size_bins_and_pred_area_ha_mode() -> None:
    """`pred_area_ha` given: pred area bin comes from p.area * pred_area_ha."""
    gt = [
        shapely.geometry.box(0, 0, 10, 10),
        shapely.geometry.box(20, 20, 30, 30),
        shapely.geometry.box(40, 40, 41, 41),
    ]
    gt_areas = [0.3, 1.0, 5.0]  # small, medium, large; the third is never matched
    matched = [shapely.geometry.box(0, 0, 10, 10), shapely.geometry.box(20, 20, 30, 30)]
    spurious = shapely.geometry.box(100, 100, 102, 102)  # area 4
    pred = [*matched, spurious]
    # spurious area 4 px^2 * pred_area_ha 0.01 = 0.04 ha -> small
    row = tally(gt, gt_areas, pred, pred_area_ha=0.01)
    assert row["n_gt"] == 3 and row["n_pred"] == 3
    assert row["tp_50_small"] == 1 and row["tp_50_medium"] == 1
    assert row["fn_50_large"] == 1
    assert row["fp_50_small"] == 1
    assert row["iou_sum_50"] == 2.0  # both matches are exact, IoU 1.0 each


def test_tally_pred_area_ha_none_uses_utm_m2_to_ha() -> None:
    """`pred_area_ha=None`: pred area bin comes from p.area * 1e-4 (UTM m^2 -> ha)."""
    gt = [shapely.geometry.box(0, 0, 141, 142)]  # area ~2ha in m^2
    pred = [shapely.geometry.box(0, 0, 141, 142)]
    row = tally(gt, [2.0], pred, pred_area_ha=None)
    assert row["tp_50_large"] == 1  # 141*142 m^2 * 1e-4 ha/m^2 = 2.0022 ha -> large


def test_build_strategies_base_set_has_expected_keys() -> None:
    shape = (30, 30)
    box = shapely.geometry.box(2, 2, 10, 10)
    cand = {"cc": [box]}
    for h in H_SWEEP:
        cand[h_tag(h)] = [box]
    opts = {"sets": ("base",), "taus": (0.5,), "drop_one": False}
    out = build_strategies(cand, shape, opts)
    assert "cc" in out
    for h in H_SWEEP:
        assert f"raw:{h_tag(h)}" in out and f"rt:{h_tag(h)}" in out
    assert "vote:t0.5" in out
    assert "vote:t0.5:k6" in out
    assert "nms" in out and "elbow" in out
    assert not any(k.startswith("raw:v:") for k in out)  # no variant set requested


def test_build_strategies_drop_one_adds_leave_one_out_votes() -> None:
    shape = (30, 30)
    box = shapely.geometry.box(2, 2, 10, 10)
    cand = {"cc": [box]}
    for h in H_SWEEP:
        cand[h_tag(h)] = [box]
    opts = {"sets": ("base",), "taus": (), "drop_one": True}
    out = build_strategies(cand, shape, opts)
    dropped = f"vote:t0.5:drop{h_tag(H_SWEEP[0])}"
    assert dropped in out


def test_build_strategies_variant_set_filters_by_tag() -> None:
    shape = (30, 30)
    box = shapely.geometry.box(2, 2, 10, 10)
    cand = {"v:nbg-none": [box], "v:arg-none": [box]}
    opts = {"sets": ("variants",), "variant_tags": {"v:nbg-none"}}
    out = build_strategies(cand, shape, opts)
    assert set(out) == {"raw:v:nbg-none", "rt:v:nbg-none"}


def test_build_strategies_variant_set_defaults_to_every_tag() -> None:
    shape = (30, 30)
    box = shapely.geometry.box(2, 2, 10, 10)
    cand = {"v:nbg-none": [box], "v:arg-none": [box]}
    opts = {"sets": ("variants",), "variant_tags": None}
    out = build_strategies(cand, shape, opts)
    assert set(out) == {
        "raw:v:nbg-none",
        "rt:v:nbg-none",
        "raw:v:arg-none",
        "rt:v:arg-none",
    }
