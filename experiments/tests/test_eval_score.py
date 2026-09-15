"""Area-bin boundaries, per-patch tally arithmetic, strategy-set wiring, chunking."""

from argparse import Namespace
from pathlib import Path

import geopandas as gpd
import pytest
import shapely.geometry

from fbp.eval import paths
from fbp.eval.candidates import tag_to_file
from fbp.eval.paths import H_SWEEP, h_tag
from fbp.eval.score import (
    AREA_EDGES,
    area_bin_of,
    build_strategies,
    load_candidates,
    tally,
)


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
    out = build_strategies(cand, shape, {"sets": ("base",)})
    assert "cc" in out
    for h in H_SWEEP:
        assert f"raw:{h_tag(h)}" in out and f"rt:{h_tag(h)}" in out
    assert len(out) == 1 + 2 * len(H_SWEEP)
    assert not any(k.startswith("raw:v:") for k in out)  # no variant set requested


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


def test_load_candidates_missing_tag_filled_with_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every patch must carry every tag after loading, even if empty in some files."""
    monkeypatch.setenv("BOUNDARYVOTE_OUT", str(tmp_path))
    cdir = paths.candidates_dir("r/test") / "xx"
    cdir.mkdir(parents=True)
    box = shapely.geometry.box(0, 0, 2, 2)
    # patch 'a' has polygons under both tags; patch 'b' only under the later one
    gpd.GeoDataFrame(
        {"patch_id": ["a"], "inst_id": [1]}, geometry=[box], crs=None
    ).to_parquet(cdir / tag_to_file("cc"), index=False)
    gpd.GeoDataFrame(
        {"patch_id": ["a", "b"], "inst_id": [1, 1]}, geometry=[box, box], crs=None
    ).to_parquet(cdir / tag_to_file("v:nbg-none"), index=False)
    out = load_candidates("r", "test", "xx")
    assert set(out) == {"a", "b"}
    assert out["b"]["cc"] == [] and len(out["b"]["v:nbg-none"]) == 1
    assert len(out["a"]["cc"]) == 1


def test_load_candidates_tags_filter_restricts_to_those_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BOUNDARYVOTE_OUT", str(tmp_path))
    cdir = paths.candidates_dir("r/test") / "xx"
    cdir.mkdir(parents=True)
    box = shapely.geometry.box(0, 0, 2, 2)
    gpd.GeoDataFrame(
        {"patch_id": ["a"], "inst_id": [1]}, geometry=[box], crs=None
    ).to_parquet(cdir / tag_to_file("cc"), index=False)
    gpd.GeoDataFrame(
        {"patch_id": ["a"], "inst_id": [1]}, geometry=[box], crs=None
    ).to_parquet(cdir / tag_to_file("v:nbg-none"), index=False)
    out = load_candidates("r", "test", "xx", tags=("cc",))
    assert set(out["a"]) == {"cc"}


def test_chunked_scoring_equals_unchunked_on_synthetic_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scoring 3 one-tag variant chunks must equal one pass over all 3, order aside."""
    import numpy as np

    from fbp.eval import score as score_mod

    monkeypatch.setenv("BOUNDARYVOTE_OUT", str(tmp_path))
    run, split, country = "r", "test", "xx"

    cache = paths.cache_dir(f"{run}/{split}") / country
    cache.mkdir(parents=True)
    gt = np.zeros((20, 20), dtype=np.uint8)
    np.savez(
        cache / "p0.npz",
        prob=np.zeros((3, 20, 20), dtype=np.float16),
        seg=np.zeros((20, 20), dtype=np.uint8),
        gt=gt,
        transform=np.array([1.0, 0, 0, 0, -1.0, 0], dtype=np.float64),
        crs=np.array("EPSG:32633"),
    )

    cdir = paths.candidates_dir(f"{run}/{split}") / country
    cdir.mkdir(parents=True)
    box = shapely.geometry.box(1, 1, 5, 5)
    for tag in ("cc", *(h_tag(h) for h in H_SWEEP), "v:a", "v:b", "v:c"):
        gpd.GeoDataFrame(
            {"patch_id": ["p0"], "inst_id": [1]}, geometry=[box], crs=None
        ).to_parquet(cdir / tag_to_file(tag), index=False)

    monkeypatch.setattr(
        score_mod.refs,
        "vector",
        lambda grid: ([shapely.geometry.box(0, 0, 4, 4)], [1.0]),
    )

    def make_args(chunk_tags: int) -> Namespace:
        return Namespace(
            run="r",
            split="test",
            protocol="pixel",
            gsd_m=3.0,
            workers=1,
            sets=("base", "variants"),
            variants=None,
            chunk_tags=chunk_tags,
        )

    chunked = score_mod.run_country(make_args(1), country)
    single = score_mod.run_country(make_args(100), country)

    key = ["strategy"]
    a = chunked.sort_values(key).reset_index(drop=True)
    b = single.sort_values(key).reset_index(drop=True)
    assert sorted(a.strategy) == sorted(b.strategy)
    for col in ("n_gt", "n_pred", "tp_50", "fp_50", "fn_50", "iou_sum_50"):
        assert a.set_index("strategy")[col].equals(b.set_index("strategy")[col])
