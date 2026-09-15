"""Every patch must carry every tag after loading, even if it is empty in some files."""

from pathlib import Path

import geopandas as gpd
import pytest
import shapely.geometry

from boundaryvote import paths
from boundaryvote.candidates import tag_to_file
from boundaryvote.run_eval import load_candidates


def test_missing_tag_filled_with_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
