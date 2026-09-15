"""Reference-loading path construction and pixel-area arithmetic.

Vector ground truth (``vector``/``vector_utm``) needs the real FTW parcel
data and is not covered here.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import pytest
import shapely.geometry
from rasterio.transform import Affine

from fbp.eval.references import (
    PatchGrid,
    gt_raster,
    load_prob,
    pixel_area_ha,
    polygons_root,
    utm_crs,
)


def make_grid(
    country: str = "xx",
    patch_id: str = "p1",
    size: int = 10,
    transform: Affine | None = None,
    crs: str = "EPSG:32633",
) -> PatchGrid:
    gt = np.zeros((size, size), dtype=np.uint8)
    return PatchGrid(
        country=country,
        patch_id=patch_id,
        height=size,
        width=size,
        transform=transform or Affine(1.0, 0, 0, 0, -1.0, 0),
        crs=crs,
        gt=gt,
    )


def test_polygons_root_honours_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FTW_PLANET_REPO", str(tmp_path))
    assert polygons_root() == tmp_path.resolve() / "data" / "ftw_polygons_clipped"


def test_load_prob_reads_npz(tmp_path: Path) -> None:
    npz = tmp_path / "p1.npz"
    arr = np.random.default_rng(3).random((3, 4, 4))
    np.savez(npz, prob=arr)
    got = load_prob(npz)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, arr.astype(np.float32))


def test_utm_crs_missing_and_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FTW_PLANET_REPO", str(tmp_path))
    grid = make_grid()
    assert utm_crs(grid) is None

    pdir = polygons_root() / grid.country
    pdir.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"a": [1]}, geometry=[shapely.geometry.box(0, 0, 1, 1)], crs="EPSG:32633"
    ).to_parquet(pdir / f"{grid.patch_id}.parquet")
    crs = utm_crs(grid)
    assert crs is not None
    assert crs.to_epsg() == 32633


def test_pixel_area_ha_planet_grid_uses_transform_directly() -> None:
    grid = make_grid(transform=Affine(3.0, 0, 0, 0, -3.0, 0))
    assert pixel_area_ha(grid, None) == pytest.approx(9.0 / 1e4)


def test_pixel_area_ha_with_utm_crs() -> None:
    grid = make_grid(size=10, transform=Affine(2.0, 0, 0, 0, -2.0, 0))
    # 10x10 px at 2 m/px = 400 m^2 total / 100 px = 4 m^2/px = 0.0004 ha/px
    assert pixel_area_ha(grid, pyproj.CRS.from_epsg(32633)) == pytest.approx(0.0004)


def test_gt_raster_matches_manual_rasterize(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one-label-per-polygon rasterize gallery.py/method_fig.py used to inline."""
    grid = make_grid(size=20)
    shapes = [shapely.geometry.box(0, 0, 5, 5), shapely.geometry.box(10, 10, 15, 15)]

    monkeypatch.setattr("fbp.eval.references.vector", lambda g: (shapes, [1.0, 1.0]))
    out = gt_raster(grid)
    assert out.dtype == np.int32
    assert set(np.unique(out[out > 0]).tolist()) == {1, 2}
    assert (out[0:5, 0:5] == 1).all()
    assert (out[10:15, 10:15] == 2).all()


def test_gt_raster_empty_shapes_is_all_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = make_grid(size=8)
    monkeypatch.setattr("fbp.eval.references.vector", lambda g: ([], []))
    out = gt_raster(grid)
    assert out.shape == grid.shape
    assert out.dtype == np.int32
    assert not out.any()
