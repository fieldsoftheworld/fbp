"""Reference-loading path construction and pure-array reference extraction.

Vector ground truth (``vector``/``vector_utm``) needs the real FTW parcel
data and is not covered here; everything below is testable with ``tmp_path``
and ``monkeypatch``, following the pattern in ``test_load_candidates.py``.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import pytest
import rasterio
import shapely.geometry
from rasterio.transform import Affine

from boundaryvote.references import (
    PatchGrid,
    ftw_root,
    instance,
    load_prob,
    mask,
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


def test_polygons_root_and_ftw_root_honour_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FTW_PLANET_REPO", str(tmp_path))
    assert polygons_root() == tmp_path.resolve() / "data" / "ftw_polygons_clipped"
    assert ftw_root() == tmp_path.resolve() / "data" / "ftw"


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


def test_mask_extracts_field_class_components() -> None:
    grid = make_grid()
    grid.gt[2:6, 2:6] = 1  # field
    grid.gt[7, 7] = 2  # boundary, not field -> excluded
    shapes = mask(grid)
    assert len(shapes) == 1
    assert shapes[0].area == pytest.approx(16.0)


def write_instance_tif(
    tif_dir: Path, patch_id: str, transform: Affine, crs: str
) -> None:
    tif_dir.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((10, 10), dtype=np.int32)
    arr[2:5, 2:5] = 1
    arr[6:9, 6:9] = 2
    with rasterio.open(
        tif_dir / f"{patch_id}.tif",
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="int32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr, 1)


def test_instance_reprojects_and_polygonizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FTW_PLANET_REPO", str(tmp_path))
    grid = make_grid()
    write_instance_tif(
        ftw_root() / grid.country / "label_masks" / "instance",
        grid.patch_id,
        grid.transform,
        grid.crs,
    )
    shapes = instance(grid)
    assert len(shapes) == 2
    assert sum(s.area for s in shapes) == pytest.approx(18.0)


def test_instance_missing_file_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FTW_PLANET_REPO", str(tmp_path))
    grid = make_grid(country="yy", patch_id="missing")
    assert instance(grid) == []


def test_pixel_area_ha_planet_grid_uses_transform_directly() -> None:
    grid = make_grid(transform=Affine(3.0, 0, 0, 0, -3.0, 0))
    assert pixel_area_ha(grid, None) == pytest.approx(9.0 / 1e4)


def test_pixel_area_ha_with_utm_crs() -> None:
    grid = make_grid(size=10, transform=Affine(2.0, 0, 0, 0, -2.0, 0))
    # 10x10 px at 2 m/px = 400 m^2 total / 100 px = 4 m^2/px = 0.0004 ha/px
    assert pixel_area_ha(grid, pyproj.CRS.from_epsg(32633)) == pytest.approx(0.0004)
