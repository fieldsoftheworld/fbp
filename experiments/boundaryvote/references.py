"""Reference (ground-truth) polygon sets for one cached patch.

Three conventions, all returned in the eval-grid pixel frame unless noted:

* ``vector``   true FTW parcels from ``data/ftw_polygons_clipped`` mapped into the
               pixel frame with FTP's ``_true_gt_shapes`` (FTP Table 1 protocol for
               Planet). ``areas_ha`` are true UTM areas.
* ``mask``     connected components of the semantic field class in the label raster
               (FTP's default mode; the legacy ICVR "semantic-mask" footing).
* ``instance`` FTW instance label raster (S2 grid) reprojected nearest-neighbour
               onto the eval grid and polygonized per id (legacy ICVR
               ``rescore_instance_gt``).

``vector_utm`` returns the parcels in their UTM CRS for FTP's native-GSD mode.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyproj
import rasterio
import rasterio.features
import rasterio.warp
import shapely.geometry
from rasterio.transform import Affine

from . import ftp_compat as ftp
from .paths import ftw_planet_repo

REFERENCES = ("vector", "mask", "instance")


@dataclass(frozen=True)
class PatchGrid:
    """One cached patch's eval-grid geometry plus its remapped label raster."""

    country: str
    patch_id: str
    height: int
    width: int
    transform: Affine
    crs: str
    gt: np.ndarray  # uint8 (H, W), FTP-remapped semantic label (0/1/3)

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width)."""
        return self.height, self.width


def load_grid(npz_path: Path, country: str) -> PatchGrid:
    """Load one patch's grid geometry and label raster from its npz cache."""
    with np.load(npz_path) as d:
        gt = d["gt"]
        t = d["transform"]
        crs = str(d["crs"])
    return PatchGrid(
        country=country,
        patch_id=npz_path.stem,
        height=int(gt.shape[0]),
        width=int(gt.shape[1]),
        transform=Affine(*[float(x) for x in t[:6]]),
        crs=crs,
        gt=gt,
    )


def load_prob(npz_path: Path) -> np.ndarray:
    """Cached softmax for one patch."""
    with np.load(npz_path) as d:
        return d["prob"].astype(np.float32)


def polygons_root() -> Path:
    """Root of the clipped FTW parquet ground truth."""
    return ftw_planet_repo() / "data" / "ftw_polygons_clipped"


def ftw_root() -> Path:
    """Root of the FTW (S2) dataset."""
    return ftw_planet_repo() / "data" / "ftw"


def vector(grid: PatchGrid) -> tuple[list, list[float]]:
    """True FTW parcels in the eval-grid pixel frame, FTP Table 1 protocol."""
    return ftp.true_gt_shapes(
        str(polygons_root()), grid.country, grid.patch_id, grid.crs, grid.transform
    )


def vector_utm(grid: PatchGrid) -> tuple[list, list[float], object]:
    """True FTW parcels in their native UTM CRS, for FTP's native-GSD mode."""
    return ftp.true_gt_utm(str(polygons_root()), grid.country, grid.patch_id)


def utm_crs(grid: PatchGrid) -> pyproj.CRS | None:
    """CRS of the patch's clipped GT parquet (UTM); None if the parquet is missing."""
    import geopandas as gpd

    p = polygons_root() / grid.country / f"{grid.patch_id}.parquet"
    if not p.exists():
        return None
    return gpd.read_parquet(p, columns=["geometry"]).crs


def mask(grid: PatchGrid) -> list:
    """Connected components of the field class in the label raster."""
    return ftp.extract_shapes((grid.gt == 1).astype(np.uint8))


def instance(grid: PatchGrid) -> list:
    """FTW instance raster reprojected onto the eval grid and polygonized."""
    src_path = (
        ftw_root() / grid.country / "label_masks" / "instance" / f"{grid.patch_id}.tif"
    )
    if not src_path.exists():
        return []
    with rasterio.open(src_path) as src:
        inst = src.read(1)
        dst = np.zeros(grid.shape, dtype=inst.dtype)
        rasterio.warp.reproject(
            inst,
            dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=rasterio.warp.Resampling.nearest,
        )
    shapes = []
    for geom, _val in rasterio.features.shapes(dst.astype(np.int32), mask=dst > 0):
        shapes.append(shapely.geometry.shape(geom))
    return shapes


def pixel_area_ha(grid: PatchGrid, utm: pyproj.CRS | None) -> float:
    """True ground area (ha) of one eval-grid pixel via the patch extent in UTM."""
    if utm is None:
        # Planet grids are already UTM metres.
        return abs(grid.transform.a * grid.transform.e) / 1e4
    import geopandas as gpd

    b = rasterio.transform.array_bounds(grid.height, grid.width, grid.transform)
    box = gpd.GeoSeries([shapely.geometry.box(*b)], crs=grid.crs)
    if str(box.crs) != str(utm):
        box = box.to_crs(utm)
    return float(box.area.iloc[0]) / (grid.height * grid.width) / 1e4
