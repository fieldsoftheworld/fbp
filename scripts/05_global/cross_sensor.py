"""Quantitative cross-sensor evaluation. For each FTW ground-truth patch of a
country, read the corresponding window from the GLOBAL Sentinel-2 prediction
store (Zarr), run single-best and threshold-consensus polygonization on that S2
softmax, and write georeferenced polygons tagged with the patch id. The existing
scorer then scores them against the FTW ground truth, giving a polygon-level PQ
for consensus on Sentinel-2 predictions (the model elsewhere is Planet-based).
"""

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
import shapely.geometry
import zarr
from affine import Affine
from scipy.ndimage import distance_transform_edt

REPO = Path("/u/agiri1/github/ftw-planet")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/inference"))
import rasterio  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from postprocess_eval import watershed_instances  # noqa: E402
import multi_hmin_ensemble as mhe  # noqa: E402

URL = "https://data.source.coop/ftw/global-data/predictions/zarr/alpha/global.zarr"
TX = 8.98311982e-05
X0, Y0 = -180.0, 83.748345
HMINS = [2.0, 4.0, 8.0, 12.0, 16.0, 24.0]


def read_window(v, t, yi0, yi1, xi0, xi1, tries=5):
    for k in range(tries):
        try:
            return np.asarray(v[t, :, yi0:yi1, xi0:xi1])
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def field_polys(fm, transform):
    fm = fm.astype(np.uint8)
    return [shapely.geometry.shape(g) for g, val in
            rasterio.features.shapes(fm, mask=(fm > 0), transform=transform) if val == 1]


def inst_polys(mask3, transform, h_min):
    boundary = (mask3 == 2).astype(np.uint8)
    dist = distance_transform_edt(boundary == 0).astype(np.float32)
    inst = watershed_instances(mask3, dist, h_min=h_min, field_class=1)
    return [shapely.geometry.shape(g) for g, _ in
            rasterio.features.shapes(inst.astype(np.int32), mask=(inst > 0), transform=transform)]


def consensus_single(mask, transform):
    covs = []
    for h in HMINS:
        boundary = (mask == 2).astype(np.uint8)
        dist = distance_transform_edt(boundary == 0).astype(np.float32)
        covs.append(watershed_instances(mask, dist, h_min=h, field_class=1) > 0)
    vote = np.mean(np.stack(covs, 0), 0)
    # Consensus 3-class mask: field where the vote agrees, boundary kept from the
    # prediction, so watershed re-separates touching fields into instances rather
    # than merging them via connected components.
    cons_mask = (vote >= 0.5).astype(np.uint8)
    cons_mask[mask == 2] = 2
    cons = inst_polys(cons_mask, transform, h_min=8.0)
    single = inst_polys(mask, transform, h_min=12.0)
    return cons, single


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--t", type=int, default=0)
    args = ap.parse_args()

    g = zarr.open_group(URL, mode="r")
    v = g["variables"]
    # Iterate the actual TEST split so patch_ids and extents match the scorer.
    ds = mhe.FTWPlanet(root="data", countries=[args.country], split="test",
                       transforms=None, load_boundaries=True)
    cons_rows, single_rows = [], []
    n_ok = 0
    for s in ds.records:
        pid = s.get("patch_id")
        with rasterio.open(s["label"]) as src:
            gtcrs = src.crs
            b = src.bounds
        w, so, e, n = transform_bounds(gtcrs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
        xi0, xi1 = int((w - X0) / TX), int((e - X0) / TX)
        yi0, yi1 = int((Y0 - n) / TX), int((Y0 - so) / TX)
        if xi1 - xi0 < 4 or yi1 - yi0 < 4:
            continue
        prob = read_window(v, args.t, yi0, yi1, xi0, xi1)
        if prob.shape[1] < 4 or prob.shape[2] < 4:
            continue
        mask = prob.argmax(0).astype(np.uint8)
        transform = Affine(TX, 0, X0 + xi0 * TX, 0, -TX, Y0 - yi0 * TX)
        cp, sp = consensus_single(mask, transform)
        for poly in cp:
            cons_rows.append({"geometry": poly, "patch_id": pid, "country": args.country})
        for poly in sp:
            single_rows.append({"geometry": poly, "patch_id": pid, "country": args.country})
        n_ok += 1
    patches = ds.records

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag, rows in [("s2_consensus", cons_rows), ("s2_single", single_rows)]:
        if rows:
            gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        else:
            gdf = gpd.GeoDataFrame({"patch_id": [], "country": [], "geometry": []},
                                   geometry="geometry", crs="EPSG:4326")
        gdf.to_file(args.out_dir / f"{args.country}_{tag}.geojson", driver="GeoJSON")
    print(f"{args.country}: {n_ok}/{len(patches)} test patches read, consensus={len(cons_rows)} single={len(single_rows)} polys")
    return 0


if __name__ == "__main__":
    sys.exit(main())
