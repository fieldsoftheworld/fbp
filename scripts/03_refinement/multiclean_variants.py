"""Produce field polygons from saved softmax with and without MultiClean
pre-vectorization cleanup, so the cleanup-before-vectorize stage can be scored on
polygon-level metrics. Reads predict-v2 npz (prob, mask, transform), writes
<country>_vraw.geojson (watershed on the argmax mask) and <country>_vmc.geojson
(MultiClean(mask) then watershed). The existing scorer scores both. Private scratch.
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import shapely.geometry
from affine import Affine
from multiclean import clean_array
from scipy.ndimage import distance_transform_edt

REPO = Path("/u/agiri1/github/ftw-planet")
sys.path.insert(0, str(REPO / "scripts"))
from postprocess_eval import watershed_instances  # noqa: E402


def mask_to_polys(mask: np.ndarray, transform: Affine, h_min: float) -> list:
    boundary = (mask == 2).astype(np.uint8)
    dist = distance_transform_edt(boundary == 0).astype(np.float32)
    inst = watershed_instances(mask, dist, h_min=h_min, field_class=1)
    polys = []
    for geom, _ in rasterio.features.shapes(
        inst.astype(np.int32), mask=(inst > 0), transform=transform
    ):
        polys.append(shapely.geometry.shape(geom))
    return polys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", required=True)
    ap.add_argument("--softmax-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=REPO / "logs/inference/results_summary.csv")
    ap.add_argument("--min-island", type=int, default=100)
    ap.add_argument("--smooth-edge", type=int, default=2)
    args = ap.parse_args()

    summ = pd.read_csv(args.summary)
    row = summ[summ["country"] == args.country]
    h_min = float(row.iloc[0]["h_min_predict"]) if not row.empty else 12.0

    cdir = args.softmax_dir / args.country
    index = json.loads((cdir / "index.json").read_text())
    crs = index[0]["crs"] if index else None

    rows_raw: list = []
    rows_mc: list = []
    for entry in index:
        pid = entry["patch_id"]
        d = np.load(cdir / f"{pid}.npz")
        mask = d["mask"].astype(np.uint8)
        transform = Affine(*[float(x) for x in d["transform"][:6]])
        for poly in mask_to_polys(mask, transform, h_min):
            rows_raw.append({"geometry": poly, "patch_id": pid, "country": args.country})
        mc = clean_array(mask, min_island_size=args.min_island, smooth_edge_size=args.smooth_edge)
        for poly in mask_to_polys(mc.astype(np.uint8), transform, h_min):
            rows_mc.append({"geometry": poly, "patch_id": pid, "country": args.country})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(rows_raw, crs=crs).to_file(
        args.out_dir / f"{args.country}_vraw.geojson", driver="GeoJSON"
    )
    gpd.GeoDataFrame(rows_mc, crs=crs).to_file(
        args.out_dir / f"{args.country}_vmc.geojson", driver="GeoJSON"
    )
    print(f"{args.country}: h_min={h_min} raw_polys={len(rows_raw)} mc_polys={len(rows_mc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
