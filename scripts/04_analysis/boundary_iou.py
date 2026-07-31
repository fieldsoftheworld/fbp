"""Boundary IoU (Cheng et al.) for a prediction GeoJSON against ground truth.

Semantic field-band version: rasterize the predicted field mask and the GT field
mask (label > 0) on the GT grid, take each mask's boundary band (the mask minus
its erosion by d pixels), and report the IoU of the two bands, averaged over
patches by pooling intersection and union. Appends one row to a shared CSV.
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from scipy.ndimage import binary_erosion

REPO = Path("/u/agiri1/github/ftw-planet")
sys.path.insert(0, str(REPO / "scripts/inference"))
import multi_hmin_ensemble as mhe  # noqa: E402
import pandas as pd  # noqa: E402
from score_minarea_at_best_hmin import country_best_geojson  # noqa: E402


def best_geojson(country):
    s = pd.read_csv(REPO / "logs/inference/results_summary.csv")
    h = float(s[s["country"] == country].iloc[0]["h_min_predict"])
    return REPO / country_best_geojson(country, h)


def band(mask, d):
    if mask.sum() == 0:
        return mask
    return mask & ~binary_erosion(mask, iterations=d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", required=True)
    ap.add_argument("--geojson", type=Path, default=None)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gj = args.geojson if args.geojson is not None else best_geojson(args.country)
    if not Path(gj).exists():
        print(f"{args.country} {args.tag}: geojson missing ({gj}), skipping")
        return 0
    gdf = gpd.read_file(gj)
    by_patch = {}
    for _, r in gdf.iterrows():
        by_patch.setdefault(r["patch_id"], []).append(r.geometry)

    ds = mhe.FTWPlanet(root="data", countries=[args.country], split="test",
                       transforms=None, load_boundaries=True)
    inter = union = 0
    for s in ds.records:
        pid = s.get("patch_id")
        with rasterio.open(s["label"]) as src:
            gt = src.read(1)
            transform = src.transform
            shape = gt.shape
        gt_field = gt > 0
        polys = by_patch.get(pid, [])
        if polys:
            pred_field = rasterio.features.rasterize(
                polys, out_shape=shape, transform=transform, dtype=np.uint8) > 0
        else:
            pred_field = np.zeros(shape, bool)
        gb = band(gt_field, args.d)
        pb = band(pred_field, args.d)
        inter += int((gb & pb).sum())
        union += int((gb | pb).sum())

    biou = inter / union if union else 0.0
    print(f"{args.country} {args.tag}: BoundaryIoU={biou:.4f}")
    out = Path("/u/agiri1/fieldvote/analysis/boundary_iou.csv")
    with out.open("a") as f:
        f.write(f"{args.country},{args.tag},{biou:.4f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
