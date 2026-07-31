"""Per-polygon uncertainty calibration. For the best-threshold polygons of a
country, compute each polygon's threshold-agreement fraction across the h-minima
sweep and its best IoU against the ground-truth instances, then report whether
agreement predicts correctness (Spearman, and mean IoU per agreement bin). This
decides whether threshold agreement is a useful label-free confidence signal.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import shapely.geometry
from scipy import stats

REPO = Path("/u/agiri1/github/ftw-planet")
sys.path.insert(0, str(REPO / "scripts/inference"))
import multi_hmin_ensemble as mhe  # noqa: E402


def iou(a, b):
    if not a.intersects(b):
        return 0.0
    i = a.intersection(b).area
    u = a.area + b.area - i
    return i / u if u else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", required=True)
    args = ap.parse_args()

    summ = pd.read_csv(REPO / "logs/inference/results_summary.csv")
    hmin = float(summ[summ["country"] == args.country].iloc[0]["h_min_predict"])
    gjs = mhe.find_geojsons(args.country, REPO / "logs/inference")
    polys_per_h = {h: mhe.load_polygons_by_patch(g) for h, g in gjs.items()}
    best_h = min(gjs.keys(), key=lambda h: abs(h - hmin))
    best = polys_per_h[best_h]
    n_other = len(polys_per_h) - 1

    ds = mhe.FTWPlanet(root="data", countries=[args.country], split="test",
                       transforms=None, load_boundaries=True)
    rows = []
    for s in ds.records:
        pid = s.get("patch_id")
        with rasterio.open(s["label"]) as src:
            gt = src.read(1)
            transform = src.transform
        gt_shapes = []
        for v in np.unique(gt):
            if v <= 0:
                continue
            for geom, _ in rasterio.features.shapes((gt == v).astype(np.uint8),
                                                     mask=(gt == v), transform=transform):
                gt_shapes.append(shapely.geometry.shape(geom))
        for poly in best.get(pid, []):
            agree = sum(1 for h in polys_per_h if h != best_h
                        and any(iou(poly, q) >= 0.5 for q in polys_per_h[h].get(pid, [])))
            best_iou = max((iou(poly, g) for g in gt_shapes), default=0.0)
            rows.append({"patch_id": pid,
                         "agree_frac": agree / n_other if n_other else 0.0,
                         "iou": best_iou})

    d = pd.DataFrame(rows)
    out = Path("/u/agiri1/fieldvote/analysis")
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / f"calib_{args.country}.csv", index=False)
    rho, p = stats.spearmanr(d["agree_frac"], d["iou"])
    d["bin"] = pd.cut(d["agree_frac"], [-0.01, 0.5, 0.8, 1.01])
    binned = d.groupby("bin", observed=True)["iou"].agg(["mean", "count"])
    print(f"{args.country}: n={len(d)} Spearman(agree,iou)={rho:.3f} p={p:.4f}")
    print(binned.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
