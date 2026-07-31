"""Label-free per-field confidence from threshold agreement (the watershed jury).

For each best-threshold field polygon, count how many of the other h_min sweep
outputs contain an overlapping polygon (IoU >= 0.5). That count, normalized by
the number of sweep thresholds, is a label-free confidence score: fields that
survive across many thresholds are more likely real. This script computes and
saves the signal per polygon; calibration against ground truth is a separate
step. Reads the existing per-h_min GeoJSONs. Private scratch.
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path("/u/agiri1/github/ftw-planet")
INF = REPO / "logs/inference"
OUT = Path("/u/agiri1/fieldvote/analysis/agreement")
sys.path.insert(0, str(REPO / "scripts/inference"))
from score_minarea_at_best_hmin import country_best_geojson  # noqa: E402


def best_hmin_file(country: str) -> Path:
    summ = pd.read_csv(REPO / "logs/inference/results_summary.csv")
    row = summ[summ["country"] == country]
    h_min = float(row.iloc[0]["h_min_predict"]) if not row.empty else 12.0
    return REPO / country_best_geojson(country, h_min)


def list_hmin_files(country: str) -> list:
    files = sorted(INF.glob(f"{country}_fields_hmin*.geojson"))
    full = INF / f"{country}_fields_full.geojson"
    if full.exists():
        files.append(full)
    return files


def iou(a, b) -> float:
    if not a.intersects(b):
        return 0.0
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", required=True)
    ap.add_argument("--best-hmin-file", type=Path, default=None)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    args = ap.parse_args()

    best_file = args.best_hmin_file or best_hmin_file(args.country)
    if not best_file.exists():
        print(f"{args.country}: best-hmin file missing: {best_file}")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_hmin_files(args.country) if f != best_file]
    n_sets = len(files)
    if n_sets == 0:
        print(f"{args.country}: no other h_min files found")
        return 1

    best = gpd.read_file(best_file)
    others = []
    for f in files:
        g = gpd.read_file(f)
        others.append(g.set_index("patch_id").sort_index() if "patch_id" in g.columns else g)

    rows = []
    for _, p in best.iterrows():
        pid = p.get("patch_id")
        geom = p.geometry
        agree = 0
        for g in others:
            try:
                cand = g.loc[[pid]] if pid in g.index else g.iloc[0:0]
            except KeyError:
                cand = g.iloc[0:0]
            if any(iou(geom, q) >= args.iou_thresh for q in cand.geometry):
                agree += 1
        rows.append({
            "patch_id": pid,
            "area_m2": geom.area,
            "agreement_count": agree,
            "n_other_sets": n_sets,
            "agreement_frac": agree / n_sets,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"agreement_{args.country}.csv", index=False)
    print(f"{args.country}: {len(df)} polygons, {n_sets} sweep sets, "
          f"mean agreement_frac={df['agreement_frac'].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
