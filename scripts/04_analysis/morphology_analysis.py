"""Per-country field-morphology descriptors joined with the regularization gain.

Tests the hypothesis that edge regularization helps rectilinear fields and harms
organic ones. For each country: median rectangularity (area / minimum rotated
rectangle area), median convexity (area / convex hull area), and mean vertices
per polygon, computed from the raw postprocessing-stage polygons; joined with the
regularization effect (PQ at the regularize stage minus PQ at the raw stage) read
from the postprocessing-stage metrics. Reads existing scratch outputs only.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PP = Path("/u/agiri1/fieldvote/pp2")
OUT = Path("/u/agiri1/fieldvote/analysis")


def poly_descriptors(geom) -> tuple:
    if geom is None or geom.is_empty or geom.area <= 0:
        return None
    mrr = geom.minimum_rotated_rectangle
    hull = geom.convex_hull
    rect = geom.area / mrr.area if mrr.area > 0 else np.nan
    conv = geom.area / hull.area if hull.area > 0 else np.nan
    n = len(geom.exterior.coords) if geom.geom_type == "Polygon" else sum(
        len(g.exterior.coords) for g in geom.geoms
    )
    return rect, conv, n


def stage_pq(country: str, stage_tag: str) -> float:
    f = PP / f"metrics_{country}.csv"
    if not f.exists():
        return np.nan
    df = pd.read_csv(f)
    row = df[df["geojson"].str.contains(f"_pp{stage_tag}.geojson")]
    return float(row.iloc[0]["pq"]) if not row.empty else np.nan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raws = sorted(PP.glob("*_pp0_raw.geojson"))
    rows = []
    for f in raws:
        country = f.name.replace("_pp0_raw.geojson", "")
        gdf = gpd.read_file(f)
        descs = [poly_descriptors(g) for g in gdf.geometry]
        descs = [d for d in descs if d is not None]
        if not descs:
            continue
        rect, conv, nv = zip(*descs)
        raw_pq = stage_pq(country, "0_raw")
        reg_pq = stage_pq(country, "2_regularize")
        rows.append({
            "country": country,
            "n_polygons": len(descs),
            "median_rectangularity": float(np.median(rect)),
            "median_convexity": float(np.median(conv)),
            "mean_vertices": float(np.mean(nv)),
            "raw_pq": raw_pq,
            "regularize_pq": reg_pq,
            "delta_pq_regularize": reg_pq - raw_pq,
        })
    out = pd.DataFrame(rows).sort_values("median_rectangularity", ascending=False)
    out.to_csv(OUT / "morphology.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {OUT / 'morphology.csv'}")
    # association between rectangularity and regularization gain
    sub = out.dropna(subset=["median_rectangularity", "delta_pq_regularize"])
    if len(sub) >= 3:
        r = np.corrcoef(sub["median_rectangularity"], sub["delta_pq_regularize"])[0, 1]
        print(f"Pearson r(rectangularity, delta_PQ_regularize) = {r:.3f} over {len(sub)} countries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
