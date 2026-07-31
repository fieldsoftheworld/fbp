"""Novel method: soft-boundary watershed.

The standard pipeline derives watershed barriers from the argmax boundary class,
then floods a Euclidean distance transform. That discards the network's boundary
CONFIDENCE. Here we instead threshold the boundary-class softmax at tau to define
barriers, so a lower tau trusts faint boundaries and splits more, a higher tau
merges. Sweeping tau turns a learned, continuous signal into the watershed
control. Reads predict-v2 softmax (3-class: bg/field/boundary). Writes one
GeoJSON per tau; the existing scorer scores them. Private scratch.
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
from scipy.ndimage import distance_transform_edt

REPO = Path("/u/agiri1/github/ftw-planet")
sys.path.insert(0, str(REPO / "scripts"))
from postprocess_eval import watershed_instances  # noqa: E402


def soft_boundary_polys(mask, prob, transform, h_min, tau):
    boundary = prob[2] > tau                       # boundary-class softmax
    dist = distance_transform_edt(~boundary).astype(np.float32)
    inst = watershed_instances(mask, dist, h_min=h_min, field_class=1)
    out = []
    for geom, _ in rasterio.features.shapes(
        inst.astype(np.int32), mask=(inst > 0), transform=transform
    ):
        out.append(shapely.geometry.shape(geom))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", required=True)
    ap.add_argument("--softmax-dir", type=Path, default=Path("/u/agiri1/fieldvote/predict_v2"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--taus", default="0.20,0.33,0.50")
    args = ap.parse_args()

    summ = pd.read_csv(REPO / "logs/inference/results_summary.csv")
    row = summ[summ["country"] == args.country]
    h_min = float(row.iloc[0]["h_min_predict"]) if len(row) else 12.0

    cdir = args.softmax_dir / args.country
    index = json.loads((cdir / "index.json").read_text())
    crs = index[0]["crs"] if index else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for tau in [float(t) for t in args.taus.split(",")]:
        rows = []
        for e in index:
            pid = e["patch_id"]
            d = np.load(cdir / f"{pid}.npz")
            prob = d["prob"].astype(np.float32)
            mask = d["mask"].astype(np.uint8)
            transform = Affine(*[float(x) for x in d["transform"][:6]])
            for p in soft_boundary_polys(mask, prob, transform, h_min, tau):
                rows.append({"geometry": p, "patch_id": pid, "country": args.country})
        tag = str(tau).replace(".", "p")
        gpd.GeoDataFrame(rows, crs=crs).to_file(
            args.out_dir / f"{args.country}_softb{tag}.geojson", driver="GeoJSON"
        )
        print(f"{args.country} tau={tau}: {len(rows)} polygons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
