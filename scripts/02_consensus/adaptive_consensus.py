"""Size-adaptive consensus. The probability blend helps where fields are small or
the prediction over-fragments, and the pure majority vote helps where fields are
large and clean. We pick between them per patch using a label-free signal, the
mean area of the vote-consensus instances: below a size threshold T we use the
blend, otherwise the pure vote. We sweep T to find the operating point. From
predict-v2 softmax; the existing scorer scores the output. Private scratch.
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
import shapely.geometry
from affine import Affine
from scipy.ndimage import distance_transform_edt

REPO = Path("/u/agiri1/github/ftw-planet")
sys.path.insert(0, str(REPO / "scripts"))
from postprocess_eval import watershed_instances  # noqa: E402

HMINS = [2.0, 4.0, 8.0, 12.0, 16.0, 24.0]
THRESHOLDS = {"a_T20k": 20000.0, "a_T50k": 50000.0, "a_T100k": 100000.0}


def field_polys(fm, transform):
    fm = fm.astype(np.uint8)
    return [shapely.geometry.shape(g) for g, val in
            rasterio.features.shapes(fm, mask=(fm > 0), transform=transform) if val == 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", required=True)
    ap.add_argument("--softmax-dir", type=Path, default=Path("/u/agiri1/fieldvote/predict_v2"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import json
    cdir = args.softmax_dir / args.country
    index = json.loads((cdir / "index.json").read_text())
    crs = index[0]["crs"] if index else None
    variants = {k: [] for k in THRESHOLDS}

    for e in index:
        pid = e["patch_id"]
        d = np.load(cdir / f"{pid}.npz")
        prob = d["prob"].astype(np.float32)
        mask = d["mask"].astype(np.uint8)
        transform = Affine(*[float(x) for x in d["transform"][:6]])
        covs = []
        for h in HMINS:
            boundary = (mask == 2).astype(np.uint8)
            dist = distance_transform_edt(boundary == 0).astype(np.float32)
            covs.append(watershed_instances(mask, dist, h_min=h, field_class=1) > 0)
        vote = np.mean(np.stack(covs, 0), 0)
        vote_field = vote >= 0.5
        blend_field = (0.4 * vote + 0.6 * prob[1]) >= 0.5
        vote_polys = field_polys(vote_field, transform)
        mean_area = float(np.mean([p.area for p in vote_polys])) if vote_polys else 0.0
        blend_polys = field_polys(blend_field, transform)
        for tag, T in THRESHOLDS.items():
            chosen = blend_polys if (0 < mean_area < T) else vote_polys
            for poly in chosen:
                variants[tag].append({"geometry": poly, "patch_id": pid, "country": args.country})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag, rows in variants.items():
        gpd.GeoDataFrame(rows, crs=crs).to_file(
            args.out_dir / f"{args.country}_{tag}.geojson", driver="GeoJSON")
    print(f"{args.country}: " + " ".join(f"{t}={len(r)}" for t, r in variants.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
