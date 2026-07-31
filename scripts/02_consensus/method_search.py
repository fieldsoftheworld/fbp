"""Method search: try new consensus combination rules and see if any beats the
hard per-pixel majority vote. From predict-v2 softmax, for each patch we compute
per-threshold field coverage, then form several consensus fields:
  v0_hardvote   : pixel is field if >= 50% of thresholds cover it (current best)
  v1_blend_a40  : 0.4*vote + 0.6*field_prob >= 0.5  (lean on network confidence)
  v2_blend_a60  : 0.6*vote + 0.4*field_prob >= 0.5  (lean on vote stability)
  v3_probfloor  : vote >= 0.5 OR field_prob >= 0.8  (rescue high-confidence pixels)
Each field set is written as a GeoJSON; the existing scorer scores them so the
macro PQ is directly comparable to the hard-vote consensus. Private scratch.
"""

import argparse
import json
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


def field_polys(field_mask, transform):
    fm = field_mask.astype(np.uint8)
    return [shapely.geometry.shape(g) for g, val in
            rasterio.features.shapes(fm, mask=(fm > 0), transform=transform) if val == 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", required=True)
    ap.add_argument("--softmax-dir", type=Path, default=Path("/u/agiri1/fieldvote/predict_v2"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    cdir = args.softmax_dir / args.country
    index = json.loads((cdir / "index.json").read_text())
    crs = index[0]["crs"] if index else None
    variants = {"v0_hardvote": [], "v1_blend_a40": [], "v2_blend_a60": [], "v3_probfloor": []}

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
            inst = watershed_instances(mask, dist, h_min=h, field_class=1)
            covs.append(inst > 0)
        vote = np.mean(np.stack(covs, 0), 0)
        fp = prob[1]
        fields = {
            "v0_hardvote": vote >= 0.5,
            "v1_blend_a40": (0.4 * vote + 0.6 * fp) >= 0.5,
            "v2_blend_a60": (0.6 * vote + 0.4 * fp) >= 0.5,
            "v3_probfloor": (vote >= 0.5) | (fp >= 0.8),
        }
        for tag, fm in fields.items():
            for poly in field_polys(fm, transform):
                variants[tag].append({"geometry": poly, "patch_id": pid, "country": args.country})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag, rows in variants.items():
        gpd.GeoDataFrame(rows, crs=crs).to_file(
            args.out_dir / f"{args.country}_{tag}.geojson", driver="GeoJSON")
    print(f"{args.country}: " + " ".join(f"{t}={len(r)}" for t, r in variants.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
