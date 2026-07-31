"""Run the threshold-consensus postprocessing on the global Fields of The World
Sentinel-2 prediction store (Zarr v3). For a lon/lat window it reads the 3-class
softmax (background/field/boundary), runs single-best watershed and the
multi-threshold consensus, saves both polygon sets as GeoJSON (EPSG:4326),
records scale statistics, and renders a single-best-vs-consensus before/after
figure over the field-probability map. This is the global, cross-sensor
(Sentinel-2) demonstration of the technique developed on Planet.
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.features
import shapely.geometry
import zarr
from affine import Affine
from scipy.ndimage import distance_transform_edt

REPO = Path("/u/agiri1/github/ftw-planet")
sys.path.insert(0, str(REPO / "scripts"))
from postprocess_eval import watershed_instances  # noqa: E402

URL = "https://data.source.coop/ftw/global-data/predictions/zarr/alpha/global.zarr"
TX = 8.98311982e-05
X0, Y0 = -180.0, 83.748345
HMINS = [2, 4, 8, 12, 16, 24]
BEST_H = 12


def watershed_polys(mask, transform, h_min):
    boundary = (mask == 2).astype(np.uint8)
    dist = distance_transform_edt(boundary == 0).astype(np.float32)
    inst = watershed_instances(mask, dist, h_min=h_min, field_class=1)
    return [shapely.geometry.shape(g) for g, _ in
            rasterio.features.shapes(inst.astype(np.int32), mask=(inst > 0), transform=transform)]


def n_vertices(polys):
    return sum(len(p.exterior.coords) for p in polys if p.geom_type == "Polygon")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--size", type=int, default=1536)
    ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    g = zarr.open_group(URL, mode="r")
    xi = int((args.lon - X0) / TX)
    yi = int((Y0 - args.lat) / TX)
    prob = np.asarray(g["variables"][args.t, :, yi:yi + args.size, xi:xi + args.size])
    mask = prob.argmax(0).astype(np.uint8)
    transform = Affine(TX, 0, X0 + xi * TX, 0, -TX, Y0 - yi * TX)

    vote = np.zeros(mask.shape, np.int32)
    for h in HMINS:
        polys = watershed_polys(mask, transform, h)
        cov = rasterio.features.rasterize(
            polys, out_shape=mask.shape, transform=transform, dtype=np.uint8) > 0 if polys else np.zeros(mask.shape, bool)
        vote += cov.astype(np.int32)
    k = int(np.ceil(len(HMINS) * 0.5))
    consensus = (vote >= k).astype(np.uint8)
    cons_polys = [shapely.geometry.shape(geom) for geom, _ in
                  rasterio.features.shapes(consensus, mask=(consensus > 0), transform=transform)]
    single_polys = watershed_polys(mask, transform, BEST_H)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag, polys in [("single", single_polys), ("consensus", cons_polys)]:
        if polys:
            gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326").to_file(
                args.out_dir / f"{args.name}_{tag}.geojson", driver="GeoJSON")

    agree = vote / len(HMINS)
    stats = {
        "region": args.name, "lon": args.lon, "lat": args.lat,
        "field_frac": float((mask == 1).mean()),
        "single_polys": len(single_polys), "consensus_polys": len(cons_polys),
        "single_vertices": n_vertices(single_polys), "consensus_vertices": n_vertices(cons_polys),
        "mean_agreement": float(agree[mask == 1].mean()) if (mask == 1).any() else 0.0,
    }
    pd.DataFrame([stats]).to_csv(args.out_dir / f"stats_{args.name}.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.7))
    for a, (title, polys) in zip(ax, [("single threshold", single_polys), ("consensus", cons_polys)]):
        a.imshow(prob[1], cmap="Greens", vmin=0, vmax=1)
        for p in polys:
            if p.geom_type == "Polygon":
                xs, ys = p.exterior.xy
                px = [(x - transform.c) / transform.a for x in xs]
                py = [(y - transform.f) / transform.e for y in ys]
                a.plot(px, py, color="red", lw=0.3)
        a.set_title(f"{args.name}: {title} (n={len(polys)})", fontsize=8)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"fig_{args.name}.png", dpi=140, bbox_inches="tight")
    print(f"{args.name}: single={len(single_polys)} consensus={len(cons_polys)} "
          f"verts {stats['single_vertices']}->{stats['consensus_vertices']} agree={stats['mean_agreement']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
