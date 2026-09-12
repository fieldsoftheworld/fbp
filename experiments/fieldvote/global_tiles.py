"""List the inner chunks of the global store that hold any field signal.

Reads the 2048x overview (one pixel per inner chunk), keeps chunks whose mean
field probability for the chosen year exceeds ``--min-field``, tags each with
the country its centre falls in (Natural Earth 1:110m), and assigns shards of
``--per-shard`` tiles for the array job.

Writes ``out/global/tiles.parquet`` with one row per tile.

Regenerate: python -m fieldvote.global_tiles --time-index 1
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .global_store import open_variables, pixel_area_m2, tile_center
from .paths import out_root

NE = Path(
    "/projects/bgtj/neuralftw/hf_migration/raw/natural_earth/ne_110m_admin_0_countries.geojson"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--time-index", type=int, default=1, help="0 = 2024, 1 = 2025")
    p.add_argument("--min-field", type=float, default=0.01)
    p.add_argument("--per-shard", type=int, default=300)
    p.add_argument("--natural-earth", type=Path, default=NE)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ov = open_variables("2048x")
    field = np.asarray(ov[args.time_index, 1, :, :])
    rows, cols = np.nonzero(np.isfinite(field) & (field > args.min_field))
    lon, lat = zip(*(tile_center(r, c) for r, c in zip(rows, cols)), strict=True)
    df = pd.DataFrame(
        {
            "row": rows.astype(np.int32),
            "col": cols.astype(np.int32),
            "lon": lon,
            "lat": lat,
            "field_mean": field[rows, cols],
        }
    )
    df["px_m2"] = [pixel_area_m2(la) for la in df.lat]
    df["tile_id"] = np.arange(len(df), dtype=np.int64)

    ne = gpd.read_file(args.natural_earth)[["ADMIN", "ISO_A3", "CONTINENT", "geometry"]]
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    tagged = gpd.sjoin(pts, ne, how="left", predicate="within")
    tagged = tagged[~tagged.index.duplicated(keep="first")]
    df["country"] = tagged["ADMIN"].fillna("(sea/unassigned)").to_numpy()
    df["iso3"] = tagged["ISO_A3"].fillna("---").to_numpy()
    df["continent"] = tagged["CONTINENT"].fillna("(none)").to_numpy()

    # Shards are drawn at random so every array task sees a mix of regions and
    # densities, and the partial result is a global sample at any point.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(df))
    shard = np.empty(len(df), dtype=np.int32)
    shard[order] = np.arange(len(df)) // args.per_shard
    df["shard"] = shard

    dst = out_root() / "global"
    dst.mkdir(parents=True, exist_ok=True)
    f = dst / "tiles.parquet"
    df.to_parquet(f, index=False)
    print(
        f"{len(df)} tiles with field mean > {args.min_field} in {df.shard.max() + 1} "
        f"shards of {args.per_shard}; wrote {f}"
    )
    print(df.groupby("continent").size().sort_values(ascending=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
