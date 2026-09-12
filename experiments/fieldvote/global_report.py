"""Aggregate the global shards into method totals, per-country tables, and cost.

Reads every ``out/global/shards/*.parquet`` and writes to ``out/global/``:

* ``summary.csv``      per method: tiles, parcels, area, share under 0.5 ha,
                       ratio of parcels to the global map's conversion and to
                       the released pipeline, median and total cost
* ``by_country.csv``   the same per (country, method), for countries with at
                       least ``--min-tiles`` tiles
* ``area_hist.csv``    parcel counts per area bin and method

Totals are over the tiles every method has finished, so methods that ran on a
subsample (stochastic watershed) are compared on their own tiles only where
noted. Stochastic watershed rows are marked with their sampling fraction.

Regenerate: python -m fieldvote.global_report
"""

import argparse
from pathlib import Path

import pandas as pd

from .paths import out_root


def load(shards: Path) -> pd.DataFrame:
    files = sorted(shards.glob("shard_*.parquet"))
    if not files:
        raise SystemExit(f"no shards under {shards}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    print(f"{len(files)} shards, {df.tile_id.nunique()} tiles, {len(df)} rows")
    return df


def totals(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("method")
    out = pd.DataFrame(
        {
            "tiles": g.tile_id.nunique(),
            "parcels": g.n_parcels.sum(),
            "area_Mha": g.area_ha.sum() / 1e6,
            "share_lt_0p5ha": g.apply(
                lambda x: (x.n_parcels * x.share_lt_0p5ha.fillna(0)).sum() / max(x.n_parcels.sum(), 1)
            ),
            "median_ms_per_tile": g.ms.median(),
            "cpu_hours_total": g.ms.sum() / 3.6e6,
            "peak_mb_p95": g.peak_mb.quantile(0.95),
        }
    )
    # every method on the same tiles as the reference conversion
    common = df.groupby("method").tile_id.apply(set)
    ref_tiles = common.get("global_cc", set())
    for ref in ("global_cc", "released"):
        if ref not in common:
            continue
        r = df[df.method == ref].set_index("tile_id").n_parcels
        ratio = {}
        for m, tiles in common.items():
            sub = df[(df.method == m) & df.tile_id.isin(tiles & set(r.index))]
            ratio[m] = sub.n_parcels.sum() / max(r.loc[sub.tile_id].sum(), 1)
        out[f"parcels_vs_{ref}"] = pd.Series(ratio)
    out["tile_fraction"] = out.tiles / max(len(ref_tiles), 1)
    return out


def by_country(df: pd.DataFrame, min_tiles: int) -> pd.DataFrame:
    g = df.groupby(["country", "method"])
    out = pd.DataFrame(
        {
            "tiles": g.tile_id.nunique(),
            "parcels": g.n_parcels.sum(),
            "area_Mha": g.area_ha.sum() / 1e6,
            "median_ms_per_tile": g.ms.median(),
        }
    ).reset_index()
    keep = out[out.method == "global_cc"].set_index("country").tiles
    out = out[out.country.isin(keep[keep >= min_tiles].index)]
    ref = out[out.method == "global_cc"].set_index("country").parcels
    out["parcels_vs_global_cc"] = out.parcels / out.country.map(ref).to_numpy()
    return out.sort_values(["country", "method"])


def area_hist(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("n_ge_")]
    return df.groupby("method")[cols].sum()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", type=Path, default=out_root() / "global" / "shards")
    p.add_argument("--min-tiles", type=int, default=20)
    args = p.parse_args()
    df = load(args.shards)
    dst = args.shards.parent
    s = totals(df)
    s.to_csv(dst / "summary.csv")
    by_country(df, args.min_tiles).to_csv(dst / "by_country.csv", index=False)
    area_hist(df).to_csv(dst / "area_hist.csv")
    pd.set_option("display.width", 200)
    print(s.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
