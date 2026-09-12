"""Run every post-processing method on one shard of global tiles.

For each tile the 3-band softmax is read once, then each method in
``--methods`` is timed (one call, wall clock) with peak resident memory sampled
in a background thread, and its parcels are summarised: count, total area, and
a histogram of parcel area in hectares. No polygons are written; the point is
the difference between conversions at scale and their cost, not a product.

Methods are the strategy names of the letter with the shared 900 m^2 step,
plus ``global_cc``, the conversion of the released global map (components of
the field argmax, at least 4 pixels). Stochastic watershed is ten times slower
than the rest, so it runs on every ``--swst-every``-th tile of the shard.

Writes ``out/global/shards/shard_<k>.parquet``, one row per (tile, method).
Existing shard files are skipped, so a resubmitted array resumes.

Run one shard: python -m fieldvote.global_run --shard 0
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .global_store import YEARS, open_variables, read_window, tile_window
from .paths import out_root
from .profile_methods import RSSSampler
from .variants import foreground, run_variant

# label -> strategy; settings are the validation choices behind Table II
METHODS = {
    "global_cc": "arg-none+a4",
    "released": "arg-ws2+A900",
    "thr": "arg-thr-te0.5-tb0.7+A900",
    "ucm": "nbg-ucm-k0.7+A900",
    "iterh": "arg-iterh-a50000+A900",
    "swst": "nbg-swst-n80-m30-t0.3+A900",
    "ours": "nbg-sup-h2-t0.7+A900",
}
# parcel-area histogram edges in hectares (last bin open)
AREA_EDGES_HA = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, np.inf)


SUB = 512  # sub-tile side in pixels, the letter's Sentinel-2 patch size


def summarise(inst: np.ndarray, px_m2: float) -> dict:
    sizes = np.bincount(inst.ravel())[1:]
    sizes = sizes[sizes > 0]
    ha = sizes * (px_m2 / 1e4)
    hist, _ = np.histogram(ha, bins=AREA_EDGES_HA)
    out = {
        "n_parcels": int(len(sizes)),
        "area_ha": float(ha.sum()),
        "share_lt_0p5ha": float((ha < 0.5).mean()) if len(ha) else 0.0,
    }
    for lo, n in zip(AREA_EDGES_HA[:-1], hist, strict=True):
        out[f"n_ge_{lo:g}ha"] = int(n)
    return out


def _add(acc: dict, part: dict) -> None:
    for k, val in part.items():
        acc[k] = acc.get(k, 0) + val


def run_methods(prob: np.ndarray, seg: np.ndarray, px_m2: float, methods: list[str]) -> dict:
    """Each method on the 512 x 512 sub-tiles of one chunk, the letter's patch size.

    The prior methods' validated settings (marker counts, pixel-area thresholds)
    mean what they meant in Table II only at that size. Counts and histograms
    are summed over sub-tiles, wall time is summed, and peak memory is the
    maximum over sub-tiles.
    """
    n = prob.shape[1] // SUB
    subs = [
        (
            np.ascontiguousarray(prob[:, i * SUB : (i + 1) * SUB, j * SUB : (j + 1) * SUB]),
            np.ascontiguousarray(seg[i * SUB : (i + 1) * SUB, j * SUB : (j + 1) * SUB]),
        )
        for i in range(n)
        for j in range(n)
    ]
    out: dict[str, dict] = {}
    for m in methods:
        acc: dict = {"ms": 0.0, "n_lt_0p5": 0.0}
        # one memory sampler per method and tile; starting a thread and
        # collecting garbage per sub-tile would cost more than the methods
        with RSSSampler(hz=200) as smp:
            for p, s in subs:
                t0 = time.perf_counter()
                inst = run_variant(METHODS[m], s, p, px_m2=px_m2)
                acc["ms"] += 1000 * (time.perf_counter() - t0)
                summ = summarise(inst, px_m2)
                acc["n_lt_0p5"] += summ["n_parcels"] * summ["share_lt_0p5ha"]
                _add(acc, {k: v for k, v in summ.items() if k.startswith("n_") or k == "area_ha"})
        acc["peak_mb"] = smp.mb
        acc["share_lt_0p5ha"] = acc.pop("n_lt_0p5") / max(acc.get("n_parcels", 0), 1)
        acc["n_subtiles"] = n * n
        out[m] = acc
    return out


def run_tile(v, tile, t: int, methods: list[str], do_swst: bool) -> list[dict]:
    y0, y1, x0, x1 = tile_window(int(tile.row), int(tile.col))
    t0 = time.perf_counter()
    prob = read_window(v, t, y0, y1, x0, x1)
    read_s = time.perf_counter() - t0
    seg = prob.argmax(axis=0).astype(np.uint8)
    base = {
        "tile_id": int(tile.tile_id),
        "row": int(tile.row),
        "col": int(tile.col),
        "lon": float(tile.lon),
        "lat": float(tile.lat),
        "country": tile.country,
        "iso3": tile.iso3,
        "continent": tile.continent,
        "year": YEARS[t],
        "px_m2": float(tile.px_m2),
        "px": int(seg.size),
        "n_fg_field": int((seg == 1).sum()),
        "n_fg_nbg": int(foreground(seg, prob, "nbg").sum()),
        "read_s": read_s,
    }
    todo = [m for m in methods if m != "swst" or do_swst]
    res = run_methods(prob, seg, float(tile.px_m2), todo)
    return [{**base, "method": m, "variant": METHODS[m]} | res[m] for m in todo]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--tiles", type=Path, default=out_root() / "global" / "tiles.parquet")
    p.add_argument("--time-index", type=int, default=1)
    p.add_argument("--methods", nargs="*", default=list(METHODS))
    p.add_argument("--swst-every", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="tiles per shard (smoke tests)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    tiles = pd.read_parquet(args.tiles)
    tiles = tiles[tiles.shard == args.shard].sort_values("tile_id")
    if args.limit:
        tiles = tiles.head(args.limit)
    dst = args.out or out_root() / "global" / "shards" / f"shard_{args.shard:05d}.parquet"
    if dst.exists():
        print(f"{dst} exists, nothing to do")
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)

    v = open_variables()
    rows = []
    t_start = time.time()
    for i, tile in enumerate(tiles.itertuples(index=False)):
        try:
            rows += run_tile(v, tile, args.time_index, args.methods, i % args.swst_every == 0)
        except Exception as e:  # one bad tile must not sink the shard
            print(f"tile {tile.tile_id} ({tile.row},{tile.col}) failed: {e!r}", flush=True)
            continue
        if (i + 1) % 10 == 0:
            print(
                f"{i + 1}/{len(tiles)} tiles, {time.time() - t_start:.0f}s elapsed",
                flush=True,
            )
    df = pd.DataFrame(rows)
    tmp = dst.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.rename(dst)
    print(f"wrote {dst}: {len(df)} rows from {len(tiles)} tiles in {time.time() - t_start:.0f}s")
    if len(df):
        print(df.groupby("method")[["ms", "peak_mb", "n_parcels"]].median().round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
