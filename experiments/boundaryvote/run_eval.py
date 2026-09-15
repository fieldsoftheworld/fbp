"""Stage 3 (CPU): score every strategy per patch -> long-form parquet.

For each cached patch the candidate polygon sets (stage 2) are matched against
the reference parcels with FTP's greedy rule. One row per (patch, strategy)
holding the raw tallies needed to pool PQ/SQ/RQ/F1 at any level, per-size-bin
tallies, and split/merge/miss/spurious diagnostics. Nothing is averaged here;
see ``scoring.py``.

Strategies:
* ``cc``            connected components of the field class
* ``raw:h{h}``      per-instance watershed polygons at depth ``h``
* ``rt:h{h}``       the same after rasterize -> polygonize, which is what FTP
                    scores (``rt:h2`` is the FTP pipeline)
* ``raw:v:<name>``  a ``variants.run_variant`` result (``rt:v:`` likewise)

Protocols:
* ``pixel``  match in the eval-grid pixel frame (FTP Table 1 Planet rows)
* ``gsd``    FTP native-GSD mode: predictions rendered at ``--gsd-m`` in UTM,
             matched against UTM parcels (FTP Table 1 S2 rows; use 10 for S2)

Example:
    python -m boundaryvote.run_eval --run planet_b3 --split test --workers 16
    python -m boundaryvote.run_eval --run planet_b3 --split test --workers 16 \
        --sets variants --tag sweep --grids sweep
"""

import argparse
import time
from argparse import Namespace
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import ftp_compat as ftp
from . import references as refs
from . import strategies as s
from .candidates import file_to_tag
from .grids import names as grid_names
from .matching import THRESHOLDS, diagnostics, match
from .paths import DENSE10, H_SWEEP, cache_dir, candidates_dir, h_tag, scores_dir

BINS = ("small", "medium", "large")
AREA_EDGES = (0.5, 2.0)  # ha, FTP's --area-bins 0.5,2

# Module globals populated before forking the worker pool.
_CANDS: dict[str, dict[str, list]] = {}
_OPTS: dict = {}


def build_strategies(
    cand: dict[str, list], shape: tuple[int, int], opts: dict
) -> dict[str, list]:
    """Return {strategy_name: polygons} for the requested strategy sets."""
    out: dict[str, list] = {}
    if "variants" in opts["sets"]:
        wanted = opts.get("variant_tags")
        for tag in sorted(t for t in cand if t.startswith("v:")):
            if wanted is not None and tag not in wanted:
                continue
            out[f"raw:{tag}"] = cand[tag]
            out[f"rt:{tag}"] = s.roundtrip(cand[tag], shape)
    if "base" not in opts["sets"]:
        return out
    out["cc"] = cand["cc"]
    for h in H_SWEEP:
        tag = h_tag(h)
        out[f"raw:{tag}"] = cand[tag]
        out[f"rt:{tag}"] = s.roundtrip(cand[tag], shape)
    return out


def area_bin_of(area_ha: float) -> str:
    """FTP's small/medium/large bin for one parcel area."""
    return ftp.area_bin(area_ha, AREA_EDGES)


def tally(
    gt: list, gt_areas: list[float], pred: list, pred_area_ha: float | None
) -> dict:
    """Raw TP/FP/FN and diagnostic counts for one (gt, pred) polygon set."""
    m = match(gt, pred)
    row: dict = {"n_gt": m.n_gt, "n_pred": m.n_pred}
    for t in THRESHOLDS:
        k = f"{int(round(t * 100)):02d}"
        row[f"tp_{k}"] = m.tp[t]
        row[f"fp_{k}"] = m.fp[t]
        row[f"fn_{k}"] = m.fn[t]
    row["iou_sum_50"] = float(sum(iou for _, _, iou in m.pairs_50))
    row.update(diagnostics(gt, pred))
    # size bins at IoU 0.5
    gt_bins = [area_bin_of(a) for a in gt_areas]
    if pred_area_ha is None:  # UTM m^2 -> ha
        pred_bins = [area_bin_of(p.area * 1e-4) for p in pred]
    else:
        pred_bins = [area_bin_of(p.area * pred_area_ha) for p in pred]
    matched_gt = {i: iou for i, _, iou in m.pairs_50}
    matched_pred = {j for _, j, _ in m.pairs_50}
    for b in BINS:
        row[f"tp_50_{b}"] = row[f"fp_50_{b}"] = row[f"fn_50_{b}"] = 0
        row[f"iou_sum_50_{b}"] = 0.0
    for i, b in enumerate(gt_bins):
        if i in matched_gt:
            row[f"tp_50_{b}"] += 1
            row[f"iou_sum_50_{b}"] += matched_gt[i]
        else:
            row[f"fn_50_{b}"] += 1
    for j, b in enumerate(pred_bins):
        if j not in matched_pred:
            row[f"fp_50_{b}"] += 1
    return row


def score_patch(job: tuple[str, str, str]) -> list[dict]:
    """Every strategy row for one cached patch, against the vector reference."""
    country, pid, npz = job
    grid = refs.load_grid(Path(npz), country)
    cand = _CANDS[pid]
    opts = _OPTS
    t0 = time.perf_counter()
    strategies = build_strategies(cand, grid.shape, opts)
    t_build = time.perf_counter() - t0

    utm = refs.utm_crs(grid)
    px_ha = refs.pixel_area_ha(grid, utm)
    if opts["protocol"] == "gsd":
        gt, areas, _ = refs.vector_utm(grid)
    else:
        gt, areas = refs.vector(grid)

    rows = []
    for sname, pred in strategies.items():
        if opts["protocol"] == "gsd":
            pred_s = ftp.cap_pred_to_gsd(
                pred, grid.transform, grid.crs, utm, opts["gsd_m"]
            )
            row = tally(gt, areas, pred_s, None)
        else:
            row = tally(gt, areas, pred, px_ha)
        row.update(
            country=country,
            patch_id=pid,
            reference="vector",
            strategy=sname,
            px_ha=px_ha,
            t_build=t_build,
        )
        rows.append(row)
    return rows


def load_candidates(run: str, split: str, country: str) -> dict[str, dict[str, list]]:
    """{patch_id: {tag: [polygons]}} from every ``polys_*.parquet`` on disk."""
    cdir = candidates_dir(f"{run}/{split}") / country
    files = sorted(cdir.glob("polys_*.parquet"))
    tags = [file_to_tag(f.name) for f in files]
    out: dict[str, dict[str, list]] = {}
    for tag, f in zip(tags, files, strict=True):
        gdf = gpd.read_parquet(f)
        grouped = gdf.groupby("patch_id").geometry.apply(list).to_dict()
        for pid in grouped:
            out.setdefault(pid, {})
        for pid, d in out.items():
            d[tag] = grouped.get(pid, [])
    # a patch may first appear in a later file (e.g. empty under `arg`, non-empty
    # under `nbg`); every patch must carry every tag, empty lists included
    for d in out.values():
        for tag in tags:
            d.setdefault(tag, [])
    return out


def run_country(args: Namespace, country: str) -> pd.DataFrame:
    """Score every patch of one country; workers share ``_CANDS``/``_OPTS`` by fork."""
    global _CANDS, _OPTS
    src = cache_dir(f"{args.run}/{args.split}") / country
    npzs = sorted(src.glob("*.npz"))
    cands = load_candidates(args.run, args.split, country)
    all_tags = sorted({t for d in cands.values() for t in d})
    for p in npzs:
        cands.setdefault(p.stem, {t: [] for t in all_tags})
    _CANDS = cands
    _OPTS = {
        "sets": tuple(args.sets),
        "variant_tags": (
            {f"v:{v}" for v in args.variants} if args.variants is not None else None
        ),
        "protocol": args.protocol,
        "gsd_m": args.gsd_m,
    }
    jobs = [(country, p.stem, str(p)) for p in npzs]
    rows: list[dict] = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(score_patch, jobs, chunksize=2):
                rows.extend(r)
    else:
        for j in jobs:
            rows.extend(score_patch(j))
    return pd.DataFrame(rows)


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    p.add_argument("--protocol", default="pixel", choices=["pixel", "gsd"])
    p.add_argument("--gsd-m", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--sets",
        nargs="*",
        default=["base"],
        choices=["base", "variants"],
        help="base = cc and the depth sweep; variants = boundaryvote.variants tags",
    )
    p.add_argument("--tag", default="", help="output file infix, e.g. 'sweep'")
    p.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="score only these variant names (default: every v: tag on disk)",
    )
    p.add_argument(
        "--grids", nargs="*", default=(), help="variant groups from boundaryvote.grids"
    )
    args = p.parse_args()
    if args.grids:
        args.variants = [*(args.variants or []), *grid_names(args.grids)]

    out = scores_dir(f"{args.run}/{args.split}/{args.protocol}")
    out.mkdir(parents=True, exist_ok=True)
    infix = f"{args.tag}_" if args.tag else ""
    for c in args.countries:
        t0 = time.time()
        df = run_country(args, c)
        df.insert(0, "run", args.run)
        df.insert(1, "split", args.split)
        df.insert(2, "protocol", args.protocol)
        df.to_parquet(out / f"patches_{infix}{c}.parquet", index=False)
        n_pat = df.patch_id.nunique()
        n_str = df.strategy.nunique()
        print(
            f"{c}: {n_pat} patches x {n_str} strategies in {time.time() - t0:.0f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
