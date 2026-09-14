"""Stage 3 (CPU): score every strategy x reference per patch -> long-form parquet.

For each cached patch the candidate polygon sets (stage 2) are combined by every
strategy in ``build_strategies`` and matched against each requested reference
with FTP's greedy rule. One row per (patch, reference, strategy) holding the
raw tallies needed to pool PQ/SQ/RQ/F1 at any level, per-size-bin tallies, and
split/merge/miss/spurious diagnostics. Nothing is averaged here; see
``scoring.py``.

Protocols:
* ``pixel``  match in the eval-grid pixel frame (FTP Table 1 Planet rows)
* ``gsd``    FTP native-GSD mode: predictions rendered at ``--gsd-m`` in UTM,
             matched against UTM parcels (FTP Table 1 S2 rows; use 10 for S2)

Example:
    python -m fieldvote.run_eval --run planet_b3 --split test --workers 16
    python -m fieldvote.run_eval --run s2_b7 --split test --protocol gsd --gsd-m 10
"""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import ftp_compat as ftp
from . import references as refs
from . import strategies as S
from .candidates import file_to_tag
from .matching import THRESHOLDS, diagnostics, match
from .paths import DENSE10, H_SWEEP, cache_dir, candidates_dir, h_tag, scores_dir

BINS = ("small", "medium", "large")
AREA_EDGES = (0.5, 2.0)  # ha, FTP's --area-bins 0.5,2

SUBSETS: dict[str, tuple[float, ...]] = {
    "k6": (
        2.0,
        4.0,
        8.0,
        12.0,
        16.0,
        24.0,
    ),  # legacy 6-depth sweep (02_consensus, 05_global)
    "k4": (2.0, 4.0, 8.0, 12.0),  # legacy figure export set
    "k3": (1.0, 4.0, 16.0),
    "lo7": H_SWEEP[:7],
    "hi7": H_SWEEP[7:],
    "odd": H_SWEEP[0::2],
    "even": H_SWEEP[1::2],
    "uniform7": (0.5, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0),
}

# Module globals populated before forking the worker pool.
_CANDS: dict[str, dict[str, list]] = {}
_OPTS: dict = {}


def build_strategies(
    cand: dict[str, list], shape: tuple[int, int], opts: dict
) -> dict[str, list]:
    """Return {strategy_name: polygons}. Order of ``H_SWEEP`` defines K."""
    out: dict[str, list] = {}
    if "variants" in opts["sets"]:
        wanted = opts.get("variant_tags")
        for tag in sorted(t for t in cand if t.startswith("v:")):
            if wanted is not None and tag not in wanted:
                continue
            out[f"raw:{tag}"] = cand[tag]
            out[f"rt:{tag}"] = S.roundtrip(cand[tag], shape)
    if "base" not in opts["sets"]:
        return out
    sweep = [cand[h_tag(h)] for h in H_SWEEP]
    out["cc"] = cand["cc"]
    for h in H_SWEEP:
        tag = h_tag(h)
        out[f"raw:{tag}"] = cand[tag]
        out[f"rt:{tag}"] = S.roundtrip(cand[tag], shape)
    for tau in opts["taus"]:
        out[f"vote:t{tau:g}"] = S.vote(sweep, shape, tau)
    for name, hs in SUBSETS.items():
        out[f"vote:t0.5:{name}"] = S.vote([cand[h_tag(h)] for h in hs], shape, 0.5)
    if opts["drop_one"]:
        for h in H_SWEEP:
            out[f"vote:t0.5:drop{h_tag(h)}"] = S.vote(
                [cand[h_tag(g)] for g in H_SWEEP if g != h], shape, 0.5
            )
    out["nms"] = S.nms_pool(sweep, 0.5)
    out["elbow"] = S.elbow(sweep)
    return out


def area_bin_of(area_ha: float) -> str:
    return ftp.area_bin(area_ha, AREA_EDGES)


def tally(
    gt: list, gt_areas: list[float], pred: list, pred_area_ha: float | None
) -> dict:
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
    country, pid, npz = job
    grid = refs.load_grid(Path(npz), country)
    cand = _CANDS[pid]
    opts = _OPTS
    t0 = time.perf_counter()
    strategies = build_strategies(cand, grid.shape, opts)
    t_build = time.perf_counter() - t0

    utm = refs.utm_crs(grid)
    px_ha = refs.pixel_area_ha(grid, utm)
    gts: dict[str, tuple[list, list[float]]] = {}
    if "vector" in opts["references"]:
        if opts["protocol"] == "gsd":
            g, a, _ = refs.vector_utm(grid)
        else:
            g, a = refs.vector(grid)
        gts["vector"] = (g, a)
    if "mask" in opts["references"]:
        g = refs.mask(grid)
        gts["mask"] = (g, [p.area * px_ha for p in g])
    if "instance" in opts["references"]:
        g = refs.instance(grid)
        gts["instance"] = (g, [p.area * px_ha for p in g])

    rows = []
    for sname, pred in strategies.items():
        for rname, (gt, areas) in gts.items():
            if rname == "vector" and opts["protocol"] == "gsd":
                pred_s = ftp.cap_pred_to_gsd(
                    pred, grid.transform, grid.crs, utm, opts["gsd_m"]
                )
                row = tally(gt, areas, pred_s, None)
            else:
                row = tally(gt, areas, pred, px_ha)
            row.update(
                country=country,
                patch_id=pid,
                reference=rname,
                strategy=sname,
                px_ha=px_ha,
                t_build=t_build,
            )
            rows.append(row)
    return rows


def load_candidates(run: str, split: str, country: str) -> dict[str, dict[str, list]]:
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


def run_country(args, country: str) -> pd.DataFrame:
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
        "taus": tuple(args.taus),
        "drop_one": args.drop_one,
        "references": tuple(args.references),
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
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    p.add_argument(
        "--references", nargs="*", default=["vector", "mask"], choices=refs.REFERENCES
    )
    p.add_argument("--protocol", default="pixel", choices=["pixel", "gsd"])
    p.add_argument("--gsd-m", type=float, default=3.0)
    p.add_argument("--taus", nargs="*", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument(
        "--drop-one", action="store_true", help="add leave-one-depth-out votes"
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--sets",
        nargs="*",
        default=["base"],
        choices=["base", "variants"],
        help="base = depth sweep/vote/nms/elbow; variants = fieldvote.variants tags",
    )
    p.add_argument("--tag", default="", help="output file infix, e.g. 'variants'")
    p.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="score only these variant names (default: every v: tag on disk)",
    )
    args = p.parse_args()

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
            f"{c}: {n_pat} patches x {n_str} strategies x {df.reference.nunique()} refs in {time.time() - t0:.0f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
