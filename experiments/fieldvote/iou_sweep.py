"""Recognition quality across IoU matching thresholds, including below 0.5.

The scored tables stop at 0.5 because that is the threshold the letter reports.
A split that halves a reference field gives each piece an IoU near 0.5, so a
rule that adds splits can be punished by the threshold itself rather than by
the quality of its polygons. This scores a few variants from 0.2 upward to see
whether the ranking between them depends on where the threshold sits.

Writes ``out/iou_sweep/<run>_<split>.csv``: one row per (variant, country,
threshold) with matched, predicted, and reference counts.

Run: python -m fieldvote.iou_sweep --run planet_b3 --split val \
        --variants nbg-sup-h2-t0.7 nbg-sup-h2-t0.7+H40
"""

import argparse
from multiprocessing import Pool

import pandas as pd

from . import references as refs
from .matching import match
from .paths import DENSE10, cache_dir, candidates_dir, out_root

THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7)


def score_country(job: tuple[str, str, str, tuple[str, ...]]) -> list[dict]:
    """TP/FP/FN rows for one country, every variant x IoU threshold."""
    run, split, country, variants = job
    cdir = candidates_dir(f"{run}/{split}") / country
    polys = {}
    for v in variants:
        f = cdir / f"polys_v_{v}.parquet"
        if not f.exists():
            continue
        import geopandas as gpd

        gdf = gpd.read_parquet(f)
        polys[v] = gdf.groupby("patch_id").geometry.apply(list).to_dict()
    rows = []
    for npz in sorted((cache_dir(f"{run}/{split}") / country).glob("*.npz")):
        grid = refs.load_grid(npz, country)
        gt, _ = refs.vector(grid)
        if not gt:
            continue
        for v, by_patch in polys.items():
            pred = by_patch.get(grid.patch_id, [])
            m = match(gt, pred, thresholds=THRESHOLDS)
            for t in THRESHOLDS:
                rows.append(
                    {
                        "variant": v,
                        "country": country,
                        "patch_id": grid.patch_id,
                        "threshold": t,
                        "tp": m.tp[t],
                        "fp": m.fp[t],
                        "fn": m.fn[t],
                    }
                )
    return rows


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="planet_b3")
    p.add_argument("--split", default="val")
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    jobs = [(args.run, args.split, c, tuple(args.variants)) for c in args.countries]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_country, jobs) for r in part]
    df = pd.DataFrame(rows)
    dst = out_root() / "iou_sweep"
    dst.mkdir(parents=True, exist_ok=True)
    f = dst / f"{args.run}_{args.split}.csv"
    df.to_csv(f, index=False)

    # macro recognition quality: pool patches within a country, weight countries equally
    g = df.groupby(["variant", "threshold", "country"])[["tp", "fp", "fn"]].sum()
    prec = g.tp / (g.tp + g.fp).clip(lower=1)
    rec = g.tp / (g.tp + g.fn).clip(lower=1)
    f1 = (2 * prec * rec / (prec + rec).where((prec + rec) > 0, 1)).fillna(0)
    rq = (f1.groupby(["variant", "threshold"]).mean() * 100).unstack()
    print(f"wrote {f}\n")
    print("macro RQ by IoU matching threshold")
    print(rq.round(2).to_string())
    if len(rq) > 1:
        base = rq.loc[args.variants[0]]
        print("\ngap against", args.variants[0])
        print((rq - base).drop(index=args.variants[0]).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
