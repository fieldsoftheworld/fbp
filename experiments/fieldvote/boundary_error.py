"""Boundary error and polygon-count delta, scored with FTP's own definitions.

Two metrics that do not reward a rule for simply cutting more or less:

* **Boundary error (m)** is FTP's symmetric chamfer between the outlines of a
  matched prediction and reference, in pixels, scaled by the patch's ground
  sample distance. It measures how well an outline sits on the reference
  outline, not whether the instance matching accepted it.
* **Polygon count delta** is the per-patch difference between the number of
  predicted and reference polygons, reported both absolutely (FTP's form) and
  relative to the reference count.

Chamfer is defined only on matched pairs, so a rule that matches more fields is
measured over a larger and harder set of pairs than one that matches fewer.
Both are therefore reported twice: over each rule's own matches, and over the
references that every rule compared here matched, which is the like-for-like
number.

Writes ``out/boundary/<run>_<split>.csv``: one row per (variant, country, patch).

Run: python -m fieldvote.boundary_error --run planet_b3 --split test \
        --variants arg-ws2+A900 nbg-sup-h2-t0.7+A900
"""

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import rasterio.features
import shapely.geometry.base

from . import references as refs
from .ftp_compat import symmetric_chamfer
from .matching import match
from .paths import DENSE10, cache_dir, candidates_dir, out_root


def outline_mask(
    shape: tuple[int, int], geom: shapely.geometry.base.BaseGeometry
) -> np.ndarray | None:
    """Boundary pixels of one polygon. Shapes are already in pixel coordinates."""
    filled = rasterio.features.rasterize(
        [(geom, 1)], out_shape=shape, dtype="uint8"
    ).astype(bool)
    if not filled.any():
        return None
    from scipy.ndimage import binary_erosion

    return filled & ~binary_erosion(filled, border_value=0)


def score_country(job: tuple[str, str, str, tuple[str, ...]]) -> list[dict]:
    """Boundary error and count-delta rows for one country, run over a Pool."""
    run, split, country, variants = job
    import geopandas as gpd

    cdir = candidates_dir(f"{run}/{split}") / country
    polys = {}
    for v in variants:
        f = cdir / f"polys_v_{v}.parquet"
        if f.exists():
            polys[v] = (
                gpd.read_parquet(f).groupby("patch_id").geometry.apply(list).to_dict()
            )
    rows = []
    for npz in sorted((cache_dir(f"{run}/{split}") / country).glob("*.npz")):
        grid = refs.load_grid(npz, country)
        gt, _ = refs.vector(grid)
        if not gt:
            continue
        utm = refs.utm_crs(grid)
        px_m = float(np.sqrt(refs.pixel_area_ha(grid, utm) * 1e4)) if utm else np.nan
        gt_masks: dict[int, np.ndarray] = {}
        for v, by_patch in polys.items():
            pred = by_patch.get(grid.patch_id, [])
            m = match(gt, pred, thresholds=(0.5,))
            for i, j, _ in m.pairs_50:
                if i not in gt_masks:
                    gm = outline_mask(grid.shape, gt[i])
                    if gm is None:
                        continue
                    gt_masks[i] = gm
                gm = gt_masks.get(i)
                pm = outline_mask(grid.shape, pred[j])
                if gm is None or pm is None:
                    continue
                c = symmetric_chamfer(pm, gm)
                if c is None:
                    continue
                rows.append(
                    {
                        "variant": v,
                        "country": country,
                        "patch_id": grid.patch_id,
                        "gt_id": i,
                        "chamfer_m": c * px_m,
                    }
                )
            rows.append(
                {
                    "variant": v,
                    "country": country,
                    "patch_id": grid.patch_id,
                    "gt_id": -1,  # patch-level row: counts, no chamfer
                    "n_pred": len(pred),
                    "n_gt": len(gt),
                }
            )
    return rows


def report(df: pd.DataFrame, variants: list[str]) -> None:
    """Print boundary error and count delta per variant, own matches and common."""
    pairs = df[df.gt_id >= 0]
    counts = df[df.gt_id < 0]
    common = None
    for v in variants:
        keys = set(
            map(
                tuple,
                pairs[pairs.variant == v][["country", "patch_id", "gt_id"]].to_numpy(),
            )
        )
        common = keys if common is None else (common & keys)
    print(
        f"{'variant':26s} {'bnd err m':>10s} {'on common':>10s} "
        f"{'|dN|':>7s} {'dN/N':>8s}"
    )
    for v in variants:
        p = pairs[pairs.variant == v]
        own = p.chamfer_m.mean()
        key = list(map(tuple, p[["country", "patch_id", "gt_id"]].to_numpy()))
        on_common = p.chamfer_m[[k in common for k in key]].mean()
        c = counts[counts.variant == v]
        d_abs = (c.n_pred - c.n_gt).abs().mean()
        d_rel = ((c.n_pred - c.n_gt).abs() / c.n_gt.clip(lower=1)).mean()
        print(f"{v:26s} {own:10.2f} {on_common:10.2f} {d_abs:7.1f} {d_rel:8.3f}")
    print(f"\npairs matched by every variant: {len(common)}")


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="planet_b3")
    p.add_argument("--split", default="test")
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    jobs = [(args.run, args.split, c, tuple(args.variants)) for c in args.countries]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_country, jobs) for r in part]
    df = pd.DataFrame(rows)
    dst = out_root() / "boundary"
    dst.mkdir(parents=True, exist_ok=True)
    f = dst / f"{args.run}_{args.split}.csv"
    df.to_csv(f, index=False)
    print(f"wrote {f}\n")
    report(df, args.variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
