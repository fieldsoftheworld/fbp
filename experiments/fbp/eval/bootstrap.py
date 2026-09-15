"""Patch-level bootstrap of the macro PQ gain over the FTP pipeline.

Patches are resampled with replacement within each country, PQ is pooled per
country the way ``scoring.pool_country`` does it, and the ten countries are
averaged. The official splits stay as they are, so every score remains
comparable with the published ones. Writes ``out/bootstrap/<run>.csv`` with a
95% interval for each method's PQ and for the difference between them.

    python -m fbp.eval.bootstrap --run planet_b3
"""

import argparse

import numpy as np
import pandas as pd

from fbp.paper.shipped import FTP_MIN, OURS

from .paths import DENSE10, out_root, scores_dir
from .scoring import load_patches

COLS = ["tp_50", "fp_50", "fn_50", "iou_sum_50"]


def pq(sums: np.ndarray) -> np.ndarray:
    """PQ from summed (tp, fp, fn, iou) along the last axis, zero when undefined."""
    tp, fp, fn, iou = np.moveaxis(sums, -1, 0)
    denom = tp + 0.5 * (fp + fn)
    with np.errstate(invalid="ignore", divide="ignore"):
        rq = np.where(denom > 0, tp / denom, 0.0)
        sq = np.where(tp > 0, iou / tp, 0.0)
    return sq * rq


def country_arrays(
    df: pd.DataFrame, strategies: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Per country, a ``(n_patches, n_strategies, 4)`` array aligned on patch id."""
    out = {}
    for country, g in df.groupby("country"):
        wide = g.pivot(index="patch_id", columns="strategy", values=COLS)
        arr = np.stack([wide[c][list(strategies)].to_numpy() for c in COLS], axis=-1)
        out[str(country)] = arr.astype(np.float64)
    return out


def draw_macro(
    arrays: dict[str, np.ndarray], countries: tuple[str, ...], draws: int, seed: int
) -> np.ndarray:
    """``(draws + 1, n_strategies)`` macro PQ; row 0 is the unresampled estimate."""
    rng = np.random.default_rng(seed)
    per = []
    for c in countries:
        a = arrays[c]
        n = len(a)
        idx = rng.integers(0, n, size=(draws, n))
        sums = np.concatenate([a.sum(axis=0)[None], a[idx].sum(axis=1)], axis=0)
        per.append(pq(sums))
    return np.mean(per, axis=0)


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", default="planet_b3")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    args = p.parse_args()
    countries = tuple(args.countries)

    df = load_patches(scores_dir(f"{args.run}/test/pixel"), "patches*.parquet")
    df = df[(df.reference == "vector") & df.strategy.isin((FTP_MIN, OURS))]
    df = df.drop_duplicates(subset=["country", "patch_id", "strategy"], keep="last")
    strategies = (FTP_MIN, OURS)
    macro = 100 * draw_macro(
        country_arrays(df, strategies), countries, args.draws, args.seed
    )
    point, boot = macro[0], macro[1:]
    gain = boot[:, 1] - boot[:, 0]
    rows = [
        {
            "run": args.run,
            "quantity": name,
            "point": pt,
            "lo": float(np.percentile(b, 2.5)),
            "hi": float(np.percentile(b, 97.5)),
            "frac_positive": float((b > 0).mean()),
            "draws": args.draws,
            "seed": args.seed,
        }
        for name, pt, b in (
            ("ftp_min_size", point[0], boot[:, 0]),
            ("ours", point[1], boot[:, 1]),
            ("gain", point[1] - point[0], gain),
        )
    ]
    out = pd.DataFrame(rows)
    dst = out_root() / "bootstrap"
    dst.mkdir(parents=True, exist_ok=True)
    f = dst / f"{args.run}.csv"
    out.to_csv(f, index=False)
    print(out.round(2).to_string(index=False))
    print(f"\nwrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
