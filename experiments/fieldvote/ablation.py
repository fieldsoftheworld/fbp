"""Ablation report for the model-free variants across runs.

For every run: macro PQ/SQ/RQ of each variant (per-instance ``raw`` and
round-tripped ``rt``), the reference rows (``cc``, ``rt:h2``, ``rt:h4``), the
per-country worst change vs ``rt:h2``, the number of countries improved, and
the error-type rates. Then a cross-run table ranking variants by mean PQ gain
so model-agnostic winners stand out.

Example:
    python -m fieldvote.ablation --runs planet_b3 planet_b7 planet_b37 --protocol pixel
    python -m fieldvote.ablation --runs s2_b7 s2_b3 --protocol gsd
"""

import argparse

import pandas as pd

from .paths import DENSE10, scores_dir
from .scoring import load_patches, macro, pool_country

BASE = "rt:h2"
REFROWS = ["cc", "rt:h2", "rt:h4", "vote:t0.5"]


def per_run(
    run: str,
    protocol: str,
    split: str,
    reference: str,
    countries: tuple[str, ...] = DENSE10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-strategy macro table for one run, plus the per-country pool."""
    root = scores_dir(f"{run}/{split}/{protocol}")
    df = load_patches(root, "patches*.parquet")
    df = df[df.reference == reference]
    country = pool_country(df).drop_duplicates(subset=["strategy", "country"])
    mac = macro(country, countries)
    base = country[country.strategy == BASE].set_index("country").pq
    rows = []
    for s, g in country.groupby("strategy"):
        d = g.set_index("country").pq - base
        m = mac[mac.strategy == s].iloc[0]
        rows.append(
            {
                "strategy": s,
                "pq": m.pq,
                "sq": m.sq,
                "rq": m.rq,
                "d_pq": m.pq - mac[mac.strategy == BASE].iloc[0].pq,
                "worst_country_d": d.min(),
                "best_country_d": d.max(),
                "n_improved": int((d > 0).sum()),
                "pq_small": m.pq_small,
                "count_bias": m.count_bias,
                "split": m.split_rate,
                "merge": m.merge_rate,
                "miss": m.miss_rate,
                "spurious": m.spurious_rate,
            }
        )
    tab = pd.DataFrame(rows)
    keep = tab.strategy.isin(REFROWS) | tab.strategy.str.contains(":v:")
    tab = tab[keep].sort_values("pq", ascending=False).reset_index(drop=True)
    return tab, country


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--protocol", default="pixel")
    p.add_argument("--split", default="test")
    p.add_argument("--reference", default="vector")
    p.add_argument("--out", default=None, help="CSV path for the cross-run table")
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    args = p.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)

    tabs = {}
    for run in args.runs:
        tab, _ = per_run(
            run, args.protocol, args.split, args.reference, tuple(args.countries)
        )
        tabs[run] = tab
        print(
            f"\n== {run} / {args.protocol} / reference={args.reference} "
            f"(Δ vs {BASE}, PQ x100) =="
        )
        show = tab.copy()
        for c in (
            "pq",
            "sq",
            "rq",
            "d_pq",
            "worst_country_d",
            "best_country_d",
            "pq_small",
        ):
            show[c] = (show[c] * 100).round(2)
        for c in ("split", "merge", "miss", "spurious"):
            show[c] = show[c].round(3)
        show["count_bias"] = show["count_bias"].round(1)
        print(show.to_string(index=False))

    if len(tabs) > 1:
        cross = None
        for run, tab in tabs.items():
            t = tab.set_index("strategy")[["d_pq", "worst_country_d"]].rename(
                columns={"d_pq": f"d_{run}", "worst_country_d": f"worst_{run}"}
            )
            cross = t if cross is None else cross.join(t, how="inner")
        assert cross is not None
        dcols = [c for c in cross.columns if c.startswith("d_")]
        wcols = [c for c in cross.columns if c.startswith("worst_")]
        cross["mean_d"] = cross[dcols].mean(axis=1)
        cross["min_d"] = cross[dcols].min(axis=1)
        cross["min_worst"] = cross[wcols].min(axis=1)
        cross = cross.sort_values("mean_d", ascending=False)
        print(
            f"\n== cross-run: Δ PQ vs {BASE} (x100); "
            "min_d = worst run, min_worst = worst country in worst run =="
        )
        print((cross * 100).round(2).to_string())
        if args.out:
            cross.to_csv(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
