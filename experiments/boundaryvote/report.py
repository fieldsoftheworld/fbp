"""Stage 4: pool scores, add the test-selected depth, gate against FTP, print tables.

Writes under ``out/scores/<run>/<split>/<protocol>/``:

* ``country.csv``  every (strategy, country), plus ``rt_test``
* ``macro.csv``    dense-10 macro for every strategy
* ``ftp_gate.csv`` per-country comparison of ``rt:h2`` (== FTP's ``inst>0``
                   components) against FTP's published per-country CSV

``rt_test`` is the FTP pipeline at the watershed depth that scores best on each
country's own test labels, the depth-tuning ceiling the letter quotes.

Example:
    python -m boundaryvote.report --run planet_b3 --split test
"""

import argparse

import pandas as pd

from .paths import DENSE10, ftw_planet_repo, scores_dir
from .scoring import load_patches, macro, pool_country, select_depth

# Published per-country numbers exist under one scoring protocol per sensor:
# PlanetScope in the prediction (pixel) frame, Sentinel-2 capped to 10 m.
FTP_CONDITION = {
    "planet_b3": ("planet3m", "pixel"),
    "planet_b7": ("planetb7_3m", "pixel"),
    "s2_b7": ("s2nat10", "gsd"),
    "s2_b3": ("s2b3_10m", "gsd"),
}

HEADLINE = ["cc", "raw:h2", "rt:h2", "rt_test"]


def ftp_gate(country: pd.DataFrame, run: str, protocol: str) -> pd.DataFrame | None:
    """Per-country ``rt:h2`` vs FTP's published CSV; None if this run has no gate."""
    entry = FTP_CONDITION.get(run)
    if entry is None or entry[1] != protocol:
        return None
    cond = entry[0]
    d = ftw_planet_repo() / "logs" / "resolution_ablation" / cond
    ours = country[
        (country.reference == "vector") & (country.strategy == "rt:h2")
    ].set_index("country")
    rows = []
    for c in ours.index:
        f = d / f"{c}.csv"
        if not f.exists():
            continue
        ref = pd.read_csv(f).iloc[-1]
        o = ours.loc[c]
        rows.append(
            {
                "country": c,
                "n_ours": int(o.n_patches),
                "n_ftp": int(ref.n_patches),
                "pq_ours": o.pq,
                "pq_ftp": ref.pq,
                "d_pq": o.pq - ref.pq,
                "sq_ours": o.sq,
                "sq_ftp": ref.pq_sq,
                "rq_ours": o.rq,
                "rq_ftp": ref.pq_rq,
                "n_pred_mean_ours": o.n_pred / o.n_patches,
                "n_pred_mean_ftp": ref.n_pred_mean,
                "n_gt_mean_ours": o.n_gt / o.n_patches,
                "n_gt_mean_ftp": ref.n_gt_mean,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--protocol", default="pixel")
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    args = p.parse_args()
    countries = tuple(args.countries)

    root = scores_dir(f"{args.run}/{args.split}/{args.protocol}")
    country = pool_country(load_patches(root))
    if country.strategy.str.startswith("rt:h").any():
        country = pd.concat(
            [country, select_depth(country, country, "rt", "rt_test")],
            ignore_index=True,
        )
    country.to_csv(root / "country.csv", index=False)
    # A strategy scored for only some countries, because a job is still running or
    # was run on a subset, cannot be macro-averaged; drop it and say so.
    have = country.groupby(["reference", "strategy"]).country.nunique()
    partial = have[have < len(countries)]
    if len(partial):
        print(
            f"[report] skipping {len(partial)} strategy/reference pairs "
            "with partial country coverage"
        )
        keep = country.set_index(["reference", "strategy"]).index.isin(
            have[have == len(countries)].index
        )
        country = country[keep]
    mac = macro(country, countries)
    mac.to_csv(root / "macro.csv", index=False)

    # FTP publishes test-split numbers only
    gate = ftp_gate(country, args.run, args.protocol) if args.split == "test" else None
    if gate is not None and len(gate):
        gate.to_csv(root / "ftp_gate.csv", index=False)
        pd.set_option("display.width", 200)
        print(
            "\n== FTP reproduction gate (reference=vector, strategy=rt:h2 "
            "vs FTP per-country CSV) =="
        )
        print(gate.round(4).to_string(index=False))
        print(
            f"macro PQ ours {gate.pq_ours.mean():.4f} vs FTP {gate.pq_ftp.mean():.4f}"
        )

    print(f"\n== {args.run}/{args.split}/{args.protocol}: dense-10 macro ==")
    cols = [
        "strategy",
        "pq",
        "sq",
        "rq",
        "f1_5_95",
        "pq_small",
        "pq_medium",
        "pq_large",
        "count_mae",
        "count_bias",
        "split_rate",
        "merge_rate",
        "miss_rate",
        "spurious_rate",
        "n_scored",
    ]
    show = mac[mac.strategy.isin(HEADLINE)].copy()
    show["strategy"] = pd.Categorical(show.strategy, HEADLINE, ordered=True)
    print(show.sort_values("strategy")[cols].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
