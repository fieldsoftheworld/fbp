"""Print a macro comparison row for a handful of methods.

Reads ``results/<run>/<split>/<protocol>/macro.csv`` if it exists (the report
stage's output); otherwise pools directly from ``out/scores`` so a comparison
can be printed before ``report`` has run.

    python -m fbp.eval.compare --run planet_b3 --split test --protocol pixel \
        --methods rt:h2 raw:v:nbg-pb-h0.01-t0.3+A900
"""

import argparse
import sys

import pandas as pd

from .paths import DENSE10, results_root, scores_dir
from .scoring import load_patches, macro, pool_country

COLS = [
    "strategy",
    "pq",
    "sq",
    "rq",
    "pq_small",
    "pq_medium",
    "pq_large",
    "miss_rate",
    "merge_rate",
]


def load_macro(
    run: str, split: str, protocol: str, countries: tuple[str, ...]
) -> pd.DataFrame:
    """``macro.csv`` if the report stage already wrote one, else pool it fresh."""
    f = results_root() / run / split / protocol / "macro.csv"
    if f.exists():
        return pd.read_csv(f)
    root = scores_dir(f"{run}/{split}/{protocol}")
    df = load_patches(root)
    df = df[df.reference == "vector"]
    country = pool_country(df)
    return macro(country, countries)


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--protocol", default="pixel")
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    p.add_argument("--methods", nargs="+", required=True)
    args = p.parse_args()

    mac = load_macro(args.run, args.split, args.protocol, tuple(args.countries))
    found = mac[mac.strategy.isin(args.methods)].copy()
    found["strategy"] = pd.Categorical(found.strategy, args.methods, ordered=True)
    missing = sorted(set(args.methods) - set(found.strategy))
    if missing:
        print(f"[compare] not scored: {missing}", file=sys.stderr)
    print(found.sort_values("strategy")[COLS].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
