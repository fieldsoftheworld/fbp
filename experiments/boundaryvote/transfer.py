"""Does a method's setting transfer across countries?

For each family of settings, compare three ways of choosing one:

* ``val``    one setting for every country, the best on the validation split
* ``test``   one setting for every country, the best on the test split
* ``per``    the best setting for each country, chosen on that country's test
             split (not deployable; it needs local reference polygons)

``per - val`` is what per-region tuning would buy over shipping one setting, and
``n_settings`` is how many distinct settings win across the ten countries. A
method whose gap is large, or whose winning setting changes from country to
country, needs local labels to reach the accuracy its best row reports.

Writes ``out/transfer/<run>.csv``.

Regenerate: python -m boundaryvote.transfer --run planet_b3
"""

import argparse
import re

import pandas as pd

from .paths import DENSE10, out_root, scores_dir
from .scoring import load_patches, macro, pool_country

# Each family is the grid of that method's own parameters, before the shared
# minimum-size step; settings carrying that step are excluded so the comparison
# is between parameter choices and not between post-steps.
FAMILIES = {
    "ours": r"^raw:v:nbg-sup-h[\d.]+-t[\d.]+$",
    "ucm": r"^raw:v:nbg-ucm-k[\d.]+$",
    "swst": r"^raw:v:nbg-swst-[^+]+$",
    "iterh": r"^raw:v:(nbg|arg)-iterh-[^+]+$",
    "thr": r"^raw:v:arg-thr-[^+]+$",
    # searched with the minimum-size step: its over-segmentation leaves slivers
    # that only that step removes, so the bare grid ranks settings by sliver count
    "owt": r"^raw:v:nbg-owt-l[\d.]+-h[\d.]+\+A900$",
    "depth": r"^rt:h[\d.p]+$",
}


def country_table(run: str, split: str, patterns: tuple[str, ...]) -> pd.DataFrame:
    """Per-country PQ for every strategy matching ``patterns``, vector reference."""
    root = scores_dir(f"{run}/{split}/pixel")
    df = load_patches(root, "patches*.parquet")
    df = df[df.reference == "vector"]
    c = pool_country(df).drop_duplicates(subset=["strategy", "country"])
    return c[c.strategy.str.match("|".join(patterns))]


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", default="planet_b3")
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    args = p.parse_args()
    countries = tuple(args.countries)
    pats = tuple(FAMILIES.values())

    test = country_table(args.run, "test", pats)
    val = country_table(args.run, "val", pats)
    mac_test = macro(test, countries).set_index("strategy").pq
    mac_val = macro(val, countries).set_index("strategy").pq

    rows = []
    for name, pat in FAMILIES.items():
        cands = [s for s in mac_test.index if re.match(pat, s)]
        if not cands:
            continue
        cands_val = [s for s in cands if s in mac_val.index]
        best_val = max(cands_val, key=lambda s: mac_val[s]) if cands_val else None
        best_test = max(cands, key=lambda s: mac_test[s])
        sub = test[test.strategy.isin(cands) & test.country.isin(countries)]
        per = sub.loc[sub.groupby("country").pq.idxmax()]
        rows.append(
            {
                "family": name,
                "n_settings_tried": len(cands),
                "val_choice": best_val,
                "pq_val_choice": 100 * mac_test[best_val] if best_val else float("nan"),
                "test_choice": best_test,
                "pq_test_choice": 100 * mac_test[best_test],
                "pq_per_country": 100 * per.pq.mean(),
                "gap_per_minus_val": 100 * (per.pq.mean() - mac_test[best_val])
                if best_val
                else float("nan"),
                "n_distinct_winners": per.strategy.nunique(),
                "winners": ", ".join(sorted({s.split(":")[-1] for s in per.strategy})),
            }
        )
    out = pd.DataFrame(rows)
    dst = out_root() / "transfer"
    dst.mkdir(parents=True, exist_ok=True)
    f = dst / f"{args.run}.csv"
    out.to_csv(f, index=False)
    pd.set_option("display.width", 250)
    print(out.round(2).to_string(index=False))
    print(f"\nwrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
