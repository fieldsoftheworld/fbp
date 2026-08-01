"""Emit the LaTeX body of the aggregation-ablation table (main_grsl.tex, Table III).

Why this exists: HEADLINE_ensemble_macro.txt reports the four aggregation
strategies as HELDOUT_11 macros, but the Letter scores everything on DENSE_10
(HELDOUT_11 minus Kenya, the presence-only region). Converting between the two
by hand requires assuming Kenya contributes 0.0 to every strategy. That holds for
the two strategies with a per-country breakdown in the txt file, but A/B/D have
no breakdown there, so the assumption is unverified and must not be hand-applied.
This script recomputes the DENSE_10 macro from the per-country CSV directly and
never needs the assumption.

Usage:
    python scripts/04_analysis/aggregation_ablation.py --csv <ensemble_per_country.csv>

Prints the tabular rows to stdout; paste them over the placeholder body.
"""

import argparse
from pathlib import Path

import pandas as pd

# HELDOUT_11 minus kenya (presence-only; unlabeled background breaks polygon metrics).
DENSE_10 = [
    "belgium", "cambodia", "croatia", "germany", "latvia",
    "lithuania", "portugal", "slovenia", "south_africa", "sweden",
]

# strategy key in the CSV -> (row label, labels-used column) in paper order
STRATEGIES = [
    ("A_nms_pool", "Polygon pool $+$ agreement NMS", "none"),
    ("D_elbow", "Per-patch elbow selection", "none"),
    ("B_oracle", "Per-patch oracle depth", "ground truth"),
    ("C_pixel_vote", r"\textbf{Per-pixel vote (ours)}", "none"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="ensemble_per_country.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = {"country", "strategy", "pq"} - set(df.columns)
    if missing:
        raise SystemExit(
            f"{args.csv} is missing required column(s): {sorted(missing)}. "
            f"Found: {sorted(df.columns)}. See scripts/results/README_ablation.md."
        )

    df["country"] = df["country"].str.strip().str.lower()
    rows = []
    for key, label, labels_used in STRATEGIES:
        sub = df[(df["strategy"] == key) & (df["country"].isin(DENSE_10))]
        have = sorted(sub["country"].unique())
        if len(have) != len(DENSE_10):
            raise SystemExit(
                f"strategy {key}: expected {len(DENSE_10)} DENSE_10 countries, got "
                f"{len(have)}. Missing: {sorted(set(DENSE_10) - set(have))}. "
                f"Do not fill the table from a partial macro."
            )
        pq = 100.0 * sub["pq"].mean()
        cell = rf"\textbf{{{pq:.1f}}}" if "textbf" in label else f"{pq:.1f}"
        rows.append(f"{label} & {labels_used} & {cell} \\\\")

    print(r"\toprule")
    print(r"Strategy & Labels used & PQ \\")
    print(r"\midrule")
    for r in rows[:-1]:
        print(r)
    print(r"\midrule")
    print(rows[-1])
    print(r"\bottomrule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
