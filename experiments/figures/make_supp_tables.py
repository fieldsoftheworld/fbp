"""Generate the supplementary tables from experiments/out and experiments/results.

Writes tables/supp_*.tex:
  supp_sweep_rows    every tau and h setting, all four model/sensor runs
  supp_priors_rows   every prior-method setting, validation and test
  supp_country_rows  per-country PQ, released and ours, all four runs
  supp_timing_rows   median ms/patch in four density bins, both sensors

Regenerate: ~/github/ftw-planet/.venv/bin/python experiments/figures/make_supp_tables.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
OUT = ROOT / "tables"
RUNS = [
    ("planet_b3", "test", "Planet B3"),
    ("planet_b7", "test", "Planet B7"),
    ("s2_b7", "test/pixel", "S2 B7"),
    ("s2_b3", "test/pixel", "S2 B3"),
]
OURS, BASE = "raw:v:nbg-sup-h2-t0.7+A900", "rt:h2"
OURS_NOFILTER = "raw:v:nbg-sup-h2-t0.7"
NAMES = {
    "belgium": "Belgium",
    "cambodia": "Cambodia",
    "croatia": "Croatia",
    "germany": "Germany",
    "latvia": "Latvia",
    "lithuania": "Lithuania",
    "portugal": "Portugal",
    "slovenia": "Slovenia",
    "south_africa": "South Africa",
    "sweden": "Sweden",
}


def macro(run: str, sub: str) -> pd.DataFrame:
    m = pd.read_csv(RES / run / sub / "macro.csv")
    return m[m.reference == "vector"].set_index("strategy")


def pq(m: pd.DataFrame, strat: str) -> str:
    return f"{100 * m.loc[strat].pq:.1f}" if strat in m.index else "--"


def sweep_rows() -> list[str]:
    ms = {label: macro(r, s) for r, s, label in RUNS}
    rows = []
    blocks = [
        (
            "Support threshold $\\tau$ ($h=2$)",
            [
                (f"$\\tau={t}$", f"raw:v:nbg-sup-h2-t{t}")
                for t in ("0.3", "0.5", "0.7", "0.8", "0.9", "1")
            ],
        ),
        (
            "Marker depth $h$ ($\\tau=0.7$)",
            [(f"$h={h}$", f"raw:v:nbg-sup-h{h}-t0.7") for h in ("1", "2", "4", "8")],
        ),
        (
            "Foreground and split rule",
            [
                ("Field, components", "cc"),
                ("Field, watershed instances", "raw:h2"),
                ("Field, watershed components", BASE),
                ("Field, supported splits", "raw:v:arg-sup-h2-t0.7"),
                ("Not-background, components", "raw:v:nbg-none"),
                ("Not-background, supported splits", OURS_NOFILTER),
                ("Ours, with minimum size", OURS),
            ],
        ),
    ]
    for title, items in blocks:
        rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for label, strat in items:
            rows.append(
                f"{label} & "
                + " & ".join(pq(ms[lbl], strat) for _, _, lbl in RUNS)
                + r" \\"
            )
    return rows


def priors_rows() -> list[str]:
    val = (
        macro("planet_b3", "val")
        if (RES / "planet_b3" / "val" / "macro.csv").exists()
        else None
    )
    test = macro("planet_b3", "test")
    fams = [
        ("Extent/boundary thresholds", "raw:v:arg-thr-"),
        ("Hierarchical merging", "raw:v:nbg-ucm-"),
        ("Iterative marker depth", "raw:v:"),
        ("Stochastic watershed", "raw:v:nbg-swst-"),
    ]
    rows = []
    for title, prefix in fams:
        keys = [
            k
            for k in test.index
            if k.startswith(prefix) and ("iterh" in k if "Iterative" in title else True)
        ]
        if "Iterative" in title:
            keys = [k for k in test.index if "iterh" in k]
        keys = sorted(keys)
        if not keys:
            continue
        rows.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for k in keys:
            short = k.replace("raw:v:", "").replace("nbg-", "").replace("arg-", "")
            v = (
                f"{100 * val.loc[k].pq:.1f}"
                if val is not None and k in val.index
                else "--"
            )
            rows.append(f"\\texttt{{{short}}} & {v} & {100 * test.loc[k].pq:.1f} \\\\")
    return rows


def country_rows() -> list[str]:
    tabs = {}
    for run, sub, label in RUNS:
        c = pd.read_csv(RES / run / sub / "country.csv")
        c = c[c.reference == "vector"]
        base = c[c.strategy == BASE].set_index("country").pq
        ours = c[c.strategy == OURS].set_index("country").pq
        tabs[label] = (100 * base, 100 * ours)
    rows = []
    for country, name in NAMES.items():
        cells = []
        for _, _, label in RUNS:
            b, o = tabs[label]
            cells += [f"{b[country]:.1f}", f"{o[country]:.1f}"]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return rows


def timing_rows() -> list[str]:
    order = ["1-9", "10-29", "30-99", "100+"]
    labels = {
        "released": "Released",
        "thr": "Thresholds",
        "ucm": "Hierarchical merging",
        "iterh": "Iterative marker depth",
        "swst": "Stochastic watershed",
        "ours": "Ours",
    }
    frames = {}
    for run, label in (("planet_b3", "Planet B3"), ("s2_b7", "S2 B7")):
        f = RES / "cost" / f"{run}_test.csv"
        if f.exists():
            df = pd.read_csv(f)
            frames[label] = df.groupby(["method", "bin"]).ms.median().unstack()[order]
    rows = []
    for meth, label in labels.items():
        cells = []
        for _, fr in frames.items():
            cells += [f"{fr.loc[meth, b]:.0f}" for b in order]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    return rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, fn in (
        ("sweep", sweep_rows),
        ("priors", priors_rows),
        ("country", country_rows),
        ("timing", timing_rows),
    ):
        lines = fn()
        (OUT / f"supp_{name}_rows.tex").write_text("\n".join(lines) + "\n")
        print(f"tables/supp_{name}_rows.tex: {len(lines)} lines")


if __name__ == "__main__":
    main()
