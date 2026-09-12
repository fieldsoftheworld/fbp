"""Generate the supplementary tables from experiments/out and experiments/results.

Writes tables/supp_*.tex:
  supp_sweep_rows    every tau and h setting, all four model/sensor runs
  supp_priors_rows   every prior-method setting, validation and test
  supp_variants_rows every alternative named in the ablation, validation and test
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
# Fig. 3 of the letter (experiments/figures/per_country.py) compares ours
# against the released pipeline WITH the minimum-size step, since that is the
# like-for-like comparison the letter reports; the per-country table below is
# captioned as reproducing that figure, so it must use the same baseline.
COUNTRY_BASE = "raw:v:arg-ws2+A900"
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
        # The grid is searched without the minimum-size step, which is added
        # to the chosen setting afterwards, so +A900 rows are not part of it.
        keys = [
            k
            for k in test.index
            if k.startswith(prefix)
            and "+A" not in k
            and ("iterh" in k if "Iterative" in title else True)
        ]
        keys = sorted(keys)
        if not keys:
            continue
        rows.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for k in keys:
            short = k.replace("raw:v:", "")
            if "iterh" not in k:  # only the iterative family varies its foreground
                short = short.replace("nbg-", "").replace("arg-", "")
            v = (
                f"{100 * val.loc[k].pq:.1f}"
                if val is not None and k in val.index
                else "--"
            )
            rows.append(f"\\texttt{{{short}}} & {v} & {100 * test.loc[k].pq:.1f} \\\\")
    return rows


def variants_rows() -> list[str]:
    """Every alternative the letter's ablation names, validation and test PQ."""
    val = macro("planet_b3", "val")
    test = macro("planet_b3", "test")
    ours = OURS_NOFILTER
    rel = "raw:v:arg-ws2"
    blocks = [
        (
            "Released pipeline and ours, no minimum size",
            [("Released", BASE), ("Ours", ours)],
        ),
        (
            "Alternatives to the argmax foreground",
            [
                ("Otsu threshold on field probability", "raw:v:otsu-none"),
                ("Field probability at least 0.5", "raw:v:pf50-none"),
                ("Boundary class closed, radius 1 px", "raw:v:arg-close1"),
                ("Boundary class closed, radius 2 px", "raw:v:arg-close2"),
                ("Watershed on boundary probability, $h=0.1$", "raw:v:arg-pb-h0.1-t0.5"),
                ("Watershed on boundary probability, $h=0.2$", "raw:v:arg-pb-h0.2-t0.5"),
                ("Not-background, boundary closed, radius 1 px", "raw:v:nbg-close1"),
                ("Not-background, boundary-probability watershed", "raw:v:nbg-pb-h0.2-t0.5"),
            ],
        ),
        (
            "Steps added to ours",
            [
                ("Kept line at least 3 px", f"{ours}+l3"),
                ("Kept line at least 5 px", f"{ours}+l5"),
                ("Kept line at least 10 px", f"{ours}+l10"),
                ("Support tolerance band, 1 px", f"{ours}+b1"),
                ("Support tolerance band, 2 px", f"{ours}+b2"),
                ("Fill holes under 50 px", f"{ours}+f50"),
                ("Fill holes under 200 px", f"{ours}+f200"),
                ("Fill every hole", f"{ours}+f-1"),
                ("Merge slivers under 20 px into a neighbor", f"{ours}+m20"),
                ("Merge slivers under 50 px into a neighbor", f"{ours}+m50"),
                ("Merge slivers under 200 px into a neighbor", f"{ours}+m200"),
            ],
        ),
        (
            "Minimum parcel size",
            [(f"Released, {a} m$^2$", f"{rel}+A{a}") for a in (450, 900, 1800, 3600)]
            + [(f"Ours, {a} m$^2$", f"{ours}+A{a}") for a in (450, 900, 1800, 3600)],
        ),
    ]
    rows = []
    for title, items in blocks:
        rows.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for label, k in items:
            rows.append(f"{label} & {pq(val, k)} & {pq(test, k)} \\\\")
    return rows


def country_rows() -> list[str]:
    tabs = {}
    for run, sub, label in RUNS:
        c = pd.read_csv(RES / run / sub / "country.csv")
        c = c[c.reference == "vector"]
        base = c[c.strategy == COUNTRY_BASE].set_index("country").pq
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
        ("variants", variants_rows),
        ("country", country_rows),
        ("timing", timing_rows),
    ):
        lines = fn()
        (OUT / f"supp_{name}_rows.tex").write_text("\n".join(lines) + "\n")
        print(f"tables/supp_{name}_rows.tex: {len(lines)} lines")


if __name__ == "__main__":
    main()
