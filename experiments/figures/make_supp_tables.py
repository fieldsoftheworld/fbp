"""Generate the supplementary tables from experiments/out and experiments/results.

Writes tables/supp_*.tex:
  supp_sweep_rows    every tau and h setting, all four model/sensor runs
  supp_priors_rows   every prior-method setting, validation and test
  supp_variants_rows every alternative named in the ablation, validation and test
  supp_size_rows     PQ and RQ in FTP's hectare bins, all four runs
  supp_country_rows  per-country PQ, FTP pipeline and boundary veto, all four runs
  supp_timing_rows   median ms/patch in four density bins, both sensors

Regenerate: ~/github/ftw-planet/.venv/bin/python experiments/figures/make_supp_tables.py
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
OUT = ROOT / "tables"
RUNS = [
    ("planet_b3", "test", "PS B3"),
    ("planet_b7", "test", "PS B7"),
    ("s2_b7", "test/pixel", "S2 B7"),
    ("s2_b3", "test/pixel", "S2 B3"),
]
OURS, BASE = "raw:v:nbg-sup-h2-t0.7+A900", "rt:h2"
# Fig. 3 of the letter (experiments/figures/per_country.py) compares the
# boundary veto against the FTP pipeline WITH the minimum-size step, since that
# is the like-for-like comparison the letter reports; the per-country table
# below is captioned as reproducing that figure, so it must use the same
# baseline.
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
            "Support Threshold $\\tau$ ($h=2$)",
            [
                (f"$\\tau={t}$", f"raw:v:nbg-sup-h2-t{t}")
                for t in ("0.3", "0.5", "0.7", "0.8", "0.9", "1")
            ],
        ),
        (
            "Marker Depth $h$ ($\\tau=0.7$)",
            [(f"$h={h}$", f"raw:v:nbg-sup-h{h}-t0.7") for h in ("1", "2", "4", "8")],
        ),
        (
            "Foreground and Split Rule",
            [
                ("Field, Components", "cc"),
                ("Field, Watershed Instances", "raw:h2"),
                ("Field, Watershed Components", BASE),
                ("Field, Supported Splits", "raw:v:arg-sup-h2-t0.7"),
                ("Not-Background, Components", "raw:v:nbg-none"),
                ("Not-Background, Supported Splits", OURS_NOFILTER),
                ("Boundary Veto + Min Size", OURS),
            ],
        ),
    ]
    for title, items in blocks:
        if rows:
            rows.append(r"\addlinespace[2pt]")
        rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for label, strat in items:
            rows.append(
                f"{label} & "
                + " & ".join(pq(ms[lbl], strat) for _, _, lbl in RUNS)
                + r" \\"
            )
    return rows


FOREGROUND = {"arg": "field", "nbg": "not-background"}


def natural_key(s: str) -> list[object]:
    """Split digits out so numbers in a strategy id sort by value."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]


def thousands(x: str) -> str:
    """20000 -> 20{,}000, for use inside math mode."""
    return f"{int(x):,}".replace(",", "{,}")


def prior_label(key: str) -> str:
    """A prior method's setting in the letter's notation, not its strategy id."""
    s = key.replace("raw:v:", "")
    m = re.fullmatch(r"(?:arg|nbg)-thr-te([\d.]+)-tb([\d.]+)", s)
    if m:
        return f"$t_e{{=}}{m[1]}$, $t_b{{=}}{m[2]}$"
    m = re.fullmatch(r"(?:arg|nbg)-ucm-k([\d.]+)", s)
    if m:
        return f"$k{{=}}{m[1]}$"
    m = re.fullmatch(r"(arg|nbg)-iterh-a(\d+)", s)
    if m:
        return f"$a{{=}}{thousands(m[2])}$, {FOREGROUND[m[1]]}"
    m = re.fullmatch(r"(?:arg|nbg)-swst-n(\d+)-m(\d+)-t([\d.]+)", s)
    if m:
        return f"$n{{=}}{m[1]}$, $m{{=}}{m[2]}$, $t{{=}}{m[3]}$"
    return f"\\texttt{{{s}}}"


def priors_rows() -> list[str]:
    val = (
        macro("planet_b3", "val")
        if (RES / "planet_b3" / "val" / "macro.csv").exists()
        else None
    )
    test = macro("planet_b3", "test")
    fams = [
        ("Extent/Boundary Thresholds", "raw:v:arg-thr-"),
        ("Hierarchical Merging", "raw:v:nbg-ucm-"),
        ("Iterative Marker Depth", "raw:v:"),
        ("Stochastic Watershed", "raw:v:nbg-swst-"),
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
        # Natural order, so a=5000 sorts before a=20000 rather than after it.
        keys = sorted(keys, key=natural_key)
        if not keys:
            continue
        if rows:
            rows.append(r"\addlinespace[2pt]")
        rows.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for k in keys:
            v = (
                f"{100 * val.loc[k].pq:.1f}"
                if val is not None and k in val.index
                else "--"
            )
            rows.append(f"{prior_label(k)} & {v} & {100 * test.loc[k].pq:.1f} \\\\")
    return rows


def variants_rows() -> list[str]:
    """Every alternative the letter's ablation names, validation and test PQ."""
    val = macro("planet_b3", "val")
    test = macro("planet_b3", "test")
    ours = OURS_NOFILTER
    rel = "raw:v:arg-ws2"
    blocks = [
        (
            "FTP Pipeline and Boundary Veto, No Minimum Size",
            [("FTP Pipeline", BASE), ("Boundary Veto", ours)],
        ),
        (
            "Alternatives to the Argmax Foreground",
            [
                ("Otsu Threshold on Field Probability", "raw:v:otsu-none"),
                ("Field Probability at Least 0.5", "raw:v:pf50-none"),
                ("Boundary Class Closed, Radius 1 px", "raw:v:arg-close1"),
                ("Boundary Class Closed, Radius 2 px", "raw:v:arg-close2"),
                (
                    "Watershed on Boundary Probability, $h=0.1$",
                    "raw:v:arg-pb-h0.1-t0.5",
                ),
                (
                    "Watershed on Boundary Probability, $h=0.2$",
                    "raw:v:arg-pb-h0.2-t0.5",
                ),
                ("Not-Background, Boundary Closed, Radius 1 px", "raw:v:nbg-close1"),
                (
                    "Not-Background, Boundary-Probability Watershed",
                    "raw:v:nbg-pb-h0.2-t0.5",
                ),
            ],
        ),
        (
            "Steps Added to the Boundary Veto",
            [
                ("Kept Line at Least 3 px", f"{ours}+l3"),
                ("Kept Line at Least 5 px", f"{ours}+l5"),
                ("Kept Line at Least 10 px", f"{ours}+l10"),
                ("Support Tolerance Band, 1 px", f"{ours}+b1"),
                ("Support Tolerance Band, 2 px", f"{ours}+b2"),
                ("Fill Holes Under 50 px", f"{ours}+f50"),
                ("Fill Holes Under 200 px", f"{ours}+f200"),
                ("Fill Every Hole", f"{ours}+f-1"),
                ("Merge Slivers Under 20 px Into a Neighbor", f"{ours}+m20"),
                ("Merge Slivers Under 50 px Into a Neighbor", f"{ours}+m50"),
                ("Merge Slivers Under 200 px Into a Neighbor", f"{ours}+m200"),
            ],
        ),
        (
            "Minimum Parcel Size",
            [(f"FTP, {a} m$^2$", f"{rel}+A{a}") for a in (450, 900, 1800, 3600)]
            + [
                (f"Boundary Veto, {a} m$^2$", f"{ours}+A{a}")
                for a in (450, 900, 1800, 3600)
            ],
        ),
    ]
    rows = []
    for title, items in blocks:
        if rows:
            rows.append(r"\addlinespace[2pt]")
        rows.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for label, k in items:
            rows.append(f"{label} & {pq(val, k)} & {pq(test, k)} \\\\")
    return rows


def size_rows() -> list[str]:
    """PQ and RQ in FTP's own hectare bins: small <0.5, medium 0.5-2, large >2."""
    cols = ["pq_small", "pq_medium", "pq_large", "rq_small", "rq_medium", "rq_large"]
    methods = [
        ("FTP, $h{=}2$", BASE),
        ("FTP + Min Size", "raw:v:arg-ws2+A900"),
        ("Boundary Veto", OURS),
    ]
    rows = []
    for run, sub, label in RUNS:
        m = macro(run, sub)
        if rows:
            rows.append(r"\addlinespace[2pt]")
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{label}}}}} \\\\")
        for name, strat in methods:
            if strat not in m.index:
                continue
            r = m.loc[strat]
            cells = []
            for c in cols:
                cell = f"{100 * r[c]:.1f}"
                best = max(m.loc[s, c] for _, s in methods if s in m.index)
                if r[c] == best:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            rows.append(f"{name} & " + " & ".join(cells) + r" \\")
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
        cells: list[str] = []
        for _, _, label in RUNS:
            b, o = tabs[label]
            pair = [b[country], o[country]]
            cells += [
                f"\\textbf{{{value:.1f}}}" if value == max(pair) else f"{value:.1f}"
                for value in pair
            ]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return rows


def timing_rows() -> list[str]:
    order = ["1-9", "10-29", "30-99", "100+"]
    labels = {
        "released": "FTP Pipeline",
        "thr": "Thresholds",
        "ucm": "Hierarchical Merging",
        "iterh": "Iterative Marker Depth",
        "swst": "Stochastic Watershed",
        "ours": "Boundary Veto",
    }
    frames = {}
    for run, label in (("planet_b3", "PS"), ("s2_b7", "S2")):
        f = RES / "cost" / f"{run}_test.csv"
        if f.exists():
            df = pd.read_csv(f)
            frames[label] = df.groupby(["method", "bin"]).ms.median().unstack()[order]
    rows = []
    for meth, label in labels.items():
        cells: list[str] = []
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
        ("size", size_rows),
        ("country", country_rows),
        ("timing", timing_rows),
    ):
        lines = fn()
        (OUT / f"supp_{name}_rows.tex").write_text("\n".join(lines) + "\n")
        print(f"tables/supp_{name}_rows.tex: {len(lines)} lines")


if __name__ == "__main__":
    main()
