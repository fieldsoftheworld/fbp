"""Generate the supplementary tables from experiments/results.

Writes tables/supp_*.tex:
  supp_sweep_rows    every tau and h setting, all four model/sensor runs
  supp_priors_rows   every prior-method setting, validation and test
  supp_variants_rows every alternative named in the ablation, validation and test
  supp_size_rows     PQ and RQ in FTP's hectare bins, all four runs
  supp_country_rows  per-country PQ, FTP pipeline and BoundaryVote, all four runs
  supp_timing_rows   median ms/patch in four density bins, both sensors
  supp_bootstrap_rows patch-bootstrap intervals for both pipelines and the gain

Regenerate: ~/github/ftw-planet/.venv/bin/python -m fbp.paper.supp_tables
"""

import re

import pandas as pd

from fbp.paper.common import (
    NAMES,
    OURS_LABEL,
    OURS_ROW,
    OUT,
    RES,
    SENSOR_SHORT,
    country,
    macro,
)
from fbp.paper.common import RUNS as _COMMON_RUNS
from fbp.paper.shipped import FTP as BASE
from fbp.paper.shipped import FTP_MIN as COUNTRY_BASE
from fbp.paper.shipped import OURS, OURS_DISTANCE, OURS_NOFILTER

# (run, split, shown label), always the test split here.
RUNS = [
    (run, "test", f"{SENSOR_SHORT[sensor]} {enc}") for run, sensor, enc in _COMMON_RUNS
]
# Fig. 3 of the letter (experiments/fbp/paper/figures/per_country.py) compares
# BoundaryVote against the FTP pipeline WITH the minimum-size step, since that
# is the like-for-like comparison the letter reports; the per-country table
# below is captioned as reproducing that figure, so it must use the same
# baseline.
# Zhu et al. at its validation-chosen setting, scored on PS B3 and S2 B7 only.
OWT = "raw:v:nbg-owt-l0.5-h0+A900"


def pq(m: pd.DataFrame, strat: str) -> str:
    return f"{100 * m.loc[strat].pq:.1f}" if strat in m.index else "--"


def fill(strat: str) -> str:
    """Row color prefix for the shipped method, empty otherwise."""
    return OURS_ROW if strat == OURS else ""


def sweep_rows() -> list[str]:
    ms = {label: macro(r, s) for r, s, label in RUNS}
    rows = []
    blocks = [
        (
            "Support Threshold $\\tau$ ($h=0.01$)",
            [
                (f"$\\tau={t}$", f"raw:v:nbg-pb-h0.01-t{t}+A900")
                for t in ("0", "0.1", "0.2", "0.3", "0.5", "0.7", "0.9")
            ],
        ),
        (
            "Seed Depth $h$ ($\\tau=0.3$)",
            [
                (f"$h={h}$", f"raw:v:nbg-pb-h{h}-t0.3+A900")
                for h in ("0", "0.01", "0.02", "0.05", "0.1", "0.2")
            ],
        ),
        (
            "Foreground, Surface, and Split Rule",
            [
                ("Field, Components, No Min Size", "cc"),
                ("FTP Pipeline, No Min Size", BASE),
                ("FTP + Min Size", COUNTRY_BASE),
                ("Field, Prob. Watershed + Vote", "raw:v:arg-pb-h0.01-t0.3+A900"),
                ("Not-Bg, Components", "raw:v:nbg-none+A900"),
                ("Not-Bg, Distance Watershed + Vote", OURS_DISTANCE),
                (
                    "Not-Bg, Prob. Watershed, $\\tau{=}0$",
                    "raw:v:nbg-pb-h0.01-t0+A900",
                ),
                ("Same, $h=0$", "raw:v:nbg-pb-h0-t0+A900"),
                ("BoundaryVote, $h=0$", "raw:v:nbg-pb-h0-t0.3+A900"),
                (OURS_LABEL, OURS),
            ],
        ),
    ]
    for title, items in blocks:
        if rows:
            rows.append(r"\addlinespace[2pt]")
        rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for label, strat in items:
            rows.append(
                f"{fill(strat)}{label} & "
                + " & ".join(pq(ms[lbl], strat) for _, _, lbl in RUNS)
                + r" \\"
            )
    return rows


FOREGROUND = {"arg": "field", "nbg": "not-background"}


def natural_key(s: str) -> list[object]:
    """Split numbers out so a strategy id sorts by value, 0.05 before 0.2."""
    parts = re.split(r"(\d+(?:\.\d+)?)", s)
    return [float(p) if re.fullmatch(r"\d+(?:\.\d+)?", p) else p for p in parts]


def thousands(x: str) -> str:
    """20000 -> 20{,}000, for use inside math mode."""
    return f"{int(x):,}".replace(",", "{,}")


def prior_label(key: str) -> str:
    """A prior method's setting in the letter's notation, not its strategy id."""
    s = key.replace("raw:v:", "").replace("+A900", "")
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
    m = re.fullmatch(r"(?:arg|nbg)-owt-l([\d.]+)-h([\d.]+)", s)
    if m:
        return f"$\\lambda{{=}}{m[1]}$, $h{{=}}{m[2]}$"
    fixed = {"nbg-gmm": "$t_e$, $t_b$ fitted per country"}
    return fixed.get(s, f"\\texttt{{{s}}}")


def priors_rows() -> list[str]:
    val = macro("planet_b3", "val")
    test = macro("planet_b3", "test")
    fams = [
        ("Extent/Boundary Thresholds", "raw:v:arg-thr-", False),
        ("Hierarchical Merging", "raw:v:nbg-ucm-", False),
        ("Iterative Marker Depth", "raw:v:", False),
        ("Stochastic Watershed", "raw:v:nbg-swst-", False),
        ("Oriented Watershed, With Minimum Size", "raw:v:nbg-owt-", True),
        ("Gaussian-Mixture Thresholds", "raw:v:nbg-gmm", False),
    ]
    rows = []
    for title, prefix, with_min in fams:
        # A grid is searched without the minimum-size step, which is added to
        # the chosen setting afterwards, except for the oriented watershed,
        # whose slivers make the bare grid meaningless (see transfer.FAMILIES).
        keys = [
            k
            for k in test.index
            if k.startswith(prefix)
            and (("+A900" in k) if with_min else ("+A" not in k))
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
            rows.append(f"{prior_label(k)} & {pq(val, k)} & {pq(test, k)} \\\\")
    return rows


def variants_rows() -> list[str]:
    """Every alternative the letter's ablation names, validation and test PQ."""
    val = macro("planet_b3", "val")
    test = macro("planet_b3", "test")
    ours = OURS_NOFILTER
    dist = OURS_DISTANCE.replace("+A900", "")
    rel = "raw:v:arg-ws2"
    blocks = [
        (
            "FTP Pipeline and BoundaryVote, No Minimum Size",
            [
                ("FTP Pipeline", BASE),
                ("BoundaryVote", ours),
                (
                    "BoundaryVote, Distance Proposals",
                    OURS_DISTANCE.replace("+A900", ""),
                ),
            ],
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
            "Line Statistic for the Vote (Distance Proposals, No Minimum Size)",
            [
                ("Vote Share, $\\tau=0.7$", dist),
            ]
            + [
                (f"Mean Share, $\\tau={t}$", f"raw:v:nbg-ssup-h2-t{t}")
                for t in ("0.5", "0.6", "0.7")
            ]
            + [("Mean Margin at Least 0", "raw:v:nbg-dsup-h2-t0.5")]
            + [
                (
                    f"Mean $p_{{\\mathrm{{bnd}}}}$ at Least {t}",
                    f"raw:v:nbg-asup-h2-t{t}",
                )
                for t in ("0.3", "0.4", "0.5", "0.6", "0.7")
            ],
        ),
        (
            "Split Proposals for the Vote, With Minimum Size",
            [
                ("Boundary Probability, $h=0.01$, $\\tau=0.3$ (Ours)", OURS),
                (
                    "Boundary Probability, $h=0.1$, $\\tau=0.7$",
                    "raw:v:nbg-pb-h0.1-t0.7+A900",
                ),
                (
                    "Distance to Predicted Boundary, $h=2$ px, $\\tau=0.7$",
                    OURS_DISTANCE,
                ),
                (
                    "Oriented Boundary Signal, $h=0.05$, $\\tau=0.7$",
                    "raw:v:nbg-osup-h0.05-t0.7+A900",
                ),
                (
                    "Oriented Boundary Signal, $h=0.02$, $\\tau=0.7$",
                    "raw:v:nbg-osup-h0.02-t0.7+A900",
                ),
                (
                    "Oriented Boundary Signal, $h=0$, $\\tau=0.7$",
                    "raw:v:nbg-osup-h0-t0.7+A900",
                ),
            ],
        ),
        (
            "Steps Added to BoundaryVote (Distance Proposals, No Minimum Size)",
            [
                ("Kept Line at Least 3 px", f"{dist}+l3"),
                ("Kept Line at Least 5 px", f"{dist}+l5"),
                ("Kept Line at Least 10 px", f"{dist}+l10"),
                ("Support Tolerance Band, 1 px", f"{dist}+b1"),
                ("Support Tolerance Band, 2 px", f"{dist}+b2"),
                ("Fill Holes Under 50 px", f"{dist}+f50"),
                ("Fill Holes Under 200 px", f"{dist}+f200"),
                ("Fill Every Hole", f"{dist}+f-1"),
                ("Merge Slivers Under 20 px Into a Neighbor", f"{dist}+m20"),
                ("Merge Slivers Under 50 px Into a Neighbor", f"{dist}+m50"),
                ("Merge Slivers Under 200 px Into a Neighbor", f"{dist}+m200"),
            ],
        ),
        (
            "Minimum Parcel Size",
            [(f"FTP, {a} m$^2$", f"{rel}+A{a}") for a in (450, 900, 1800, 3600)]
            + [
                (f"BoundaryVote, Distance Proposals, {a} m$^2$", f"{dist}+A{a}")
                for a in (450, 900, 1800, 3600)
            ]
            + [
                (
                    f"{OURS_LABEL if a == 900 else 'BoundaryVote'}, {a} m$^2$",
                    f"{ours}+A{a}",
                )
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
            rows.append(f"{fill(k)}{label} & {pq(val, k)} & {pq(test, k)} \\\\")
    return rows


def size_rows() -> list[str]:
    """PQ and RQ in FTP's own hectare bins: small <0.5, medium 0.5-2, large >2."""
    cols = ["pq_small", "pq_medium", "pq_large", "rq_small", "rq_medium", "rq_large"]
    methods = [
        ("FTP, $h{=}2$", BASE),
        ("FTP + Min Size", COUNTRY_BASE),
        ("Oriented Watershed + Min Size", OWT),
        ("BoundaryVote, Distance Proposals", OURS_DISTANCE),
        (OURS_LABEL, OURS),
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
            rows.append(f"{fill(strat)}{name} & " + " & ".join(cells) + r" \\")
    return rows


def country_rows() -> list[str]:
    tabs = {}
    for run, sub, label in RUNS:
        c = country(run, sub)
        base = c[c.strategy == COUNTRY_BASE].set_index("country").pq
        ours = c[c.strategy == OURS].set_index("country").pq
        tabs[label] = (100 * base, 100 * ours)
    rows = []
    for country_id, name in NAMES.items():
        cells: list[str] = []
        for _, _, label in RUNS:
            b, o = tabs[label]
            pair = [b[country_id], o[country_id]]
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
        "owt": "Oriented Watershed",
        "gmm": "GMM Thresholds",
        "ours": OURS_LABEL,
    }
    frames = {}
    for run, label in (("planet_b3", "PS"), ("s2_b7", "S2")):
        df = pd.read_csv(RES / "profile" / f"{run}_test.csv")
        frames[label] = df.groupby(["method", "bin"]).ms.median().unstack()[order]
    rows = []
    for meth, label in labels.items():
        cells: list[str] = []
        for _, fr in frames.items():
            cells += [f"{fr.loc[meth, b]:.0f}" for b in order]
        lead = OURS_ROW if meth == "ours" else ""
        rows.append(f"{lead}{label} & " + " & ".join(cells) + r" \\")
    return rows


def bootstrap_rows() -> list[str]:
    """Point estimate and 95% patch-bootstrap interval per run, both pipelines and gain."""
    rows = []
    for run, _sub, label in RUNS:
        b = pd.read_csv(RES / "bootstrap" / f"{run}.csv").set_index("quantity")
        cells = [
            f"{b.loc[q].point:.1f} [{b.loc[q].lo:.1f}, {b.loc[q].hi:.1f}]"
            for q in ("ftp_min_size", "ours", "gain")
        ]
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
        ("bootstrap", bootstrap_rows),
    ):
        lines = fn()
        (OUT / f"supp_{name}_rows.tex").write_text("\n".join(lines) + "\n")
        print(f"tables/supp_{name}_rows.tex: {len(lines)} lines")


if __name__ == "__main__":
    main()
