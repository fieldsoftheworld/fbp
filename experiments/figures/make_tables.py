"""Generate the revised letter's table rows and value macros from experiments/results.

Writes tables/rev_main_rows.tex, tables/rev_ablation_rows.tex, tables/rev_values.tex.
Regenerate: ~/github/ftw-planet/.venv/bin/python experiments/figures/make_tables.py
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
OUT = ROOT / "tables"

RUNS = [
    ("planet_b3", "test", "PlanetScope", "B3"),
    ("planet_b7", "test", "PlanetScope", "B7"),
    ("s2_b7", "test/pixel", "Sentinel-2", "B7"),
    ("s2_b3", "test/pixel", "Sentinel-2", "B3"),
]
OURS = "raw:v:nbg-sup-h2-t0.7+A900"
OURS_NOFILTER = "raw:v:nbg-sup-h2-t0.7"
RELEASED_FILTER = "raw:v:arg-ws2+A900"
BASE = "rt:h2"
ORACLE = "rt_test"
# The shipped method's row: bold label, light green fill (oursbg in the preamble).
OURS_LABEL = r"\textbf{BoundaryVote (Ours)}"
OURS_ROW = r"\rowcolor{oursbg}"


def macro_vec(run: str, sub: str) -> pd.DataFrame:
    m = pd.read_csv(RES / run / sub / "macro.csv")
    return m[m.reference == "vector"].set_index("strategy")


def country_vec(run: str, sub: str) -> pd.DataFrame:
    c = pd.read_csv(RES / run / sub / "country.csv")
    return c[c.reference == "vector"]


def f(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}"


COMPARE = [
    # label, knobs, transfer family, profile method, macro strategy on test
    ("FTP Pipeline", "$h$", "depth", "released", "raw:v:arg-ws2"),
    ("Thresholds~\\cite{waldner2020decode}", "$t_e$, $t_b$", "thr", "thr", None),
    (
        "Hierarchical Merging~\\cite{arbelaez2011hierarchical}",
        "$k$",
        "ucm",
        "ucm",
        None,
    ),
    (
        "Iterative Depth~\\cite{koyuncu2016iterative}",
        "$a$, sched.",
        "iterh",
        "iterh",
        None,
    ),
    (
        "Stochastic Watershed~\\cite{angulo2007stochastic}",
        "$n$, $m$, $t$",
        "swst",
        "swst",
        None,
    ),
    (
        "Oriented Watershed~\\cite{zhu2024gfafd}",
        "$\\lambda$, $h$",
        "owt",
        "owt",
        None,
    ),
    (
        "GMM Thresholds~\\cite{amin2025earlyseason}",
        "fitted",
        "gmm",
        "gmm",
        None,
    ),
    (OURS_LABEL, "$h$, $\\tau$", "ours", "ours", OURS_NOFILTER),
]
PRIOR_FAMILIES = ("thr", "ucm", "iterh", "swst", "owt", "gmm")


def with_filter(strat: str, m: pd.DataFrame) -> str:
    """The same method with the minimum-size step, when it was scored."""
    cand = strat + "+A900"
    return cand if cand in m.index else strat


def compare_rows() -> list[str]:
    """Accuracy, per-country tuning headroom, and cost for each method.

    Every row carries the same minimum-size step, so the comparison is
    like-for-like with the method as shipped.
    """
    tr_f = ROOT / "experiments" / "out" / "transfer" / "planet_b3.csv"
    pr_f = ROOT / "experiments" / "out" / "profile" / "planet_b3_test.csv"
    if not (tr_f.exists() and pr_f.exists()):
        return []
    tr = pd.read_csv(tr_f).set_index("family")
    pr = pd.read_csv(pr_f)
    ms = pr.groupby(["method", "bin"]).ms.median().unstack()
    mb3 = macro_vec("planet_b3", "test")
    ms2 = macro_vec("s2_b7", "test/pixel")
    lines = []
    for label, knobs, fam, meth, strat in COMPARE:
        if fam not in tr.index:
            continue  # a family whose grid has not been scored yet
        resolved_strat = tr.loc[fam].val_choice if strat is None else strat
        s3, s2 = with_filter(resolved_strat, mb3), with_filter(resolved_strat, ms2)
        pq3, pq2 = (
            f(mb3.loc[s3].pq),
            f(ms2.loc[s2].pq) if s2 in ms2.index else "--",
        )
        # The FTP pipeline row is reported at its default h=2, so its headroom
        # is measured from h=2, not from the validation-best depth.
        gap_val = (
            tr.loc[fam].pq_per_country - 100 * mb3.loc[BASE].pq
            if fam == "depth"
            else tr.loc[fam].gap_per_minus_val
        )
        gap = f"{gap_val:+.1f}"
        timed = meth in ms.index
        cells = [
            pq3,
            pq2,
            gap,
            f"{ms.loc[meth, '10-29']:.0f}" if timed else "--",
            f"{ms.loc[meth, '100+']:.0f}" if timed else "--",
        ]
        if fam == "ours":
            cells = [f"\\textbf{{{c}}}" for c in cells[:2]] + cells[
                2:
            ]  # plain numbers, not math
        lead = OURS_ROW if fam == "ours" else ""
        lines.append(f"{lead}{label} & {knobs} & " + " & ".join(cells) + r" \\")
    return lines


def compare_macros() -> dict[str, str]:
    """Numbers the comparison paragraph quotes, from the same two CSVs."""
    tr = pd.read_csv(
        ROOT / "experiments" / "out" / "transfer" / "planet_b3.csv"
    ).set_index("family")
    pr = pd.read_csv(ROOT / "experiments" / "out" / "profile" / "planet_b3_test.csv")
    ms = pr.groupby(["method", "bin"]).ms.median().unstack()
    priors = [f for f in PRIOR_FAMILIES if f in tr.index]
    gaps = tr.loc[priors].gap_per_minus_val
    dense = "100+"
    return {
        "OursTune": f"{tr.loc['ours'].gap_per_minus_val:.1f}",
        "PriorTuneMin": f"{gaps.min():.1f}",
        "PriorTuneMax": f"{gaps.max():.1f}",
        "OursSlowdown": f"{ms.loc['ours', dense] / ms.loc['released', dense]:.1f}$\\times$",
        "SwstSlowdown": f"{ms.loc['swst', dense] / ms.loc['ours', dense]:.0f}$\\times$",
        "PeakMB": f"{max(pd.read_csv(g).peak_mb.max() for g in (ROOT / 'experiments' / 'out' / 'profile').glob('*_test.csv')):.0f}",
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    main_rows: list[str] = []
    vals: dict[str, str] = {}
    gains: dict[str, float] = {}
    merge_up: list[float] = []
    merge_vs_min: list[float] = []
    for run, sub, sensor, enc in RUNS:
        m = macro_vec(run, sub)
        c = country_vec(run, sub)
        rows = []
        # macro PQ/SQ/RQ, then the small-field bin (<0.5 ha) the gain lands in
        cols = ["pq", "sq", "rq", "pq_small", "rq_small", "miss_rate", "merge_rate"]
        higher = [True, True, True, True, True, False, False]
        for name, strat in (
            ("FTP, $h{=}2$", BASE),
            ("FTP + Min Size", RELEASED_FILTER),
            # the model column already sets the table apart, so no "(Ours)" here
            (r"\textbf{BoundaryVote}", OURS),
        ):
            r = m.loc[strat]
            cells = []
            for col, hi in zip(cols, higher, strict=True):
                v = f(r[col])
                # Equal numeric widths align decimals in centered metric columns.
                if len(v.partition(".")[0]) == 1:
                    v = rf"\phantom{{0}}{v}"
                shown = [
                    m.loc[t][col] for t in (BASE, RELEASED_FILTER, OURS) if t in m.index
                ]
                best = max(shown) if hi else min(shown)
                if r[col] == best:
                    v = f"\\textbf{{{v}}}"
                cells.append(v)
            rows.append((name, cells, strat == OURS))
        sensor_short = {"PlanetScope": "PS", "Sentinel-2": "S2"}[sensor]
        for i, (name, cells, ours) in enumerate(rows):
            # The model label sits in the last (shaded) row with a negative
            # span, since a later row's fill would paint over text that
            # multirow hangs down from the first row.
            last = i == len(rows) - 1
            lead = f"\\multirow{{-3}}{{*}}{{{sensor_short} {enc}}}" if last else ""
            fill = OURS_ROW if ours else ""
            main_rows.append(f"{fill}{lead} & {name} & " + " & ".join(cells) + r" \\")
        main_rows.append(r"\midrule" if run != RUNS[-1][0] else "")
        # like-for-like: both with the minimum-size step, so the split rule is
        # what the headline gain measures
        ref = RELEASED_FILTER if RELEASED_FILTER in m.index else BASE
        d = m.loc[OURS].pq - m.loc[ref].pq
        base_c = c[c.strategy == ref].set_index("country").pq
        ours_c = c[c.strategy == OURS].set_index("country").pq
        dc = ours_c - base_c
        tag = {"PlanetScope": "Planet", "Sentinel-2": "Sentinel"}[sensor] + {
            "B3": "Three",
            "B7": "Seven",
        }[enc]
        vals[f"Gain{tag}"] = f(d)
        vals[f"Base{tag}"] = f(m.loc[ref].pq)
        vals[f"Published{tag}"] = f(m.loc[BASE].pq)
        vals[f"GainPub{tag}"] = f"{100 * (m.loc[OURS].pq - m.loc[BASE].pq):.1f}"
        vals[f"Ours{tag}"] = f(m.loc[OURS].pq)
        vals[f"Oracle{tag}"] = f(m.loc[ORACLE].pq)
        vals[f"Up{tag}"] = str(int((dc > 0).sum()))
        vals[f"Worst{tag}"] = f"{100 * dc.min():+.1f}"
        vals[f"OracleGain{tag}"] = f"{100 * (m.loc[ORACLE].pq - m.loc[BASE].pq):.1f}"
        gains[tag] = d
        # merge rate against the raw FTP output and against its minimum-size row
        merge_up.append(100 * (m.loc[OURS].merge_rate - m.loc[BASE].merge_rate))
        merge_vs_min.append(100 * (m.loc[OURS].merge_rate - m.loc[ref].merge_rate))
        if run == "planet_b3":
            for k, s in (("Base", BASE), ("Ours", OURS)):
                vals[f"{k}Miss"] = f(m.loc[s].miss_rate)
                vals[f"{k}Merge"] = f(m.loc[s].merge_rate)
                vals[f"{k}Split"] = f(m.loc[s].split_rate)
                vals[f"{k}Small"] = f(m.loc[s].pq_small)
                vals[f"{k}Bias"] = f"{m.loc[s].count_bias:.0f}"
                vals[f"{k}BiasAbs"] = f"{abs(m.loc[s].count_bias):.0f}"
            vals["RawInstPQ"] = f(m.loc["raw:h2"].pq)
            vals["CCPQ"] = f(m.loc["cc"].pq)
    while main_rows and main_rows[-1] == "":
        main_rows.pop()
    (OUT / "rev_main_rows.tex").write_text("\n".join(main_rows) + "\n")

    # Ablation: Δ PQ vs BASE on PlanetScope B3 and Sentinel-2 B7.
    mb3 = macro_vec("planet_b3", "test")
    ms2 = macro_vec("s2_b7", "test/pixel")

    def delta(m: pd.DataFrame, s: str) -> str:
        r"""Signed PQ change, colored green when it helps and red when it hurts.

        \\gainc and \\lossc are defined in the preamble of the letter, so the
        two colors are set in one place rather than per cell.
        """
        if s not in m.index:
            return "--"
        d = 100 * (m.loc[s].pq - m.loc[BASE].pq)
        # a change that rounds to 0.0 is shown as neither a gain nor a loss,
        # so the color never claims more than the printed digit supports
        cmd = "flatc" if abs(d) < 0.05 else ("gainc" if d > 0 else "lossc")
        return f"\\{cmd}{{${d:+.1f}$}}"

    abl = [
        (r"\multicolumn{3}{@{}l}{\emph{Foreground and Split Rule}} \\", None),
        ("Field, Components", "cc"),
        ("Field, Watershed Instances", "raw:h2"),
        ("Field, Watershed Components ($h{=}2$)", BASE),
        ("Field, Supported Splits", "raw:v:arg-sup-h2-t0.7"),
        ("Not-Background, Components", "raw:v:nbg-none"),
        ("Not-Background, Supported Splits", OURS_NOFILTER),
        (r"\multicolumn{3}{@{}l}{\emph{Minimum Parcel Size of 900~m$^2$}} \\", None),
        ("FTP, With Minimum Size", RELEASED_FILTER),
        ("BoundaryVote, Without Minimum Size", OURS_NOFILTER),
        # with the step as well, so the block shows what it is worth to each
        # pipeline rather than leaving our half to be inferred
        (f"{OURS_LABEL}, With Minimum Size", OURS),
    ]
    lines = []
    for name, strat in abl:
        if strat is None:
            if lines:
                lines.append(r"\addlinespace[2pt]")
            lines.append(name)
            continue
        a, b = delta(mb3, strat), delta(ms2, strat)
        fill = ""
        if strat == OURS:
            # bold inside math needs \mathbf; \textbf around $...$ does nothing
            a, b = (x.replace("{$", "{$\\mathbf{").replace("$}", "}$}") for x in (a, b))
            fill = OURS_ROW
        lines.append(f"{fill}{name} & {a} & {b} \\\\")
    (OUT / "rev_ablation_rows.tex").write_text("\n".join(lines) + "\n")

    cmp_lines = compare_rows()
    if cmp_lines:
        (OUT / "rev_compare_rows.tex").write_text("\n".join(cmp_lines) + "\n")
        vals.update(compare_macros())

    # each alternative line statistic at its own validation-best threshold,
    # and how far it falls short of the vote share on test
    val_b3 = macro_vec("planet_b3", "val")
    losses = []
    for fam in ("ssup", "asup", "dsup"):
        cands = [
            k
            for k in val_b3.index
            if re.match(rf"^raw:v:nbg-{fam}-h2-t[\d.]+$", k) and k in mb3.index
        ]
        if cands:
            best = max(cands, key=lambda k: val_b3.loc[k].pq)
            losses.append(100 * (mb3.loc[OURS_NOFILTER].pq - mb3.loc[best].pq))
    if losses:
        vals["LineStatLossMin"] = f"{min(losses):.1f}"
        vals["LineStatLossMax"] = f"{max(losses):.1f}"
    tr_f = ROOT / "experiments" / "out" / "transfer" / "planet_b3.csv"
    if tr_f.exists():
        tr = pd.read_csv(tr_f).set_index("family")
        if "owt" in tr.index:
            owt = tr.loc["owt"].val_choice
            vals["OwtNoFilterPQ"] = f(mb3.loc[owt.replace("+A900", "")].pq)
            vals["OwtGapPS"] = f(mb3.loc[OURS].pq - mb3.loc[owt].pq)
            if owt in ms2.index:
                vals["OwtGapSS"] = f(ms2.loc[OURS].pq - ms2.loc[owt].pq)
    vals["ArgSupGain"] = delta(mb3, "raw:v:arg-sup-h2-t0.7")
    if RELEASED_FILTER in mb3.index:
        vals["BaseAreaGain"] = (
            f"{100 * (mb3.loc[RELEASED_FILTER].pq - mb3.loc[BASE].pq):.1f}"
        )
    if OURS in mb3.index and OURS_NOFILTER in mb3.index:
        vals["OursAreaGain"] = (
            f"{100 * (mb3.loc[OURS].pq - mb3.loc[OURS_NOFILTER].pq):.1f}"
        )
    vals["NbgNonePQ"] = f(mb3.loc["raw:v:nbg-none"].pq)
    vals["GainMin"] = f(min(gains.values()))
    vals["GainMax"] = f(max(gains.values()))
    vals["MergeUpMin"] = f"{min(merge_up):.1f}"
    vals["MergeUpMax"] = f"{max(merge_up):.1f}"
    vals["MergeVsMinAbsMax"] = f"{max(abs(x) for x in merge_vs_min):.1f}"
    boot = pd.concat(pd.read_csv(g) for g in sorted((RES / "bootstrap").glob("*.csv")))
    gain_ci = boot[boot.quantity == "gain"]
    vals["GainCILow"] = f"{gain_ci.lo.min():.1f}"
    vals["GainCIHigh"] = f"{gain_ci.hi.max():.1f}"
    total_up = sum(int(vals[f"Up{t}"]) for t in gains)
    vals["UpTotal"] = str(total_up)
    vals["WorstAll"] = f"{min(float(vals[f'Worst{t}']) for t in gains):+.1f}"
    vals["OracleGainMax"] = f"{max(float(vals[f'OracleGain{t}']) for t in gains):.1f}"
    macros = "% Generated by experiments/figures/make_tables.py\n" + "\n".join(
        f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in vals.items()
    )
    (OUT / "rev_values.tex").write_text(macros + "\n")
    print(macros)


if __name__ == "__main__":
    main()
