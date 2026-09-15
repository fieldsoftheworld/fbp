"""Generate the letter's table rows and value macros from experiments/results.

Writes tables/rev_main_rows.tex, tables/rev_compare_rows.tex,
tables/rev_ablation_rows.tex, tables/rev_values.tex.
Regenerate: ~/github/ftw-planet/.venv/bin/python -m fbp.paper.tables
"""

import pandas as pd

from fbp.paper.common import (
    OURS_LABEL,
    OURS_ROW,
    OUT,
    RES,
    RUNS,
    SENSOR_SHORT,
    country,
    f,
    macro,
)
from fbp.paper.shipped import FTP as BASE
from fbp.paper.shipped import FTP_MIN as RELEASED_FILTER
from fbp.paper.shipped import OURS, OURS_DISTANCE, OURS_NOFILTER, TAU

ORACLE = "rt_test"  # the FTP pipeline at each country's test-best depth

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


def transfer() -> pd.DataFrame:
    return pd.read_csv(RES / "transfer" / "planet_b3.csv").set_index("family")


def profile_ms(run: str) -> pd.DataFrame:
    """Median ms per patch by (method, density bin)."""
    pr = pd.read_csv(RES / "profile" / f"{run}_test.csv")
    return pr.groupby(["method", "bin"]).ms.median().unstack()


def compare_rows() -> list[str]:
    """Accuracy, per-country tuning headroom, and cost for each method.

    Every row carries the same minimum-size step, so the comparison is
    like-for-like with the method as shipped.
    """
    tr = transfer()
    ms = profile_ms("planet_b3")
    mb3 = macro("planet_b3")
    ms2 = macro("s2_b7")
    lines = []
    for label, knobs, fam, meth, strat in COMPARE:
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
        cells = [
            pq3,
            pq2,
            f"{gap_val:+.1f}",
            f"{ms.loc[meth, '10-29']:.0f}",
            f"{ms.loc[meth, '100+']:.0f}",
        ]
        if fam == "ours":
            cells = [f"\\textbf{{{c}}}" for c in cells[:2]] + cells[2:]
        lead = OURS_ROW if fam == "ours" else ""
        lines.append(f"{lead}{label} & {knobs} & " + " & ".join(cells) + r" \\")
    return lines


def compare_macros() -> dict[str, str]:
    """Numbers the comparison paragraph quotes, from the same CSVs."""
    tr = transfer()
    ms = profile_ms("planet_b3")
    dense = "100+"
    peak = max(
        pd.read_csv(g).peak_mb.max() for g in (RES / "profile").glob("*_test.csv")
    )
    return {
        "OursTune": f"{tr.loc['ours'].gap_per_minus_val:.1f}",
        "PriorTuneMax": f"{tr.loc[list(PRIOR_FAMILIES)].gap_per_minus_val.max():.1f}",
        "OursSlowdown": f"{ms.loc['ours', dense] / ms.loc['released', dense]:.1f}$\\times$",
        "SwstSlowdown": f"{ms.loc['swst', dense] / ms.loc['ours', dense]:.0f}$\\times$",
        "PeakMB": f"{peak:.0f}",
    }


def main_rows() -> tuple[list[str], dict[str, str]]:
    """Table II rows and the macros summarised over the four runs."""
    rows_out: list[str] = []
    vals: dict[str, str] = {}
    gains: list[float] = []
    ups: list[int] = []
    oracle_gains: list[float] = []
    vote_gain: list[float] = []
    cols = ["pq", "sq", "rq", "pq_small", "rq_small", "miss_rate", "merge_rate"]
    higher = [True, True, True, True, True, False, False]
    for run, sensor, enc in RUNS:
        m = macro(run)
        c = country(run)
        rows = []
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
                shown = [m.loc[t][col] for t in (BASE, RELEASED_FILTER, OURS)]
                best = max(shown) if hi else min(shown)
                if r[col] == best:
                    v = f"\\textbf{{{v}}}"
                cells.append(v)
            rows.append((name, cells, strat == OURS))
        sensor_short = SENSOR_SHORT[sensor]
        for i, (name, cells, ours) in enumerate(rows):
            # The model label sits in the last (shaded) row with a negative
            # span, since a later row's fill would paint over text that
            # multirow hangs down from the first row.
            last = i == len(rows) - 1
            lead = f"\\multirow{{-3}}{{*}}{{{sensor_short} {enc}}}" if last else ""
            fill = OURS_ROW if ours else ""
            rows_out.append(f"{fill}{lead} & {name} & " + " & ".join(cells) + r" \\")
        if run != RUNS[-1][0]:
            rows_out.append(r"\midrule")
        # like-for-like: both with the minimum-size step, so the split rule is
        # what the headline gain measures
        gains.append(100 * (m.loc[OURS].pq - m.loc[RELEASED_FILTER].pq))
        base_c = c[c.strategy == RELEASED_FILTER].set_index("country").pq
        ours_c = c[c.strategy == OURS].set_index("country").pq
        ups.append(int((ours_c - base_c > 0).sum()))
        oracle_gains.append(100 * (m.loc[ORACLE].pq - m.loc[BASE].pq))
        # what the vote itself is worth at the shipped seed depth
        vote_gain.append(
            100 * (m.loc[OURS].pq - m.loc[OURS.replace(f"-t{TAU:g}", "-t0")].pq)
        )
        if run == "planet_b3":
            vals["OursPlanetThree"] = f(m.loc[OURS].pq)
            vals["BaseMiss"] = f(m.loc[BASE].miss_rate)
            for k, s in (("Base", BASE), ("Ours", OURS)):
                vals[f"{k}Small"] = f(m.loc[s].pq_small)
                vals[f"{k}Large"] = f(m.loc[s].pq_large)
            vals["RawInstPQ"] = f(m.loc["raw:h2"].pq)
            vals["CCPQ"] = f(m.loc["cc"].pq)
    vals["GainMin"] = f"{min(gains):.1f}"
    vals["GainMax"] = f"{max(gains):.1f}"
    vals["UpTotal"] = str(sum(ups))
    vals["OracleGainMax"] = f"{max(oracle_gains):.1f}"
    vals["VoteGainMin"] = f"{min(vote_gain):+.1f}"
    vals["VoteGainMax"] = f"{max(vote_gain):+.1f}"
    return rows_out, vals


def delta(m: pd.DataFrame, s: str) -> str:
    r"""Signed PQ change vs the FTP pipeline, green when it helps, red when it hurts.

    \\gainc, \\lossc and \\flatc are defined in the letter's preamble.
    """
    if s not in m.index:
        return "--"
    d = 100 * (m.loc[s].pq - m.loc[BASE].pq)
    # a change that rounds to 0.0 is shown as neither a gain nor a loss,
    # so the color never claims more than the printed digit supports
    cmd = "flatc" if abs(d) < 0.05 else ("gainc" if d > 0 else "lossc")
    return f"\\{cmd}{{${d:+.1f}$}}"


def ablation_rows(mb3: pd.DataFrame, ms2: pd.DataFrame) -> list[str]:
    """Table III: PQ change vs the FTP pipeline on PlanetScope B3 and Sentinel-2 B7."""
    abl = [
        (
            r"\multicolumn{3}{@{}l}{\emph{Foreground, Surface, and Split Rule "
            r"(With Minimum Size)}} \\",
            None,
        ),
        ("FTP + Min Size", RELEASED_FILTER),
        ("Field, Probability Watershed + Vote", "raw:v:arg-pb-h0.01-t0.3+A900"),
        ("Not-Background, Components", "raw:v:nbg-none+A900"),
        ("Not-Background, Distance Watershed + Vote", OURS_DISTANCE),
        (
            "Not-Background, Probability Watershed, No Vote",
            "raw:v:nbg-pb-h0.01-t0+A900",
        ),
        ("Same, Every Minimum Seeds ($h{=}0$)", "raw:v:nbg-pb-h0-t0+A900"),
        ("BoundaryVote, Every Minimum Seeds ($h{=}0$)", "raw:v:nbg-pb-h0-t0.3+A900"),
        (OURS_LABEL, OURS),
        (r"\multicolumn{3}{@{}l}{\emph{Without the Minimum Parcel Size}} \\", None),
        ("FTP Pipeline ($h{=}2$)", BASE),
        ("BoundaryVote, Distance Proposals", OURS_DISTANCE.replace("+A900", "")),
        ("BoundaryVote", OURS_NOFILTER),
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
    return lines


def ablation_macros(mb3: pd.DataFrame, ms2: pd.DataFrame) -> dict[str, str]:
    """Numbers the ablation and comparison paragraphs quote."""
    vals: dict[str, str] = {}
    owt = transfer().loc["owt"].val_choice
    vals["OwtNoFilterPQ"] = f(mb3.loc[owt.replace("+A900", "")].pq)
    vals["OwtGapPS"] = f(mb3.loc[OURS].pq - mb3.loc[owt].pq)
    vals["OwtGapSS"] = f(ms2.loc[OURS].pq - ms2.loc[owt].pq)
    # the vote's two roles: accuracy at a tuned depth and rescue at h = 0
    for key, strat in (
        ("NoVotePQ", "raw:v:nbg-pb-h0.01-t0+A900"),
        ("NoVoteZeroPQ", "raw:v:nbg-pb-h0-t0+A900"),
        ("OursZeroPQ", "raw:v:nbg-pb-h0-t0.3+A900"),
        ("NbgNoneMinPQ", "raw:v:nbg-none+A900"),
        ("DistancePQ", OURS_DISTANCE),
        ("OursNoFilterPQ", OURS_NOFILTER),
    ):
        vals[key] = f(mb3.loc[strat].pq)
    dist_bare = OURS_DISTANCE.replace("+A900", "")
    vals["BaseAreaGain"] = (
        f"{100 * (mb3.loc[RELEASED_FILTER].pq - mb3.loc[BASE].pq):.1f}"
    )
    vals["DistanceAreaGain"] = (
        f"{100 * (mb3.loc[OURS_DISTANCE].pq - mb3.loc[dist_bare].pq):.1f}"
    )
    return vals


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows, vals = main_rows()
    (OUT / "rev_main_rows.tex").write_text("\n".join(rows) + "\n")

    mb3, ms2 = macro("planet_b3"), macro("s2_b7")
    (OUT / "rev_ablation_rows.tex").write_text(
        "\n".join(ablation_rows(mb3, ms2)) + "\n"
    )
    (OUT / "rev_compare_rows.tex").write_text("\n".join(compare_rows()) + "\n")
    vals.update(compare_macros())
    vals.update(ablation_macros(mb3, ms2))

    boot = pd.concat(pd.read_csv(g) for g in sorted((RES / "bootstrap").glob("*.csv")))
    vals["GainCILow"] = f"{boot[boot.quantity == 'gain'].lo.min():.1f}"
    macros = "% Generated by experiments/figures/make_tables.py\n" + "\n".join(
        f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in vals.items()
    )
    (OUT / "rev_values.tex").write_text(macros + "\n")
    print(macros)


if __name__ == "__main__":
    main()
