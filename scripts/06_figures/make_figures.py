"""Regenerate the four vector figures in the paper.

Values are the ones plotted in the submitted figures; they are recorded here
as literals so the figures rebuild without access to the run store. The
provenance of each block is given in its comment. Metrics are on the 0-100
scale throughout.

    python scripts/06_figures/make_figures.py [--out figures]
"""

import argparse
from pathlib import Path

import numpy as np

import style
from style import COL, C_BROWN, C_DEL, C_FV, C_GREY, C_TUNED, nice

style.use()
import matplotlib.pyplot as plt  # noqa: E402  (after rcParams)

# --- Data -------------------------------------------------------------------
# Per-country PQ on DENSE_10 (presence-only Kenya excluded).
# Source: multi_hmin_ensemble headline run; see scripts/02_consensus.
COUNTRIES = ["belgium", "sweden", "latvia", "south_africa", "lithuania",
             "croatia", "germany", "slovenia", "cambodia", "portugal"]
DELANY = np.array([9.2, 16.0, 9.1, 1.2, 4.7, 5.4, 0.7, 1.8, 0.0, 2.0])
TUNED = np.array([30.6, 41.7, 41.6, 45.8, 33.4, 20.5, 32.5, 14.0, 17.1, 12.9])
FIELDVOTE = np.array([52.9, 50.5, 50.2, 47.6, 46.6, 35.8, 31.2, 24.0, 9.5, 9.7])

# Uniform vote vs size-adaptive vote on the softmax-derived pipeline
# (T = 100k m^2). Source: scripts/03_refinement adaptive sweep.
UNIFORM = np.array([49.05, 49.64, 48.77, 48.24, 42.62, 30.40, 30.56, 19.99, 7.38, 7.15])
ADAPTIVE = np.array([52.03, 49.03, 48.57, 47.85, 45.12, 34.42, 30.49, 22.73, 9.32, 12.07])

# Vote-threshold sweep: macro PQ vs fraction of thresholds required to agree.
VOTE_FRAC = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
VOTE_PQ = np.array([35.97, 35.97, 35.78, 33.29, 33.29])

# Douglas-Peucker tolerance sweep, mean over the countries present at every
# tolerance: PQ and mean vertices per polygon.
SIMP_TOL = np.array([0.0, 1.0, 3.0, 6.0])
SIMP_PQ = np.array([36.61, 36.61, 36.46, 36.13])
SIMP_VERT = np.array([187.2, 187.2, 28.5, 11.4])


def dumbbell(ax, labels, lo, hi, c_lo, c_hi, lab_lo, lab_hi, extra=None,
             lab_extra=None, delta_header=r"$\Delta$"):
    """Sorted change plot: one row per country, baseline -> ours.

    Rows are ordered by the change, so gains and losses separate visually and
    the reader does not have to difference two bar lengths by eye.
    """
    d = hi - lo
    order = np.argsort(d)
    y = np.arange(len(labels))

    for i, j in enumerate(order):
        gain = d[j] >= 0
        ax.plot([lo[j], hi[j]], [i, i],
                color=C_FV if gain else C_GREY,
                lw=2.0, alpha=0.85 if gain else 0.55,
                solid_capstyle="butt", zorder=2)
    if extra is not None:
        ax.scatter(extra[order], y, s=13, color=C_DEL, zorder=3,
                   edgecolor="white", linewidth=0.4, label=lab_extra)
    # Baseline marker is smaller and sits on top, so it stays visible on the
    # rows where the change is too small to separate the two dots.
    ax.scatter(hi[order], y, s=26, color=c_hi, zorder=4,
               edgecolor="white", linewidth=0.5, label=lab_hi)
    ax.scatter(lo[order], y, s=13, color=c_lo, zorder=5,
               edgecolor="white", linewidth=0.5, label=lab_lo)

    ax.set_yticks(y)
    ax.set_yticklabels([nice(labels[j]) for j in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Panoptic Quality")
    ax.grid(axis="x")
    ax.set_ylim(-0.7, len(labels) - 0.3)

    # Signed change in a right-hand column, so the per-country effect is a
    # number rather than a length difference.
    xmax = max(lo.max(), hi.max())
    if extra is not None:
        xmax = max(xmax, extra.max())
    xcol = xmax * 1.24
    ax.set_xlim(0, xcol)
    # Grid and ticks stop at the data; the delta column is not a value axis.
    step = 10
    ax.set_xticks(np.arange(0, int(xmax / step) * step + 1, step))
    for i, j in enumerate(order):
        ax.text(xcol, i, f"{d[j]:+.1f}", va="center", ha="right", fontsize=7.5,
                color=C_FV if d[j] >= 0 else C_GREY)
    ax.text(xcol, len(labels) - 0.35, delta_header, va="center", ha="right",
            fontsize=7.5, color=C_BROWN)
    return order


def fig_percountry(out):
    fig, ax = plt.subplots(figsize=(COL, 2.75))
    dumbbell(ax, COUNTRIES, TUNED, FIELDVOTE, C_TUNED, C_FV,
             "Tuned watershed", "FieldVote", extra=DELANY,
             lab_extra="DelineateAnything")
    h, l = ax.get_legend_handles_labels()
    ax.legend([h[l.index(k)] for k in ("DelineateAnything", "Tuned watershed", "FieldVote")],
              ["DelineateAnything", "Tuned watershed", "FieldVote"],
              loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3,
              handletextpad=0.3, columnspacing=1.0)
    fig.savefig(out / "fig_percountry.pdf")
    plt.close(fig)


def fig_adaptive(out):
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    dumbbell(ax, COUNTRIES, UNIFORM, ADAPTIVE, C_TUNED, C_FV,
             "Uniform vote", "Size-adaptive")
    h, l = ax.get_legend_handles_labels()
    keys = ["Uniform vote", "Size-adaptive"]
    ax.legend([h[l.index(k)] for k in keys], keys,
              loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=2,
              handletextpad=0.3, columnspacing=1.2)
    fig.savefig(out / "fig_adaptive.pdf")
    plt.close(fig)


def fig_votethreshold(out):
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    ax.axvspan(0.28, 0.52, color=C_TUNED, alpha=0.10, lw=0)
    ax.plot(VOTE_FRAC, VOTE_PQ, "-", color=C_FV, zorder=3)
    ax.plot(VOTE_FRAC, VOTE_PQ, "o", color=C_FV, ms=4, zorder=4,
            markeredgecolor="white", markeredgewidth=0.5)

    # Mark the operating point used everywhere else in the paper with a rule
    # to the axis, so no leader line crosses the curve.
    ax.axvline(0.5, color=C_BROWN, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.text(0.49, 33.05, "majority vote,\nused throughout", fontsize=7,
            color=C_BROWN, ha="right", va="bottom")
    ax.text(0.40, 36.35, "flat region", fontsize=7, color=C_TUNED,
            ha="center", va="center")
    ax.set_xlabel(r"Vote threshold $\tau_v$ (fraction of thresholds agreeing)")
    ax.set_ylabel("Macro PQ")
    ax.set_xticks(VOTE_FRAC)
    ax.set_ylim(32.8, 36.6)
    ax.grid(axis="y")
    fig.savefig(out / "fig_votethreshold.pdf")
    plt.close(fig)


def fig_simplify(out):
    fig, ax = plt.subplots(figsize=(COL, 2.05))

    # Vertices on a log axis: the sixteen-fold drop is the story.
    ax.plot(SIMP_TOL, SIMP_VERT, "s--", color=C_BROWN, ms=3.6,
            label="Vertices / polygon (left)")
    ax.set_yscale("log")
    ax.set_ylim(8, 400)
    ax.set_yticks([10, 30, 100, 300])
    ax.set_yticklabels(["10", "30", "100", "300"])
    ax.set_ylabel("Vertices / polygon", color=C_BROWN)
    ax.set_xlabel("Douglas–Peucker tolerance (m)")
    ax.set_xticks(SIMP_TOL)
    ax.grid(axis="y")
    for x, v in zip(SIMP_TOL, SIMP_VERT):
        if x in (0.0, 6.0):
            ax.annotate(f"{v:.0f}", (x, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color=C_BROWN)

    # PQ on a wide, untruncated axis: the point is that it does not move.
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(SIMP_TOL, SIMP_PQ, "o-", color=C_FV, ms=3.6,
             label="Panoptic Quality (right)")
    ax2.set_ylim(28, 40)
    ax2.set_yticks([30, 34, 38])
    ax2.set_ylabel("Panoptic Quality", color=C_FV)
    ax2.tick_params(axis="y", colors=C_FV)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C_FV)
    ax2.annotate(f"{SIMP_PQ[0]:.1f}", (0.0, SIMP_PQ[0]), textcoords="offset points",
                 xytext=(2, -10), ha="left", fontsize=7, color=C_FV)
    ax2.annotate(f"{SIMP_PQ[-1]:.1f}", (6.0, SIMP_PQ[-1]), textcoords="offset points",
                 xytext=(-2, -10), ha="right", fontsize=7, color=C_FV)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, handletextpad=0.4, columnspacing=1.0)
    fig.savefig(out / "fig_simplify.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_percountry(args.out)
    fig_adaptive(args.out)
    fig_votethreshold(args.out)
    fig_simplify(args.out)
    print("macro PQ  tuned %.1f  fieldvote %.1f" % (TUNED.mean(), FIELDVOTE.mean()))
    print("macro PQ  uniform %.1f  adaptive %.1f" % (UNIFORM.mean(), ADAPTIVE.mean()))
    print("figures written to", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
