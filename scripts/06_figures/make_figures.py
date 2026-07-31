"""Regenerate the paper figures: Taylor Geospatial brand colors, serif (Times)
font, bold publication styling, metrics on the 0-100 scale, FieldVote/tuned-
watershed legend labels. Writes vector PDFs to analysis/."""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12, "axes.labelweight": "bold", "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.6,
    "grid.color": "#b8b8ac", "axes.axisbelow": True,
    "figure.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
    "lines.linewidth": 2.5, "lines.markersize": 6,
})
OUT = "/u/agiri1/fieldvote/analysis"
# Taylor Geospatial brand palette
C_DEL = "#a7d0dc"    # light blue - DelineateAnything (weakest baseline)
C_TUNED = "#80a0d8"  # periwinkle - tuned watershed / uniform vote
C_FV = "#ff4f2c"     # red        - FieldVote (ours, highlight)
C_BROWN = "#3b1e1c"  # brown      - secondary series (vertices)


def nice(names):
    return [n.replace("_", " ").title() for n in names]


def pq(f, frag):
    d = pd.read_csv(f)
    r = d[d.geojson.str.contains(frag)]
    return float(r.iloc[0].pq) * 100 if len(r) else np.nan


# --- Fig: per-country (DENSE_10, presence-only Kenya excluded) ---
co = ["belgium", "sweden", "latvia", "south_africa", "lithuania", "croatia",
      "germany", "slovenia", "cambodia", "portugal"]
delany = np.array([.092, .160, .091, .012, .047, .054, .007, .018, .0, .020]) * 100
single = np.array([.306, .417, .416, .458, .334, .205, .325, .140, .171, .129]) * 100
cons = np.array([.529, .505, .502, .476, .466, .358, .312, .240, .095, .097]) * 100
order = np.argsort(cons)
y = np.arange(len(co))
h = 0.27
fig, ax = plt.subplots(figsize=(3.6, 4.4))
ax.barh(y + h, delany[order], h, label="DelineateAnything", color=C_DEL)
ax.barh(y, single[order], h, label="Tuned watershed", color=C_TUNED)
ax.barh(y - h, cons[order], h, label="FieldVote", color=C_FV)
ax.set_yticks(y)
ax.set_yticklabels([nice(co)[i] for i in order])
ax.set_xlabel("Panoptic Quality")
ax.grid(axis="y", alpha=0)
ax.legend(loc="lower right")
fig.savefig(f"{OUT}/fig_percountry.pdf")
plt.close(fig)

# --- Fig: vote-threshold ---
rows = [pd.read_csv(f) for f in glob.glob(f"{OUT}/../consensus_sweep/*.csv")]
if rows:
    sweep = pd.concat(rows)
    if "country" in sweep.columns:
        sweep = sweep[sweep["country"] != "kenya"]
    macro = sweep.groupby("frac").pq.mean() * 100
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(macro.index, macro.values, "o-", color=C_FV)
    ax.set_xlabel("Vote threshold (fraction agreeing)")
    ax.set_ylabel("Macro PQ")
    fig.savefig(f"{OUT}/fig_votethreshold.pdf")
    plt.close(fig)

# --- Fig: simplify tradeoff ---
TOLS = [0, 1, 3, 6]
DIRS = {0: "pp_tol0", 1: "pp_tol1", 3: "pp2", 6: "pp_tol6"}
base = "/u/agiri1/fieldvote"
sets = [{os.path.basename(f).replace("metrics_", "").replace(".csv", "")
         for f in glob.glob(f"{base}/{DIRS[t]}/metrics_*.csv")} for t in TOLS]
common = sorted(set.intersection(*sets))
pqv, vv = [], []
for t in TOLS:
    pqs = [pq(f"{base}/{DIRS[t]}/metrics_{c}.csv", "_pp1_simplify.geojson") for c in common]
    vs = []
    for c in common:
        vc = pd.read_csv(f"{base}/{DIRS[t]}/vertices_{c}.csv")
        rr = vc[vc.stage == "1_simplify"]
        vs.append(float(rr.iloc[0].mean_vertices) if len(rr) else np.nan)
    pqv.append(np.nanmean(pqs))
    vv.append(np.nanmean(vs))
fig, ax1 = plt.subplots(figsize=(3.4, 2.7))
ax1.plot(TOLS, pqv, "o-", color=C_FV, label="PQ")
ax1.set_xlabel("Simplification tolerance (m)")
ax1.set_ylabel("Panoptic Quality", color=C_FV)
ax1.tick_params(axis="y", colors=C_FV)
ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(TOLS, vv, "s--", color=C_BROWN, label="vertices")
ax2.set_ylabel("Vertices / polygon", color=C_BROWN)
ax2.tick_params(axis="y", colors=C_BROWN)
ax2.grid(False)
fig.savefig(f"{OUT}/fig_simplify.pdf")
plt.close(fig)

# --- Fig: size-adaptive vs uniform vote (per country) ---
hv, ad, names = [], [], []
for c in co:
    fm = f"{base}/msearch2/metrics_{c}.csv"
    fa = f"{base}/adapt/metrics_{c}.csv"
    if os.path.exists(fm) and os.path.exists(fa):
        hv.append(pq(fm, c + "_v0_hardvote"))
        ad.append(pq(fa, c + "_a_T100k"))
        names.append(c)
if names:
    yy = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.6, 4.0))
    ax.barh(yy + 0.2, hv, 0.4, label="Uniform vote", color=C_TUNED)
    ax.barh(yy - 0.2, ad, 0.4, label="Size-adaptive FieldVote", color=C_FV)
    ax.set_yticks(yy)
    ax.set_yticklabels(nice(names))
    ax.grid(axis="y", alpha=0)
    ax.set_xlabel("Panoptic Quality")
    ax.legend(loc="lower right")
    fig.savefig(f"{OUT}/fig_adaptive.pdf")
    plt.close(fig)
    print("adaptive fig: hardvote macro %.1f adaptive macro %.1f" % (np.nanmean(hv), np.nanmean(ad)))

print("figures written to", OUT)
