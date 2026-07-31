"""Compute macro PQ / SQ / RQ / AP and mean polygon counts for FieldVote
(consensus) and the tuned watershed (best single h per country) over Dense-10,
on the SAME semantic-GT footing as the 35.8 headline. Reuses the validated
scorer verbatim. Read-only."""

import sys
from pathlib import Path

import numpy as np

REPO = Path("/u/agiri1/github/ftw-planet")
sys.path.insert(0, str(REPO / "scripts/inference"))
import multi_hmin_ensemble as mhe  # noqa: E402

DENSE10 = ["belgium", "sweden", "latvia", "south_africa", "lithuania",
           "croatia", "germany", "slovenia", "cambodia", "portugal"]

cons_m, single_m = [], []
for c in DENSE10:
    gjs = mhe.find_geojsons(c, REPO / "logs/inference")
    pph = {h: mhe.load_polygons_by_patch(g) for h, g in gjs.items()}
    ds = mhe.FTWPlanet(root="data", countries=[c], split="test",
                       transforms=None, load_boundaries=True)
    samples = ds.records
    cons = mhe.strategy_c_pixel_vote(pph, samples, threshold_frac=0.5)
    cons_m.append(mhe.score_polygons_against_gt(cons, samples))
    best = None
    for h in pph:
        m = mhe.score_polygons_against_gt(pph[h], samples)
        if best is None or m["pq"] > best["pq"]:
            best = m
    single_m.append(best)


def macro(ms, k):
    return float(np.mean([m[k] for m in ms]))


print(f"{'method':16s} {'PQ':>6s} {'SQ':>6s} {'RQ':>6s} {'AP':>6s} "
      f"{'npred':>7s} {'ngt':>7s} {'dN':>6s}")
for name, ms in [("FieldVote", cons_m), ("Tuned watershed", single_m)]:
    npred = macro(ms, "n_pred_mean")
    ngt = macro(ms, "n_gt_mean")
    print(f"{name:16s} {macro(ms,'pq')*100:6.1f} {macro(ms,'pq_sq')*100:6.1f} "
          f"{macro(ms,'pq_rq')*100:6.1f} {macro(ms,'ap_5_95')*100:6.1f} "
          f"{npred:7.1f} {ngt:7.1f} {abs(npred-ngt):6.1f}")
