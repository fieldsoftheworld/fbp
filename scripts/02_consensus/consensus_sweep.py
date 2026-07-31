"""Consensus vote-threshold ablation: sweep the fraction of watershed thresholds
that must agree for a pixel to be labeled field, from 0.3 to 0.7. Tests whether
the 50 percent majority is the right operating point. Reuses the validated
strategy_c_pixel_vote and scorer from the ensemble script.
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path("/u/agiri1/github/ftw-planet")
sys.path.insert(0, str(REPO / "scripts/inference"))
import multi_hmin_ensemble as mhe  # noqa: E402

country = sys.argv[1]
out_dir = Path("/u/agiri1/fieldvote/consensus_sweep")
out_dir.mkdir(parents=True, exist_ok=True)

gjs = mhe.find_geojsons(country, REPO / "logs/inference")
polys = {h: mhe.load_polygons_by_patch(g) for h, g in gjs.items()}
ds = mhe.FTWPlanet(root="data", countries=[country], split="test",
                   transforms=None, load_boundaries=True)
samples = ds.records

rows = []
for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
    ens = mhe.strategy_c_pixel_vote(polys, samples, threshold_frac=frac)
    m = mhe.score_polygons_against_gt(ens, samples)
    rows.append({"country": country, "frac": frac, "pq": m["pq"],
                 "pq_sq": m["pq_sq"], "pq_rq": m["pq_rq"], "ap_5_95": m["ap_5_95"]})
    print(f"{country} frac={frac}: PQ={m['pq']:.4f}")

pd.DataFrame(rows).to_csv(out_dir / f"{country}.csv", index=False)
print(f"wrote {out_dir / f'{country}.csv'}")
