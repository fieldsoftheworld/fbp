"""Choose the gallery patches and freeze every figure input into the repo.

Gallery rows cover all ten dense-label countries on PlanetScope B3 except
Portugal (two test patches), eight countries on Sentinel-2 B7, and one failure
case. Patch choice per (model, country): the largest per-patch PQ gain
(BoundaryVote minus FTP pipeline) among test patches with 10-120 reference
parcels whose densest square window shows at least eight of them covering at
least a quarter of the window. The method figure's three crops are fixed in
``method_fig.EXAMPLES``.

Writes ``results/figure_inputs/gallery.json`` and one npz per patch under
``results/figure_inputs/<run>/<country>/``, and removes bundles no figure asks
for any more. Needs the prediction cache, the per-patch scores and the
ftw-planet data; the figure scripts themselves need only the frozen files.

Regenerate: ~/github/ftw-planet/.venv/bin/python -m fbp.paper.figures.freeze
"""

import numpy as np
import pandas as pd

from fbp.eval import references as refs
from fbp.eval.paths import DENSE10, cache_dir, scores_dir
from fbp.paper.figures import inputs
from fbp.paper.figures.gallery import best_window
from fbp.paper.figures.method_fig import EXAMPLES as METHOD_EXAMPLES
from fbp.paper.shipped import FTP as BASE
from fbp.paper.shipped import OURS

S2_COUNTRIES = [
    "belgium",
    "cambodia",
    "croatia",
    "germany",
    "latvia",
    "lithuania",
    "slovenia",
    "sweden",
]
FAILURE = ("planet_b3", "latvia", "g26_00052_7")


def per_patch_delta(run: str, proto: str) -> pd.DataFrame:
    """Per-patch PQ of the FTP pipeline and BoundaryVote, and their difference."""
    fs = scores_dir(f"{run}/test/{proto}").glob("patches_*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in fs])
    df = df[(df.reference == "vector") & df.strategy.isin([BASE, OURS])].copy()
    tp, fp, fn = df.tp_50, df.fp_50, df.fn_50
    p = tp / (tp + fp).clip(lower=1)
    r = tp / (tp + fn).clip(lower=1)
    f1 = (2 * p * r / (p + r).where((p + r) > 0, 1)).fillna(0)
    df["pq"] = (df.iou_sum_50 / tp.clip(lower=1)) * f1
    w = df.pivot_table(
        index=["country", "patch_id"], columns="strategy", values=["pq", "n_gt"]
    )
    w.columns = ["_".join(c) for c in w.columns]
    w["d"] = w[f"pq_{OURS}"] - w[f"pq_{BASE}"]
    w["n_gt"] = w[f"n_gt_{BASE}"]
    return w.reset_index()


def patch_pq(w: pd.DataFrame, country: str, pid: str) -> tuple[float, float]:
    """Full-patch PQ in percent: (FTP pipeline, BoundaryVote)."""
    r = w[(w.country == country) & (w.patch_id == pid)].iloc[0]
    return 100 * r[f"pq_{BASE}"], 100 * r[f"pq_{OURS}"]


def pick(w: pd.DataFrame, run: str, country: str, min_in_crop: int = 8) -> str:
    """Largest per-patch gain whose best square window still shows >= min_in_crop parcels."""
    c = w[(w.country == country) & (w.n_gt >= 10) & (w.n_gt <= 120)]
    if c.empty:
        c = w[w.country == country]
    c = c.sort_values("d", ascending=False)
    for pid in c.patch_id.head(25):
        npz = cache_dir(f"{run}/test") / country / f"{pid}.npz"
        gt = refs.gt_raster(refs.load_grid(npz, country))
        crop = gt[best_window(gt)]
        if len(np.unique(crop[crop > 0])) >= min_in_crop and (crop > 0).mean() >= 0.25:
            return pid
    return c.iloc[0].patch_id


def main() -> None:
    wp = per_patch_delta("planet_b3", "pixel")
    ws = per_patch_delta("s2_b7", "pixel")
    rows = [
        ("planet_b3", c, pick(wp, "planet_b3", c), "PS")
        for c in DENSE10
        if c != "portugal"
    ]
    rows += [("s2_b7", c, pick(ws, "s2_b7", c), "S2") for c in S2_COUNTRIES]
    rows.append((*FAILURE, "PS, Failure"))
    picks = []
    for run, country, pid, tag in rows:
        pq_b, pq_o = patch_pq(wp if run == "planet_b3" else ws, country, pid)
        picks.append(
            {
                "run": run,
                "country": country,
                "patch_id": pid,
                "tag": tag,
                "pq_base": round(float(pq_b), 2),
                "pq_ours": round(float(pq_o), 2),
            }
        )
    inputs.write_picks(picks)

    wanted = {(r, c, p) for r, c, p, _ in rows}
    wanted |= {(r, c, p) for r, c, p, _, _ in METHOD_EXAMPLES}
    keep = {inputs.path(*k) for k in wanted}
    for stale in inputs.FROZEN.rglob("*.npz"):
        if stale not in keep:
            stale.unlink()
            print("removed", stale)
    for key in sorted(wanted):
        print(inputs.freeze_patch(*key))


if __name__ == "__main__":
    main()
