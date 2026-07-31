"""Re-score FieldVote (consensus) and the tuned watershed (best single h per
country) against PER-FIELD instance-mask ground truth, on the Dense-10 set.

Motivation: the headline scorer takes GT as connected components of the semantic
field class (gt==1), which MERGES touching fields (Isaac's warning). Here we
instead take the FTW instance mask (unique id per field), NN-reproject it onto
the Planet patch grid, and polygonize per id -> per-field GT with no merge.
Everything else (consensus rule, shape matching, PQ = SQ x RQ) is reused verbatim
from multi_hmin_ensemble so the comparison is apples-to-apples.

Read-only. Prints Dense-10 macro consensus vs tuned-watershed and compares to the
paper's 35.8 / 29.0 (which used the semantic-GT footing).
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import shapely.geometry
from rasterio.warp import Resampling, reproject

REPO = Path("/u/agiri1/github/ftw-planet")
sys.path.insert(0, str(REPO / "scripts/inference"))
import multi_hmin_ensemble as mhe  # noqa: E402

DENSE10 = ["belgium", "sweden", "latvia", "south_africa", "lithuania",
           "croatia", "germany", "slovenia", "cambodia", "portugal"]
FTW = REPO / "data/ftw"


def instance_gt_shapes(country, pid, ref_crs, ref_transform, ref_shape):
    """FTW instance mask -> NN-reprojected to the Planet grid -> per-field polys."""
    p = FTW / country / "label_masks" / "instance" / f"{pid}.tif"
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        src_arr = src.read(1).astype(np.int32)
        src_crs, src_transform = src.crs, src.transform
    dst = np.zeros(ref_shape, dtype=np.int32)
    reproject(source=src_arr, destination=dst,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=ref_transform, dst_crs=ref_crs,
              resampling=Resampling.nearest)
    shapes = []
    for geom, val in rasterio.features.shapes(dst, mask=(dst > 0), transform=ref_transform):
        if val == 0:
            continue
        g = shapely.geometry.shape(geom)
        if g.is_valid and g.area > 0:
            shapes.append(g)
    return shapes


def score(polys_by_patch, samples, country):
    counts = {t: [0, 0, 0] for t in mhe.AP_IOU_THRESHOLDS}
    matched = []
    npat = 0
    for s in samples:
        pid = s.get("patch_id") or Path(s["window_a"]).stem
        with rasterio.open(s["label"]) as src:
            ref = (src.crs, src.transform, (src.height, src.width))
        gt = instance_gt_shapes(country, pid, *ref)
        if gt is None:
            continue
        pred = polys_by_patch.get(pid, [])
        if not pred and not gt:
            continue
        npat += 1
        m = mhe._match_shapes(gt, pred, mhe.AP_IOU_THRESHOLDS)
        for t, (tp, fp, fn) in m["per_t"].items():
            counts[t][0] += tp
            counts[t][1] += fp
            counts[t][2] += fn
        for _i, _j, iou in m["matched_pairs_low"]:
            matched.append(iou)
    t05 = mhe.AP_IOU_THRESHOLDS[0]
    tp, fp, fn = counts[t05]
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    rq = (2 * p * r / (p + r)) if (p + r) else 0.0
    sq = float(np.mean(matched)) if matched else 0.0
    return sq * rq, npat


cons_macro, single_macro = [], []
print(f"{'country':14s} {'consensus':>10s} {'tuned-ws':>10s}  best_h  n")
for c in DENSE10:
    gjs = mhe.find_geojsons(c, REPO / "logs/inference")
    polys_per_h = {h: mhe.load_polygons_by_patch(g) for h, g in gjs.items()}
    ds = mhe.FTWPlanet(root="data", countries=[c], split="test",
                       transforms=None, load_boundaries=True)
    samples = ds.records
    cons = mhe.strategy_c_pixel_vote(polys_per_h, samples, threshold_frac=0.5)
    pq_c, npat = score(cons, samples, c)
    pq_h = {h: score(polys_per_h[h], samples, c)[0] for h in polys_per_h}
    best_h = max(pq_h, key=pq_h.get)
    pq_s = pq_h[best_h]
    cons_macro.append(pq_c)
    single_macro.append(pq_s)
    print(f"{c:14s} {pq_c*100:10.1f} {pq_s*100:10.1f}  {best_h:>5}  {npat}")

print()
print(f"DENSE-10 MACRO (instance-mask per-field GT):  "
      f"consensus {np.mean(cons_macro)*100:.1f}   tuned-watershed {np.mean(single_macro)*100:.1f}")
print(f"paper headline (semantic-GT footing):         consensus 35.8   tuned-watershed 29.0")
