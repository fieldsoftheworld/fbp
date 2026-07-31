"""Re-score FieldVote (consensus) and the tuned watershed against the TRUE
vector ground-truth polygons (per-patch <pid>.polygons.parquet from the public
HF ftw-planet release), on Dense-10.

GT matching is spatial: each parquet ships a .json sidecar with bounds_4326;
for each of our test patches we take the label raster bounds, transform to
EPSG:4326, and pick the GT patch with maximum bounds-IoU. Polygons are read in
their native UTM CRS (same zone as our predictions) and clipped to the patch
box. Matching/PQ reuse the validated code verbatim. Read-only inputs; writes
one CSV to fieldvote.
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely.geometry
from rasterio.warp import transform_bounds

REPO = Path("/u/agiri1/github/ftw-planet")
sys.path.insert(0, str(REPO / "scripts/inference"))
import multi_hmin_ensemble as mhe  # noqa: E402

GTROOT = Path("/u/agiri1/fieldvote/gt_parquet")
OUT = Path("/u/agiri1/fieldvote/vector_gt_rescore.csv")
DENSE10 = ["belgium", "sweden", "latvia", "south_africa", "lithuania",
           "croatia", "germany", "slovenia", "cambodia", "portugal"]


def bounds_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt_index(country):
    idx = []
    for j in sorted((GTROOT / country).glob("*.json")):
        meta = json.loads(j.read_text())
        idx.append((tuple(meta["bounds_4326"]), j.with_suffix("").name, meta["crs"]))
    return idx


def gt_shapes_for(country, sample, gt_index):
    with rasterio.open(sample["label"]) as src:
        crs, tfm = src.crs, src.transform
        H, W = src.height, src.width
    x0, y0 = tfm * (0, 0)
    x1, y1 = tfm * (W, H)
    box_native = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    b4326 = transform_bounds(crs, "EPSG:4326", *box_native)
    best, best_iou = None, 0.0
    for gb, stem, gcrs in gt_index:
        i = bounds_iou(b4326, gb)
        if i > best_iou:
            best, best_iou = (stem, gcrs), i
    if best is None or best_iou < 0.25:
        return None, best_iou
    stem, gcrs = best
    pq_file = GTROOT / country / f"{stem}.polygons.parquet"
    g = gpd.read_parquet(pq_file)
    if len(g) == 0:
        return [], best_iou
    if str(g.crs).upper() != str(crs).upper():
        g = g.to_crs(crs)
    clip = shapely.geometry.box(*box_native)
    out = []
    for geom in g.geometry:
        if geom is None or geom.is_empty:
            continue
        gg = geom.intersection(clip)
        if gg.is_empty:
            continue
        if gg.geom_type == "Polygon":
            out.append(gg)
        elif gg.geom_type == "MultiPolygon":
            out.extend(list(gg.geoms))
    return out, best_iou


def score(polys_by_patch, samples, country, gt_index):
    counts = {t: [0, 0, 0] for t in mhe.AP_IOU_THRESHOLDS}
    matched = []
    npat = skipped = 0
    npred_l, ngt_l, pdelta = [], [], []
    for s in samples:
        pid = s.get("patch_id") or Path(s["window_a"]).stem
        gt, _ = gt_shapes_for(country, s, gt_index)
        if gt is None:
            skipped += 1
            continue
        pred = polys_by_patch.get(pid, [])
        if not pred and not gt:
            continue
        npat += 1
        npred_l.append(len(pred))
        ngt_l.append(len(gt))
        pdelta.append(abs(len(pred) - len(gt)))
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
    ap = float(np.mean([
        (lambda tp2, fp2, fn2: (
            (2 * (tp2 / max(tp2 + fp2, 1)) * (tp2 / max(tp2 + fn2, 1)))
            / max((tp2 / max(tp2 + fp2, 1)) + (tp2 / max(tp2 + fn2, 1)), 1e-9)
        ))(*counts[t]) for t in mhe.AP_IOU_THRESHOLDS
    ]))
    return dict(pq=sq * rq, sq=sq, rq=rq, ap=ap, n=npat, skipped=skipped,
                dn=float(np.mean(pdelta)) if pdelta else 0.0,
                npred=float(np.mean(npred_l)) if npred_l else 0.0,
                ngt=float(np.mean(ngt_l)) if ngt_l else 0.0)


rows = []
print(f"{'country':14s} {'FV_pq':>6s} {'TW_pq':>6s} {'FV_sq':>6s} {'FV_rq':>6s} "
      f"{'npred':>6s} {'ngt':>6s} {'bestH':>5s} {'n':>4s} {'skip':>4s}")
for c in DENSE10:
    gt_index = load_gt_index(c)
    if not gt_index:
        print(f"{c:14s} NO GT PARQUET - skipping")
        continue
    gjs = mhe.find_geojsons(c, REPO / "logs/inference")
    pph = {h: mhe.load_polygons_by_patch(g) for h, g in gjs.items()}
    ds = mhe.FTWPlanet(root="data", countries=[c], split="test",
                       transforms=None, load_boundaries=True)
    samples = ds.records
    cons = mhe.strategy_c_pixel_vote(pph, samples, threshold_frac=0.5)
    mc = score(cons, samples, c, gt_index)
    per_h = {h: score(pph[h], samples, c, gt_index) for h in pph}
    best_h = max(per_h, key=lambda h: per_h[h]["pq"])
    ms = per_h[best_h]
    rows.append((c, mc, ms, best_h))
    print(f"{c:14s} {mc['pq']*100:6.1f} {ms['pq']*100:6.1f} {mc['sq']*100:6.1f} "
          f"{mc['rq']*100:6.1f} {mc['npred']:6.1f} {mc['ngt']:6.1f} {best_h:>5} "
          f"{mc['n']:4d} {mc['skipped']:4d}")

fv = np.mean([r[1]["pq"] for r in rows]) * 100
tw = np.mean([r[2]["pq"] for r in rows]) * 100
fvsq = np.mean([r[1]["sq"] for r in rows]) * 100
fvrq = np.mean([r[1]["rq"] for r in rows]) * 100
fvap = np.mean([r[1]["ap"] for r in rows]) * 100
fvdn = np.mean([r[1]["dn"] for r in rows])
twdn = np.mean([r[2]["dn"] for r in rows])
print()
print(f"DENSE-10 MACRO (VECTOR GT): FieldVote PQ {fv:.1f} (SQ {fvsq:.1f} RQ {fvrq:.1f} AP {fvap:.1f})  "
      f"TunedWS PQ {tw:.1f}")
print(f"polygon-count delta |Npred-Ngt| per patch: FieldVote {fvdn:.1f}  TunedWS {twdn:.1f}")
print("reference footings: semantic-mask 35.8 / 29.0; instance-mask 30.0 / 24.4")

import csv
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["country", "fv_pq", "fv_sq", "fv_rq", "fv_ap", "tw_pq", "best_h",
                "npred_mean", "ngt_mean", "fv_dn", "tw_dn", "n_patches", "skipped"])
    for c, mc, ms, bh in rows:
        w.writerow([c, mc["pq"], mc["sq"], mc["rq"], mc["ap"], ms["pq"], bh,
                    mc["npred"], mc["ngt"], mc["dn"], ms["dn"], mc["n"], mc["skipped"]])
print("wrote", OUT)
