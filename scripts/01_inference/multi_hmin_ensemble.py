"""Multi-h_min polygon ensembling for FTW-Planet field segmentation.

Implements 4 strategies that combine existing GeoJSON outputs across
multiple h_min values into a single per-patch prediction set:

  A. NMS-pool with agreement weighting — pool polygons from all h_min
     values, score each by how many h_min outputs have a similar
     polygon (multi-scale stability selection / persistence-style),
     then greedy NMS by IoU>=0.5.
  B. Oracle h_min — per patch, pick the h_min whose polygons best
     match GT (upper bound; not deployable).
  C. Pixel-vote consensus — rasterize each h_min's polygons to GT grid,
     vote per pixel, polygonize the majority mask.
  D. n_pred elbow auto-select — per patch, pick the h_min on the
     "plateau" of the n_pred(h_min) curve (self-supervised).

Reads:
  - logs/inference/<country>_fields_hmin{N}.geojson for h_min in
    {1, 2, 4, 8, 12, 16, 24, 32, 48, 64} where present
  - FTW v2 GT via FTWPlanet (for scoring + as reference grid for C)

Writes:
  - logs/inference/<country>_ensemble_summary.csv (one row per strategy)

DRAFT (untracked).
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import shapely.geometry

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from score_country_geojson import _crs_match, _extract_shapes_with_transform  # noqa: E402

sys.path.insert(0, str(_HERE.parent))
from polygon_metrics_eval import (  # noqa: E402
    AP_IOU_THRESHOLDS,
    _boundary_pixels,
    _match_shapes,
    _symmetric_chamfer,
)
from ftw_planet.datasets import FTWPlanet  # noqa: E402


H_MIN_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]


def _h_min_suffix(h_min: float) -> str:
    if h_min == int(h_min):
        return str(int(h_min))
    return str(h_min).replace(".", "p")


def find_geojsons(country: str, root: Path) -> dict[float, Path]:
    """Find all GeoJSONs for a country across h_min values."""
    found: dict[float, Path] = {}
    if country == "rwanda":
        baseline = root / "rwanda_fields.geojson"
    else:
        baseline = root / f"{country}_fields_full.geojson"
    if baseline.exists():
        found[2.0] = baseline
    for h_min in H_MIN_VALUES:
        if h_min == 2.0:
            continue
        suffix = _h_min_suffix(h_min)
        p = root / f"{country}_fields_hmin{suffix}.geojson"
        if p.exists():
            found[h_min] = p
    return found


def load_polygons_by_patch(geojson_path: Path) -> dict[str, list]:
    """patch_id -> list of shapely polygons."""
    gdf = gpd.read_file(geojson_path)
    out: dict[str, list] = {}
    for _, row in gdf.iterrows():
        out.setdefault(row["patch_id"], []).append(row.geometry)
    return out


def score_polygons_against_gt(
    polygons_by_patch: dict[str, list],
    samples: list[dict],
    skip_chamfer: bool = True,
    gsd_m: float = 3.0,
) -> dict:
    """Score a per-patch prediction set against FTW v2 GT.

    Returns the same metric dict as score_country_geojson.
    """
    counts: dict[float, list[int]] = {t: [0, 0, 0] for t in AP_IOU_THRESHOLDS}
    matched_ious_05: list[float] = []
    chamfer_pixels: list[float] = []
    polygon_deltas: list[int] = []
    n_pred_per_patch: list[int] = []
    n_gt_per_patch: list[int] = []
    n_patches = 0
    missing_predictions = 0

    for sample in samples:
        patch_id = sample.get("patch_id") or Path(sample["window_a"]).stem
        with rasterio.open(sample["label"]) as src:
            gt = src.read(1).astype(np.uint8)
            transform = src.transform
        gt_field = (gt == 1).astype(np.uint8)
        gt_shapes = _extract_shapes_with_transform(gt_field, transform, value=1)
        pred_shapes = polygons_by_patch.get(patch_id, [])

        if not pred_shapes and not gt_shapes:
            continue
        if not pred_shapes:
            missing_predictions += 1

        n_patches += 1
        n_pred_per_patch.append(len(pred_shapes))
        n_gt_per_patch.append(len(gt_shapes))
        polygon_deltas.append(abs(len(pred_shapes) - len(gt_shapes)))

        m = _match_shapes(gt_shapes, pred_shapes, AP_IOU_THRESHOLDS)
        for t, (tps, fps, fns) in m["per_t"].items():
            counts[t][0] += tps
            counts[t][1] += fps
            counts[t][2] += fns
        for _i, _j, iou in m["matched_pairs_low"]:
            matched_ious_05.append(iou)

    def _f1(tps: int, fps: int, fns: int) -> float:
        p = tps / max(tps + fps, 1)
        r = tps / max(tps + fns, 1)
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    t05 = AP_IOU_THRESHOLDS[0]
    tps, fps, fns = counts[t05]
    rq = _f1(tps, fps, fns)
    sq = float(np.mean(matched_ious_05)) if matched_ious_05 else 0.0
    pq = sq * rq
    ap = float(np.mean([_f1(*counts[t]) for t in AP_IOU_THRESHOLDS]))

    return {
        "n_patches": n_patches,
        "pq_sq": sq,
        "pq_rq": rq,
        "pq": pq,
        "ap_5_95": ap,
        "n_pred_mean": float(np.mean(n_pred_per_patch)) if n_pred_per_patch else 0.0,
        "n_gt_mean": float(np.mean(n_gt_per_patch)) if n_gt_per_patch else 0.0,
        "missing_predictions": missing_predictions,
    }


# ------------------------------ STRATEGY A: NMS-pool ------------------------------


def strategy_a_nms_pool(
    polygons_per_h_min: dict[float, dict[str, list]],
    iou_thresh: float = 0.5,
) -> dict[str, list]:
    """Pool polygons across h_min, score by agreement, greedy NMS."""
    out: dict[str, list] = {}
    all_patches = set()
    for h_min in polygons_per_h_min:
        all_patches.update(polygons_per_h_min[h_min].keys())

    for patch_id in all_patches:
        # Pool polygons + their source h_min
        pooled: list[tuple[shapely.geometry.base.BaseGeometry, float]] = []
        for h_min, by_patch in polygons_per_h_min.items():
            for poly in by_patch.get(patch_id, []):
                pooled.append((poly, h_min))
        if not pooled:
            continue

        # Compute agreement count for each polygon
        scores: list[float] = []
        for i, (poly_i, _) in enumerate(pooled):
            count = 0
            for j, (poly_j, _) in enumerate(pooled):
                if i == j or not poly_i.intersects(poly_j):
                    continue
                inter = poly_i.intersection(poly_j).area
                if inter <= 0:
                    continue
                union = poly_i.area + poly_j.area - inter
                if union <= 0:
                    continue
                if inter / union >= iou_thresh:
                    count += 1
            scores.append(float(count))

        # Greedy NMS by score descending
        order = np.argsort(scores)[::-1]
        kept: list[shapely.geometry.base.BaseGeometry] = []
        for idx in order:
            poly = pooled[idx][0]
            ok = True
            for k in kept:
                if not poly.intersects(k):
                    continue
                inter = poly.intersection(k).area
                if inter <= 0:
                    continue
                union = poly.area + k.area - inter
                if union > 0 and inter / union >= iou_thresh:
                    ok = False
                    break
            if ok:
                kept.append(poly)

        out[patch_id] = kept
    return out


# ------------------------------ STRATEGY B: Oracle h_min ------------------------------


def strategy_b_oracle(
    polygons_per_h_min: dict[float, dict[str, list]],
    samples: list[dict],
) -> dict[str, list]:
    """Per patch, pick h_min with highest per-patch PQ. Uses GT — UPPER BOUND ONLY."""
    out: dict[str, list] = {}
    for sample in samples:
        patch_id = sample.get("patch_id") or Path(sample["window_a"]).stem
        with rasterio.open(sample["label"]) as src:
            gt = src.read(1).astype(np.uint8)
            transform = src.transform
        gt_field = (gt == 1).astype(np.uint8)
        gt_shapes = _extract_shapes_with_transform(gt_field, transform, value=1)
        if not gt_shapes:
            out[patch_id] = []
            continue

        best_pq = -1.0
        best_polys: list = []
        for h_min, by_patch in polygons_per_h_min.items():
            pred_shapes = by_patch.get(patch_id, [])
            if not pred_shapes:
                continue
            m = _match_shapes(gt_shapes, pred_shapes, AP_IOU_THRESHOLDS)
            t05 = AP_IOU_THRESHOLDS[0]
            tps, fps, fns = m["per_t"][t05]
            if tps + fns == 0:
                continue
            p = tps / max(tps + fps, 1)
            r = tps / max(tps + fns, 1)
            rq = (2 * p * r / (p + r)) if (p + r) else 0.0
            sq = float(np.mean([iou for _, _, iou in m["matched_pairs_low"]])) if m["matched_pairs_low"] else 0.0
            pq = sq * rq
            if pq > best_pq:
                best_pq = pq
                best_polys = pred_shapes
        out[patch_id] = best_polys
    return out


# ------------------------------ STRATEGY C: Pixel-vote consensus ------------------------------


def strategy_c_pixel_vote(
    polygons_per_h_min: dict[float, dict[str, list]],
    samples: list[dict],
    threshold_frac: float = 0.5,
) -> dict[str, list]:
    """Rasterize each h_min, vote per pixel, polygonize majority."""
    out: dict[str, list] = {}
    n_h = len(polygons_per_h_min)
    k_min = max(1, int(np.ceil(n_h * threshold_frac)))

    for sample in samples:
        patch_id = sample.get("patch_id") or Path(sample["window_a"]).stem
        with rasterio.open(sample["label"]) as src:
            gt = src.read(1).astype(np.uint8)
            transform = src.transform

        vote_map = np.zeros(gt.shape, dtype=np.int32)
        for h_min, by_patch in polygons_per_h_min.items():
            polys = by_patch.get(patch_id, [])
            if not polys:
                continue
            raster = rasterio.features.rasterize(
                polys, out_shape=gt.shape, transform=transform, dtype=np.uint8
            )
            vote_map += (raster > 0).astype(np.int32)

        consensus = (vote_map >= k_min).astype(np.uint8)
        # Polygonize the consensus
        shapes_iter = rasterio.features.shapes(consensus, mask=(consensus > 0), transform=transform)
        out_polys = []
        for geom, val in shapes_iter:
            if val == 1:
                out_polys.append(shapely.geometry.shape(geom))
        out[patch_id] = out_polys
    return out


# ------------------------------ STRATEGY D: n_pred elbow ------------------------------


def strategy_d_elbow(
    polygons_per_h_min: dict[float, dict[str, list]],
) -> dict[str, list]:
    """Pick the h_min on the plateau of n_pred(h_min) — self-supervised."""
    out: dict[str, list] = {}
    all_patches = set()
    for h_min in polygons_per_h_min:
        all_patches.update(polygons_per_h_min[h_min].keys())

    h_mins_sorted = sorted(polygons_per_h_min.keys())

    for patch_id in all_patches:
        n_pred = [
            len(polygons_per_h_min[h].get(patch_id, []))
            for h in h_mins_sorted
        ]
        # Find the index where the derivative |dn| is smallest (and n>0)
        best_idx = 0
        best_dn = float("inf")
        for i in range(len(h_mins_sorted) - 1):
            if n_pred[i] == 0:
                continue
            dn = abs(n_pred[i + 1] - n_pred[i]) / max(n_pred[i], 1)
            if dn < best_dn:
                best_dn = dn
                best_idx = i
        chosen_h = h_mins_sorted[best_idx]
        out[patch_id] = polygons_per_h_min[chosen_h].get(patch_id, [])
    return out


# ------------------------------ Main ------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", required=True)
    p.add_argument("--root", default="data", type=str)
    p.add_argument("--inference-root", type=Path, default=Path("logs/inference"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="test")
    args = p.parse_args()

    # Discover all available GeoJSONs for this country.
    geojsons = find_geojsons(args.country, args.inference_root)
    if not geojsons:
        print(f"ERROR: no GeoJSONs found for {args.country}")
        return 1
    print(f"loaded {len(geojsons)} GeoJSONs for {args.country}: h_min={sorted(geojsons.keys())}")

    # Load polygons per (h_min, patch_id).
    polys_per_h: dict[float, dict[str, list]] = {}
    for h_min, gj in geojsons.items():
        polys_per_h[h_min] = load_polygons_by_patch(gj)
        print(f"  h_min={h_min}: {sum(len(v) for v in polys_per_h[h_min].values())} polygons")

    # Load GT samples
    ds = FTWPlanet(
        root=args.root, countries=[args.country], split=args.split,
        transforms=None, load_boundaries=True,
    )
    samples = ds.records
    print(f"FTWPlanet test split: {len(samples)} samples")

    results = []

    print("\n=== Strategy A: NMS-pool with agreement weighting ===")
    ens_a = strategy_a_nms_pool(polys_per_h)
    m_a = score_polygons_against_gt(ens_a, samples)
    results.append({"strategy": "A_nms_pool", "country": args.country, **m_a})
    print(f"  PQ={m_a['pq']:.4f} (SQ={m_a['pq_sq']:.3f}, RQ={m_a['pq_rq']:.3f}) AP={m_a['ap_5_95']:.4f}")

    print("\n=== Strategy B: Oracle h_min (upper bound) ===")
    ens_b = strategy_b_oracle(polys_per_h, samples)
    m_b = score_polygons_against_gt(ens_b, samples)
    results.append({"strategy": "B_oracle", "country": args.country, **m_b})
    print(f"  PQ={m_b['pq']:.4f} (SQ={m_b['pq_sq']:.3f}, RQ={m_b['pq_rq']:.3f}) AP={m_b['ap_5_95']:.4f}")

    print("\n=== Strategy C: Pixel-vote consensus (50% threshold) ===")
    ens_c = strategy_c_pixel_vote(polys_per_h, samples, threshold_frac=0.5)
    m_c = score_polygons_against_gt(ens_c, samples)
    results.append({"strategy": "C_pixel_vote", "country": args.country, **m_c})
    print(f"  PQ={m_c['pq']:.4f} (SQ={m_c['pq_sq']:.3f}, RQ={m_c['pq_rq']:.3f}) AP={m_c['ap_5_95']:.4f}")

    print("\n=== Strategy D: n_pred elbow auto-select ===")
    ens_d = strategy_d_elbow(polys_per_h)
    m_d = score_polygons_against_gt(ens_d, samples)
    results.append({"strategy": "D_elbow", "country": args.country, **m_d})
    print(f"  PQ={m_d['pq']:.4f} (SQ={m_d['pq_sq']:.3f}, RQ={m_d['pq_rq']:.3f}) AP={m_d['ap_5_95']:.4f}")

    # Append-only CSV (one row per strategy per country)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    if args.out.exists():
        df.to_csv(args.out, mode="a", header=False, index=False)
    else:
        df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
