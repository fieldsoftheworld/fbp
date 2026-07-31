"""Score a predicted-country GeoJSON against FTW v2 ground truth.

Companion to ``scripts/inference/predict_and_polygonize.py``. Closes the
"we have an artifact, not a result" gap: takes the per-country GeoJSON
that ``predict_and_polygonize.py`` produced, reads the FTW v2 ground-
truth label masks for the same country/split, and reports the same
polygon-quality metrics as ``scripts/polygon_metrics_eval.py``
(PQ / SQ / RQ, AP@[0.5:0.95], polygon-count delta, boundary error).

Why a separate scorer (instead of re-running polygon_metrics_eval):

* Decouples scoring from inference. We can iterate on watershed knobs,
  noise filters, multi-country runs, etc. without re-running the model.
* Lets us compare *any* polygon source (our model, DelAny, S2 baseline,
  even hand-labelled tiles) on the same footing, as long as it produces
  a GeoJSON keyed by ``patch_id``.

DRAFT (untracked; ``.local.py`` suffix marks personal). Mirrors the
metric definitions in ``scripts/polygon_metrics_eval.py`` exactly so
numbers are directly comparable to the paper's headline polygon table.

Example:
    uv run scripts/inference/score_country_geojson.local.py \\
        --country rwanda \\
        --geojson logs/inference/rwanda_fields_full.geojson \\
        --out logs/inference/rwanda_score.csv \\
        --split test
"""

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import shapely.geometry
from tqdm import tqdm

# Reuse the exact match / chamfer machinery the paper's eval uses.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from polygon_metrics_eval import (  # noqa: E402
    AP_IOU_THRESHOLDS,
    GSD_M,
    _boundary_pixels,
    _match_shapes,
    _symmetric_chamfer,
)

from ftw_planet.datasets import FTWPlanet  # noqa: E402


def _extract_shapes_with_transform(
    mask: np.ndarray, transform, value: int = 1
) -> list[shapely.geometry.base.BaseGeometry]:
    """Connected-component shapes of value==value, georeferenced via transform.

    Same primitive as polygon_metrics_eval._extract_shapes but uses the
    affine transform so output shapes live in the raster's CRS (matching
    our pred GeoJSON, which is also in the source raster's CRS).
    """
    shapes: list = []
    for geom, val in rasterio.features.shapes(
        mask.astype(np.uint8), transform=transform
    ):
        if val == value:
            shapes.append(shapely.geometry.shape(geom))
    return shapes


def _crs_match(crs_a, crs_b) -> bool:
    """Tolerant CRS comparison: same EPSG or string-equal."""
    if crs_a is None or crs_b is None:
        return False
    try:
        return crs_a.to_epsg() == crs_b.to_epsg() and crs_a.to_epsg() is not None
    except Exception:
        return str(crs_a) == str(crs_b)


def score_country_geojson(
    country: str,
    geojson_path: Path,
    root: str,
    split: str,
    *,
    skip_chamfer: bool = False,
    min_area_m2: float = 0.0,
    dataset_backend: str = "planet",
) -> dict:
    """Score predicted polygons against FTW ground-truth label masks.

    Strategy:
      1. Load the GeoJSON; group polygons by ``patch_id`` column.
      2. Instantiate FTWPlanet for the country/split to resolve label
         file paths and verify which patches are actually in this split.
      3. For each split sample: read label raster (mask + transform + CRS),
         extract GT shapes in that CRS, pull pred shapes for this patch_id
         from the GeoJSON, sanity-check CRS, run greedy IoU matching,
         optionally compute chamfer on IoU>=0.5 matches.
      4. Aggregate per-threshold counts into PQ / SQ / RQ / AP.
    """
    # ---- 1. Load + group predicted polygons --------------------------------
    if not geojson_path.exists():
        msg = f"GeoJSON not found: {geojson_path}"
        raise FileNotFoundError(msg)
    pred_gdf = gpd.read_file(geojson_path)
    if "patch_id" not in pred_gdf.columns:
        msg = "GeoJSON must have a 'patch_id' column (set by predict_and_polygonize.py)"
        raise ValueError(msg)
    if min_area_m2 > 0:
        n_before = len(pred_gdf)
        pred_gdf = pred_gdf[pred_gdf.geometry.area >= min_area_m2].copy()
        print(
            f"  noise filter (>={min_area_m2:.0f} m^2): {n_before} -> {len(pred_gdf)} polygons"
        )
    pred_crs = pred_gdf.crs
    pred_by_patch: dict[str, list] = {}
    for _, row in pred_gdf.iterrows():
        pred_by_patch.setdefault(row["patch_id"], []).append(row.geometry)
    print(
        f"  loaded {len(pred_gdf)} predicted polygons across "
        f"{len(pred_by_patch)} patches; pred CRS={pred_crs}"
    )

    # ---- 2. Resolve label files via FTWPlanet ------------------------------
    if dataset_backend != "planet":
        msg = "Only dataset_backend='planet' is currently supported."
        raise NotImplementedError(msg)
    ds = FTWPlanet(
        root=root,
        countries=[country],
        split=split,
        transforms=None,
        load_boundaries=True,
    )
    print(f"  FTWPlanet split='{split}' has {len(ds.records)} samples for {country}")

    # ---- 3. Per-patch matching ---------------------------------------------
    counts: dict[float, list[int]] = {t: [0, 0, 0] for t in AP_IOU_THRESHOLDS}
    matched_ious_05: list[float] = []
    chamfer_pixels: list[float] = []
    polygon_deltas: list[int] = []
    n_pred_per_patch: list[int] = []
    n_gt_per_patch: list[int] = []
    n_patches = 0
    missing_predictions = 0
    crs_mismatch_warned = False

    gsd_m = GSD_M[dataset_backend]

    for sample in tqdm(ds.records, desc=country, leave=False):
        patch_id = sample.get("patch_id") or Path(sample["window_a"]).stem
        with rasterio.open(sample["label"]) as src:
            gt = src.read(1).astype(np.uint8)
            transform = src.transform
            label_crs = src.crs

        if not _crs_match(pred_crs, label_crs) and not crs_mismatch_warned:
            print(
                f"  WARN: pred CRS {pred_crs} != label CRS {label_crs} for "
                f"{patch_id}. Will not reproject — matches may be wrong."
            )
            crs_mismatch_warned = True

        # FTW 3-class convention: 0=bg, 1=field, 2=boundary, 3=ignore.
        # For polygon eval we look at field-interior only (matches
        # polygon_metrics_eval which folds class 2 into 0 and excludes 3).
        gt_field = (gt == 1).astype(np.uint8)
        gt_shapes = _extract_shapes_with_transform(gt_field, transform, value=1)

        pred_shapes = pred_by_patch.get(patch_id, [])
        if not pred_shapes and not gt_shapes:
            # Empty patch on both sides — skip cleanly, contributes nothing.
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

        for i, j, iou in m["matched_pairs_low"]:
            matched_ious_05.append(iou)
            if skip_chamfer:
                continue
            # Rasterize each matched shape into the gt pixel grid and chamfer.
            try:
                pred_mask = rasterio.features.rasterize(
                    [pred_shapes[j]], out_shape=gt.shape, transform=transform, dtype=np.uint8
                )
                gt_mask = rasterio.features.rasterize(
                    [gt_shapes[i]], out_shape=gt.shape, transform=transform, dtype=np.uint8
                )
            except ValueError:
                # Empty geometry intersection at this resolution; skip chamfer.
                continue
            pb = _boundary_pixels(pred_mask)
            gb = _boundary_pixels(gt_mask)
            c = _symmetric_chamfer(pb, gb)
            if c is not None:
                chamfer_pixels.append(c)

    if missing_predictions:
        print(
            f"  NOTE: {missing_predictions}/{n_patches} patches had GT but no "
            f"predictions in the GeoJSON (counted as full FN sets)."
        )

    # ---- 4. Aggregate ------------------------------------------------------
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

    if chamfer_pixels:
        chamfer_m = np.array(chamfer_pixels, dtype=np.float64) * gsd_m
        bnd_mean = float(chamfer_m.mean())
        bnd_p95 = float(np.percentile(chamfer_m, 95))
    else:
        bnd_mean = float("nan")
        bnd_p95 = float("nan")

    return {
        "n_patches": n_patches,
        "pq_sq": sq,
        "pq_rq": rq,
        "pq": pq,
        "ap_5_95": ap,
        "n_pred_mean": float(np.mean(n_pred_per_patch)) if n_pred_per_patch else 0.0,
        "n_gt_mean": float(np.mean(n_gt_per_patch)) if n_gt_per_patch else 0.0,
        "polygon_count_delta_mean": (
            float(np.mean(polygon_deltas)) if polygon_deltas else 0.0
        ),
        "polygon_count_delta_median": (
            float(np.median(polygon_deltas)) if polygon_deltas else 0.0
        ),
        "boundary_error_m_mean": bnd_mean,
        "boundary_error_m_p95": bnd_p95,
        "missing_predictions": missing_predictions,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", required=True, help="Country slug, e.g. rwanda")
    p.add_argument(
        "--geojson", required=True, type=Path,
        help="Predicted-polygon GeoJSON (must have patch_id column)",
    )
    p.add_argument("--root", default="data", type=str)
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--out", required=True, type=Path, help="CSV path to append to")
    p.add_argument(
        "--min-area-m2", type=float, default=0.0,
        help="Drop predicted polygons smaller than this before scoring (e.g. 100 to remove noise)",
    )
    p.add_argument(
        "--skip-chamfer", action="store_true",
        help="Skip boundary-error (chamfer) computation; faster",
    )
    p.add_argument(
        "--dataset-backend", type=str, default="planet", choices=["planet"],
        help="Currently only 'planet' is supported (3m).",
    )
    args = p.parse_args()

    print(
        f"scoring country={args.country} split={args.split} "
        f"geojson={args.geojson} root={args.root}"
    )

    tic = time.time()
    m = score_country_geojson(
        country=args.country,
        geojson_path=args.geojson,
        root=args.root,
        split=args.split,
        skip_chamfer=args.skip_chamfer,
        min_area_m2=args.min_area_m2,
        dataset_backend=args.dataset_backend,
    )
    print(f"scored in {time.time() - tic:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "geojson",
        "country",
        "split",
        "min_area_m2",
        "n_patches",
        "pq_sq",
        "pq_rq",
        "pq",
        "ap_5_95",
        "n_pred_mean",
        "n_gt_mean",
        "polygon_count_delta_mean",
        "polygon_count_delta_median",
        "boundary_error_m_mean",
        "boundary_error_m_p95",
        "missing_predictions",
    ]
    if not args.out.exists():
        with args.out.open("w") as f:
            f.write(",".join(cols) + "\n")
    row = [
        str(args.geojson),
        args.country,
        args.split,
        f"{args.min_area_m2:.1f}",
        str(m["n_patches"]),
        f"{m['pq_sq']:.6f}",
        f"{m['pq_rq']:.6f}",
        f"{m['pq']:.6f}",
        f"{m['ap_5_95']:.6f}",
        f"{m['n_pred_mean']:.4f}",
        f"{m['n_gt_mean']:.4f}",
        f"{m['polygon_count_delta_mean']:.4f}",
        f"{m['polygon_count_delta_median']:.4f}",
        f"{m['boundary_error_m_mean']:.4f}",
        f"{m['boundary_error_m_p95']:.4f}",
        str(m["missing_predictions"]),
    ]
    with args.out.open("a") as f:
        f.write(",".join(row) + "\n")
    print(
        f"  PQ={m['pq']:.4f} (SQ={m['pq_sq']:.3f}, RQ={m['pq_rq']:.3f})  "
        f"AP={m['ap_5_95']:.4f}  bnd_err={m['boundary_error_m_mean']:.2f}m  "
        f"({m['n_patches']} patches)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
