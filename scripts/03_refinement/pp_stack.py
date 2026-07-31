"""Postprocessing stack for FTW-Planet field polygons (Paper #2).

Takes a country's BEST single-h_min GeoJSON (the current deployed output) and
runs it through a cumulative chain of geometric postprocessing stages, writing
one GeoJSON per stage so each can be (a) scored with the project's
score_country_geojson CLI and (b) rendered for before/after visuals:

  stage 0_raw        : the best-h_min polygons, untouched
  stage 1_simplify   : shapely.simplify (Douglas-Peucker), removes pixel staircase
  stage 2_regularize : BuildingRegulariser (edge orthogonalization / straightening)
  stage 3_smooth     : Smoothify (Chaikin corner-cutting, area-preserving)

Each stage builds on the previous one. Vertex counts per stage are recorded so
we can report the point-reduction / "jagged -> clean" story Isaac asked for.

Standalone, private (lives in scratch, not the tracked repo). Geometric ops run
in the GeoJSON's native CRS, which is per-country UTM (meters), so tolerances are
in meters. patch_id is preserved through every stage (scoring requires it).
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from buildingregulariser import regularize_geodataframe
from smoothify import smoothify

# country_best_geojson lives in the repo (symlinked .py created in a prior session)
REPO = Path("/u/agiri1/github/ftw-planet")
sys.path.insert(0, str(REPO / "scripts/inference"))
from score_minarea_at_best_hmin import country_best_geojson  # noqa: E402


def count_vertices(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == "Polygon":
        n = len(geom.exterior.coords)
        for ring in geom.interiors:
            n += len(ring.coords)
        return n
    if geom.geom_type == "MultiPolygon":
        return sum(count_vertices(g) for g in geom.geoms)
    return 0


def stage_row(gdf: gpd.GeoDataFrame, country: str, stage: str) -> dict:
    total = int(sum(count_vertices(g) for g in gdf.geometry))
    n = len(gdf)
    return {
        "country": country,
        "stage": stage,
        "n_polygons": n,
        "total_vertices": total,
        "mean_vertices": (total / n if n else 0.0),
    }


def robust_regularize(gdf: gpd.GeoDataFrame, tol: float) -> tuple[gpd.GeoDataFrame, int]:
    """BuildingRegulariser asserts/raises on some pathological field polygons and
    fails the WHOLE batch. Try the batch first; on a known geometry error fall
    back to per-geometry regularization, keeping the original polygon for any
    that still fail. Returns (result, n_fallback). The fallback count is logged
    (deliberate robustness for a flaky third-party tool, not a silent failure)."""
    # Root-cause repair: the tool errors on self-intersecting / multi-part input
    # ("repair with make_valid() first"). make_valid can yield Multi/Collection,
    # so explode to single Polygons and drop non-polygon debris.
    g = gdf.copy()
    g["geometry"] = g.geometry.make_valid()
    g = g.explode(index_parts=False)
    g = g[(g.geometry.geom_type == "Polygon") & g.geometry.notna() & (~g.geometry.is_empty)]
    # simplify=False disables BuildingRegulariser's INTERNAL simplify, which at
    # tolerance>0 can split a polygon into a self-intersecting multipolygon and
    # trip the tool's own assertion (geometries are already simplified upstream
    # in stage 1, so this step is redundant anyway). num_cores=1 keeps it
    # single-threaded so any residual error is catchable here.
    try:
        return regularize_geodataframe(
            g, simplify=False, allow_45_degree=True, allow_circles=False,
            num_cores=1,
        ), 0
    except (ValueError, AssertionError):
        geoms = []
        n_fail = 0
        for i in range(len(g)):
            try:
                r = regularize_geodataframe(
                    g.iloc[[i]], simplify=False,
                    allow_45_degree=True, allow_circles=False, num_cores=1,
                )
                geoms.append(r.geometry.iloc[0])
            except (ValueError, AssertionError):
                geoms.append(g.geometry.iloc[i])
                n_fail += 1
        out = g.copy()
        out["geometry"] = geoms
        return out, n_fail


def ensure_attrs(gdf: gpd.GeoDataFrame, country: str) -> gpd.GeoDataFrame:
    """Keep each polygon tagged with its patch_id through a tool call.

    Both tools preserve patch_id natively and propagate it to any split parts:
    BuildingRegulariser may explode a MultiPolygon into several rows, each
    correctly inheriting its parent patch_id, so the row count can legitimately
    grow. We therefore trust the tool's own patch_id column rather than
    re-aligning by position (which would break whenever a polygon is split)."""
    if "patch_id" not in gdf.columns:
        raise RuntimeError("postprocessing tool dropped the patch_id column")
    out = gdf.copy()
    out["country"] = country
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--summary", type=Path,
        default=REPO / "logs/inference/results_summary.csv",
    )
    p.add_argument("--simplify-tol-m", type=float, default=1.0)
    p.add_argument("--smooth-seg-len-m", type=float, default=3.0)  # Planet GSD
    p.add_argument("--smooth-iters", type=int, default=3)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summ = pd.read_csv(args.summary)
    row = summ[summ["country"] == args.country]
    if row.empty:
        print(f"ERROR: {args.country} not in {args.summary}")
        return 1
    best_h_min = float(row.iloc[0]["h_min_predict"])

    src = REPO / country_best_geojson(args.country, best_h_min)
    if not src.exists():
        print(f"ERROR: best-h_min GeoJSON missing: {src}")
        return 1
    print(f"{args.country}: best h_min={best_h_min}  src={src.name}")

    raw = gpd.read_file(src)
    orig_crs = raw.crs
    if orig_crs is None:
        raise RuntimeError(f"{src} has no CRS")
    # Native CRS is per-country UTM (meters); confirm it's projected so the
    # meter-valued tolerances are meaningful.
    work_crs = orig_crs if orig_crs.is_projected else raw.estimate_utm_crs()
    work = raw.to_crs(work_crs)

    stats: list[dict] = []

    # stage 0 — raw
    stage0 = work
    stats.append(stage_row(stage0, args.country, "0_raw"))

    # stage 1 — shapely simplify (Douglas-Peucker, topology-preserving)
    stage1 = stage0.copy()
    stage1["geometry"] = stage1.geometry.simplify(
        args.simplify_tol_m, preserve_topology=True
    )
    stats.append(stage_row(stage1, args.country, "1_simplify"))

    # stage 2 — BuildingRegulariser (orthogonalize/straighten edges)
    # Fields are not circles, so disable circle-fitting; keep 45-degree edges.
    # BuildingRegulariser raises through joblib on some geometries even after
    # make_valid; a country-level fallback to the simplified polygons guarantees
    # the stage completes (the failure is logged, not silent).
    try:
        reg, n_reg_fallback = robust_regularize(stage1, args.simplify_tol_m)
        if n_reg_fallback:
            print(f"  regularize: {n_reg_fallback} polygon(s) kept un-regularized (tool errored)")
        stage2 = ensure_attrs(reg, args.country).to_crs(work_crs)
    except Exception as e:  # noqa: BLE001 -- third-party tool, unpredictable type via joblib
        print(f"  regularize stage failed for {args.country} ({type(e).__name__}); using simplified polygons as stage 2")
        stage2 = stage1.copy()
    stats.append(stage_row(stage2, args.country, "2_regularize"))

    # stage 3 — Smoothify (Chaikin corner-cutting, area-preserving).
    # Disable all merging so each field polygon is smoothed independently and
    # the row<->patch_id mapping is preserved (defaults dissolve geometries).
    smoothed = smoothify(
        stage2,
        segment_length=args.smooth_seg_len_m,
        smooth_iterations=args.smooth_iters,
        merge_collection=False,
        merge_multipolygons=False,
        merge_holes=False,
    )
    stage3 = ensure_attrs(smoothed, args.country)
    stats.append(stage_row(stage3, args.country, "3_smooth"))

    # write each stage back in the ORIGINAL crs so scoring sees the same frame
    written = {}
    for stage_name, gdf in [
        ("0_raw", stage0), ("1_simplify", stage1),
        ("2_regularize", stage2), ("3_smooth", stage3),
    ]:
        out_path = args.out_dir / f"{args.country}_pp{stage_name}.geojson"
        gdf.to_crs(orig_crs)[["patch_id", "country", "geometry"]].to_file(
            out_path, driver="GeoJSON"
        )
        written[stage_name] = out_path

    vert_csv = args.out_dir / f"vertices_{args.country}.csv"
    pd.DataFrame(stats).to_csv(vert_csv, index=False)

    print(f"wrote {len(written)} stage geojsons + {vert_csv.name}")
    for s in stats:
        print(f"  {s['stage']:14s} n={s['n_polygons']:6d}  total_verts={s['total_vertices']:8d}  mean={s['mean_vertices']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
