"""Stage 2 (CPU): polygon candidates from the cached softmax.

For every cached patch: ``seg = argmax(prob)``, ``dist = EDT(seg != boundary)``,
then for each depth ``h`` in ``H_SWEEP`` run FTP's ``watershed_instances`` and
polygonize **each instance label separately** (pixel coordinates of the eval
grid, identity transform, same frame as FTP's ``_extract_shapes``). The raw
field-mask connected components are stored as depth tag ``cc``. Variants
(``boundaryvote.variants``) are stored under ``v:<name>``.

Per country under ``out/candidates/<run>/<split>/<country>/``:

* ``polys_<tag>.parquet``  columns ``patch_id, inst_id, geometry`` (GeoParquet,
                           crs=None)
* ``manifest.csv``         one row per (patch, tag): field pixels, marker count,
                           whether the connected-component fallback fired, instance
                           and polygon counts, wall time.

Example:
    python -m boundaryvote.candidates --run planet_b3 --split test --workers 8
    python -m boundaryvote.candidates --run planet_b3 --split test --workers 8 \
        --variants-only --grids main sweep
"""

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import shapely.geometry
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from . import variants as v
from .grids import names as grid_names
from .paths import DENSE10, H_SWEEP, cache_dir, candidates_dir, h_tag


def tag_to_file(tag: str) -> str:
    """Depth/variant tag -> its parquet filename."""
    return "polys_" + tag.replace(":", "_") + ".parquet"


def file_to_tag(name: str) -> str:
    """Parquet filename -> its depth/variant tag."""
    stem = name.removeprefix("polys_").removesuffix(".parquet")
    return "v:" + stem[2:] if stem.startswith("v_") else stem


MANIFEST_COLS = (
    "patch_id",
    "depth",
    "n_field_px",
    "n_boundary_px",
    "n_markers",
    "fallback",
    "n_inst",
    "n_polys",
    "seconds",
)


def watershed_with_meta(
    seg: np.ndarray, dist: np.ndarray, h: float
) -> tuple[np.ndarray, int, bool]:
    """Line-for-line ``postprocess_eval.watershed_instances`` plus marker bookkeeping.

    Returns ``(inst, n_markers, fallback)``. ``tests/test_candidates.py`` asserts the
    instance map equals FTP's function exactly.
    """
    field_mask = seg == 1
    if not field_mask.any():
        return np.zeros_like(seg, dtype=np.int32), 0, False
    surface = (dist * field_mask).astype(np.float32)
    seeds = h_maxima(surface, h=h)
    markers, n_markers = label(seeds)
    fallback = n_markers == 0
    if fallback:
        markers, _ = label(field_mask)
    inst = watershed(-surface, markers=markers, mask=field_mask)
    return inst.astype(np.int32), int(n_markers), bool(fallback)


def instance_polygons(
    inst: np.ndarray,
) -> list[tuple[int, shapely.geometry.base.BaseGeometry]]:
    """One polygon per (instance id, connected piece) in pixel coordinates."""
    out = []
    for geom, val in rasterio.features.shapes(inst, mask=inst > 0):
        out.append((int(val), shapely.geometry.shape(geom)))
    return out


def binary_polygons(
    mask: np.ndarray,
) -> list[tuple[int, shapely.geometry.base.BaseGeometry]]:
    """One polygon per connected component of a binary mask."""
    out = []
    for k, (geom, val) in enumerate(
        rasterio.features.shapes(mask.astype(np.uint8), mask=mask > 0)
    ):
        if val == 1:
            out.append((k + 1, shapely.geometry.shape(geom)))
    return out


def pixel_area_m2(npz_path: Path, country: str) -> float:
    """Ground area of one pixel of this patch's evaluation grid."""
    from . import references as refs

    grid = refs.load_grid(npz_path, country)
    return refs.pixel_area_ha(grid, refs.utm_crs(grid)) * 1e4


def process_patch(
    npz_path: Path,
    depths: tuple[float, ...],
    variant_names: tuple[str, ...] = (),
    px_m2: float | None = None,
    gmm_thr: tuple[float, float] | None = None,
) -> tuple[dict[str, list], list[tuple]]:
    """Returns ``({tag: [(patch_id, inst_id, geom), ...]}, manifest_rows)``."""
    pid = npz_path.stem
    with np.load(npz_path) as d:
        prob = d["prob"].astype(np.float32)
        seg = d["seg"].astype(np.uint8)
    field = seg == 1
    boundary = seg == 2
    dist = distance_transform_edt(~boundary).astype(np.float32)

    polys: dict[str, list] = {}
    rows: list[tuple] = []

    def record(
        tag: str, pp: list, n_markers: int, fallback: bool, n_inst: int, t0: float
    ) -> None:
        polys[tag] = [(pid, i, g) for i, g in pp]
        rows.append(
            (
                pid,
                tag,
                int(field.sum()),
                int(boundary.sum()),
                n_markers,
                fallback,
                n_inst,
                len(pp),
                time.perf_counter() - t0,
            )
        )

    t0 = time.perf_counter()
    cc = binary_polygons(field)
    record("cc", cc, 0, False, len(cc), t0)
    for h in depths:
        t0 = time.perf_counter()
        inst, n_markers, fallback = watershed_with_meta(seg, dist, h)
        record(
            h_tag(h), instance_polygons(inst), n_markers, fallback, int(inst.max()), t0
        )
    for name in variant_names:
        t0 = time.perf_counter()
        inst = v.run_variant(name, seg, prob, px_m2, gmm_thr)
        record(f"v:{name}", instance_polygons(inst), 0, False, int(inst.max()), t0)
    return polys, rows


def run_patch_job(
    args: tuple[
        Path,
        tuple[float, ...],
        tuple[str, ...],
        float | None,
        tuple[float, float] | None,
    ],
) -> tuple[dict[str, list], list[tuple]]:
    """``process_patch(*args)``, unpacked for ``Pool.imap_unordered``."""
    return process_patch(*args)


def process_country(
    run: str,
    split: str,
    country: str,
    depths: tuple[float, ...],
    workers: int,
    variant_names: tuple[str, ...] = (),
    gmm_thr: tuple[float, float] | None = None,
) -> int:
    """Run the depth sweep and/or variants for one country; returns patch count."""
    src = cache_dir(f"{run}/{split}") / country
    npzs = sorted(src.glob("*.npz"))
    if not npzs:
        raise FileNotFoundError(f"no cached patches under {src}")
    dst = candidates_dir(f"{run}/{split}") / country
    dst.mkdir(parents=True, exist_ok=True)

    per_tag: dict[str, list] = {
        "cc": [],
        **{h_tag(h): [] for h in depths},
        **{f"v:{n}": [] for n in variant_names},
    }
    manifest: list[tuple] = []
    needs_area = any("+A" in n for n in variant_names)
    jobs = [
        (
            p,
            depths,
            variant_names,
            pixel_area_m2(p, country) if needs_area else None,
            gmm_thr,
        )
        for p in npzs
    ]
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.imap_unordered(run_patch_job, jobs, chunksize=4)
            for polys, rows in results:
                for tag, lst in polys.items():
                    per_tag[tag].extend(lst)
                manifest.extend(rows)
    else:
        for job in jobs:
            polys, rows = run_patch_job(job)
            for tag, lst in polys.items():
                per_tag[tag].extend(lst)
            manifest.extend(rows)

    for tag, lst in per_tag.items():
        gdf = gpd.GeoDataFrame(
            {"patch_id": [r[0] for r in lst], "inst_id": [r[1] for r in lst]},
            geometry=[r[2] for r in lst],
            crs=None,
        )
        gdf.to_parquet(dst / tag_to_file(tag), index=False)
    mpath = dst / "manifest.csv"
    mdf = pd.DataFrame(manifest, columns=MANIFEST_COLS)
    if variant_names and not depths and mpath.exists():
        # variants-only pass: keep the depth rows already on disk
        old = pd.read_csv(mpath)
        new_tags = {f"v:{n}" for n in variant_names}
        mdf = pd.concat(
            [old[~old.depth.isin(new_tags)], mdf[mdf.depth.isin(new_tags)]],
            ignore_index=True,
        )
    mdf.sort_values(["patch_id", "depth"]).to_csv(
        mpath, index=False, quoting=csv.QUOTE_MINIMAL
    )
    return len(npzs)


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    p.add_argument("--depths", nargs="*", type=float, default=list(H_SWEEP))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--variants",
        nargs="*",
        default=(),
        help="variant names (boundaryvote.variants)",
    )
    p.add_argument(
        "--grids", nargs="*", default=(), help="variant groups from boundaryvote.grids"
    )
    p.add_argument(
        "--variants-only",
        action="store_true",
        help="skip the depth sweep (reuse existing polys_h*.parquet)",
    )
    args = p.parse_args()
    depths = () if args.variants_only else tuple(args.depths)
    variant_names = tuple(dict.fromkeys([*args.variants, *grid_names(args.grids)]))
    gmm: dict[str, tuple[float, float]] = {}
    if any(n.split("+")[0].endswith("-gmm") for n in variant_names):
        from .gmm_thresholds import ensure

        gmm = ensure(args.run, args.split, tuple(args.countries))
    for c in args.countries:
        t0 = time.time()
        n = process_country(
            args.run, args.split, c, depths, args.workers, variant_names, gmm.get(c)
        )
        n_tags = (len(depths) + 1 if depths else 0) + len(variant_names)
        print(
            f"{c}: {n} patches x {n_tags} tags in {time.time() - t0:.0f}s", flush=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
