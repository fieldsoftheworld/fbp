"""Stage 1 (CPU): cache the global store's softmax over the FTW test patches.

The global map's own model (PRUE EfficientNet-B7 U-Net, run at 10 m) never saw
the released checkpoints' inference path, so this is a third model for the
letter's comparison, with reference polygons. For every FTW patch of the
dense-10 countries the store window covering the label's bounds is read at
``--time-index`` and written in the ``fieldvote.cache`` layout:

* ``prob``      float16 (3, H, W) softmax on the store grid
* ``seg``       uint8   (H, W) argmax of the float32 softmax
* ``gt``        uint8   (H, W) label resampled (nearest) onto the store grid,
                after FTP's remap (2 -> 0, 3 kept)
* ``transform`` float64 (6,) affine of the store window, EPSG:4326
* ``crs``       "EPSG:4326"

Downstream stages (candidates, run_eval, report) run unchanged on the cache.

Example: python -m fieldvote.cache_global --run global_s2 --time-index 1
"""

import argparse
import csv
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from . import ftp_compat as ftp
from .global_store import TX, URL, X0, Y0, YEARS, affine, open_variables, read_window
from .paths import DENSE10, cache_dir, ftw_planet_repo

INDEX_COLS = ("country", "patch_id", "height", "width", "crs", "transform", "label_path")


def store_window(bounds) -> tuple[int, int, int, int]:
    """Store pixel window ``(y0, y1, x0, x1)`` covering ``bounds`` (EPSG:4326)."""
    # FTW labels carry a positive y pixel size, so ``bottom`` can exceed ``top``
    w, e = min(bounds.left, bounds.right), max(bounds.left, bounds.right)
    s, n = min(bounds.bottom, bounds.top), max(bounds.bottom, bounds.top)
    x0, x1 = math.floor((w - X0) / TX), math.ceil((e - X0) / TX)
    y0, y1 = math.floor((Y0 - n) / TX), math.ceil((Y0 - s) / TX)
    return y0, y1, x0, x1


def label_on_grid(mask_path: str, tr, shape: tuple[int, int]) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        if str(src.crs).upper() != "EPSG:4326":
            raise ValueError(f"{mask_path}: expected EPSG:4326, got {src.crs}")
        gt = np.zeros(shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=gt,
            dst_transform=tr,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )
    out = gt.copy()
    out[gt == 2] = 0
    out[gt == 3] = 3
    return out


def cache_country(v, t: int, root: str, country: str, split: str, out: Path, limit: int) -> int:
    from ftw_tools.training.datasets import FTW

    ds = FTW(
        root=root,
        countries=[country],
        split=split,
        transforms=None,
        load_boundaries=True,
        temporal_options="stacked",
    )
    cdir = out / country
    cdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, fn in enumerate(ds.filenames):
        if limit and idx >= limit:
            break
        patch_id = Path(fn["window_a"]).stem
        with rasterio.open(fn["mask"]) as src:
            bounds = src.bounds
        y0, y1, x0, x1 = store_window(bounds)
        prob = read_window(v, t, y0, y1, x0, x1)
        seg = prob.argmax(axis=0).astype(np.uint8)
        tr = affine(y0, x0)
        gt = label_on_grid(str(fn["mask"]), tr, seg.shape)
        np.savez_compressed(
            cdir / f"{patch_id}.npz",
            prob=prob.astype(np.float16),
            seg=seg,
            gt=gt,
            transform=np.array([tr.a, tr.b, tr.c, tr.d, tr.e, tr.f], dtype=np.float64),
            crs=np.array("EPSG:4326"),
        )
        h, w = seg.shape
        rows.append(
            (country, patch_id, h, w, "EPSG:4326", json.dumps(list(tr)[:6]), str(fn["mask"]))
        )
    with open(cdir / "index.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(INDEX_COLS)
        wr.writerows(rows)
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", default="global_s2")
    p.add_argument("--time-index", type=int, default=1, help="0 = 2024, 1 = 2025")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    p.add_argument("--root", default=None, help="FTW root; default <ftw-planet>/data/ftw")
    p.add_argument("--limit", type=int, default=0, help="patches per country (smoke tests)")
    args = p.parse_args()

    repo = ftw_planet_repo()
    root = args.root or str(repo / "data" / "ftw")
    v = open_variables()
    out = cache_dir(f"{args.run}/{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for c in args.countries:
        t0 = time.time()
        counts[c] = cache_country(v, args.time_index, root, c, args.split, out, args.limit)
        print(f"{c}: {counts[c]} patches in {time.time() - t0:.0f}s", flush=True)
    manifest = {
        "run": args.run,
        "split": args.split,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": platform.node(),
        "source": URL,
        "time_index": args.time_index,
        "year": YEARS[args.time_index],
        "grid": "store grid, EPSG:4326, %.8g deg/px" % TX,
        "icvr_git": ftp.git_sha(Path(__file__).resolve().parents[2]),
        "args": vars(args),
        "patches_per_country": counts,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}/manifest.json; total {sum(counts.values())} patches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
