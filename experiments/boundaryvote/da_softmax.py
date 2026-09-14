"""Turn a DelineateAnything top-k cache into a standard (prob, seg) cache run.

Pseudo-softmax rules over the per-pixel confidences ``c1 >= c2 >= c3`` of the
(1-px dilated) instances covering a pixel. Detection confidences are scores,
not calibrated probabilities (at ``--conf 0.05`` most are below 0.5), so they
enter only *relatively*: presence is DA's own decision (any instance covers
the pixel) and confidence only arbitrates contests between instances.

* ``ratio`` p_bg = [no instance]; within foreground r = c2 / c1 (0 when only
            one instance), p_boundary = r, p_field = 1 - r. A seam between two
            similarly confident instances is boundary; a weak overlapping
            duplicate leaves its dominant neighbour's pixels as field.
* ``hard``  p_bg = [no instance], p_field = [exactly one], p_boundary = [two or
            more] — confidence-free control

Also writes DA's native partition (``id1_raw``, conf-ranked painting of the raw
masks) as candidate tag ``v:da-native`` so it can be scored alongside.

Example:
    python -m boundaryvote.da_softmax --src da_v2 --rule c2 --out da_v2_c2
"""

import argparse
import json
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd

from .candidates import MANIFEST_COLS, instance_polygons, tag_to_file
from .paths import DENSE10, cache_dir, candidates_dir

RULES = ("ratio", "hard")


def pseudo_softmax(confs: np.ndarray, rule: str) -> np.ndarray:
    """``confs`` (k, H, W) sorted descending per pixel -> (3, H, W) probabilities."""
    c = confs.astype(np.float32)
    c1 = c[0]
    c2 = c[1] if c.shape[0] > 1 else np.zeros_like(c1)
    present = c1 > 0
    if rule == "hard":
        n = (c > 0).sum(axis=0)
        return np.stack([(n == 0), (n == 1), (n >= 2)]).astype(np.float32)
    if rule == "ratio":
        r = np.zeros_like(c1)
        np.divide(c2, c1, out=r, where=present)
        r = np.clip(r, 0.0, 1.0)
        return np.stack([~present, present * (1.0 - r), present * r]).astype(np.float32)
    raise ValueError(rule)


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", default="da_v2")
    p.add_argument("--rule", default="c2", choices=RULES)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    args = p.parse_args()

    src = cache_dir(f"{args.src}/{args.split}")
    dst = cache_dir(f"{args.out}/{args.split}")
    counts = {}
    for c in args.countries:
        (dst / c).mkdir(parents=True, exist_ok=True)
        cdir = candidates_dir(f"{args.out}/{args.split}") / c
        cdir.mkdir(parents=True, exist_ok=True)
        native_rows, manifest_rows = [], []
        files = sorted((src / c).glob("*.npz"))
        for f in files:
            with np.load(f) as d:
                confs, id1_raw, gt, tr, crs = (
                    d["confs"],
                    d["id1_raw"],
                    d["gt"],
                    d["transform"],
                    d["crs"],
                )
            prob = pseudo_softmax(confs, args.rule)
            np.savez_compressed(
                dst / c / f.name,
                prob=prob.astype(np.float16),
                seg=prob.argmax(axis=0).astype(np.uint8),
                markers=id1_raw.astype(
                    np.int32
                ),  # DA's own partition, for *-msup-* variants
                gt=gt,
                transform=tr,
                crs=crs,
            )
            pp = instance_polygons(id1_raw.astype(np.int32))
            native_rows += [(f.stem, i, g) for i, g in pp]
            manifest_rows.append(
                (
                    f.stem,
                    "v:da-native",
                    int((id1_raw > 0).sum()),
                    0,
                    0,
                    False,
                    int(id1_raw.max()),
                    len(pp),
                    0.0,
                )
            )
        gdf = gpd.GeoDataFrame(
            {
                "patch_id": [r[0] for r in native_rows],
                "inst_id": [r[1] for r in native_rows],
            },
            geometry=[r[2] for r in native_rows],
            crs=None,
        )
        gdf.to_parquet(cdir / tag_to_file("v:da-native"), index=False)
        pd.DataFrame(manifest_rows, columns=MANIFEST_COLS).to_csv(
            cdir / "manifest_da_native.csv", index=False
        )
        shutil.copy(src / c / "index.csv", dst / c / "index.csv")
        counts[c] = len(files)
        print(f"{c}: {len(files)} patches", flush=True)
    with (src / "manifest.json").open() as f:
        src_manifest = json.load(f)
    with (dst / "manifest.json").open("w") as f:
        json.dump(
            {
                "run": args.out,
                "split": args.split,
                "rule": args.rule,
                "source": src_manifest,
                "patches_per_country": counts,
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
