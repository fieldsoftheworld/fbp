"""Build a cache run by averaging the softmax of existing cache runs (e.g. B3+B7).

The grids are identical across Planet checkpoints (same patches, same eval
transform), so the average is pixel-wise. ``seg`` is the argmax of the average.

Example:
    python -m fieldvote.ensemble --runs planet_b3 planet_b7 --out planet_b37 \
        --split test
"""

import argparse
import json
import shutil
from datetime import UTC, datetime

import numpy as np

from .paths import DENSE10, cache_dir


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    args = p.parse_args()

    srcs = [cache_dir(f"{r}/{args.split}") for r in args.runs]
    if not srcs:
        raise ValueError("--runs requires at least one run")
    dst = cache_dir(f"{args.out}/{args.split}")
    counts = {}
    for c in args.countries:
        (dst / c).mkdir(parents=True, exist_ok=True)
        files = sorted((srcs[0] / c).glob("*.npz"))
        for f in files:
            probs = []
            gt = tr = crs = None
            for s in srcs:
                with np.load(s / c / f.name) as d:
                    probs.append(d["prob"].astype(np.float32))
                    if s == srcs[0]:
                        gt, tr, crs = d["gt"], d["transform"], d["crs"]
            if gt is None or tr is None or crs is None:
                raise RuntimeError(f"{srcs[0]} missing from its own source list")
            avg = np.mean(probs, axis=0)
            np.savez_compressed(
                dst / c / f.name,
                prob=avg.astype(np.float16),
                seg=avg.argmax(axis=0).astype(np.uint8),
                gt=gt,
                transform=tr,
                crs=crs,
            )
        shutil.copy(srcs[0] / c / "index.csv", dst / c / "index.csv")
        counts[c] = len(files)
        print(f"{c}: {len(files)} patches", flush=True)
    manifests = []
    for s in srcs:
        with (s / "manifest.json").open() as f:
            manifests.append(json.load(f))
    with (dst / "manifest.json").open("w") as f:
        json.dump(
            {
                "run": args.out,
                "split": args.split,
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "ensemble_of": args.runs,
                "rule": "mean softmax, argmax of mean",
                "sources": manifests,
                "patches_per_country": counts,
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
