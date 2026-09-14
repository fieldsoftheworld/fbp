"""Wall time and peak memory per post-processing method, by parcel density.

For each method and each sampled patch: one warm-up call, then ``--repeats``
timed calls (the minimum is reported, so scheduler noise inflates nothing), and
one instrumented call whose peak resident memory is sampled by a background
thread at 1 kHz. Peak memory is reported as the increase over the resting set
of the process, so the cached arrays already in memory are not counted.

Patches are stratified by the number of reference parcels they contain, which
is the quantity every method's cost depends on, and ``--per-bin`` patches are
drawn from each bin with a fixed seed.

Writes ``out/profile/<run>_<split>.csv``: one row per (method, patch).

Regenerate: python -m boundaryvote.profile_methods --run planet_b3 --per-bin 25
"""

import argparse
import gc
import threading
import time
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from . import references as refs
from .candidates import pixel_area_m2
from .paths import DENSE10, cache_dir, out_root
from .variants import foreground, run_variant

# The FTP pipeline, BoundaryVote, and the prior families at the setting
# each one reaches its best validation score with. The first element of each
# pair is the ``method`` key written to the CSV, so it stays as it is.
METHODS = [
    ("released", "arg-none-ws"),
    ("ours", "nbg-sup-h2-t0.7+A900"),
    ("ucm", "nbg-ucm-k0.5"),
    ("swst", "nbg-swst-n80-m30-t0.3"),
    ("iterh", "arg-iterh-a20000"),
    ("thr", "arg-thr-te0.5-tb0.5"),
]
BINS = [(1, 10), (10, 30), (30, 100), (100, 10**9)]
BIN_NAMES = ["1-9", "10-29", "30-99", "100+"]


def ftp_pipeline(seg: np.ndarray, _prob: np.ndarray) -> np.ndarray:
    """Field argmax, watershed at h=2, connected components of the union."""
    from scipy.ndimage import distance_transform_edt, label

    from . import ftp_compat as ftp

    dist = distance_transform_edt(seg != 2).astype(np.float32)
    inst = ftp.watershed_instances(seg, dist, h_min=2.0)
    out, _ = label(inst > 0)
    return out.astype(np.int32)


def call(
    name: str, seg: np.ndarray, prob: np.ndarray, px_m2: float | None = None
) -> np.ndarray:
    """Dispatch to the FTP pipeline or a model-free variant by name."""
    if name == "arg-none-ws":
        return ftp_pipeline(seg, prob)
    return run_variant(name, seg, prob, px_m2=px_m2)


class RSSSampler:
    """Peak resident memory of this process while the block runs."""

    def __init__(self, hz: int = 1000) -> None:
        """Sample at ``hz`` Hz while the ``with`` block runs."""
        self.interval = 1.0 / hz
        self.peak = 0

    def read_rss(self) -> int:
        """Current resident set size, in bytes."""
        with Path("/proc/self/statm").open() as f:
            return int(f.read().split()[1]) * 4096  # pages -> bytes

    def __enter__(self) -> Self:
        """Start the background sampling thread."""
        gc.collect()
        self.base = self.read_rss()
        self.peak = self.base
        self._stop = threading.Event()
        self._t = threading.Thread(target=self.sample_loop, daemon=True)
        self._t.start()
        return self

    def sample_loop(self) -> None:
        """Poll RSS at ``self.interval`` until stopped."""
        while not self._stop.is_set():
            self.peak = max(self.peak, self.read_rss())
            time.sleep(self.interval)

    def __exit__(self, *exc: object) -> None:
        """Stop sampling and record the peak increase over the resting set in ``mb``."""
        self._stop.set()
        self._t.join()
        self.mb = max(self.peak - self.base, 0) / 2**20


def sample_patches(
    run: str, split: str, per_bin: int, seed: int
) -> list[tuple[str, str, int, str, str]]:
    """One row per sampled patch: (country, patch_id, n_gt, npz path, bin name)."""
    rows = []
    for country in DENSE10:
        src = cache_dir(f"{run}/{split}") / country
        for npz in sorted(src.glob("*.npz")):
            grid = refs.load_grid(npz, country)
            shapes, _ = refs.vector(grid)
            rows.append((country, npz.stem, len(shapes), str(npz)))
    df = pd.DataFrame(rows, columns=["country", "patch_id", "n_gt", "path"])
    rng = np.random.default_rng(seed)
    out = []
    for (lo, hi), name in zip(BINS, BIN_NAMES, strict=True):
        sub = df[(df.n_gt >= lo) & (df.n_gt < hi)]
        if sub.empty:
            continue
        take = sub.iloc[rng.permutation(len(sub))[:per_bin]]
        out += [
            (r.country, r.patch_id, r.n_gt, r.path, name) for r in take.itertuples()
        ]
    return out


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", default="planet_b3")
    p.add_argument("--split", default="test")
    p.add_argument("--per-bin", type=int, default=25)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    patches = sample_patches(args.run, args.split, args.per_bin, args.seed)
    print(f"{len(patches)} patches sampled", flush=True)
    rows = []
    for country, pid, n_gt, path, bin_name in patches:
        with np.load(path) as d:
            prob, seg = d["prob"].astype(np.float32), d["seg"].astype(np.uint8)
        n_px = int(foreground(seg, prob, "nbg").sum())
        px_m2 = pixel_area_m2(Path(path), country)
        for label_name, method in METHODS:
            call(
                method, seg, prob, px_m2
            )  # warm-up: first call pays import and cache costs
            times = []
            out = None
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                out = call(method, seg, prob, px_m2)
                times.append(time.perf_counter() - t0)
            if out is None:
                raise RuntimeError("no timed calls ran; --repeats must be >= 1")
            with RSSSampler() as s:
                call(method, seg, prob, px_m2)
            rows.append(
                {
                    "run": args.run,
                    "split": args.split,
                    "country": country,
                    "patch_id": pid,
                    "bin": bin_name,
                    "n_gt": n_gt,
                    "n_fg_px": n_px,
                    "px": seg.size,
                    "method": label_name,
                    "variant": method,
                    "ms": 1000 * min(times),
                    "peak_mb": s.mb,
                    "n_pred": int(out.max()),
                }
            )
        print(f"{country}/{pid} ({n_gt} parcels) done", flush=True)
    df = pd.DataFrame(rows)
    dst = out_root() / "profile"
    dst.mkdir(parents=True, exist_ok=True)
    f = dst / f"{args.run}_{args.split}.csv"
    df.to_csv(f, index=False)
    print(f"wrote {f}")
    print(df.groupby(["method", "bin"]).ms.median().unstack().round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
