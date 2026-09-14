"""Stage 1 (GPU): cache softmax predictions per patch with FTP's inference path.

One ``.npz`` per patch under ``out/cache/<run>/<country>/<patch_id>.npz`` with

* ``prob``      float16 (3, H, W) softmax on the eval grid, un-padded
* ``seg``       uint8   (H, W) argmax of the float32 softmax (FTP's ``seg_pred``)
* ``gt``        uint8   (H, W) label after FTP's remap (2 -> 0, 3 kept)
* ``transform`` float64 (6,) rasterio affine (a, b, c, d, e, f) of the eval grid
* ``crs``       str     WKT/EPSG of the eval grid

plus ``<country>/index.csv`` and a run-level ``manifest.json`` recording the
checkpoint hash, git SHAs, and every inference flag. Downstream stages are
CPU-only and re-runnable from this cache.

Example (Planet B3, dense-10 test):
    python -m fieldvote.cache --run planet_b3 \
        --ckpt logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt

Example (S2 B7, FTP native-GSD protocol grid):
    python -m fieldvote.cache --run s2_b7 --backend s2 --upsample-to 512 \
        --ckpt logs/best_checkpoints/s2_efnet7_best.ckpt
"""

import argparse
import csv
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import ftp_compat as ftp
from .paths import DENSE10, cache_dir, ftw_planet_repo

INDEX_COLS = (
    "country",
    "patch_id",
    "height",
    "width",
    "crs",
    "transform",
    "label_path",
)


def _dataset(backend: str, root: str, country: str, split: str):
    if backend == "s2":
        from ftw_tools.training.datasets import FTW

        return FTW(
            root=root,
            countries=[country],
            split=split,
            transforms=None,
            load_boundaries=True,
            temporal_options="stacked",
        )
    return ftp.FTWPlanet(
        root=root,
        countries=[country],
        split=split,
        transforms=None,
        load_boundaries=True,
    )


def _label_path(ds, idx: int, backend: str) -> str:
    if backend == "s2":
        return str(ds.filenames[idx]["mask"]) if "mask" in ds.filenames[idx] else ""
    return ds.records[idx]["label"]


def cache_country(task, model, device, args, country: str, out: Path) -> int:
    ds = _dataset(args.backend, args.root, country, args.split)
    dl = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    scale = (
        ftp.PLANET_SR_SCALE if args.backend == "planet" else float(args.s2_data_scale)
    )
    cdir = out / country
    cdir.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    for idx, batch in enumerate(tqdm(dl, desc=country, leave=False)):
        if args.limit and idx >= args.limit:
            break
        image = batch["image"].to(device) / scale
        mask = batch["mask"].to(device)
        if args.upsample_to is not None:
            image = torch.nn.functional.interpolate(
                image,
                size=(args.upsample_to,) * 2,
                mode="bilinear",
                align_corners=False,
            )
            mask = (
                torch.nn.functional.interpolate(
                    mask.unsqueeze(1).float(),
                    size=(args.upsample_to,) * 2,
                    mode="nearest",
                )
                .squeeze(1)
                .to(mask.dtype)
            )
        image, mask, H, W = ftp.pad_min32(
            image, mask, min_size=args.min_pad_size, pad_mode=args.pad_mode
        )
        if args.tta:
            probs, _ = ftp.predict_tta(task, model, image, args.sdf_clip)
        else:
            probs, _ = ftp.predict(task, model, image, args.sdf_clip)
        mask_eval = mask.clone()
        mask_eval[mask_eval == 2] = 0
        mask_eval[mask == 3] = 3
        # argmax on float32 exactly as FTP does; float16 probs are for soft variants only
        seg_np = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
        prob_np = probs.squeeze(0).cpu().numpy().astype(np.float16)[:, :H, :W]
        gt_np = mask_eval.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]

        patch_id, crs, tr, _ = ftp.eval_grid(
            ds,
            idx,
            country,
            args.backend,
            args.root,
            args.planet_window,
            args.upsample_to,
        )
        np.savez_compressed(
            cdir / f"{patch_id}.npz",
            prob=prob_np,
            seg=seg_np,
            gt=gt_np,
            transform=np.array([tr.a, tr.b, tr.c, tr.d, tr.e, tr.f], dtype=np.float64),
            crs=np.array(str(crs)),
        )
        rows.append(
            (
                country,
                patch_id,
                H,
                W,
                str(crs),
                json.dumps(list(tr)[:6]),
                _label_path(ds, idx, args.backend),
            )
        )
        n += 1
    with open(cdir / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(INDEX_COLS)
        w.writerows(rows)
    return n


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True, help="cache name, e.g. planet_b3")
    p.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="path relative to ftw-planet repo or absolute",
    )
    p.add_argument("--backend", default="planet", choices=["planet", "s2"])
    p.add_argument(
        "--root",
        default=None,
        help="dataset root; default data (planet) / data/ftw (s2)",
    )
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--min-pad-size", type=int, default=512)
    p.add_argument("--pad-mode", default="zero", choices=["zero", "replicate"])
    p.add_argument("--sdf-clip", type=float, default=20.0)
    p.add_argument("--s2-data-scale", type=float, default=3000.0)
    p.add_argument(
        "--upsample-to", type=int, default=None, help="S2: 512 for the FTP protocol"
    )
    p.add_argument("--planet-window", default="a")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--limit", type=int, default=0, help="patches per country (smoke tests)"
    )
    args = p.parse_args()
    args.tta = not args.no_tta

    repo = ftw_planet_repo()
    ckpt = args.ckpt if args.ckpt.is_absolute() else repo / args.ckpt
    if args.root is None:
        args.root = str(repo / ("data" if args.backend == "planet" else "data/ftw"))
    elif not Path(args.root).is_absolute():
        args.root = str(repo / args.root)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    tic = time.time()
    task, model, kind = ftp.load_task(ckpt, device)
    print(
        f"loaded {ckpt.name} as {kind} on {device} in {time.time() - tic:.1f}s",
        flush=True,
    )

    out = cache_dir(f"{args.run}/{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for c in args.countries:
        t0 = time.time()
        counts[c] = cache_country(task, model, device, args, c, out)
        print(f"{c}: {counts[c]} patches in {time.time() - t0:.0f}s", flush=True)

    manifest = {
        "run": args.run,
        "split": args.split,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": platform.node(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "torch": torch.__version__,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": ftp.sha256_file(ckpt),
        "task_class": kind,
        "ftw_planet_git": ftp.git_sha(repo),
        "icvr_git": ftp.git_sha(Path(__file__).resolve().parents[2]),
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "patches_per_country": counts,
        "channel_order": "window_b then window_a, PSScene [B,G,R,NIR]"
        if args.backend == "planet"
        else "ftw_tools stacked",
        "normalization": f"/{ftp.PLANET_SR_SCALE:g}"
        if args.backend == "planet"
        else f"/{args.s2_data_scale:g}",
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}/manifest.json; total {sum(counts.values())} patches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
