"""Stage 1b (GPU): DelineateAnything instances -> per-pixel top-k confidence cache.

Runs ultralytics exactly like FTP's ``eval_delineate_anything.py`` (window B ->
RGB uint8 at ceiling 3000, ``retina_masks``, ``max_det 2000``) and stores, per
patch, the instances painted in descending confidence:

* ``id1_raw, conf1_raw``  top instance per pixel from the raw masks (DA's own
                          partition; conf-ranked painting resolves overlaps)
* ``id{1,2,3}, conf{1,2,3}`` top-3 instances per pixel after a 1-px dilation of
                          every mask, so touching instances contest their seam
* ``n_det``, ``gt``, ``transform``, ``crs`` as in ``fieldvote.cache``

``fieldvote.da_softmax`` turns this into a standard (prob, seg) cache run.

Example (run inside ftw-planet with ultralytics available):
    uv run --no-sync --with ultralytics --with huggingface_hub \
        python -m fieldvote.cache_da --run da_v2 --imgsz 1024 --conf 0.05
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
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import ftp_compat as ftp
from .paths import DENSE10, cache_dir, ftw_planet_repo

RGB_REFL_CEILING = 3000.0
RGB_CHANNELS = {"planet": [2, 1, 0], "s2": [0, 1, 2]}
TOPK = 3


def window_b_to_rgb_uint8(
    window_8ch: np.ndarray, ceiling: float, backend: str
) -> np.ndarray:
    """Copy of FTP's helper: window B bands -> (H, W, 3) uint8 RGB."""
    rgb = window_8ch[:4][RGB_CHANNELS[backend]]
    rgb = np.clip(rgb / ceiling, 0.0, 1.0)
    return np.transpose((rgb * 255.0).astype(np.uint8), (1, 2, 0))


def paint_topk(
    masks: np.ndarray, confs: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Paint instances in descending confidence into k layers.

    Returns ``ids (k, H, W) int32`` (1-based instance index, 0 = none) and
    ``confs (k, H, W) float32``. A pixel covered by m instances fills layers
    1..min(m, k) with the m most confident of them.
    """
    n, h, w = masks.shape
    ids = np.zeros((k, h, w), dtype=np.int32)
    out_conf = np.zeros((k, h, w), dtype=np.float32)
    order = np.argsort(-confs, kind="stable")
    for rank in order:
        m = masks[rank].astype(bool)
        if not m.any():
            continue
        remaining = m
        for layer in range(k):
            free = remaining & (ids[layer] == 0)
            if not free.any():
                continue
            ids[layer][free] = int(rank) + 1
            out_conf[layer][free] = float(confs[rank])
            remaining = remaining & ~free
            if not remaining.any():
                break
    return ids, out_conf


def cache_country(yolo, device, args, country: str, out: Path) -> int:
    ds = ftp.FTWPlanet(
        root=args.root,
        countries=[country],
        split=args.split,
        transforms=None,
        load_boundaries=True,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    cdir = out / country
    cdir.mkdir(parents=True, exist_ok=True)
    rows = []
    struct = np.ones((3, 3), dtype=bool)
    n = 0
    for idx, batch in enumerate(tqdm(dl, desc=country, leave=False)):
        if args.limit and idx >= args.limit:
            break
        image = batch["image"][0].numpy()
        mask = batch["mask"][0].numpy().astype(np.int64)
        H, W = mask.shape
        rgb = window_b_to_rgb_uint8(image, args.rgb_ceiling, "planet")
        r = yolo.predict(
            source=rgb[..., ::-1],  # ultralytics assumes BGR for numpy input
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=str(device) if device.type != "cpu" else "cpu",
            verbose=False,
            retina_masks=True,
            max_det=2000,
            half=device.type == "cuda",
        )[0]
        if r.masks is not None and r.masks.data is not None and len(r.masks.data) > 0:
            masks = r.masks.data.cpu().numpy().astype(np.uint8)
            confs = r.boxes.conf.cpu().numpy().astype(np.float32)
            masks = masks[:, :H, :W]
            if masks.shape[1:] != (H, W):
                masks = np.pad(
                    masks, ((0, 0), (0, H - masks.shape[1]), (0, W - masks.shape[2]))
                )
        else:
            masks = np.zeros((0, H, W), dtype=np.uint8)
            confs = np.zeros((0,), dtype=np.float32)
        ids_raw, conf_raw = paint_topk(masks, confs, 1)
        if len(masks):
            dil = np.stack(
                [binary_dilation(m.astype(bool), structure=struct) for m in masks]
            )
        else:
            dil = masks
        ids, conf_k = paint_topk(dil, confs, TOPK)

        gt = mask.copy()
        gt[gt == 2] = 0
        gt[mask == 3] = 3
        patch_id, crs, tr, _ = ftp.eval_grid(
            ds, idx, country, "planet", args.root, "a", None
        )
        np.savez_compressed(
            cdir / f"{patch_id}.npz",
            id1_raw=ids_raw[0],
            conf1_raw=conf_raw[0].astype(np.float16),
            ids=ids,
            confs=conf_k.astype(np.float16),
            n_det=np.int32(len(masks)),
            gt=gt.astype(np.uint8),
            transform=np.array([tr.a, tr.b, tr.c, tr.d, tr.e, tr.f], dtype=np.float64),
            crs=np.array(str(crs)),
        )
        rows.append(
            (country, patch_id, H, W, str(crs), json.dumps(list(tr)[:6]), len(masks))
        )
        n += 1
    with open(cdir / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ("country", "patch_id", "height", "width", "crs", "transform", "n_det")
        )
        w.writerows(rows)
    return n


def main() -> int:
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", default="da_v2")
    p.add_argument("--hf-repo", default="MykolaL/DelineateAnything")
    p.add_argument("--hf-file", default="DelineateAnythingv2.pt")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--rgb-ceiling", type=float, default=RGB_REFL_CEILING)
    p.add_argument("--root", default=None)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    repo = ftw_planet_repo()
    args.root = str(repo / "data") if args.root is None else args.root

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    ckpt = Path(hf_hub_download(args.hf_repo, args.hf_file))
    yolo = YOLO(str(ckpt))
    print(f"loaded {ckpt.name} on {device}", flush=True)

    out = cache_dir(f"{args.run}/{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for c in args.countries:
        t0 = time.time()
        counts[c] = cache_country(yolo, device, args, c, out)
        print(f"{c}: {counts[c]} patches in {time.time() - t0:.0f}s", flush=True)
    manifest = {
        "run": args.run,
        "split": args.split,
        "kind": "delineate-anything instances",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": platform.node(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": ftp.sha256_file(ckpt),
        "hf_repo": args.hf_repo,
        "hf_file": args.hf_file,
        "ftw_planet_git": ftp.git_sha(repo),
        "icvr_git": ftp.git_sha(Path(__file__).resolve().parents[2]),
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "topk": TOPK,
        "dilation_px": 1,
        "patches_per_country": counts,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}/manifest.json; total {sum(counts.values())} patches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
