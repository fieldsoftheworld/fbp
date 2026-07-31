"""predict-v2: run the frozen B7 Planet model and SAVE per-patch softmax + argmax
mask (+ affine transform + CRS) per country.

Why: the original predict pipeline threw away the per-pixel output after argmax.
By persisting softmax once on the GPU, the ENTIRE postprocessing sweep becomes
CPU-only and re-runnable offline: watershed at any h_min, MultiClean (needs the
mask), DenseCRF (needs softmax), the multi-h_min pixel-vote, and the label-free
agreement/uncertainty map. One GPU pass per country; everything downstream is CPU.

Reuses the validated model-load + forward from predict_and_polygonize. Output to
private scratch only. Softmax saved as float16 to halve storage.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/u/agiri1/github/ftw-planet/scripts/inference")
from predict_and_polygonize import (  # noqa: E402
    load_patch,
    load_planet_b7_model,
    pad_to_multiple,
    predict_tta_d4,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", required=True)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data-root", type=Path, default=Path("data/planet"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} ckpt={args.ckpt} tta={args.tta}")
    model = load_planet_b7_model(str(args.ckpt), device=device)

    cdir = args.data_root / args.country
    if not (cdir / "window_a").exists():
        sys.exit(f"window_a/ not found under {cdir}")
    pids = sorted(p.stem for p in (cdir / "window_a").glob("*.tif"))
    if args.limit:
        pids = pids[: args.limit]

    outc = args.out_dir / args.country
    outc.mkdir(parents=True, exist_ok=True)
    index: list = []
    for pid in tqdm(pids, desc=args.country):
        tensor, transform, crs = load_patch(cdir, pid, device=device)
        padded, H, W = pad_to_multiple(tensor)
        with torch.no_grad():
            logits = predict_tta_d4(model, padded) if args.tta else model(padded)
        logits = logits[:, :, :H, :W]
        prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float16)
        mask = prob.argmax(0).astype(np.uint8)
        np.savez_compressed(
            outc / f"{pid}.npz",
            prob=prob,
            mask=mask,
            transform=np.asarray(transform, dtype=np.float64),
        )
        index.append({"patch_id": pid, "crs": str(crs), "H": int(H), "W": int(W)})

    (outc / "index.json").write_text(json.dumps(index))
    print(f"{args.country}: saved {len(pids)} softmax+mask npz to {outc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
