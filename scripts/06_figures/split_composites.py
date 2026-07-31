"""Split the two multi-panel PNGs into single-panel files.

Both composites were saved straight from matplotlib with the debug titles
still on ("CLEAN: g6_00089_17 (RGB)", "pampas: single threshold (n=426)") and
with wide gutters between the panels. Roughly half of `qualitative_example`
is empty canvas. Splitting them lets the paper set the gutters in LaTeX and
carry the panel labels as typeset text at the document font.

Panel boxes are found from the pixels: title rows and gutters are the runs
with no ink, so the crop cannot silently clip imagery.

    python scripts/06_figures/split_composites.py [--dir figures]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# source file -> (output stems, left to right)
COMPOSITES = {
    "qualitative_example.png": ["fig_qual_rgb", "fig_qual_pred", "fig_qual_gt"],
    "global_pampas.png": ["fig_global_tuned", "fig_global_fv"],
}
TOL = 8
MIN_PANEL = 20  # px; anything narrower is text, not a panel


def runs(flags):
    out, start = [], None
    for i, on in enumerate(flags):
        if on and start is None:
            start = i
        if not on and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def split(path, stems, outdir):
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    ink = np.abs(im - 255).max(axis=2) > TOL

    # The title strip is the first short run of inked rows; the panels are the
    # tall run below it.
    row_runs = [r for r in runs(ink.sum(axis=1) > 0)]
    body = max(row_runs, key=lambda r: r[1] - r[0])
    y0, y1 = body

    col_runs = [r for r in runs(ink[y0:y1].sum(axis=0) > 0)
                if r[1] - r[0] >= MIN_PANEL]
    if len(col_runs) != len(stems):
        raise SystemExit(f"{path}: found {len(col_runs)} panels, expected {len(stems)}")

    src = Image.open(path)
    for (x0, x1), stem in zip(col_runs, stems):
        panel = src.crop((x0, y0, x1, y1))
        # Each panel keeps its own blank border from the parent figure; the
        # panels are rendered at slightly different scales but the same aspect,
        # so trimming per panel and setting a common height in LaTeX registers
        # them exactly.
        ink = np.abs(np.asarray(panel.convert("RGB")).astype(int) - 255).max(axis=2) > TOL
        ys, xs = np.where(ink)
        panel = panel.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        out = outdir / f"{stem}.png"
        panel.save(out)
        w, h = panel.size
        print(f"  {out.name}: {w}x{h}  aspect {w / h:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    for name, stems in COMPOSITES.items():
        src = args.dir / name
        if not src.exists():
            print(f"{name}: already split, skipping")
            continue
        print(name)
        split(src, stems, args.dir)


if __name__ == "__main__":
    raise SystemExit(main())
