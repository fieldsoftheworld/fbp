"""Reframe the rendered panels on the area they actually annotate.

`make_method_fig_panels.py` saves each panel with `bbox_inches="tight"`, which
leaves a wide band of unpainted figure canvas on the left and bottom of every
panel. On top of that, the crop window it computes is the bounding box of every
polygon in the patch, so a single stray sliver in a corner pushes the field
cluster into the middle of an otherwise empty frame. In the paper that waste is
repeated 7 times across the teaser and 15 times across the results grid, where
it costs more page area than the imagery does.

Two rules, both measured from the pixels rather than chosen by hand:

1. Panels in the same figure row are cropped to one shared box, so the crop
   cannot break the spatial registration between them.
2. The box is the extent of the *annotation* — the saturated strokes and fills
   of the polygon panels — not of the pale probability backdrop, which covers
   the whole axes and would defeat the crop. The RGB input panels are cropped
   to the same box so the row stays registered, which frames the imagery on
   the area the figure is actually about.

Nothing drawn on a polygon panel is removed; the box is their union.

    python scripts/06_figures/crop_panels.py [--dir figures] [--dry-run]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# group -> (all panels in the row, the subset that defines the crop box).
# The RGB inputs are carried along but do not vote on the box.
GROUPS = {
    "teaser": (
        ["fig_pipe_input", "fig_pipe_prob", "fig_pipe_wslow", "fig_pipe_wshigh",
         "fig_pipe_consensus", "fig_pipe_agreement", "fig_pipe_gt"],
        ["fig_pipe_prob", "fig_pipe_wslow", "fig_pipe_wshigh",
         "fig_pipe_consensus", "fig_pipe_agreement", "fig_pipe_gt"],
    ),
}
for _c in ("be", "se", "lt", "hr", "lv"):
    GROUPS[_c] = (
        [f"fig_res_{_c}_{s}" for s in ("in", "inb", "single", "fv", "gt")],
        [f"fig_res_{_c}_{s}" for s in ("single", "fv", "gt")],
    )

# A stroke or fill counts as annotation if a channel is this far off white.
# The field-probability backdrop is a very pale green and stays below it.
INK = 40
PAD = 6  # px of breathing room around the annotated extent


def ink_box(path):
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    mask = np.abs(im - 255).max(axis=2) > INK
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("figures"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for group, (names, anchors) in GROUPS.items():
        paths = [args.dir / f"{n}.png" for n in names]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{group}: missing {[str(p) for p in missing]}")

        sizes = {Image.open(p).size for p in paths}
        if len(sizes) != 1:
            raise SystemExit(f"{group}: panels differ in size, {sizes}")
        w, h = sizes.pop()

        boxes = [ink_box(args.dir / f"{n}.png") for n in anchors]
        if any(b is None for b in boxes):
            raise SystemExit(f"{group}: an anchor panel has no annotation")
        x0 = max(0, min(b[0] for b in boxes) - PAD)
        y0 = max(0, min(b[1] for b in boxes) - PAD)
        x1 = min(w, max(b[2] for b in boxes) + PAD)
        y1 = min(h, max(b[3] for b in boxes) + PAD)

        nw, nh = x1 - x0, y1 - y0
        print(f"{group:8s} {w}x{h} -> {nw}x{nh}  aspect {nw / nh:.3f}  "
              f"keeps {nw * nh / (w * h):.0%}")
        if args.dry_run:
            continue
        for p in paths:
            Image.open(p).crop((x0, y0, x1, y1)).save(p)


if __name__ == "__main__":
    raise SystemExit(main())
