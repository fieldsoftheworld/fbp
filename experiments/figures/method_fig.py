"""Worked example of the post-processing (Fig. 2): every intermediate image on one crop.

Panels, left to right: image; boundary probability; foreground with the pixels
the field argmax would drop; distance surface with h-maxima markers; watershed
regions and lines; lines colored by their support s(l); final parcels with
reference outlines. Recomputed from the prediction cache with the same code
that produced the tables.

Regenerate: ~/github/ftw-planet/.venv/bin/python experiments/figures/method_fig.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
from matplotlib import colors as mcolors
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("FTW_PLANET_REPO", str(Path.home() / "github" / "ftw-planet"))

from fieldvote import references as refs  # noqa: E402
from fieldvote.paths import cache_dir  # noqa: E402
from fieldvote.variants import (  # noqa: E402
    _pairs_at_lines,
    foreground,
    supported_split,
)
from figstyle import PALETTE, WIDE, save, use_style  # noqa: E402
from gallery import label_colors, outlines, rgb_for  # noqa: E402

TAU, H = 0.7, 2.0
RUN, COUNTRY, PID = "planet_b3", "lithuania", "g11_00047_7"


def line_support(inst: np.ndarray, lines: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Per-pixel map of s(l) for the line each pixel belongs to (NaN elsewhere)."""
    r, c, pairs = _pairs_at_lines(inst, lines)
    valid = pairs[:, 0] > 0
    support = (prob[2] >= prob[1])[r, c]
    base = inst.max() + 1
    keys = pairs[valid, 0].astype(np.int64) * base + pairs[valid, 1]
    uniq, inv = np.unique(keys, return_inverse=True)
    n_px = np.bincount(inv, minlength=len(uniq))
    n_sup = np.bincount(inv, weights=support[valid], minlength=len(uniq))
    s = np.full(inst.shape, np.nan, dtype=np.float32)
    s[r[valid], c[valid]] = (n_sup / n_px)[inv]
    return s


def main() -> None:
    use_style(family="serif")
    npz = cache_dir(f"{RUN}/test") / COUNTRY / f"{PID}.npz"
    grid = refs.load_grid(npz, COUNTRY)
    with np.load(npz) as d:
        prob, seg = d["prob"].astype(np.float32), d["seg"].astype(np.uint8)
    fg = foreground(seg, prob, "nbg")
    dist = distance_transform_edt(seg != 2).astype(np.float32) * fg
    markers, _ = label(h_maxima(dist, h=H))
    ws = watershed(-dist, markers=markers, mask=fg, watershed_line=True).astype(
        np.int32
    )
    lines = fg & (ws == 0)
    s_map = line_support(ws, lines, prob)
    final, _, kept = supported_split(fg, seg, prob, H, TAU, return_lines=True)
    gt_shapes, _ = refs.vector(grid)
    gt = np.zeros(grid.shape, dtype=np.int32)
    if gt_shapes:
        gt = rasterio.features.rasterize(
            ((g, k + 1) for k, g in enumerate(gt_shapes)),
            out_shape=grid.shape,
            fill=0,
            dtype="int32",
        )

    Hh, Ww = grid.shape
    side = min(Hh, Ww)
    r0, c0 = (Hh - side) // 2, (Ww - side) // 2
    sl = (slice(r0, r0 + side), slice(c0, c0 + side))
    rgb = rgb_for(RUN, COUNTRY, PID, grid.shape)[sl]
    seg, fg, dist, markers, ws, lines, s_map, final, kept, gt = (
        a[sl] for a in (seg, fg, dist, markers, ws, lines, s_map, final, kept, gt)
    )
    pb = prob[2][sl]

    # panel images
    fg_img = np.ones((*fg.shape, 3))
    fg_img[seg == 1] = mcolors.to_rgb(PALETTE["green"])
    fg_img[(seg == 2)] = mcolors.to_rgb(
        PALETTE["orange"]
    )  # boundary pixels the released mask drops

    ws_img = label_colors(ws, 4)
    ws_img[binary_dilation(lines, iterations=1) & fg] = (0.15, 0.15, 0.15)

    sup_img = np.ones((*fg.shape, 3))
    sup_img[fg] = (0.88, 0.88, 0.88)
    cmap = plt.get_cmap("coolwarm_r")
    thick = binary_dilation(lines, iterations=1) & fg
    # propagate s to the thickened line pixels from the nearest line pixel
    from scipy.ndimage import distance_transform_edt as edt

    _, (ir, ic) = edt(~lines, return_indices=True)
    s_thick = s_map[ir, ic]
    m = thick & ~np.isnan(s_thick)
    sup_img[m] = cmap(s_thick[m])[:, :3]

    final_img = label_colors(final, 2)
    final_img[binary_dilation(outlines(gt), iterations=1)] = (0.05, 0.05, 0.05)

    panels = [
        (rgb, "(a) image", None),
        (pb, "(b) boundary\nprobability $p_{\\mathrm{bnd}}$", "gray_r"),
        (fg_img, "(c) foreground; orange =\nboundary pixels kept", None),
        (
            np.where(fg, dist, np.nan),
            "(d) distance $D$ to boundary;\nred = $h$-maxima markers",
            "viridis",
        ),
        (ws_img, "(e) watershed regions\nand lines", None),
        (sup_img, "(f) line support $s(\\ell)$;\nkept if $s\\geq\\tau=0.7$", None),
        (final_img, "(g) parcels; reference\noutlined in black", None),
    ]
    n = len(panels)
    panel_w = WIDE / (n + 0.9)
    fig = plt.figure(figsize=(WIDE, panel_w * 1.45))
    gs = fig.add_gridspec(
        1, n, left=0.004, right=0.996, top=0.99, bottom=0.30, wspace=0.16
    )
    axes = [fig.add_subplot(gs[0, j]) for j in range(n)]
    for ax, (img, title, cm) in zip(axes, panels, strict=True):
        if cm is None:
            ax.imshow(img, interpolation="nearest")
        else:
            ax.imshow(
                img,
                cmap=cm,
                interpolation="nearest",
                vmin=0 if cm == "gray_r" else None,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(0.4)
            s.set_color("#888888")
        ax.text(
            0.5,
            -0.05,
            title,
            transform=ax.transAxes,
            fontsize=6,
            ha="center",
            va="top",
            linespacing=1.15,
        )
    # markers on the distance panel
    mr, mc = np.nonzero(markers > 0)
    axes[3].scatter(mc, mr, s=1.6, color=PALETTE["red"], linewidths=0)
    # arrows between panels
    for j in range(n - 1):
        a, b = axes[j].get_position(), axes[j + 1].get_position()
        y = (a.y0 + a.y1) / 2
        fig.patches.append(
            FancyArrowPatch(
                (a.x1 + 0.002, y),
                (b.x0 - 0.002, y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=5,
                color="#555555",
                linewidth=0.6,
            )
        )
    # colorbar for (f)
    pos = axes[5].get_position()
    cax = fig.add_axes(
        [
            pos.x0 + 0.12 * pos.width,
            pos.y0 - 0.30 * pos.height,
            0.76 * pos.width,
            0.045 * pos.height,
        ]
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, TAU, 1])
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    save(fig, str(ROOT / "figures" / "rev_method"), formats=("pdf",))


if __name__ == "__main__":
    main()
