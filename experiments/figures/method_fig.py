"""Worked example of the post-processing (Fig. 2): every intermediate image on three crops.

Columns, left to right: image; boundary probability; foreground with the pixels
the field class would drop; distance surface with the seed points; watershed
regions and lines; lines colored by their support s(l); final polygons with
reference outlines. One row per crop, all three chosen for dense field layouts
where the post-processing has something to get wrong. Recomputed from the prediction
cache with the same code that produced the tables.

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

from figstyle import (  # noqa: E402
    LABEL_BBOX,
    LOSS,
    PALETTE,
    PANEL_EDGE,
    PANEL_LINEWIDTH,
    WIDE,
    save,
    use_style,
)
from gallery import label_colors, outlines, rgb_for  # noqa: E402

from fieldvote import references as refs  # noqa: E402
from fieldvote.paths import cache_dir  # noqa: E402
from fieldvote.variants import (  # noqa: E402
    foreground,
    pairs_at_lines,
    supported_split,
)

TAU, H = 0.7, 2.0
# height:width of each row; wider rows let three of them fit the page budget
ASPECT = 0.62
# (run, country, patch, label, crop fraction of the patch's short side).
# Each row is cropped to a window of aspect ASPECT, so three rows fit the page
# without shrinking the panels.
# All three hold many fields per patch, so the watershed has real decisions to
# make. Two rows are PlanetScope and one Sentinel-2, and the Cambodia crop is
# windowed so its smallholder fields stay legible.
EXAMPLES = [
    ("planet_b3", "croatia", "g10-3_00071_11", "Croatia, PS", 0.63),
    ("planet_b3", "slovenia", "g13_00038_6", "Slovenia, PS", 0.63),
    ("s2_b7", "cambodia", "g60_0000000000-0000001536", "Cambodia, S2", 0.26),
]
COLUMNS = [
    "(a) Image",
    "(b) Boundary\nProbability",
    "(c) Foreground\nOrange: Boundary Class",
    "(d) Distance $D$\nWhite: Seed Points",
    "(e) Watershed Regions\nand Lines",
    "(f) Line Support $s(\\ell)$\nKept if $s\\geq\\tau$",
    "(g) Polygons\nBlack: Reference Outlines",
]
SUPPORT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "line_support", [LOSS, "white", PALETTE["blue"]]
)
SUPPORT_NORM = mcolors.TwoSlopeNorm(vmin=0, vcenter=TAU, vmax=1)


def line_support(inst: np.ndarray, lines: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Per-pixel map of s(l) for the line each pixel belongs to (NaN elsewhere)."""
    r, c, pairs = pairs_at_lines(inst, lines)
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


def build_row(
    run: str, country: str, pid: str, crop: float
) -> tuple[list[tuple[np.ndarray, str | None]], tuple[np.ndarray, ...]]:
    """The seven panel images for one crop, plus its seed-point coordinates.

    ``crop`` is the kept fraction of the patch's short side; the window is
    centered and spans the full width, giving every row the same aspect.
    """
    npz = cache_dir(f"{run}/test") / country / f"{pid}.npz"
    grid = refs.load_grid(npz, country)
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
    final = supported_split(fg, seg, prob, H, TAU)
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
    wide = min(Ww, int(round(min(Hh, Ww) * crop / ASPECT)))
    tall = int(round(wide * ASPECT))
    r0, c0 = (Hh - tall) // 2, (Ww - wide) // 2
    sl = (slice(r0, r0 + tall), slice(c0, c0 + wide))
    rgb = rgb_for(run, country, pid, grid.shape)[sl]
    seg, fg, dist, markers, ws, lines, s_map, final, gt = (
        a[sl] for a in (seg, fg, dist, markers, ws, lines, s_map, final, gt)
    )
    pb = prob[2][sl]

    fg_img = np.ones((*fg.shape, 3))
    fg_img[seg == 1] = mcolors.to_rgb(PALETTE["green"])
    # the boundary-class pixels the FTP pipeline drops from its foreground
    fg_img[seg == 2] = mcolors.to_rgb(PALETTE["orange"])

    ws_img = label_colors(ws, 4)
    ws_img[binary_dilation(lines, iterations=1) & fg] = (0.15, 0.15, 0.15)

    sup_img = np.ones((*fg.shape, 3))
    sup_img[fg] = (0.88, 0.88, 0.88)
    thick = binary_dilation(lines, iterations=1) & fg
    # propagate s to the thickened line pixels from the nearest line pixel
    from scipy.ndimage import distance_transform_edt as edt

    _, (ir, ic) = edt(~lines, return_indices=True)
    s_thick = s_map[ir, ic]
    m = thick & ~np.isnan(s_thick)
    sup_img[m] = SUPPORT_CMAP(SUPPORT_NORM(s_thick[m]))[:, :3]

    final_img = label_colors(final, 2)
    final_img[binary_dilation(outlines(gt), iterations=1)] = (0.05, 0.05, 0.05)

    panels: list[tuple[np.ndarray, str | None]] = [
        (rgb, None),
        (pb, "gray_r"),
        (fg_img, None),
        (np.where(fg, dist, np.nan), "viridis"),
        (ws_img, None),
        (sup_img, None),
        (final_img, None),
    ]
    return panels, np.nonzero(markers > 0)


def main() -> None:
    use_style(family="serif")
    rows = [(lab, *build_row(r, c, p, crop)) for r, c, p, lab, crop in EXAMPLES]
    n, nr = len(COLUMNS), len(rows)
    panel_w = WIDE / (n + 0.9)
    # the bottom strip carries the column titles and the support colorbar
    fig = plt.figure(figsize=(WIDE, panel_w * ASPECT * nr * 1.06 + 0.46))
    bottom = 0.46 / (panel_w * ASPECT * nr * 1.06 + 0.46)
    gs = fig.add_gridspec(
        nr,
        n,
        left=0.004,
        right=0.996,
        top=0.995,
        bottom=bottom,
        wspace=0.16,
        hspace=0.06,
    )
    axes = [[fig.add_subplot(gs[i, j]) for j in range(n)] for i in range(nr)]

    for i, (label_text, panels, (mr, mc)) in enumerate(rows):
        for j, (img, cm) in enumerate(panels):
            ax = axes[i][j]
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
                s.set_linewidth(PANEL_LINEWIDTH)
                s.set_color(PANEL_EDGE)
        # seed points on the distance panel
        axes[i][3].scatter(
            mc, mr, s=2.2, facecolor="white", edgecolor="black", linewidths=0.25
        )
        # name the crop inside its first panel, so it costs no layout width
        axes[i][0].text(
            0.035,
            0.965,
            label_text,
            transform=axes[i][0].transAxes,
            fontsize=5.5,
            ha="left",
            va="top",
            bbox=LABEL_BBOX,
        )
        # arrows between columns, on every row, so each row reads as one pass
        for j in range(n - 1):
            a, b = axes[i][j].get_position(), axes[i][j + 1].get_position()
            y = (a.y0 + a.y1) / 2
            fig.patches.append(
                FancyArrowPatch(
                    (a.x1 + 0.002, y),
                    (b.x0 - 0.002, y),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=4,
                    color="#555555",
                    linewidth=0.5,
                )
            )

    for j, title in enumerate(COLUMNS):
        axes[-1][j].text(
            0.5,
            -0.06,
            title,
            transform=axes[-1][j].transAxes,
            fontsize=6,
            ha="center",
            va="top",
            linespacing=1.15,
        )
    # colorbar for column (f), under its titles
    pos = axes[-1][5].get_position()
    cax = fig.add_axes(
        (pos.x0 + 0.06 * pos.width, bottom * 0.30, 0.88 * pos.width, 0.02)
    )
    sm = plt.cm.ScalarMappable(cmap=SUPPORT_CMAP, norm=SUPPORT_NORM)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, TAU, 1])
    cb.set_ticklabels(["0", r"$\tau=0.7$", "1"])
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.ax.spines["outline"].set_linewidth(0.4)
    save(fig, str(ROOT / "figures" / "rev_method"), formats=("pdf",))


if __name__ == "__main__":
    main()
