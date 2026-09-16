"""Cross-country gallery (Fig. 1): FTP pipeline vs. BoundaryVote, one patch per country.

Each example is a triple: image, the FTP pipeline, BoundaryVote; reference parcels
are drawn as black outlines on both prediction panels; titles give crop field
counts and badges give full-patch PQ. Which patches are shown, and how they were
chosen, is ``freeze.py``'s business: this script draws the rows listed in
``results/figure_inputs/gallery.json`` from the frozen inputs next to it, so it
needs neither the prediction cache nor the imagery.

Regenerate: ~/github/ftw-planet/.venv/bin/python -m fbp.paper.figures.gallery
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy import ndarray
from scipy.ndimage import binary_dilation

from fbp.methods import ftp_pipeline
from fbp.methods import run as run_method
from fbp.paper.common import NAMES
from fbp.paper.figures import inputs
from fbp.paper.figures.figstyle import (
    LABEL_BBOX,
    PALETTE,
    PANEL_EDGE,
    PANEL_LINEWIDTH,
    WIDE,
    save,
    use_style,
)
from fbp.paper.shipped import OURS

ROOT = Path(__file__).resolve().parents[4]
H = 2.0  # the FTP pipeline's watershed depth
VARIANT = OURS.removeprefix("raw:v:")
PER_ROW = 3


def label_colors(inst: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(inst.max())
    palette = np.array(
        [mcolors.to_rgb(color) for name, color in PALETTE.items() if name != "black"]
    )
    cols = palette[rng.permutation(n) % len(palette)]
    out = np.ones((*inst.shape, 3))
    m = inst > 0
    out[m] = cols[inst[m] - 1]
    return out


def outlines(inst: np.ndarray) -> np.ndarray:
    """Pixels on the inner boundary of every label (1 px)."""
    fg = inst > 0
    p = np.pad(inst, 1)
    same = (
        (p[1:-1, 1:-1] == p[:-2, 1:-1])
        & (p[1:-1, 1:-1] == p[2:, 1:-1])
        & (p[1:-1, 1:-1] == p[1:-1, :-2])
        & (p[1:-1, 1:-1] == p[1:-1, 2:])
    )
    return fg & ~same


def example(
    run: str, country: str, pid: str
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    p = inputs.load(run, country, pid)
    ours = run_method(VARIANT, p.seg, p.prob, px_m2=p.px_m2)
    base = ftp_pipeline(p.seg, H)
    sl = best_window(p.gt)
    return p.rgb[sl], base[sl], ours[sl], p.gt[sl]


def best_window(gt: np.ndarray) -> tuple[slice, slice]:
    """Square window along the long axis holding the most reference parcels."""
    Hh, Ww = gt.shape
    side = min(Hh, Ww)
    best, best_n = (slice(0, side), slice(0, side)), -1
    steps = range(0, max(Hh, Ww) - side + 1, max(1, (max(Hh, Ww) - side) // 8 or 1))
    for o in steps:
        sl = (
            (slice(o, o + side), slice(0, side))
            if Hh >= Ww
            else (slice(0, side), slice(o, o + side))
        )
        n = len(np.unique(gt[sl][gt[sl] > 0]))
        if n > best_n:
            best, best_n = sl, n
    return best


def draw_pred(
    ax: Axes, inst: ndarray, gt: ndarray, seed: int, pq: float, bold: bool = False
) -> None:
    img = label_colors(inst, seed)
    img[binary_dilation(outlines(gt), iterations=1)] = (0.05, 0.05, 0.05)
    ax.imshow(img, interpolation="nearest")
    ax.text(
        0.97,
        0.03,
        f"PQ {pq:.0f}",
        transform=ax.transAxes,
        fontsize=5.5,
        fontweight="bold" if bold else "normal",
        ha="right",
        va="bottom",
        bbox=LABEL_BBOX,
    )


def main() -> None:
    use_style(family="serif")
    examples = inputs.picks()
    n_rows = int(np.ceil(len(examples) / PER_ROW))
    ncols = PER_ROW * 3
    panel = (WIDE * 0.985) / (ncols + 0.35 * (PER_ROW - 1))
    height = n_rows * panel * 1.16 + 0.38
    fig = plt.figure(figsize=(WIDE, height))
    gs = fig.add_gridspec(
        n_rows,
        ncols,
        left=0.005,
        right=0.995,
        top=0.965,
        bottom=0.35 / height,
        wspace=0.05,
        hspace=0.24,
    )
    for k, ex in enumerate(examples):
        row, col = divmod(k, PER_ROW)
        rgb, base, ours, gt = example(ex["run"], ex["country"], ex["patch_id"])
        pq_b, pq_o = ex["pq_base"], ex["pq_ours"]
        axes = [fig.add_subplot(gs[row, col * 3 + j]) for j in range(3)]
        axes[0].imshow(rgb, interpolation="nearest")
        draw_pred(axes[1], base, gt, 1, pq_b, bold=pq_b > pq_o)
        draw_pred(axes[2], ours, gt, 2, pq_o, bold=pq_o > pq_b)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_linewidth(PANEL_LINEWIDTH)
                s.set_color(PANEL_EDGE)
        n_gt = len(np.unique(gt[gt > 0]))
        axes[1].text(
            0.5,
            1.055,
            f"{NAMES[ex['country']]}, {ex['tag']}, {n_gt} Fields",
            transform=axes[1].transAxes,
            fontsize=6,
            ha="center",
            va="bottom",
        )
        # rule under the title, spanning the triple, so each title visibly
        # belongs to the three panels beneath it rather than floating between
        left, right = axes[0].get_position(), axes[2].get_position()
        fig.add_artist(
            Line2D(
                [left.x0, right.x1],
                [left.y1 + 0.0045, left.y1 + 0.0045],
                transform=fig.transFigure,
                color="#444444",
                linewidth=0.5,
            )
        )
        if k == 0:
            # inside the panels, so the labels cost no vertical space
            for ax, t in zip(axes, ("Image", "FTP", "BoundaryVote"), strict=True):
                ax.text(
                    0.03,
                    0.97,
                    t,
                    transform=ax.transAxes,
                    fontsize=5.5,
                    ha="left",
                    va="top",
                    bbox=LABEL_BBOX,
                )
    handles = [
        Patch(
            facecolor=PALETTE["green"],
            edgecolor=PANEL_EDGE,
            label="Predicted Field (Color Fill)",
        ),
        Patch(facecolor="white", edgecolor="black", label="Reference Field (Outline)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=1.4,
    )
    save(fig, str(ROOT / "figures" / "rev_gallery"), formats=("pdf",))
    for ex in examples:
        print(ex["run"], ex["country"], ex["patch_id"], ex["tag"])


if __name__ == "__main__":
    main()
