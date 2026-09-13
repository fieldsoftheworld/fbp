"""Cross-country gallery (Fig. 1): released pipeline vs. ours on one patch per country.

Each example is a triple: image, released components, ours; reference parcels
are drawn as black outlines on both prediction panels and the numbers are
polygon counts in the crop. Rows cover all ten dense-label countries on
PlanetScope B3 except Portugal (two test patches), eight countries on
Sentinel-2 B7, and one failure case. Patch choice per (model, country): the
largest per-patch PQ gain (ours minus released) among test patches with 10-120
reference parcels whose densest square window shows at least eight of them
covering at least a quarter of the window;
the full-patch PQ of each prediction panel is printed on it.

Regenerate: ~/github/ftw-planet/.venv/bin/python experiments/figures/gallery.py
"""

import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation, distance_transform_edt, label

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("FTW_PLANET_REPO", str(Path.home() / "github" / "ftw-planet"))

from fieldvote import ftp_compat as ftp  # noqa: E402
from fieldvote import references as refs  # noqa: E402
from fieldvote.candidates import pixel_area_m2  # noqa: E402
from fieldvote.paths import (  # noqa: E402
    DENSE10,
    cache_dir,
    ftw_planet_repo,
    scores_dir,
)
from fieldvote.variants import run_variant  # noqa: E402
from figstyle import PALETTE, WIDE, save, use_style  # noqa: E402

TAU, H = 0.7, 2.0
VARIANT = "nbg-sup-h2-t0.7+A900"
OURS, BASE = "raw:v:nbg-sup-h2-t0.7+A900", "rt:h2"
PER_ROW = 3
NAMES = {
    "south_africa": "South Africa",
    "belgium": "Belgium",
    "cambodia": "Cambodia",
    "croatia": "Croatia",
    "germany": "Germany",
    "latvia": "Latvia",
    "lithuania": "Lithuania",
    "portugal": "Portugal",
    "slovenia": "Slovenia",
    "sweden": "Sweden",
}
S2_COUNTRIES = [
    "belgium",
    "cambodia",
    "croatia",
    "germany",
    "latvia",
    "lithuania",
    "slovenia",
    "sweden",
]
FAILURE = ("planet_b3", "latvia", "g26_00052_7")


def per_patch_delta(run: str, proto: str) -> pd.DataFrame:
    fs = glob.glob(str(scores_dir(f"{run}/test/{proto}") / "patches_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in fs])
    df = df[(df.reference == "vector") & df.strategy.isin([BASE, OURS])].copy()
    tp, fp, fn = df.tp_50, df.fp_50, df.fn_50
    p = tp / (tp + fp).clip(lower=1)
    r = tp / (tp + fn).clip(lower=1)
    f1 = (2 * p * r / (p + r).where((p + r) > 0, 1)).fillna(0)
    df["pq"] = (df.iou_sum_50 / tp.clip(lower=1)) * f1
    w = df.pivot_table(
        index=["country", "patch_id"], columns="strategy", values=["pq", "n_gt"]
    )
    w.columns = ["_".join(c) for c in w.columns]
    w["d"] = w[f"pq_{OURS}"] - w[f"pq_{BASE}"]
    w["n_gt"] = w[f"n_gt_{BASE}"]
    return w.reset_index()


def patch_pq(w: pd.DataFrame, country: str, pid: str) -> tuple[float, float]:
    r = w[(w.country == country) & (w.patch_id == pid)].iloc[0]
    return 100 * r[f"pq_{BASE}"], 100 * r[f"pq_{OURS}"]


def pick(w: pd.DataFrame, run: str, country: str, min_in_crop: int = 8) -> str:
    """Largest per-patch gain whose best square window still shows >= min_in_crop parcels."""
    c = w[(w.country == country) & (w.n_gt >= 10) & (w.n_gt <= 120)]
    if c.empty:
        c = w[w.country == country]
    c = c.sort_values("d", ascending=False)
    for pid in c.patch_id.head(25):
        npz = cache_dir(f"{run}/test") / country / f"{pid}.npz"
        grid = refs.load_grid(npz, country)
        gt_shapes, _ = refs.vector(grid)
        gt = (
            rasterio.features.rasterize(
                ((g, k + 1) for k, g in enumerate(gt_shapes)),
                out_shape=grid.shape,
                fill=0,
                dtype="int32",
            )
            if gt_shapes
            else np.zeros(grid.shape, dtype=np.int32)
        )
        sl = best_window(gt)
        crop = gt[sl]
        if len(np.unique(crop[crop > 0])) >= min_in_crop and (crop > 0).mean() >= 0.25:
            return pid
    return c.iloc[0].patch_id


def rgb_for(run: str, country: str, pid: str, shape: tuple[int, int]) -> np.ndarray:
    repo = ftw_planet_repo()
    if run.startswith("planet"):
        with rasterio.open(
            repo / "data" / "planet" / country / "window_a" / f"{pid}.tif"
        ) as src:
            rgb = src.read().astype(np.float32)[[2, 1, 0]]
    else:
        with rasterio.open(
            repo / "data" / "ftw" / country / "s2_images" / "window_a" / f"{pid}.tif"
        ) as src:
            rgb = src.read().astype(np.float32)[[0, 1, 2]]
        Hh, Ww = shape
        ry = np.linspace(0, rgb.shape[1] - 1, Hh).round().astype(int)
        rx = np.linspace(0, rgb.shape[2] - 1, Ww).round().astype(int)
        rgb = rgb[:, ry][:, :, rx]
    rgb = rgb[:, : shape[0], : shape[1]]
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.8
    return np.transpose(rgb, (1, 2, 0))


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


def example(run: str, country: str, pid: str):
    npz = cache_dir(f"{run}/test") / country / f"{pid}.npz"
    grid = refs.load_grid(npz, country)
    with np.load(npz) as d:
        prob, seg = d["prob"].astype(np.float32), d["seg"].astype(np.uint8)
    ours = run_variant(VARIANT, seg, prob, px_m2=pixel_area_m2(npz, country))
    dist = distance_transform_edt(seg != 2).astype(np.float32)
    base, _ = label(ftp.watershed_instances(seg, dist, h_min=H) > 0)
    gt_shapes, _ = refs.vector(grid)
    gt = np.zeros(grid.shape, dtype=np.int32)
    if gt_shapes:
        gt = rasterio.features.rasterize(
            ((g, k + 1) for k, g in enumerate(gt_shapes)),
            out_shape=grid.shape,
            fill=0,
            dtype="int32",
        )
    sl = best_window(gt)
    rgb = rgb_for(run, country, pid, grid.shape)[sl]
    return rgb, base[sl], ours[sl], gt[sl]


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


def draw_pred(ax, inst, gt, seed, pq: float, bold: bool = False):
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
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9),
    )


def main() -> None:
    use_style(family="serif")
    wp = per_patch_delta("planet_b3", "pixel")
    ws = per_patch_delta("s2_b7", "pixel")
    examples = [
        ("planet_b3", c, pick(wp, "planet_b3", c), "Planet B3")
        for c in DENSE10
        if c != "portugal"
    ]
    examples += [("s2_b7", c, pick(ws, "s2_b7", c), "S2 B7") for c in S2_COUNTRIES]
    examples.append((*FAILURE, "Planet B3, Failure"))
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
    for k, (run, country, pid, tag) in enumerate(examples):
        row, col = divmod(k, PER_ROW)
        rgb, base, ours, gt = example(run, country, pid)
        pq_b, pq_o = patch_pq(wp if run == "planet_b3" else ws, country, pid)
        axes = [fig.add_subplot(gs[row, col * 3 + j]) for j in range(3)]
        axes[0].imshow(rgb, interpolation="nearest")
        draw_pred(axes[1], base, gt, 1, pq_b)
        draw_pred(axes[2], ours, gt, 2, pq_o, bold=pq_o > pq_b)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_linewidth(0.3)
                s.set_color("#999999")
        n_gt = len(np.unique(gt[gt > 0]))
        axes[1].text(
            0.5,
            1.03,
            f"{NAMES[country]}, {tag}, {n_gt} Parcels",
            transform=axes[1].transAxes,
            fontsize=6,
            ha="center",
            va="bottom",
        )
        if k == 0:
            # inside the panels, so the labels cost no vertical space
            for ax, t in zip(axes, ("Image", "FTP", "Veto"), strict=True):
                ax.text(
                    0.03,
                    0.97,
                    t,
                    transform=ax.transAxes,
                    fontsize=5.5,
                    ha="left",
                    va="top",
                    bbox=dict(
                        boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85
                    ),
                )
    handles = [
        Patch(
            facecolor=PALETTE["green"], edgecolor="#999999", label="Predicted Parcel (Fill)"
        ),
        Patch(facecolor="white", edgecolor="black", label="Reference Parcel (Outline)"),
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
    for run, country, pid, tag in examples:
        print(run, country, pid, tag)


if __name__ == "__main__":
    main()
