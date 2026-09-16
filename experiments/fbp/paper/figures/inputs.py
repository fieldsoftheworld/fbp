"""The few cached patches the gallery and method figure draw, frozen into the repo.

The prediction cache and the Planet and Sentinel-2 imagery live outside the
repository, so ``freeze_patch`` copies everything the two figure scripts read
for their handful of patches into ``experiments/results/figure_inputs/`` and
the scripts read only from there. Refresh with ``fbp.paper.figures.freeze``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fbp.eval.paths import cache_dir, ftw_planet_repo, results_root

FROZEN = results_root() / "figure_inputs"
PICKS = FROZEN / "gallery.json"


@dataclass(frozen=True)
class Patch:
    """One patch's figure inputs: predictions, image, reference labels, pixel area."""

    prob: np.ndarray  # float32 (3, H, W) softmax
    seg: np.ndarray  # uint8 (H, W) predicted class
    rgb: np.ndarray  # float32 (H, W, 3) display-stretched image in [0, 1]
    gt: np.ndarray  # int32 (H, W), one label per reference parcel
    px_m2: float  # ground area of one pixel

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width)."""
        return self.seg.shape[0], self.seg.shape[1]


def path(run: str, country: str, pid: str) -> Path:
    """Where one patch's frozen inputs live."""
    return FROZEN / run / country / f"{pid}.npz"


def save(p: Path, patch: Patch) -> None:
    """Write one patch; prob as float16 (the cache's own precision), rgb as uint8."""
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        prob=patch.prob.astype(np.float16),
        seg=patch.seg.astype(np.uint8),
        rgb=(np.clip(patch.rgb, 0, 1) * 255).round().astype(np.uint8),
        gt=patch.gt.astype(np.int32),
        px_m2=np.float64(patch.px_m2),
    )


def read(p: Path) -> Patch:
    """Read one frozen patch."""
    with np.load(p) as d:
        return Patch(
            prob=d["prob"].astype(np.float32),
            seg=d["seg"].astype(np.uint8),
            rgb=d["rgb"].astype(np.float32) / 255,
            gt=d["gt"].astype(np.int32),
            px_m2=float(d["px_m2"]),
        )


def load(run: str, country: str, pid: str) -> Patch:
    """Read one patch's frozen inputs by id."""
    return read(path(run, country, pid))


def picks() -> list[dict]:
    """The gallery rows chosen by ``freeze``: run, country, patch_id, tag, PQs."""
    if not PICKS.exists():
        msg = f"{PICKS} missing; run fbp.paper.figures.freeze with the cache present"
        raise FileNotFoundError(msg)
    return json.loads(PICKS.read_text())


def write_picks(rows: list[dict]) -> None:
    """Record the gallery rows."""
    PICKS.parent.mkdir(parents=True, exist_ok=True)
    PICKS.write_text(json.dumps(rows, indent=1) + "\n")


def stretched_rgb(
    run: str, country: str, pid: str, shape: tuple[int, int]
) -> np.ndarray:
    """The patch's image on the eval grid, 2-98 percentile stretch, gamma 0.8."""
    import rasterio

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


def freeze_patch(run: str, country: str, pid: str) -> Path:
    """Build one patch's inputs from the live cache and ftw-planet data, and save."""
    from fbp.eval import references as refs
    from fbp.eval.candidates import pixel_area_m2

    npz = cache_dir(f"{run}/test") / country / f"{pid}.npz"
    grid = refs.load_grid(npz, country)
    with np.load(npz) as d:
        prob, seg = d["prob"].astype(np.float32), d["seg"].astype(np.uint8)
    patch = Patch(
        prob=prob,
        seg=seg,
        rgb=stretched_rgb(run, country, pid, grid.shape),
        gt=refs.gt_raster(grid),
        px_m2=pixel_area_m2(npz, country),
    )
    out = path(run, country, pid)
    save(out, patch)
    return out
