"""Import Fields of the Planet primitives from the sibling checkout.

We deliberately re-use FTP's functions instead of copying them so the rerun
is scored with the same code as FTP Table 1. Anything imported here is the
published behaviour; anything defined in this package is new.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from .paths import ftw_planet_repo

_REPO = ftw_planet_repo()
_EVAL = _REPO / "scripts" / "eval"
if not _EVAL.is_dir():
    raise FileNotFoundError(
        f"ftw-planet scripts/eval not found at {_EVAL}; set FTW_PLANET_REPO"
    )
for p in (str(_EVAL), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

# ruff: noqa: E402  (sys.path must be set before these imports)
from ftw_planet.datasets import (  # type: ignore[import-not-found]
    PLANET_SR_SCALE,
    FTWPlanet,
)
from polygon_metrics_eval import (  # type: ignore[import-not-found]
    AP_IOU_THRESHOLDS,
    GSD_M,
    _area_bin,
    _cap_pred_to_gsd,
    _symmetric_chamfer,
    _eval_grid,
    _extract_shapes,
    _match_shapes,
    _true_gt_shapes,
    _true_gt_utm,
)
from postprocess_eval import (  # type: ignore[import-not-found]
    _pad_min32,
    _predict,
    _predict_tta,
    watershed_instances,
)

__all__ = [
    "AP_IOU_THRESHOLDS",
    "GSD_M",
    "PLANET_SR_SCALE",
    "FTWPlanet",
    "_area_bin",
    "_symmetric_chamfer",
    "_cap_pred_to_gsd",
    "_eval_grid",
    "_extract_shapes",
    "_match_shapes",
    "_pad_min32",
    "_predict",
    "_predict_tta",
    "_true_gt_shapes",
    "_true_gt_utm",
    "git_sha",
    "load_task",
    "sha256_file",
    "watershed_instances",
]


def load_task(ckpt: Path, device):
    """Same task-class fallback chain as ``polygon_metrics_eval.main``."""
    from ftw_tools.training.trainers import CustomSemanticSegmentationTask

    errors = (RuntimeError, KeyError, TypeError, argparse.ArgumentError)
    task = None
    try:
        from ftw_planet.trainers import FrameFieldSegTask

        task = FrameFieldSegTask.load_from_checkpoint(str(ckpt), map_location="cpu")
        kind = "FrameFieldSegTask"
    except errors:
        task = None
    if task is None:
        try:
            from ftw_planet.trainers import SDFSegTask

            task = SDFSegTask.load_from_checkpoint(str(ckpt), map_location="cpu")
            kind = "SDFSegTask"
        except errors:
            task = CustomSemanticSegmentationTask.load_from_checkpoint(
                str(ckpt), map_location="cpu"
            )
            kind = "CustomSemanticSegmentationTask"
    task = task.eval().to(device)
    return task, task.model, kind


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_sha(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return out.stdout.strip() + ("-dirty" if dirty else "")
