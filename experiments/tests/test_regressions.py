"""Regression tests for the four unbound-name bugs the type checkers caught."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from boundaryvote import ensemble, ftp_compat, profile_methods, references
from boundaryvote.paths import DENSE10, cache_dir


def test_sample_patches_rows_unpack_into_five_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sample_patches rows are (country, patch_id, n_gt, path, bin_name), not 4."""
    monkeypatch.setenv("FIELDVOTE_OUT", str(tmp_path))
    country = DENSE10[0]
    src = cache_dir("run/test") / country
    src.mkdir(parents=True)
    np.savez(src / "patch0.npz", gt=np.zeros((4, 4), dtype=np.uint8))
    npz = sorted(src.glob("*.npz"))[0]

    monkeypatch.setattr(references, "load_grid", lambda path, c: None)
    monkeypatch.setattr(references, "vector", lambda grid: ([0, 1, 2], [0.0] * 3))

    rows = profile_methods.sample_patches("run", "test", 10, 0)

    assert len(rows) == 1
    country_out, patch_id, n_gt, path, bin_name = rows[0]  # must not raise
    assert (country_out, patch_id, n_gt, path, bin_name) == (
        country,
        "patch0",
        3,
        str(npz),
        "1-9",
    )


def test_profile_methods_rejects_zero_repeats(monkeypatch: pytest.MonkeyPatch) -> None:
    """--repeats 0 would leave `out` unbound; it must fail fast instead."""
    monkeypatch.setattr(sys, "argv", ["profile_methods", "--repeats", "0"])
    with pytest.raises(ValueError, match="--repeats"):
        profile_methods.main()


def test_ensemble_binds_gt_transform_crs_from_first_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gt/transform/crs come from the first --runs source, not left unbound."""
    monkeypatch.setenv("FIELDVOTE_OUT", str(tmp_path))
    country = "belgium"
    gt = np.array([[1, 1], [3, 1]], dtype=np.uint8)
    transform = np.arange(6, dtype=np.float64)
    for i, run in enumerate(("run_a", "run_b")):
        d = cache_dir(f"{run}/test") / country
        d.mkdir(parents=True)
        prob = np.zeros((3, 2, 2), dtype=np.float32)
        prob[i] = 1.0
        np.savez(d / "p0.npz", prob=prob, gt=gt, transform=transform, crs="EPSG:4326")
        (d / "index.csv").write_text("patch_id\np0\n")
        with (cache_dir(f"{run}/test") / "manifest.json").open("w") as f:
            json.dump({"run": run}, f)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ensemble",
            "--runs",
            "run_a",
            "run_b",
            "--out",
            "merged",
            "--split",
            "test",
            "--countries",
            country,
        ],
    )
    assert ensemble.main() == 0

    with np.load(cache_dir("merged/test") / country / "p0.npz") as d:
        assert np.array_equal(d["gt"], gt)
        assert np.array_equal(d["transform"], transform)
        assert str(d["crs"]) == "EPSG:4326"


class FakeTask:
    """Stands in for a loaded ftw-planet task."""

    def __init__(self, name: str) -> None:
        self.model = f"{name}-model"

    def eval(self) -> "FakeTask":
        return self

    def to(self, device: torch.device) -> "FakeTask":
        return self


def loads_as(name: str) -> classmethod:
    """A `load_from_checkpoint` that succeeds and reports `name`."""

    def load(cls: type, *args: object, **kwargs: object) -> FakeTask:
        return FakeTask(name)

    return classmethod(load)


def fails_to_load(cls: type, *args: object, **kwargs: object) -> FakeTask:
    """A `load_from_checkpoint` that always raises."""
    raise RuntimeError("no such checkpoint")


def test_load_task_kind_set_on_every_fallback_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`kind` must be bound however far the FrameField -> SDF -> Custom chain falls."""
    import ftw_planet.trainers as ftw_trainers
    from ftw_tools.training.trainers import CustomSemanticSegmentationTask

    device = torch.device("cpu")
    ckpt = Path("dummy.ckpt")

    monkeypatch.setattr(
        ftw_trainers.FrameFieldSegTask,
        "load_from_checkpoint",
        loads_as("FrameFieldSegTask"),
    )
    _, _, kind = ftp_compat.load_task(ckpt, device)
    assert kind == "FrameFieldSegTask"

    monkeypatch.setattr(
        ftw_trainers.FrameFieldSegTask,
        "load_from_checkpoint",
        classmethod(fails_to_load),
    )
    monkeypatch.setattr(
        ftw_trainers.SDFSegTask, "load_from_checkpoint", loads_as("SDFSegTask")
    )
    _, _, kind = ftp_compat.load_task(ckpt, device)
    assert kind == "SDFSegTask"

    monkeypatch.setattr(
        ftw_trainers.SDFSegTask, "load_from_checkpoint", classmethod(fails_to_load)
    )
    monkeypatch.setattr(
        CustomSemanticSegmentationTask,
        "load_from_checkpoint",
        loads_as("CustomSemanticSegmentationTask"),
    )
    _, _, kind = ftp_compat.load_task(ckpt, device)
    assert kind == "CustomSemanticSegmentationTask"
