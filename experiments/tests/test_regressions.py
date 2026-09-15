"""Regression tests for unbound-name bugs the type checkers caught."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from boundaryvote import ftp_compat, profile_methods, references
from boundaryvote.paths import DENSE10, cache_dir


def test_sample_patches_rows_unpack_into_five_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sample_patches rows are (country, patch_id, n_gt, path, bin_name), not 4."""
    monkeypatch.setenv("BOUNDARYVOTE_OUT", str(tmp_path))
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
