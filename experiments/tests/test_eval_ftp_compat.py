"""``load_task`` must set `kind` on every fallback branch of its trainer chain."""

from pathlib import Path

import pytest
import torch

from fbp.eval import ftp_compat


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
