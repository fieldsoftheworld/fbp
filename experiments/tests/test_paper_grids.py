"""The variant groups run, cover the shipped settings, and match the scored results."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbp.methods import run
from fbp.paper import grids
from fbp.paper.common import macro
from fbp.paper.shipped import FTP_MIN, OURS, OURS_DISTANCE, OURS_NOFILTER

RESULTS = Path(__file__).resolve().parents[1] / "results" / "planet_b3"


def two_field_scene() -> tuple[np.ndarray, np.ndarray]:
    """Two fields separated by a confident boundary line, plus a faint internal line."""
    height, width = 40, 60
    prob = np.zeros((3, height, width), dtype=np.float32)
    prob[0] = 1.0
    field = np.zeros((height, width), bool)
    field[4:36, 4:56] = True
    prob[1][field] = 0.9
    prob[0][field] = 0.05
    prob[2][field] = 0.05
    prob[2][4:36, 30] = 0.9  # strong boundary column
    prob[1][4:36, 30] = 0.05
    prob[2][20, 4:30] = 0.4  # weak internal line
    prob[1][20, 4:30] = 0.55
    seg = prob.argmax(0).astype(np.uint8)
    return seg, prob


def test_groups_are_disjoint_from_nothing_and_have_no_duplicates() -> None:
    for g, v in grids.GRIDS.items():
        assert len(set(v)) == len(v), g
    assert grids.names(["main", "main"]) == list(grids.GRIDS["main"])


def test_shipped_settings_are_in_the_grids() -> None:
    every = set(grids.names(list(grids.GRIDS)))
    for s in (OURS, OURS_NOFILTER, OURS_DISTANCE, FTP_MIN):
        assert s.removeprefix("raw:v:") in every, s


@pytest.mark.parametrize("group", sorted(grids.GRIDS))
def test_every_setting_runs(group: str) -> None:
    seg, prob = two_field_scene()
    for name in grids.GRIDS[group]:
        inst = run(name, seg, prob, px_m2=1.0, ctx={"gmm": (0.5, 0.5)})
        assert inst.dtype == np.int32 and inst.shape == seg.shape, name
        assert inst.min() >= 0, name


def test_every_setting_was_scored_on_planet_b3() -> None:
    """Every grid entry has a row in the tracked planet_b3 test or validation macro."""
    scored: set[str] = set()
    for split in ("test", "val"):
        f = RESULTS / split / "pixel" / "macro.csv"
        if f.exists():
            scored |= set(pd.read_csv(f).strategy)
    if not scored:
        pytest.skip("tracked results missing")
    missing = [n for n in grids.names(list(grids.GRIDS)) if f"raw:v:{n}" not in scored]
    assert missing == []


def test_common_readers_load_the_tracked_planet_b3_results() -> None:
    """The macro reader sees the shipped and FTP strategies on the tracked run."""
    m = macro("planet_b3")
    assert OURS in m.index
    assert FTP_MIN in m.index
