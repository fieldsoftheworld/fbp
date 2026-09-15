"""The variant groups run, cover the shipped settings, and match the scored results."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_priors import two_field_scene

from boundaryvote import grids
from boundaryvote.shipped import FTP_MIN, OURS, OURS_DISTANCE, OURS_NOFILTER
from boundaryvote.variants import run_variants

RESULTS = Path(__file__).resolve().parents[1] / "results" / "planet_b3"


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
    out = run_variants(seg, prob, grids.GRIDS[group], px_m2=1.0, gmm_thr=(0.5, 0.5))
    for name, inst in out.items():
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
