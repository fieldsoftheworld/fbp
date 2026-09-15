"""Regression coverage for two profile.py behaviours that once broke silently."""

import sys
from pathlib import Path

import numpy as np
import pytest

from fbp.eval import profile, references
from fbp.eval.paths import DENSE10, cache_dir


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

    rows = profile.sample_patches("run", "test", 10, 0)

    assert len(rows) == 1
    country_out, patch_id, n_gt, path, bin_name = rows[0]  # must not raise
    assert (country_out, patch_id, n_gt, path, bin_name) == (
        country,
        "patch0",
        3,
        str(npz),
        "1-9",
    )


def test_profile_rejects_zero_repeats(monkeypatch: pytest.MonkeyPatch) -> None:
    """--repeats 0 would leave `out` unbound; it must fail fast instead."""
    monkeypatch.setattr(sys, "argv", ["profile", "--repeats", "0"])
    with pytest.raises(ValueError, match="--repeats"):
        profile.main()
