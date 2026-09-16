"""Frozen figure inputs: the npz round trip, and the tracked set matches the figures."""

from pathlib import Path

import numpy as np
import pytest

from fbp.paper.figures import inputs
from fbp.paper.figures.method_fig import EXAMPLES


def test_save_read_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    prob = rng.random((3, 6, 5), dtype=np.float32)
    prob /= prob.sum(0)
    patch = inputs.Patch(
        prob=prob,
        seg=prob.argmax(0).astype(np.uint8),
        rgb=rng.random((6, 5, 3), dtype=np.float32),
        gt=rng.integers(0, 3, (6, 5)).astype(np.int32),
        px_m2=9.0,
    )
    p = tmp_path / "run" / "country" / "p1.npz"
    inputs.save(p, patch)
    got = inputs.read(p)
    assert got.shape == (6, 5)
    np.testing.assert_allclose(got.prob, prob, atol=1e-3)  # stored as float16
    np.testing.assert_array_equal(got.seg, patch.seg)
    np.testing.assert_array_equal(got.gt, patch.gt)
    np.testing.assert_allclose(got.rgb, patch.rgb, atol=1 / 255)
    assert got.px_m2 == 9.0


def test_picks_missing_says_how_to_make_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inputs, "PICKS", Path("/nonexistent/gallery.json"))
    with pytest.raises(FileNotFoundError, match="freeze"):
        inputs.picks()


def _frozen_or_skip() -> None:
    if not inputs.PICKS.exists():
        pytest.skip("tracked figure inputs missing")


def test_every_gallery_row_is_frozen_and_consistent() -> None:
    _frozen_or_skip()
    rows = inputs.picks()
    assert rows
    for ex in rows:
        p = inputs.load(ex["run"], ex["country"], ex["patch_id"])
        assert p.prob.shape[1:] == p.shape == p.gt.shape == p.rgb.shape[:2]
        assert p.px_m2 > 0
        assert 0 <= ex["pq_base"] <= 100
        assert 0 <= ex["pq_ours"] <= 100


def test_method_figure_crops_are_frozen() -> None:
    _frozen_or_skip()
    for run, country, pid, _, _ in EXAMPLES:
        assert inputs.path(run, country, pid).exists()
