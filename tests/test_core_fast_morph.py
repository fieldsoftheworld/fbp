"""The integer backend (``fast=True``) against scikit-image."""

import numpy as np
import pytest

pytest.importorskip("SimpleITK")
pytest.importorskip("numba")

from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from fbp import BoundaryVote
from fbp.core.fast_morph import h_maxima_itk, quantize_prob, watershed_bucket
from fbp.core.regions import watershed_regions


def _surface(
    seed: int, shape: tuple[int, int] = (96, 96)
) -> tuple[np.ndarray, np.ndarray]:
    """A quantized pb-style surface with a few basins and its foreground mask."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: shape[0], : shape[1]]
    pb = np.zeros(shape, np.float32)
    for _ in range(6):
        cy, cx = rng.integers(8, shape[0] - 8, 2)
        r = np.hypot(y - cy, x - cx)
        pb = np.maximum(pb, np.exp(-((r - 12) ** 2) / 4))  # a ring of boundary
    pb = np.clip(pb + rng.normal(0, 0.02, shape), 0, 1)
    surf = quantize_prob(np.stack([1 - pb, 1 - pb, pb]))[0]  # 1 - pb on the grid
    fg = surf > 0.3
    return surf.astype(np.float32), fg


@pytest.mark.parametrize("seed", range(3))
def test_h_maxima_itk_is_bit_identical(seed: int) -> None:
    surf, fg = _surface(seed)
    for h in (0.01, 0.05, 0.2):
        ref = h_maxima(surf * fg, h=h)
        got = h_maxima_itk(surf * fg, h=h)
        assert np.array_equal(ref.astype(bool), got.astype(bool))


@pytest.mark.parametrize("seed", range(3))
def test_watershed_bucket_matches_skimage_off_ties(seed: int) -> None:
    surf, fg = _surface(seed)
    reg = watershed_regions(surf, fg, 0.01)
    if reg.fallback:
        pytest.skip("no markers in this synthetic surface")
    ref = watershed(-surf, markers=reg.markers, mask=fg, watershed_line=True)
    got = watershed_bucket(-surf, markers=reg.markers, mask=fg, watershed_line=True)
    assert got.shape == ref.shape
    # labels agree everywhere except at exact-tie line pixels (well under 1%)
    assert (got != ref).mean() < 0.01
    assert set(np.unique(got)) == set(np.unique(ref))


def test_refuses_unquantized_surface() -> None:
    surf, _fg = _surface(0)
    with pytest.raises(ValueError, match="1/255"):
        h_maxima_itk(surf + 0.001, h=0.01)


def test_fast_runs_the_integer_backend_end_to_end() -> None:
    surf, _fg = _surface(1)
    pb = 1 - surf
    prob = np.stack([np.zeros_like(pb), 1 - pb, pb]).astype(np.float32)
    seg = prob.argmax(0).astype(np.uint8)
    m = BoundaryVote(
        foreground="not_background", seed_depth=0.01, min_line_support=0.3, fast=True
    )
    assert m.backend == "fast"
    fast = m(seg, prob)
    slow = BoundaryVote(
        foreground="not_background", seed_depth=0.01, min_line_support=0.3
    )(seg, prob)
    assert fast.shape == slow.shape
    # the two floods differ only at exact-tie line pixels; on this noisy synthetic
    # surface that still moves a few vote outcomes, so compare coverage, not counts
    assert ((fast > 0) == (slow > 0)).mean() > 0.9
