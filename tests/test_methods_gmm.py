"""GaussianMixture: extent minus thinned boundary at thresholds from ``context``."""

import numpy as np
import pytest

from fbp import GaussianMixture
from fbp.methods.gmm import gmm_split


def two_field_scene() -> tuple[np.ndarray, np.ndarray]:
    """Two fields separated by a confident boundary line, plus a faint internal line."""
    H, W = 40, 60
    prob = np.zeros((3, H, W), dtype=np.float32)
    prob[0] = 1.0
    field = np.zeros((H, W), dtype=bool)
    field[4:36, 4:56] = True
    prob[1][field] = 0.9
    prob[0][field] = 0.05
    prob[2][field] = 0.05
    prob[2][4:36, 30] = 0.9
    prob[1][4:36, 30] = 0.05
    prob[2][20, 4:30] = 0.4
    prob[1][20, 4:30] = 0.55
    seg = prob.argmax(0).astype(np.uint8)
    return seg, prob


def test_gmm_split_separates_on_the_strong_boundary() -> None:
    _seg, prob = two_field_scene()
    out = gmm_split(prob, 0.5, 0.5)
    assert out.max() == 2


def test_gaussian_mixture_reads_thresholds_from_context() -> None:
    seg, prob = two_field_scene()
    out = GaussianMixture(foreground="not_background", min_area_m2=9)(
        seg, prob, pixel_area_m2=1.0, context={"gmm": (0.5, 0.5)}
    )
    assert out.max() == 2


def test_gaussian_mixture_without_context_raises() -> None:
    seg, prob = two_field_scene()
    with pytest.raises(ValueError, match="context"):
        GaussianMixture(foreground="not_background")(seg, prob)
