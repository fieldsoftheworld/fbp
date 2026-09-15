"""The mixture fit recovers a bimodal split and the GMM variant separates fields."""

import numpy as np
from test_priors import two_field_scene

from boundaryvote.gmm_thresholds import fit_gmm1d, gmm_threshold
from boundaryvote.priors_recent import gmm_split
from boundaryvote.variants import run_variant


def test_gmm_threshold_between_modes() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0.1, 0.05, 7000), rng.normal(0.9, 0.05, 3000)]).clip(
        0, 1
    )
    _w, mu, _ = fit_gmm1d(x)
    assert abs(sorted(mu)[0] - 0.1) < 0.03 and abs(sorted(mu)[1] - 0.9) < 0.03
    t = gmm_threshold(x)
    assert abs(t - (0.7 * 0.1 + 0.3 * 0.9)) < 0.05  # weighted mean of the means


def test_gmm_split_and_dispatch() -> None:
    seg, prob = two_field_scene()
    out = gmm_split(prob, 0.5, 0.5)
    assert out.max() == 2  # the strong boundary column separates two fields
    via = run_variant("nbg-gmm+A9", seg, prob, px_m2=1.0, gmm_thr=(0.5, 0.5))
    assert via.max() == 2
    try:
        run_variant("nbg-gmm", seg, prob)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("gmm without thresholds must fail loudly")
