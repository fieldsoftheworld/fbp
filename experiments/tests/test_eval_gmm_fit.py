"""The mixture fit recovers a bimodal split."""

import numpy as np

from fbp.eval.gmm_fit import fit_gmm1d, gmm_threshold


def test_gmm_threshold_between_modes() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0.1, 0.05, 7000), rng.normal(0.9, 0.05, 3000)]).clip(
        0, 1
    )
    _w, mu, _ = fit_gmm1d(x)
    assert abs(sorted(mu)[0] - 0.1) < 0.03 and abs(sorted(mu)[1] - 0.9) < 0.03
    t = gmm_threshold(x)
    assert abs(t - (0.7 * 0.1 + 0.3 * 0.9)) < 0.05  # weighted mean of the means
