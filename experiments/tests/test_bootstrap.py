"""Bootstrap pooling matches the scoring formula and returns a point row first."""

import numpy as np
import pandas as pd

from boundaryvote.bootstrap import COLS, country_arrays, draw_macro, pq


def test_pq_matches_pooled_formula() -> None:
    sums = np.array([[6.0, 5.0, 5.0, 4.75], [0.0, 3.0, 2.0, 0.0]])
    got = pq(sums)
    assert abs(got[0] - (4.75 / 6) * (6 / (6 + 0.5 * 10))) < 1e-12
    assert got[1] == 0.0


def test_draw_macro_point_estimate_is_deterministic() -> None:
    rows = []
    for country in ("a", "b"):
        for pid in range(4):
            for strat, tp in (("x", 2), ("y", 3)):
                rows.append(
                    {
                        "country": country,
                        "patch_id": f"{country}{pid}",
                        "strategy": strat,
                        **dict(zip(COLS, [tp, 1, 1, 0.8 * tp], strict=True)),
                    }
                )
    df = pd.DataFrame(rows)
    arrays = country_arrays(df, ("x", "y"))
    m = draw_macro(arrays, ("a", "b"), draws=50, seed=1)
    assert m.shape == (51, 2)
    assert m[0, 1] > m[0, 0]  # y has more matches per patch
    m2 = draw_macro(arrays, ("a", "b"), draws=50, seed=1)
    assert np.array_equal(m, m2)
