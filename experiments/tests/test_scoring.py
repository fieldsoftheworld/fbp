"""Pooling and depth-selection arithmetic on synthetic tallies."""

from pathlib import Path

import pandas as pd
import pytest

from boundaryvote.matching import THRESHOLDS
from boundaryvote.run_eval import BINS
from boundaryvote.scoring import (
    load_patches,
    macro,
    pool_country,
    select_depth,
)


def make_row(
    country: str,
    patch: str,
    strategy: str,
    tp: int,
    fp: int,
    fn: int,
    iou_sum: float,
    split: str = "test",
) -> dict[str, str | int | float]:
    r = {
        "run": "r",
        "split": split,
        "protocol": "pixel",
        "reference": "vector",
        "strategy": strategy,
        "country": country,
        "patch_id": patch,
        "n_gt": tp + fn,
        "n_pred": tp + fp,
        "iou_sum_50": iou_sum,
        "gt_split": 0,
        "pred_merge": 0,
        "gt_missed": fn,
        "pred_spurious": fp,
        "px_ha": 0.0009,
        "t_build": 0.0,
    }
    for t in THRESHOLDS:
        k = f"{int(round(t * 100)):02d}"
        r[f"tp_{k}"], r[f"fp_{k}"], r[f"fn_{k}"] = tp, fp, fn
    for b in BINS:
        r[f"tp_50_{b}"] = tp if b == "large" else 0
        r[f"fp_50_{b}"] = fp if b == "large" else 0
        r[f"fn_50_{b}"] = fn if b == "large" else 0
        r[f"iou_sum_50_{b}"] = iou_sum if b == "large" else 0.0
    return r


def test_pool_country_pq_is_sq_times_rq() -> None:
    df = pd.DataFrame(
        [
            make_row("a", "p1", "s", tp=3, fp=1, fn=1, iou_sum=2.4),
            make_row("a", "p2", "s", tp=1, fp=1, fn=3, iou_sum=0.9),
            # an empty/empty patch still counts
            make_row("a", "p3", "s", tp=0, fp=0, fn=0, iou_sum=0.0),
        ]
    )
    c = pool_country(df).iloc[0]
    tp, fp, fn = 4, 2, 4
    p, r = tp / (tp + fp), tp / (tp + fn)
    rq = 2 * p * r / (p + r)
    sq = 3.3 / 4
    assert c.rq == pytest.approx(rq)
    assert c.sq == pytest.approx(sq)
    assert c.pq == pytest.approx(sq * rq)
    assert c.n_patches == 3 and c.n_scored == 2
    assert c.count_bias == pytest.approx(((4 - 4) + (2 - 4) + 0) / 3)
    assert c.pq_large == pytest.approx(c.pq) and c.pq_small == 0.0


def test_macro_requires_all_countries() -> None:
    df = pd.DataFrame(
        [make_row("a", "p", "s", 1, 0, 0, 0.8), make_row("b", "p", "s", 1, 1, 0, 0.7)]
    )
    c = pool_country(df)
    m = macro(c, ("a", "b"))
    assert m.iloc[0].pq == pytest.approx((c.pq.iloc[0] + c.pq.iloc[1]) / 2)
    with pytest.raises(ValueError, match="missing countries"):
        macro(c, ("a", "b", "z"))


def country_scores(
    split: str, pq_by: dict[str, dict[str, tuple[int, int, int, float]]]
) -> pd.DataFrame:
    """pq_by: {country: {strategy: (tp, fp, fn, iou_sum)}} -> pooled country frame."""
    rows = []
    for c, d in pq_by.items():
        for s, (tp, fp, fn, iou) in d.items():
            rows.append(make_row(c, "p", s, tp, fp, fn, iou, split=split))
    return pool_country(pd.DataFrame(rows))


def test_select_depth_test_and_val() -> None:
    test = country_scores(
        "test",
        {
            "a": {"raw:h2": (2, 0, 0, 1.8), "raw:h8": (1, 1, 1, 0.9)},
            "b": {"raw:h2": (1, 1, 1, 0.6), "raw:h8": (2, 0, 0, 1.9)},
        },
    )
    val = country_scores(
        "val",
        {
            "a": {"raw:h2": (1, 2, 2, 0.5), "raw:h8": (2, 0, 0, 1.9)},
            "b": {"raw:h2": (1, 2, 2, 0.5), "raw:h8": (2, 0, 0, 1.9)},
        },
    )
    ws_test = select_depth(test, test, "raw", "ws_test").set_index("country")
    assert ws_test.loc["a", "selected_depth"] == "raw:h2"
    assert ws_test.loc["b", "selected_depth"] == "raw:h8"
    assert (ws_test.strategy == "ws_test").all()
    # reported scores come from the test frame, choice from val
    ws_val = select_depth(test, val, "raw", "ws_val").set_index("country")
    assert ws_val.loc["a", "selected_depth"] == "raw:h8"
    assert ws_val.loc["a", "pq"] == pytest.approx(
        test.set_index(["country", "strategy"]).loc[("a", "raw:h8"), "pq"]
    )


def test_load_patches_missing_pattern_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_patches(tmp_path)


def test_load_patches_last_file_wins_on_duplicate_key(tmp_path: Path) -> None:
    """A strategy scored twice must not double-count; the later file's row wins."""
    old = pd.DataFrame([make_row("a", "p1", "s", tp=1, fp=0, fn=0, iou_sum=0.9)])
    new = pd.DataFrame([make_row("a", "p1", "s", tp=2, fp=1, fn=1, iou_sum=1.5)])
    old.to_parquet(tmp_path / "patches_a.parquet", index=False)
    new.to_parquet(tmp_path / "patches_b.parquet", index=False)
    df = load_patches(tmp_path)
    assert len(df) == 1
    assert df.iloc[0].n_pred == 3 and df.iloc[0].n_gt == 3
