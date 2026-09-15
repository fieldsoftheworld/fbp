"""Filtering strategies before pool_country must not change the pooled PQ."""

from pathlib import Path

import pandas as pd
import pytest

from fbp.eval.matching import THRESHOLDS
from fbp.eval.score import BINS
from fbp.eval.scoring import pool_country
from fbp.eval.transfer import FAMILIES, country_table


def _row(country: str, strategy: str, tp: int, fp: int, fn: int) -> dict:
    r = {
        "run": "r",
        "split": "test",
        "protocol": "pixel",
        "reference": "vector",
        "strategy": strategy,
        "country": country,
        "patch_id": "p",
        "n_gt": tp + fn,
        "n_pred": tp + fp,
        "iou_sum_50": float(tp),
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
        r[f"iou_sum_50_{b}"] = float(tp) if b == "large" else 0.0
    return r


def test_country_table_filter_before_pool_matches_filter_after() -> None:
    """Filtering strategies before or after pool_country must give the same PQ."""
    df = pd.DataFrame(
        [
            _row("a", "raw:v:nbg-ucm-k0.7", tp=3, fp=1, fn=0),
            _row("a", "not-a-family-member", tp=9, fp=9, fn=9),
        ]
    )
    pat = FAMILIES["ucm"]

    filtered_first = pool_country(df[df.strategy.str.match(pat)])
    filtered_after = pool_country(df)
    filtered_after = filtered_after[filtered_after.strategy.str.match(pat)]

    assert filtered_first.set_index("strategy").pq.equals(
        filtered_after.set_index("strategy").pq
    )
    assert set(filtered_first.strategy) == {"raw:v:nbg-ucm-k0.7"}


def test_country_table_reads_patches_and_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fbp.eval import paths

    monkeypatch.setenv("BOUNDARYVOTE_OUT", str(tmp_path))
    root = paths.scores_dir("r/test/pixel")
    root.mkdir(parents=True)
    df = pd.DataFrame(
        [
            _row("a", "raw:v:nbg-ucm-k0.7", tp=2, fp=0, fn=0),
            _row("a", "irrelevant", tp=1, fp=0, fn=0),
        ]
    )
    df.to_parquet(root / "patches_a.parquet", index=False)
    out = country_table("r", "test", tuple(FAMILIES.values()))
    assert set(out.strategy) == {"raw:v:nbg-ucm-k0.7"}
