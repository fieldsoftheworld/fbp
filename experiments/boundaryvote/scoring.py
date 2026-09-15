"""Pool per-patch tallies into country and macro scores.

FTP convention: within a country, TP/FP/FN are summed over patches, RQ is the
F1 of those sums, SQ is the mean matched IoU, PQ = SQ x RQ; the macro score is
the unweighted mean over countries. Every patch counts (an empty/empty patch
contributes zeros); ``n_scored`` reports how many had any object.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from .matching import THRESHOLDS
from .run_eval import BINS

KEYS = ["run", "split", "protocol", "reference", "strategy", "country"]


def threshold_key(t: float) -> str:
    """0.5 -> '50', for the ``tp_50``-style column names."""
    return f"{int(round(t * 100)):02d}"


def f1(tp: float, fp: float, fn: float) -> float:
    """F1 from raw counts."""
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / (p + r) if (p + r) else 0.0


PATCH_KEY = ["run", "split", "protocol", "reference", "strategy", "country", "patch_id"]


def load_patches(scores_root: Path, pattern: str = "patches_*.parquet") -> pd.DataFrame:
    """Every scored patch under ``scores_root``, one row per key.

    A strategy can be written by more than one scoring job, so the last file
    wins; without this a repeated strategy would double every count that is a
    sum rather than a ratio (``n_patches``, ``n_gt``).
    """
    files = sorted(scores_root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {scores_root}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates(subset=PATCH_KEY, keep="last").reset_index(drop=True)


def pool_country(df: pd.DataFrame) -> pd.DataFrame:
    """One row per KEYS with pooled PQ/SQ/RQ/F1, bins, counts, diagnostics."""
    rows = []
    for key, g in df.groupby(KEYS, sort=False):
        # pandas types a group key as Hashable; grouping on several columns
        # always yields the tuple that zip needs.
        r = dict(zip(KEYS, cast(tuple, key), strict=True))
        tp, fp, fn = g["tp_50"].sum(), g["fp_50"].sum(), g["fn_50"].sum()
        r["n_patches"] = len(g)
        r["n_scored"] = int(((g.n_gt + g.n_pred) > 0).sum())
        r["n_gt"] = int(g.n_gt.sum())
        r["n_pred"] = int(g.n_pred.sum())
        r["rq"] = f1(tp, fp, fn)
        r["sq"] = float(g.iou_sum_50.sum() / tp) if tp else 0.0
        r["pq"] = r["sq"] * r["rq"]
        r["f1_5_95"] = float(
            np.mean(
                [
                    f1(
                        g[f"tp_{threshold_key(t)}"].sum(),
                        g[f"fp_{threshold_key(t)}"].sum(),
                        g[f"fn_{threshold_key(t)}"].sum(),
                    )
                    for t in THRESHOLDS
                ]
            )
        )
        dn = g.n_pred - g.n_gt
        r["count_mae"] = float(dn.abs().mean())
        r["count_bias"] = float(dn.mean())
        r["count_rel_err"] = float(dn.abs().sum() / max(g.n_gt.sum(), 1))
        for c in ("gt_split", "pred_merge", "gt_missed", "pred_spurious"):
            r[c] = int(g[c].sum())
        r["split_rate"] = r["gt_split"] / max(r["n_gt"], 1)
        r["merge_rate"] = r["pred_merge"] / max(r["n_pred"], 1)
        r["miss_rate"] = r["gt_missed"] / max(r["n_gt"], 1)
        r["spurious_rate"] = r["pred_spurious"] / max(r["n_pred"], 1)
        for b in BINS:
            tpb, fpb, fnb = (
                g[f"tp_50_{b}"].sum(),
                g[f"fp_50_{b}"].sum(),
                g[f"fn_50_{b}"].sum(),
            )
            rqb = f1(tpb, fpb, fnb)
            sqb = float(g[f"iou_sum_50_{b}"].sum() / tpb) if tpb else 0.0
            r[f"pq_{b}"] = sqb * rqb
            r[f"rq_{b}"] = rqb
            r[f"n_gt_{b}"] = int(tpb + fnb)
        rows.append(r)
    return pd.DataFrame(rows)


def macro(country_df: pd.DataFrame, countries: tuple[str, ...]) -> pd.DataFrame:
    """Unweighted mean over ``countries``; fails loudly if any are missing."""
    keys = [k for k in KEYS if k != "country"]
    sub = country_df[country_df.country.isin(countries)]
    rows = []
    for key, g in sub.groupby(keys, sort=False):
        missing = set(countries) - set(g.country)
        if missing:
            raise ValueError(
                f"{dict(zip(keys, key, strict=True))} "
                f"missing countries {sorted(missing)}"
            )
        r = dict(zip(keys, key, strict=True))
        r["n_countries"] = len(g)
        r["n_patches"] = int(g.n_patches.sum())
        r["n_scored"] = int(g.n_scored.sum())
        for m in (
            "pq",
            "sq",
            "rq",
            "f1_5_95",
            "count_mae",
            "count_bias",
            "count_rel_err",
            "split_rate",
            "merge_rate",
            "miss_rate",
            "spurious_rate",
            *[f"pq_{b}" for b in BINS],
            *[f"rq_{b}" for b in BINS],
        ):
            r[m] = float(g[m].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def select_depth(
    country_df: pd.DataFrame, by: pd.DataFrame, prefix: str, label: str
) -> pd.DataFrame:
    """Per-country single-depth selection.

    ``by`` supplies the scores used for the choice (same schema as
    ``country_df``); ``country_df`` supplies the scores reported. ``prefix`` is
    ``raw`` or ``rt``. Returns rows with strategy renamed to ``label`` and a
    ``selected_depth`` column.
    """
    sel_keys = [k for k in KEYS if k != "strategy"]
    cand = by[by.strategy.str.startswith(prefix + ":h")]
    best = cand.loc[
        cand.groupby(sel_keys, sort=False).pq.idxmax(), [*sel_keys, "strategy"]
    ]
    best = best.rename(columns={"strategy": "selected_depth"})
    merge_on = [k for k in sel_keys if k in country_df.columns and k != "split"]
    out = country_df.merge(best.drop(columns=["split"]), on=merge_on, how="inner")
    out = out[out.strategy == out.selected_depth].copy()
    out["strategy"] = label
    return out
