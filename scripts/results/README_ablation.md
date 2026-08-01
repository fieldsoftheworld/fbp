# Data needed to fill Table III of `main_grsl.tex`

Table III ("Aggregation Strategies Over the Same 14-Depth Sweep") currently has
three `??` placeholders. This note says exactly what is needed to replace them.

## What is missing and why

`HEADLINE_ensemble_macro.txt` already reports all four strategies, but only as
**HELDOUT_11** macros (11 countries, Kenya included):

| Strategy | HELDOUT_11 macro PQ |
|---|---|
| `A_nms_pool` | 0.2256 |
| `D_elbow` | 0.2365 |
| `B_oracle` | 0.2771 |
| `C_pixel_vote` | 0.3253 |

The Letter scores everything on **DENSE_10** = HELDOUT_11 minus Kenya, which is
presence-only and whose unlabeled background makes polygon metrics unreliable.

Rescaling `× 11/10` is only valid if Kenya contributes exactly 0.0 to that
strategy. Section 4 of the txt file confirms Kenya = 0.0000 for `single_best`
and `C_pixel_vote`, which is why the headline (0.3253 → 35.8) and the tuned
watershed (0.2638 → 29.0) are safe. **`A_nms_pool`, `B_oracle`, and `D_elbow`
have no per-country breakdown anywhere in the tracked repo**, so the same
rescaling is an unverified assumption for them. Do not apply it by hand.

## What to provide

`logs/inference/ensemble_per_country.csv` from the `ftw-planet` working tree
(currently at `/u/agiri1/github/ftw-planet/`, not readable outside that account).
Copy it into `scripts/results/` so the number has a tracked provenance.

Required columns:

| Column | Type | Notes |
|---|---|---|
| `country` | str | lowercase, underscore-separated (`south_africa`) |
| `strategy` | str | one of `A_nms_pool`, `B_oracle`, `C_pixel_vote`, `D_elbow` |
| `pq` | float | per-country Panoptic Quality in **[0, 1]**, not 0–100 |

One row per (country, strategy). All ten DENSE_10 countries must be present for
every strategy — the script refuses to emit a macro from a partial set rather
than silently averaging over fewer countries.

## How to regenerate the table

```bash
python scripts/04_analysis/aggregation_ablation.py \
  --csv scripts/results/ensemble_per_country.csv
```

Paste the printed rows over the `tabular` body of Table III in `main_grsl.tex`
and delete the `TODO(avi)` comment above it.

## Sanity check

`C_pixel_vote` must come out at **35.8**, matching Table I. If it does not, the
CSV is not the run the headline came from and nothing else in the table should be
trusted.

## Second open item (unrelated to this table)

The Method section states `K = 14` depths. `multi_hmin_ensemble.py:62`
(`find_geojsons`) only admits depths whose GeoJSON actually exists for a given
country, so the effective `K` may be below 14 somewhere. Please confirm all 14
runs exist for all ten DENSE_10 countries. If they do not, the honest phrasing is
"up to 14 depths, using the runs available per country," and the abstract needs
the same change.
