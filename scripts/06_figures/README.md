# Rebuilding the paper figures

Every script in the figure chain is in this repository. This note records what
each figure is built from, which patch it came from, and the order the scripts
run in.

## Figures that need no external data

`make_figures.py` carries the plotted values as literals, so the four vector
figures rebuild anywhere with only matplotlib installed:

```
python scripts/06_figures/make_figures.py --out figures
```

writes `fig_percountry.pdf`, `fig_adaptive.pdf`, `fig_votethreshold.pdf` and
`fig_simplify.pdf`. Styling is shared through `style.py`.

## Figures that need the run store

The raster panels are rendered from real patches, so they need two inputs that
are too large to keep here and are covered by the dataset license. Both are
regenerable from scripts already in this repository.

| Input | What it is | How to get it |
| --- | --- | --- |
| `FTW_PLANET_REPO` | checkout providing `multi_hmin_ensemble`, the `polygon_metrics_eval` helpers, the `ftw_planet` package, the FTW-Planet dataset under `data/`, and the per-h_min GeoJSONs under `logs/inference/` | dataset from the FTW-Planet release on Hugging Face (`taylor-geospatial/ftw-planet`); the per-h_min GeoJSONs come from `scripts/01_inference/` |
| `FIELDVOTE_STORE` | run store holding `predict_v2/<country>/<patch>.npz`, the per-patch softmax | produced by `scripts/01_inference/predict_v2.local.py` |

Point the renderer at them:

```
export FTW_PLANET_REPO=/path/to/ftw-planet
export FIELDVOTE_STORE=/path/to/run-store
```

Both default to the paths used for the submitted figures, so the command line
is unchanged on the machine that produced them.

## Panel provenance

Each row of the results grid, and the teaser, is the top ranked candidate for
its country under the scan in `make_method_fig_panels.py`, which orders patches
by how much low h_min over-fragmentation the consensus removes. Regenerate any
row with `--country <country> --top 1`. The h_min set exported for these
countries is {2, 4, 8, 12}, so low, mid and high resolve to 2, 8 and 12.

| Figure group | Country | Patch | Low h_min polygons to consensus | Mean agreement |
| --- | --- | --- | --- | --- |
| `fig_pipe_*` (teaser) | belgium | `g6_00035_3` | 44 to 11 | 0.66 |
| `fig_res_hr_*` | croatia | `g10-3_00086_3` | 57 to 10 | 0.48 |
| `fig_res_lv_*` | latvia | `g41_00031_11` | 35 to 5 | 0.90 |
| `fig_res_be_*` | belgium | top ranked candidate | | |
| `fig_res_se_*` | sweden | top ranked candidate | | |
| `fig_res_lt_*` | lithuania | top ranked candidate | | |

The Belgium, Sweden and Lithuania rows were not captured in a retained render
log, so their patch identifiers are recorded here as the selection rule rather
than as literals. Rerunning the scan reproduces them.

`fig_qual_*` is split from `qualitative_example.png`, belgium patch
`g6_00089_17`. `fig_global_*` is split from `global_pampas.png`, the Argentina
Pampas window written by `scripts/05_global/global_postprocess.py`.

## Order of operations

```
python scripts/06_figures/make_method_fig_panels.py --country belgium --patch g6_00035_3
python scripts/06_figures/split_composites.py --dir figures
python scripts/06_figures/crop_panels.py --dir figures
python scripts/06_figures/make_figures.py --out figures
```

`make_method_fig_panels.py` writes numbered panels into `<out>/<patch>/`
(`1_input`, `1b_input` for the second temporal window, `2_prob`, `3_ws_low`,
`5_ws_high`, `6_consensus`, `7_agreement`, `8_gt`). Those are copied into
`figures/` under the `fig_pipe_*` and `fig_res_*` names the document uses,
where `fv` is the consensus panel and `single` is the tuned watershed
operating point. `split_composites.py` then divides the two multi panel PNGs,
and `crop_panels.py` reframes each figure row on a shared box measured from
the annotation so the panels stay registered.
