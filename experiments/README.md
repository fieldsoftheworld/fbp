# BoundaryVote experiments

Everything behind `main_grsl_revised.tex` and `supplementary.tex`: caching the
released models' predictions, turning them into polygons with BoundaryVote,
the FTP pipeline and six prior post-processing methods, scoring against the
FTW parcels, and writing the tables and figures.

The code runs inside the sibling [`ftw-planet`](https://github.com/fieldsoftheworld/ftw-planet)
checkout's virtual environment (`run.sh` sets `PYTHONPATH` and `cd`s there, or
point `FTW_PLANET_REPO` elsewhere) and imports Fields of the Planet's own
inference, watershed, and matching functions (`boundaryvote/ftp_compat.py`),
so the scores are directly comparable with FTP Table 1. Stage outputs go to
`out/` (gitignored, override with `BOUNDARYVOTE_OUT`); the pooled results the
paper reads are copied into `results/`.

## Stages

| # | module | device | writes |
|---|---|---|---|
| 1 | `boundaryvote.cache` | GPU | `out/cache/<run>/<split>/<country>/<pid>.npz` (softmax f16, argmax, GT, grid) + `manifest.json` (checkpoint sha256, git SHAs, flags) |
| 2 | `boundaryvote.candidates` | CPU | `out/candidates/<run>/<split>/<country>/polys_<tag>.parquet`, per-instance polygons for the FTP depth sweep (`cc`, `h*`) and every variant (`v:<name>`) |
| 3 | `boundaryvote.run_eval` | CPU | `out/scores/<run>/<split>/<protocol>/patches_<tag>_<country>.parquet`, per (patch, strategy) tallies |
| 4 | `boundaryvote.report` | CPU | `country.csv`, `macro.csv`, `ftp_gate.csv` next to the patches |
| 5 | `boundaryvote.transfer`, `boundaryvote.bootstrap`, `boundaryvote.profile_methods` | CPU | `out/transfer/<run>.csv`, `out/bootstrap/<run>.csv`, `out/profile/<run>_<split>.csv` |

Runs and checkpoints (all trained in `ftw-planet` for the Fields of the Planet
paper with the PRUE recipe; paths in `hpc/cache_preds.sbatch`):

| run | model | checkpoint |
|---|---|---|
| `planet_b3` | PlanetScope EfficientNet-B3 | `logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt` |
| `planet_b7` | PlanetScope EfficientNet-B7 | `logs/best_checkpoints/planet_efnet7_best.ckpt` |
| `s2_b7` | Sentinel-2 EfficientNet-B7 | `logs/best_checkpoints/s2_efnet7_best.ckpt` |
| `s2_b3` | Sentinel-2 EfficientNet-B3 | `logs/prue/ftw_s2-unet-efnet3-crop256-s2-v3-augmax-full/.../epoch=42*.ckpt` |

Sentinel-2 models run inference on a grid at twice the image resolution
(`--upsample-to 512`), FTP's setting, and are scored on that grid (`pixel`
protocol). The `gsd` protocol renders predictions at 10 m in UTM and is used
only for the Sentinel-2 reproduction gate against FTP's published numbers.

## Reproducing the paper

Submit from the repository root. Job sizes follow `sacct` of the runs that
produced the tracked results.

```sh
# 1. cache predictions, all four checkpoints x {test, val}
sbatch hpc/cache_preds.sbatch

# 2+3. FTP depth sweep and scores (the `cc`, `raw:h*`, `rt:h*` strategies)
for RUN in planet_b3 planet_b7 s2_b7 s2_b3; do
  sbatch --export=ALL,RUN=$RUN,SPLIT=test hpc/cpu_stages.sbatch
done
sbatch --export=ALL,RUN=planet_b3,SPLIT=val hpc/cpu_stages.sbatch
sbatch --export=ALL,RUN=s2_b7,SPLIT=test,PROTOCOL=gsd,GSD=10 hpc/cpu_stages.sbatch   # S2 gate
sbatch --export=ALL,RUN=s2_b3,SPLIT=test,PROTOCOL=gsd,GSD=10 hpc/cpu_stages.sbatch

# variants: BoundaryVote, its ablations, and the prior methods, by group
python -m boundaryvote.grids --list                 # group names and sizes
sbatch --export=ALL,RUN=planet_b3,SPLIT=test,TAG=core,GRIDS="main sweep ablation foreground" hpc/variants.sbatch
sbatch --export=ALL,RUN=planet_b3,SPLIT=test,TAG=priors,GRIDS="thr ucm iterh swst owt gmm" hpc/variants.sbatch
sbatch --export=ALL,RUN=planet_b3,SPLIT=test,TAG=supp,GRIDS="linestat proposals steps minsize" hpc/variants.sbatch
#   ... the same three for RUN=planet_b3,SPLIT=val and RUN=s2_b7,SPLIT=test;
#   RUN=planet_b7 and RUN=s2_b3 need only TAG=core (grids.RUN_GROUPS lists this)

# 5. timing and memory, one core, both sensors
sbatch --export=ALL,RUN=planet_b3 hpc/profile.sbatch
sbatch --export=ALL,RUN=s2_b7 hpc/profile.sbatch

# 4 + tables + figures + PDFs
experiments/rebuild_paper.sh
```

`rebuild_paper.sh` runs `report`, `transfer` and `bootstrap`, copies their
outputs and the profile CSVs into `results/`, regenerates `tables/` and
`figures/`, and compiles both PDFs. Each `report` prints the FTP gate:
per-country `n_patches` must match FTP's CSV exactly and PQ must agree to
about 1e-3 before anything else is interpreted.

## What is where

* `boundaryvote/shipped.py` defines the shipped configuration once
  (`nbg-pb-h0.01-t0.3+A900`: not-background foreground, watershed of the
  boundary probability seeded at minima deeper than 0.01, vote at tau = 0.3,
  900 m^2 minimum parcel size) and the FTP pipeline as a variant. Every table,
  figure, and analysis script imports it.
* `boundaryvote/variants.py` implements BoundaryVote (`supported_split`) and
  the strategy-string dispatcher `run_variant`; the module docstring lists
  every rule and modifier.
* `boundaryvote/priors.py` and `priors_recent.py` re-implement the prior
  methods on the same cached predictions (no code was released for the last
  two):

  | name | method | knobs |
  |---|---|---|
  | `thr` | extent/boundary thresholds, `p_field >= te`, `p_bnd < tb`, then components (Waldner et al. 2020) | `te`, `tb` |
  | `ucm` | hierarchical merging of a watershed over-segmentation by mean boundary probability along each border (Arbelaez et al. 2011) | `k` |
  | `iterh` | iterative marker depth: watershed at h=8, re-segment regions over `a` px at 4, 2, 1 (Koyuncu et al. 2016) | `a` |
  | `swst` | stochastic watershed: `m` draws of `n` random markers, cut where the line frequency reaches `t` (Angulo and Jeulin 2007) | `n`, `m`, `t` |
  | `owt` | oriented watershed on the boundary probability, arcs weighted at their own orientation, greedy merge below level `l` (Zhu et al. 2024) | `l`, `h` |
  | `gmm` | two-component mixtures fitted per country to the extent and boundary probabilities (`boundaryvote/gmm_thresholds.py`), thresholds, thinned boundary subtracted, components (Amin et al. 2025) | fitted |

* `boundaryvote/grids.py` lists every variant setting the paper reports, by
  group, and `tests/test_grids.py` checks that each one was scored in the
  tracked results. Prior methods are selected on the PlanetScope B3 validation
  split (`transfer.py`), the oriented watershed with the minimum-size step in
  place because its slivers make the bare grid meaningless.
* `figures/make_tables.py` and `make_supp_tables.py` write `tables/`;
  `figures/per_country.py`, `gallery.py`, `method_fig.py` write `figures/`.
  Nothing in the two `.tex` files is typed by hand except prose.

## Results layout

```
results/<run>/test/pixel/{country,macro,ftp_gate}.csv   pooled scores, prediction grid
results/<run>/test/gsd/...                              Sentinel-2 gate at 10 m
results/planet_b3/val/pixel/...                         validation split (selection)
results/<run>/test/cache_manifest.json                  checkpoint hash and inference flags
results/transfer/planet_b3.csv                          shared vs per-country setting
results/bootstrap/<run>.csv                             patch bootstrap, 1000 draws, seed 0
results/profile/{planet_b3,s2_b7}_test.csv              ms and peak MB per patch and method
```

## Tests and tooling

```sh
cd ~/github/ftw-planet && PYTHONPATH=~/github/ICVR/experiments .venv/bin/python -m pytest -q ~/github/ICVR/experiments/tests
~/github/ftw-planet/.venv/bin/ruff check experiments && ~/github/ftw-planet/.venv/bin/ty check
```

Tests that need the prediction cache skip when it is absent.
