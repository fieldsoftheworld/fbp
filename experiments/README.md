# fbp: field boundary post-processing

`fbp` turns the three-class output (background, field, boundary) of a
field-boundary network into field polygons with a chosen post-processing
strategy, and scores strategies against each other on the Fields of The World
parcels. It holds BoundaryVote, the FTP pipeline, and the six prior methods
the letter compares, behind one interface, plus the evaluation harness and the
scripts that write the paper's tables and figures.

The code runs inside the sibling [`ftw-planet`](https://github.com/fieldsoftheworld/ftw-planet)
checkout's virtual environment (`run.sh` sets `PYTHONPATH` and `cd`s there, or
point `FTW_PLANET_REPO` elsewhere) and imports Fields of the Planet's own
inference, watershed, and matching functions (`fbp/eval/ftp_compat.py`), so
the scores are directly comparable with FTP Table 1. Stage outputs go to
`out/` (gitignored, override with `BOUNDARYVOTE_OUT`); the pooled results the
paper reads are copied into `results/`.

## Package

```
fbp/
  core/      shared primitives: surfaces (boundary distance, boundary probability,
             oriented signal), watershed_regions, RAG merging, line-vote merging,
             post-steps (drop / fill / merge small parcels)
  methods/   one module per strategy behind the Method base class, and a registry
             that parses strategy ids
  eval/      cache predictions, build candidates, score, pool, gate against FTP,
             transfer study, bootstrap, timing profile, compare
  paper/     the shipped configuration, the parameter grids the paper reports,
             and the table and figure generators
```

### Strategies and their ids

A strategy id is `<foreground>-<rule>[-<param>...][+<modifier>...]`, for example
`nbg-pb-h0.01-t0.3+A900` (BoundaryVote as shipped) or `arg-ws2+A900` (the FTP
pipeline with the same minimum parcel size). `fbp.methods.parse(id)` returns a
`Method`; calling it on `(seg, prob, px_m2)` returns an int32 label image.

| foreground | meaning |
|---|---|
| `arg` | field argmax (FTP) |
| `nbg` | argmax is not background, i.e. field or boundary |
| `pf50`, `otsu` | field probability above 0.5, or above the patch's Otsu threshold |

| rule | method | knobs |
|---|---|---|
| `none` | connected components | |
| `ws{h}` | FTP pipeline: distance watershed at depth `h` on the field class, then components of the union | `h` |
| `pb-h{h}-t{tau}` | **BoundaryVote**: watershed of the boundary probability seeded at minima deeper than `h`, keep a line when at least `tau` of its pixels vote boundary | `h`, `tau` |
| `sup-h{h}-t{tau}` | the same vote on the boundary distance map (`h` in pixels); `ssup`/`asup`/`dsup` use mean share, mean `p_bnd`, mean margin as the line statistic; `osup` votes on Zhu et al.'s oriented signal | `h`, `tau` |
| `close{r}` | boundary class closed with a `(2r+1)^2` square and subtracted, then components | `r` |
| `thr-te{te}-tb{tb}` | extent/boundary thresholds, then components (Waldner et al. 2020) | `te`, `tb` |
| `ucm-k{k}` | hierarchical merging by mean boundary probability along borders (Arbelaez et al. 2011) | `k` |
| `iterh-a{a}` | iterative marker depth 8, 4, 2, 1 for regions over `a` px (Koyuncu et al. 2016) | `a` |
| `swst-n{n}-m{m}-t{t}` | stochastic watershed, `m` draws of `n` markers, cut at line frequency `t` (Angulo and Jeulin 2007) | `n`, `m`, `t` |
| `owt-l{l}-h{h}` | oriented watershed, arcs weighted at their orientation, greedy merge below `l` (Zhu et al. 2024) | `l`, `h` |
| `gmm` | per-region Gaussian-mixture thresholds on extent and boundary probability (Amin et al. 2025), fitted by `fbp.eval.gmm_fit` | fitted |

Modifiers: `+A{m2}` drops parcels under `m2` square metres (converted with the
patch's pixel area, so it means the same on either sensor), `+l{L}` requires a
kept line of at least `L` px, `+b{R}` counts a line pixel as supported when any
pixel within `R` favours boundary, `+f{A}` fills holes under `A` px (`-1` all),
`+m{A}` merges parcels under `A` px into their longest-border neighbour.

### Adding a strategy

Subclass `fbp.methods.base.Method`, set `rule` to the id keyword, implement
`split(fg, seg, prob, ctx)` and `from_parts`, and add the class to the
registry in `fbp/methods/__init__.py`. Everything downstream (candidates,
scoring, profiling, `compare`) works from the id string.

## Reproducing the paper

Submit from the repository root. Job sizes follow `sacct` of the runs that
produced the tracked results. Runs: `planet_b3`, `planet_b7`, `s2_b7`, `s2_b3`
(checkpoint paths in `hpc/cache_preds.sbatch`; all trained in `ftw-planet`
for the Fields of the Planet paper with the PRUE recipe). Sentinel-2 models
run inference at twice the image resolution and are scored on that grid
(`pixel` protocol); the `gsd` protocol renders predictions at 10 m and is used
only for the Sentinel-2 reproduction gate against FTP's published numbers.

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

# strategies: BoundaryVote, its ablations, and the prior methods, by group
experiments/run.sh fbp.paper.grids --list            # group names and sizes
sbatch --export=ALL,RUN=planet_b3,SPLIT=test,TAG=core,GRIDS="main sweep ablation foreground" hpc/variants.sbatch
sbatch --export=ALL,RUN=planet_b3,SPLIT=test,TAG=priors,GRIDS="thr ucm iterh swst owt gmm" hpc/variants.sbatch
sbatch --export=ALL,RUN=planet_b3,SPLIT=test,TAG=supp,GRIDS="linestat proposals steps minsize" hpc/variants.sbatch
#   ... the same three for RUN=planet_b3,SPLIT=val and RUN=s2_b7,SPLIT=test;
#   RUN=planet_b7 and RUN=s2_b3 need only TAG=core (fbp.paper.grids.RUN_GROUPS)

# timing and memory, one core, both sensors
sbatch --export=ALL,RUN=planet_b3 hpc/profile.sbatch
sbatch --export=ALL,RUN=s2_b7 hpc/profile.sbatch

# pool, gate, tables, figures, PDFs
experiments/rebuild_paper.sh
```

`rebuild_paper.sh` runs `report`, `transfer` and `bootstrap`, copies their
outputs and the profile CSVs into `results/`, regenerates `tables/` and
`figures/`, and compiles both PDFs. Each `report` prints the FTP gate:
per-country `n_patches` must match FTP's CSV exactly and PQ must agree to
about 1e-3 before anything else is interpreted.

## Comparing strategies

Once a run is scored, any set of strategy ids can be compared on the pooled
country-averaged scores:

```sh
experiments/run.sh fbp.eval.compare --run planet_b3 --methods arg-ws2+A900 nbg-pb-h0.01-t0.3+A900 nbg-ucm-k0.7+A900
```

New ids are scored with `hpc/variants.sbatch` (`VARIANTS="..."`) and then
appear in `compare` and `report`.

## Results layout

```
results/<run>/test/pixel/{country,macro,ftp_gate}.csv   pooled scores, prediction grid
results/<run>/test/gsd/...                              Sentinel-2 gate at 10 m
results/planet_b3/val/pixel/...                         validation split (selection)
results/<run>/<split>/cache_manifest.json               checkpoint hash and inference flags
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
