# FieldVote rerun

Self-contained rebuild of the experiments behind `main_grsl_revised.tex` and
`supplementary.tex`, addressing `EXPERIMENT_REVIEW.md`.

Everything here runs inside the sibling `~/github/ftw-planet` checkout's venv
(`experiments/run.sh` sets `PYTHONPATH` and `cd`s there) and imports FTP's own
inference, watershed, and matching functions (`fieldvote/ftp_compat.py`), so the
scores are directly comparable with FTP Table 1. Outputs go to
`experiments/out/` (gitignored); tracked results are copied to `tables/`.

## Stages

| # | module | device | writes |
|---|---|---|---|
| 1 | `fieldvote.cache` | GPU | `out/cache/<run>/<split>/<country>/<pid>.npz` (softmax f16, argmax, GT, grid) + `manifest.json` (ckpt sha256, git SHAs, flags) |
| 2 | `fieldvote.candidates` | CPU | `out/candidates/…/polys_<h>.parquet` per-instance polygons for 14 depths + `cc`; `manifest.csv` (markers, fallback, counts) |
| 3 | `fieldvote.run_eval` | CPU | `out/scores/<run>/<split>/<protocol>/patches_<country>.parquet` — per (patch, reference, strategy) tallies |
| 4 | `fieldvote.report` | CPU | `country.csv`, `macro.csv`, `ftp_gate.csv` + printed tables |

```sh
sbatch experiments/hpc/cache_preds.sbatch                       # all 4 ckpts x {test,val}
sbatch --export=ALL,RUN=planet_b3,SPLIT=test experiments/hpc/cpu_stages.sbatch
sbatch --export=ALL,RUN=s2_b7,SPLIT=test,PROTOCOL=gsd,GSD=10 experiments/hpc/cpu_stages.sbatch
experiments/run.sh fieldvote.report --run planet_b3 --split test
```

Runs: `planet_b3` (FTP headline, `planet_efnet3_augmax_full_best.ckpt`),
`planet_b7` (the checkpoint behind the current draft), `s2_b7`, `s2_b3`
(Sentinel-2, FTP native-10 m protocol via `--protocol gsd --gsd-m 10`).

## Strategies scored (per patch)

* `cc` raw field-mask components; `raw:h*` per-instance watershed polygons at
  each depth (the object the legacy baseline scored); `rt:h*` the same depth after
  FieldVote's rasterize→polygonize (equals FTP's `inst>0` components — the
  control from review item 1)
* `vote:t{τ}` FieldVote over all 14 depths; `vote:t0.5:{k6,k4,k3,lo7,hi7,odd,even,uniform7}`
  depth-subset ablations; `--drop-one` adds leave-one-depth-out
* `nms`, `elbow` — Table II alternatives ported from `multi_hmin_ensemble.py`
* selection rows added by `report`: `ws_test` (retrospective, test-selected),
  `ws_val`, `ws_loco` (other nine countries' val), `ws_gval`; `rt_*` likewise

References: `vector` (true FTW parcels, FTP protocol — primary), `mask`
(semantic-mask components, the legacy footing), `instance` (FTW instance raster).

## Gate

`report` compares `rt:h2` vs `vector` against
`ftw-planet/logs/resolution_ablation/<condition>/<country>.csv`. Per-country
`n_patches` must match exactly and PQ must agree to ~1e-4 before anything else
is interpreted.

## Tests

```sh
cd ~/github/ftw-planet && PYTHONPATH=~/github/ICVR/experiments .venv/bin/python -m pytest -q ~/github/ICVR/experiments/tests
```

## Results (2026-09-10, `results/<run>/test/`)

Gate passes for every run: per-country `n_patches` identical to FTP and PQ within
4e-4 (planet_b3 35.49 vs 35.50; planet_b7 35.36 vs 35.37; s2_b7 24.19 vs 24.20).

Dense-10 macro PQ, true-vector reference:

| strategy | planet_b3 | planet_b7 | s2_b7 (10 m) | s2_b3 (10 m) |
|---|---:|---:|---:|---:|
| `ws_test` legacy baseline (per-instance polygons, test-selected depth) | 32.1 | 32.9 | 25.0 | 21.9 |
| `vote:t0.5` FieldVote, 14 depths | 35.4 | 35.3 | 24.3 | 21.2 |
| `rt:h2` FTP default single depth, round-tripped | 35.5 | 35.4 | 24.2 | 21.0 |
| `rt_val` / `rt_loco` single depth chosen on val / other countries' val | 35.8 / 35.8 | 35.7 / 35.7 | 24.9 / 24.8 | 21.9 / 20.8 |
| `rt_test` single depth, round-tripped, test-selected | 35.9 | 35.7 | 25.0 | 21.9 |
| `nms` / `elbow` | 17.0 / 23.6 | 17.5 / 23.2 | 24.1 / 23.7 | 20.9 / 20.5 |

Gate for s2_b3: 20.99 vs FTP 21.00.

FieldVote beats the best round-tripped single depth in 0/10 countries on every
run. Pixel check (Belgium, Croatia, Latvia): the vote mask never differs from
every single-depth mask and equals `rt:h2` exactly in 63–93% of patches. The
letter's reported gain is the rasterize→polygonize conversion (discarding
instance labels), not the vote. The legacy country-selected depth was `h=64`
in 4/10 countries, where the connected-component fallback fires on 64% of
patches. Under the S2 protocol `raw == rt` by construction (`_cap_pred_to_gsd`
rasterizes).

## Model-free variant ablation (2026-09-10, `results/ablation/`, `fieldvote/variants.py`)

Variants built from `(argmax, softmax)` only: foreground rule × split rule.
Δ macro PQ vs `rt:h2` (FTP's default), true-vector reference, all runs on one
code version:

| variant | planet_b3 | planet_b7 | planet_b37 | s2_b7 (5 m grid) | s2_b3 (5 m grid) | s2_b7 (10 m cap) | worst country |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nbg-sup-h2-t0.7` not-background fg, EDT watershed h=2, keep split if ≥70 % of line is boundary-dominant | **+2.97** | **+2.51** | **+2.69** | **+3.68** | **+3.69** | −17.9 | −0.23 (Planet), +0.10 (S2) |
| `nbg-sup-h2-t0.8` | +2.96 | +2.49 | +2.64 | +3.55 | +3.61 | −17.9 | −0.23 (Planet), +0.10 (S2) |
| `nbg-sup-h2-t0.9` | +2.60 | +2.27 | +2.34 | +3.21 | +3.24 | −17.9 | **+0.18** (Planet, never hurts), −0.02 (S2) |
| `arg-sup-h4` / `rt:h4` field fg, one depth | +0.30 | +0.34 | +0.36 | +0.38 | +0.53 | +0.63 | −0.11 |
| FieldVote `vote:t0.5` | −0.08 | −0.07 | −0.06 | +0.02 | +0.11 | +0.10 | −0.82 |
| `otsu-none`, `pf50-none` (threshold controls) | −0.7 / −0.6 | −0.5 / −0.4 | +0.5 / −0.5 | −1.0 / −0.4 | −0.9 / −0.5 | ≈0 | −2.6 |
| `arg-close{1,2}` gap closing | −0.8 / −1.1 | −0.7 / −1.1 | −1.4 / −1.6 | −0.9 / – | −1.0 / – | +0.1 / +0.4 | −2.9 |
| `nbg-none`, `nbg-pb-*`, `nbg-close*` | ≤ −10 | ≤ −10 | ≤ −10 | −24.9 | ≤ −10 | ≤ −1.4 | – |

Reading: the gain of the `nbg-sup` family comes from recovering parcels the
field-argmax loses to the boundary class (miss 0.36 → 0.30, count bias −52 →
−42, PQ_small 15.7 → 18.4) while the support test throws away the geometric
watershed splits that are not backed by boundary evidence. τ has a plateau
over 0.7–0.9; h=2 (FTP's default) is the right depth (h=8 collapses through
the marker fallback). Averaging B3+B7 softmax does not help (its own baseline
is 35.26 < B3's 35.49).

Sentinel-2 is scored on the prediction grid (FTW S2 models are run at
upsampled resolution and the output is not downsampled), where the same
variant gains +3.7 on both encoders with every country improving. FTP's native-10 m cap
(`--protocol gsd`) is kept only for the Table 1 gate; under that cap the
variant loses because sub-10 m separations and thin parcels are rasterized
away, and widening the kept lines to 10–25 m (`-w2..-w5`) does not recover it.

## Instance models (DelineateAnything-v2), `fieldvote/cache_da.py`, `fieldvote/da_softmax.py`

DA-v2 (`MykolaL/DelineateAnything`, `DelineateAnythingv2.pt`, FTP's best swept
setting `--imgsz 1024 --conf 0.05`) emits instances + confidences, no boundary
class. `cache_da` runs ultralytics exactly like FTP's `eval_delineate_anything.py`
and stores, per pixel, the top-3 instances by confidence after a 1-px dilation
(so touching instances contest their seam) plus DA's raw conf-ranked partition.
`da_softmax` converts that to the standard `(prob, seg)` cache:

* `ratio` — p_bg = [no instance]; inside foreground r = c2/c1, p_boundary = r,
  p_field = 1 − r. Confidences enter only relatively: at `--conf 0.05` most are
  below 0.5, and any rule using them as absolute probabilities (tried: `1 − c1`,
  noisy-OR) labels every pixel background.
* `hard` — confidence-free control: boundary = two or more instances overlap.

Variants then run unchanged; `*-msup-t{τ}` uses DA's own partition as watershed
markers (contested seam pixels removed so a line can form) so only the support
test decides which of DA's seams survive. `v:da-native` (conf-ranked painting of
the raw masks, overlaps resolved) is DA's baseline row. Painting an outer ring at
a fixed 0.95 was rejected: it reproduces the partition it came from and the
support test becomes an identity.

Result (`results/da_v2_*/`): DA-v2 native 10.2 PQ (median 0 detections per patch
in five of ten countries). With `ratio` every variant loses 2.8–6.0 to native —
confidence disparity between adjacent instances is not evidence that a division
is false. With `hard`, `nbg-msup-t{τ}` is best at τ→0 (+0.96, 7/10, worst −4.6)
and decreases monotonically with τ: the gain is the overlap-fragment cleanup, the
support test contributes nothing. The method needs a predicted boundary class;
DA-v2 enters the letter as a baseline row.

## Revised letter (`main_grsl_revised.tex`)

Written from the rebuilt results only; FieldVote does not appear. Build with
`latexmk -pdf main_grsl_revised.tex` (5 pages incl. references). Regenerate its
inputs from `experiments/results/`:

```sh
V=~/github/ftw-planet/.venv/bin/python
$V experiments/figures/make_tables.py      # tables/rev_{main_rows,compare_rows,ablation_rows,values}.tex
$V experiments/figures/make_supp_tables.py # tables/supp_*_rows.tex
$V experiments/figures/per_country.py     # figures/rev_percountry.pdf
$V experiments/figures/gallery.py         # figures/rev_gallery.pdf: one patch per country, released vs ours, patch PQ printed
$V experiments/figures/method_fig.py      # figures/rev_method.pdf: every intermediate image on one crop
```

Gallery patch selection: largest per-patch PQ gain among test patches with
10-120 reference parcels whose densest square window shows >= 8 parcels
covering >= 25 % of the window; Portugal (two test patches) is omitted.
Figure text is Nimbus Roman (Times New Roman is not installed here). Paper
default is `nbg-sup-h2-t0.7`; "best depth" rows are `rt_test` restricted to
`rt:h*` strategies.

## Prior post-processing baselines (`fieldvote/priors.py`)

Four label-free conversions re-implemented on the same cached predictions, so
the comparison isolates the split/merge rule:

| name | method | knobs |
|---|---|---|
| `ucm` | hierarchical merging of a watershed over-segmentation by mean boundary probability along each shared border (`skimage.graph.merge_hierarchical`) | `k` |
| `swst` | stochastic watershed: `m` draws of `n` random markers, cut where the line frequency reaches `t` | `n`, `m`, `t` |
| `iterh` | iterative marker depth: watershed at `h=8`, re-segment regions over `a` px at 4, 2, 1 | `a` |
| `thr` | extent/boundary thresholds `p_field >= te`, `p_bnd < tb`, then components | `te`, `tb` |

Each grid is swept until its optimum is interior, its setting is chosen on the
**validation** split, and the same setting is reported on test for every run.
`arg-ws{h}` is the released conversion expressed as a variant, so post-steps
(`+a`, `+f`, `+m`) can be applied to it on equal terms.

## Cost and transfer

```sh
sbatch --export=ALL,RUN=planet_b3 experiments/hpc/profile.sbatch   # out/profile/<run>_<split>.csv
experiments/run.sh fieldvote.transfer --run planet_b3              # out/transfer/<run>.csv
experiments/rebuild_paper.sh                                       # every table, figure, both PDFs
```

Copies of the cost and transfer CSVs behind Table II live in `results/cost/`.

`profile_methods` times each method on one core (threads pinned to 1, minimum
of three repeats) and samples peak RSS in a background thread, on patches
stratified by how many reference parcels they contain. `transfer` compares one
shared setting against a per-country setting chosen on that country's test
labels, which is the headroom a method gives up by shipping one setting.

## Audit findings

* Junction pixels could leave two parcels 4-adjacent, so a kept split was not
  always a gap. Fixed by resolving fill candidates against each other and
  relabelling by connected components; `test_no_adjacent_labels_on_many_real_patches`
  checks 40 real patches across all ten countries.
* Modifiers use `+` because `-a` collided with `iterh-a20000`.
* `variants.sbatch` scored every `v:` tag on disk, so it re-scored the prior
  baselines a second time. Values were identical, but `n_patches` doubled;
  `load_patches` now drops duplicate keys and the job passes an explicit list.
* `report` no longer dies when a strategy covers only some countries (a job
  still running, or one scored on a subset); it skips those and says so.
* The FTP gate only runs where a published number exists for that protocol:
  PlanetScope in the prediction frame, Sentinel-2 capped to 10 m.
* Verified: reference counts agree across strategies (patch ids repeat across
  countries, so keys must be country + patch), `tp+fn=n_gt` and `tp+fp=n_pred`
  on every row, size bins partition the references, one selection row per
  country, and macro PQ recomputed from raw tallies reproduces FTP's 35.50.

## Paper artifacts

`main_grsl_revised.tex` (5 pages) and `supplementary.tex` (2 pages). The
supplement carries the full tau and h sweeps on all four runs, every prior
setting tried with its validation and test score, per-country scores, and
timings in all four density bins on both sensors.

The shipped method is `nbg-sup-h2-t0.7+A900`: not-background foreground,
boundary-supported splits at h=2 with tau=0.7, and a 900 m^2 minimum parcel
area. The threshold is in ground area, not pixels, because a pixel threshold
means different parcel sizes on the two sensors (Planet 9.0 m^2/px, the
upsampled S2 grid 5.5 m^2/px here). Headline gains compare against the
released pipeline given the same minimum-size step (`arg-ws2+A900`).

## Analysis tools

Console reports over `out/scores` and `out/candidates`; none of them writes a
paper input, so they are not part of `rebuild_paper.sh`.

```sh
experiments/run.sh fieldvote.ablation --runs planet_b3 planet_b7 planet_b37 --protocol pixel
experiments/run.sh fieldvote.iou_sweep --run planet_b3 --split val \
    --variants nbg-sup-h2-t0.7 nbg-sup-h2-t0.7+H40          # out/iou_sweep/<run>_<split>.csv
experiments/run.sh fieldvote.boundary_error --run planet_b3 --split test \
    --variants arg-ws2+A900 nbg-sup-h2-t0.7+A900            # out/boundary/<run>_<split>.csv
```

`ablation` is the cross-run variant ranking behind `results/ablation/cross_*.csv`
and the variant table above. `iou_sweep` rescores from IoU 0.2 upward, because a
rule that adds splits can be punished by the 0.5 threshold itself rather than by
its polygons; `boundary_error` reports FTP's symmetric chamfer and the polygon
count delta over matched pairs. Neither number is in the letter yet.

## Withdrawn: the global-scale run

`results/global_s2/` and the third-model row it fed are kept for provenance but
must not be reported. The public FTW global Zarr store is rasterized at 10 m
while the model predicts near 5 m, so those scores are resolution-floored rather
than a clean third-model result. The modules that produced them
(`global_*.py`, `cache_global.py`, `hpc/global*.sbatch`) were removed; recover
them from history if the store is ever republished at native resolution.

## TODO

* Runtime breakdown per stage (review item 14).
