# GRSL revision, 2026-09-08

## Scope

The submission source is `main_grsl.tex`. The expanded `main.tex` compiles to
eight pages; the previous GRSL source compiles to five, including references.
The revised source uses the standard IEEEtran journal geometry.
The revised PDF compiles to four pages including references; all four pages
were rendered and inspected. Figure audits and Ruff checks pass.

The narrative now centers on parcel split/merge errors and foreground consensus
over watershed predictions from a frozen model. Global polygon counts, geometric
regularization, adaptive variants, and confidence claims are outside this letter's
main argument. They remain in the expanded draft. No new experiments were run.

The new figure generator uses the tracked country results rather than rounded
values copied into plotting code. The method and qualitative figures embed the
existing image panels unchanged, with vector labels and consistent layouts.
Sharper, consistently styled polygon outlines require the source geometries.

All three figure PDFs now embed Times New Roman. The prose pass replaces
caption slogans and rhetorical paragraph openings with direct descriptions,
while preserving numerical results, equations, and existing citations.
The introduction and experimental setup explicitly identify Fields of the
Planet as the source of the frozen PlanetScope model. Its citation metadata
agrees with the project page and arXiv:2607.04449.

## Verified results

- `ensemble_per_country.csv`: FieldVote macro PQ 35.7828 across ten countries;
  2,215 scored patches. Pooling and count-based selection use the same per-country
  patch counts. FieldVote improves on the recorded country-selected watershed in
  seven countries.
- `HEADLINE_ensemble_macro.txt`: country-selected watershed macro PQ 29.0140.
  Its country scores are stored to four decimal places. The FieldVote values in
  this file agree with the higher-precision CSV at that precision.
- `vector_gt_rescore.csv`: FieldVote 35.4610 versus watershed 28.5783 macro PQ;
  no skipped patches recorded. The watershed depth is selected separately for
  this reference convention.
- Recomputed country statistics: exact two-sided Wilcoxon p = 0.048828125.
  A percentile bootstrap with 100,000 resamples and NumPy seed 0 gives a 95%
  mean-difference interval [1.3851, 12.1802] points. The revised paper therefore
  reports [1.4, 12.2], replacing the previous interval whose sampling settings
  were not recorded. This is country variation, not seed uncertainty.

## Resolve before submission

1. **Separate voting from rasterization.** `strategy_c_pixel_vote` rasterizes
   candidate polygons on the reference grid and polygonizes the majority mask.
   The single-depth and selection baselines do not receive that round trip.
   Score every depth after the same rasterize/polygonize operation, then select
   the country-best result. This is the most important missing control for the
   claim that voting itself produces the improvement.
2. **Record the effective sweep.** `find_geojsons` loads only exports that exist.
   The tracked results contain no per-country h-value manifest. The revised
   paper says up to 14 depths; a complete manifest is needed before asserting
   K = 14 everywhere. Document the watershed implementation, units of h,
   inference settings, and any minimum-area filters as well.
3. **Repair the oracle diagnostic.** `B_oracle` records 2,193 scored patches,
   versus 2,215 for FieldVote. The implementation returns no predictions on
   patches without reference fields, changing false-positive accounting.
   Maximizing PQ separately on each patch also does not upper-bound pooled
   country PQ. The oracle row and upper-bound claim are removed from the letter.
4. **Correct the historical AP label.** The tracked evaluator calls the mean of
   object F1 values across IoU thresholds `ap_5_95`; that is not standard average
   precision. The revised letter uses PQ/SQ/RQ and omits AP. The older draft and
   historical result field names remain unchanged; do not reuse their AP claims.
5. **Complete visual provenance.** Record the patch IDs and exact depths behind
   `fig_res_*_single.png`. They are currently illustrative single-depth outputs,
   not a verified visualization of the country-selected comparator. Export
   raw polygons, imagery, transforms, and annotation type for a clean redraw.
6. **Strengthen the deployment claim.** After the controlled baseline, evaluate a
   sweep chosen without test-set feedback, depth-subset/spacing sensitivity,
   and independent model or sensor predictions. Portugal has only two scored
   patches, so its country result should not support a broad morphology claim.

The original confidence correlations, runtime CSV, boundary metric, and external
model comparisons were not independently reproduced here and are not used to
support the revised letter's central claims. The local scripts depend on raw data,
exported polygons, and helper modules from the original `ftw-planet` environment;
this checkout alone cannot rerun those experiments.
