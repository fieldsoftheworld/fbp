# FieldVote experimental review

Review date: September 8, 2026.
Manuscript: `main_grsl.tex`, intended for IEEE Geoscience and Remote Sensing Letters.

The current experiments need additional controls before submission.
The reported arithmetic checks out, but the experiments do not yet establish how much of the improvement comes from voting rather than from discarding watershed instance divisions.
Reproducing the Fields of the Planet (FTP) baseline and isolating this effect should take priority over expanding the figure gallery.

This review applies CVPR-level scrutiny to the experimental evidence while keeping the scope of a GRSL letter in mind.
It covers the manuscript, saved result tables, evaluation scripts, FTP's paper and local code, and rendered figures.
The audit recomputed aggregate statistics and ran targeted synthetic checks.
It did not rerun the full benchmark because the raw prediction exports and exact generating environment were unavailable locally.
Synthetic counterexamples identify possible failures in the code; their effect on the reported benchmark scores remains to be measured.

Source locators below identify repositories, filenames, and functions without depending on a particular machine's filesystem.

**Comparison with FTP**

| Result | PQ |
|---|---:|
| FTW-PRUE+, Sentinel-2, EfficientNet-B3 | 21.0 |
| FTP-PRUE+, PlanetScope, EfficientNet-B3 | 35.5 |
| FTP-PRUE+, PlanetScope, EfficientNet-B7 | 35.4 |
| FieldVote study: country-selected watershed, semantic references | 29.0 |
| FieldVote study: FieldVote, semantic references | 35.8 |
| FieldVote study: country-selected watershed, vector references | 28.6 |
| FieldVote study: FieldVote, vector references | 35.5 |

The relevant PlanetScope model is FTP-PRUE+, and the published B3 result is 35.5 PQ, rather than 35.8.
FieldVote currently reaches approximately the performance already reported by FTP.
It has not demonstrated an improvement over FTP's complete prediction pipeline.
The published values appear in [FTP Table 1](https://arxiv.org/html/2607.04449v1#S2).

These rows cannot yet be treated as one controlled comparison.
Both studies name the same ten countries, but their prediction extraction and scoring paths differ.
Excluding Kenya does not explain the gap between the 29.0 baseline and FTP's results.

**Findings and required experiments**

1. **Critical — Isolate the effect of discarding instance labels.**

   FTP's evaluator converts watershed instances to `inst_pred > 0`, then extracts binary connected components.
   This removes divisions between touching instance labels.
   FieldVote's single-depth baseline scores the original polygons, while FieldVote rasterizes their union and extracts connected components.

   A synthetic check using the actual FTP watershed function changed the output from two instances to one as depth increased.
   All 14 binary masks were identical, and voting changed zero pixels.
   This demonstrates a possible failure of the proposed mechanism, but does not establish what happened in the saved exports.

   Add raw field-mask connected components and every single depth passed through FieldVote's identical rasterize–polygonize conversion.
   Select the best single depth again after this conversion.
   If these controls recover most of the gain, the contribution needs to be reconsidered.

   Sources: FTP `polygon_metrics_eval.py`, `evaluate_country`; ICVR `multi_hmin_ensemble.py`, `strategy_c_pixel_vote`.

2. **Critical — Reproduce FTP before interpreting 29.0 as the baseline.**

   Run FTP's released B3 and B7 checkpoints through the published pipeline, then apply FieldVote to the same cached predictions.
   Fix checkpoint, seasonal-channel order, normalization, padding, D4 augmentation, watershed settings, grid, and reference polygons.

   FTP's native-resolution evaluator also performs a binary rasterization round trip.
   Changing the reference column from “Mask” to “Vector” does not reproduce that pipeline.
   Explain the roughly six-point baseline deficit before claiming a six-point improvement over the existing method.

   Source: FTP `polygon_metrics_eval.py`, `_cap_pred_to_gsd` and its use in evaluation.

3. **High — Audit order-dependent scoring in Table II's pooling baseline.**

   NMS can retain overlapping polygons when their mutual IoU is below 0.5.
   The matcher then accepts the first eligible prediction for each reference polygon.

   In a synthetic check using the actual functions, NMS retained two polygons with mutual IoU 0.3.
   Reversing their order changed PQ from 40.0 to 46.7.
   This is not a measured change to Table II.

   Standard PQ's unique-matching argument assumes nonoverlapping segments.
   Check overlap in the actual pooling outputs, then use a consistent partition representation or an explicitly defined matching rule.
   Rescore Table II if ambiguous overlaps occur.

   Sources: ICVR `multi_hmin_ensemble.py`, `strategy_a_nms_pool`; FTP `polygon_metrics_eval.py`, `_match_shapes`; [Panoptic Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf).

4. **High — Separate development decisions from final test evaluation.**

   The country-selected baseline uses test labels, which the manuscript correctly discloses.
   It is a retrospective diagnostic, not the deployed FTP baseline.

   The threshold-sweep script also evaluates candidate thresholds on `split="test"`.
   This does not prove that test results determined the final threshold, but the development history needs documenting.
   Using no labels at inference does not establish independent hyperparameter selection.

   Keep the retrospective row, add validation-selected baselines, and lock the depth sweep and voting threshold before final testing.

   Source: ICVR `consensus_sweep.py`.

5. **High — Record the provenance and completeness of candidate exports.**

   Effective depths are unrecorded.
   The `h=2` candidate comes from a specially named `_fields_full.geojson`, while other depths use separate exports.
   The available records do not verify that every file shares a checkpoint, inference configuration, area filter, and complete patch coverage.

   Record a manifest containing checkpoint hash, commands, software version, patch IDs, and available depths.
   Distinguish genuinely empty predictions from missing exports because treating both as zero votes can change results.

   Source: ICVR `multi_hmin_ensemble.py`, `find_geojsons` and `load_polygons_by_patch`.

6. **High — Verify exact correspondence for vector-reference scoring.**

   The vector rescore associates reference files with evaluation patches by maximum bounding-box IoU and accepts matches as low as 0.25.
   Zero skipped patches establishes coverage, not correct correspondence.

   Use an explicit patch mapping, or record and inspect match IoUs, bounds, CRS, clipping, and duplicate matches.
   The vector-reference result is useful, but this matching step currently limits confidence in it.

   Source: ICVR `rescore_vector_gt.py`, `gt_shapes_for`.

7. **High — Verify sample identities and empty-patch accounting.**

   FTP records 2,544 evaluated patches, while FieldVote reports 2,215 scored patches.
   This is not automatically a dataset mismatch because FieldVote's scorer skips patches where predictions and references are both empty.

   Reconstructed reference-object totals agree with FTP within its CSV rounding precision in every country, which is reassuring.
   Export the exact evaluated IDs and distinguish total, empty, prediction-missing, and scored patches.
   Equal row counts across methods do not prove equal patch identities.

   Sources: FTP `per_country_metrics.csv`; ICVR `vector_gt_rescore.csv` and `score_polygons_against_gt`.

8. **High — Report the negative field-count result.**

   The saved vector results give the following country-macro mean absolute per-patch count errors.

   | Method | Absolute field-count error ↓ |
   |---|---:|
   | Country-selected watershed | 39.4 |
   | FieldVote | 61.6 |

   Cambodia increases from 259.5 to 470.0, and Portugal from 48.5 to 77.5.
   Higher PQ does not establish uniformly better parcel counting.

   Include normalized count error, signed count bias, and explicit split/merge rates.
   Define normalization and aggregation carefully because absolute country-total error and mean absolute patch error are different quantities.
   The main discussion should acknowledge this tradeoff.

   Source: ICVR `vector_gt_rescore.csv`, columns `fv_dn` and `tw_dn`.

9. **High — Quantify the proposed split/merge mechanism.**

   SQ and RQ do not distinguish false splits, false merges, missed fields, and false detections.
   Add a reference-to-prediction overlap analysis that counts these outcomes.

   Measure how often FieldVote changes foreground pixels, removes boundaries, or produces a result different from every individual depth.
   Without these measurements, the claim that consensus corrects fragmentation rests largely on selected visual examples.

10. **High — Test the choices that define the voting method.**

    Ablate voting threshold, number of depths, depth range, spacing, and missing runs.
    Compare equal budgets and include a central single depth.

    If binary supports are nested across depths, majority voting reduces to a middle support mask.
    Check whether this occurs before treating multiple depths as complementary evidence.

    Report what happens when large depths eliminate markers and trigger fallback behavior.
    Include the frequency of fallback cases in the sweep analysis.

11. **High — Add stronger, inexpensive baselines.**

    The minimum comparison should contain the released FTP pipeline, raw field-interior connected components, validation-selected single-depth watershed, the same single-depth output after identical rasterization and polygonization, and FieldVote.

    Then consider simple boundary-gap closing or region merging, and one relevant adaptive or hierarchical watershed comparator.
    Iterative multi-depth marker selection already exists, so comparison with that family would strengthen the novelty argument.
    See [Iterative h-minima-based marker-controlled watershed](https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.22824).

    The current count-based baseline ignores unequal depth spacing, breaks ties toward earlier depths, and never directly selects the final depth when multiple depths exist.
    Justify these choices or improve the baseline before using it as evidence against adaptive depth selection.

    Source: ICVR `multi_hmin_ensemble.py`, `strategy_d_elbow`.

12. **High — Measure small-parcel performance directly.**

    Add PQ and RQ by reference-field area, preferably using FTP's existing size bins, and report object counts.
    Stratify by boundary width or parcel density where feasible.

    Country-level losses alone cannot establish a small-field explanation, especially with only two Portugal patches.
    Small-field performance is directly relevant to the motivation.

13. **Medium — Test transfer beyond one model and sensor.**

    B3 is the most useful immediate addition because it is FTP's main baseline.
    A Sentinel-2 checkpoint would then test transfer across resolution and sensor.

    Held-out patches do not establish unseen-country transfer.
    Select settings without access to the target country to test the deployment argument.
    Multiple training seeds are lower priority while the contribution uses frozen released checkpoints.

14. **Medium — Measure throughput and end-to-end cost.**

    The saved runtime CSV records 45.3 ms/patch at K=1 and 456.3 ms/patch at K=14 over 40 patches.
    It lacks enough provenance to identify hardware, timing boundaries, or whether candidate generation is included.

    Measure network inference, watershed sweep, voting, polygonization, and end-to-end cost separately.
    Report representative patch dimensions and density, median and tail latency, memory, and throughput against the controlled baselines.
    An accuracy-versus-runtime plot across K would be useful.

    Source: ICVR `runtime_bench.csv`.

15. **Medium — Expand metric coverage after fixing legacy definitions.**

    Include baseline and FieldVote SQ and RQ alongside PQ in the main comparison.
    Add threshold-averaged object F1 and boundary localization error if computed consistently.

    The existing `ap_5_95` field computes mean object F1 across IoU thresholds, not average precision.
    The Boundary IoU helper uses `gt > 0`, which includes the boundary class and potentially ignore pixels, while the main scorer uses field interiors.
    It also measures the binary union rather than all parcel interfaces.

    These metrics are absent from the current letter, so these are hazards for added experiments rather than errors in its current tables.

    Sources: ICVR `score_country_geojson.py`, metric aggregation; `boundary_iou.py`, construction of `gt_field`.

16. **Medium — Treat the statistical evidence as exploratory.**

    The audit independently reproduced p = 0.048828125 and the bootstrap interval [1.385, 12.180] points.
    The result is close to the conventional cutoff and does not account for selecting methods or settings after examining test outcomes.

    Report per-country effects and uncertainty from an appropriate spatial sampling unit.
    Treat the two reference conventions as related measurements, not independent replications.
    Locking the final evaluation matters more than adding another significance test.

    Sources: ICVR `ensemble_per_country.csv`, `HEADLINE_ensemble_macro.txt`, and `make_grsl_figures.py`.

17. **Medium — Improve the gallery's evidential value before increasing its size.**

    The current single-depth panels are not verified examples of the country-selected baseline.
    The caption discloses this, but the visual comparison remains weaker than the numerical comparison.

    Include representative improvements and a clear failure, showing RGB, the actual comparator, FieldVote, and vector references.
    Give patch IDs, depth, scale bars, per-patch PQ, and counts.
    Use identical rendering and mark the particular splits or merges under discussion.
    Retain Times New Roman for figure text.

    Put a larger gallery in the supplement with a documented sampling rule.
    More selected successes would add little evidence.

**Math and table checks**

The audit found no numerical error in the current two tables or voting equation.

- The saved values reproduce 29.014 → 35.7828 PQ, a 6.7688-point gain.
- Vector scores reproduce 28.5783 → 35.4610 PQ.
- Per-country `PQ = SQ × RQ` holds in the aggregation CSV.
  Multiplying macro SQ by macro RQ is not expected to recover macro PQ.
- The `≥ 0.5` vote correctly requires `ceil(K/2)` votes and assigns even-K ties to foreground.
- The manuscript should specify watershed's surface, units, connectivity, and fallback.
  FTP implements h-maxima on a distance surface, equivalent to h-minima on its negative.
  The parameter name alone is insufficient to reproduce the method.
- Keep the removed “oracle upper bound” out.
  Its implementation changes empty-reference handling, and maximizing patch PQ does not upper-bound pooled country PQ.

**Recommended order of work**

1. Reproduce FTP and explain the baseline deficit.
2. Run the raw binary-mask and one-depth rasterization controls.
3. Resolve scoring, spatial correspondence, and export provenance issues.
4. Add size-stratified results, count errors, and split/merge diagnostics.
5. Measure sensitivity, runtime, and transfer to B3.
6. Replace illustrative comparisons with verified baseline panels and add a sampled supplementary gallery.

For a GRSL letter, the main content should retain the controlled baseline comparison, the result that isolates voting, the small-field/count tradeoff, and measured computation cost.
Detailed manifests, full sensitivity curves, and a larger gallery can accompany the paper as supplementary material.
The raw prediction exports and exact generating environment are still needed to determine how much of the reported gain survives the controls.
