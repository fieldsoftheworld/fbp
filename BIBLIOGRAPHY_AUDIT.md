# FieldVote bibliography audit

Audited 9 September 2026. Scope: all 25 original entries in `references.bib`, their publication metadata, and relevant omissions from `main_grsl.tex`. Seven verified entries were added; the GRSL paper now cites 18 works. The shared bibliography contains 32 entries because the expanded `main.tex` uses additional references.

> Historical record. The corrections below are already applied to
> `references.bib`. The two manuscripts it audits, `main_grsl.tex` and
> `main.tex`, were superseded by `main_grsl_revised.tex` and removed, so the
> citation counts quoted here describe those earlier drafts.

## Findings and corrections

**No fabricated papers, invented authors, or identifiers pointing to unrelated works were found.** All original scholarly references were matched to publisher, conference, or arXiv records. One data-product citation needed correction, and several references needed better publication locators.

- **Global FTW data product:** changed the year from 2025 to 2026 and used the repository's title, *Global Fields of The World (FTW)*. The [cited repository](https://source.coop/ftw/global-data/predictions) records creation on 4 February 2026; 2024–2025 describes the imagery coverage. Removed the stale source comment asserting that no paper exists. There is now a [companion manuscript](https://arxiv.org/abs/2605.11055). The data citation remains a data citation, with its institutional author and official URL.
- **Dropout and deep ensembles:** retained the correct final publication years, added the official proceedings URLs, and supplied the proceedings volumes. Their arXiv submissions predate publication.
- **Delineate Anything:** retained the verified 2025 preprint and added an explicit arXiv locator that IEEEtran actually prints. An `eprint` field alone is ignored by this bibliography style.
- **FTP and FTW titles:** protected the dataset names from automatic lowercasing.
- **No author lists were shortened or replaced.** The full author lists of the added works were checked against their primary records. PRUE uses the CVF proceedings metadata, including its 13 authors and pages 6484–6495.

The year-bearing citation keys were preserved. For example, `kerner2024ftw` correctly refers to the final 2025 AAAI publication; the key itself is not printed in the paper.

## Related work added to the GRSL paper

| Work | Why it belongs | How the text uses it |
| --- | --- | --- |
| Angulo and Jeulin, *Stochastic watershed segmentation*, ISMM 2007, pp. 265–276. [Author-hosted paper](https://people.cmm.minesparis.psl.eu/users/jesus/publicat/AnguloJeulin_ISMM07.pdf) | The closest missing precedent for aggregating watershed runs with different markers. It estimates contour recurrence and then segments that density. | Explicitly distinguishes random-marker contour aggregation from the prescribed h-minima sweep and binary foreground vote in FieldVote. |
| Persello et al., *Delineation of agricultural fields in smallholder farms from satellite images using fully convolutional networks and combinatorial grouping*, RSE 2019. [Publisher DOI](https://doi.org/10.1016/j.rse.2019.111253) | Direct agricultural precedent for learned contours, oriented watershed, hierarchical region merging, and grouping. | Connects the generic hierarchical-segmentation citations to actual field delineation. |
| Waldner and Diakogiannis, *Deep learning on edge*, RSE 2020. [Publisher DOI](https://doi.org/10.1016/j.rse.2020.111741) | Field mapping already uses consensus across predictions from different acquisition dates. | Identifies the temporal source of variation, which differs from changing postprocessing depth on fixed predictions. |
| Warfield, Zou, and Wells, *Simultaneous truth and performance level estimation (STAPLE)*, TMI 2004. [Publisher DOI](https://doi.org/10.1109/TMI.2004.828354) | A standard segmentation-fusion method that estimates both consensus and input reliability. | Places equal-weight voting in the label-fusion literature and clarifies that the vote fraction is not a calibrated probability of correctness. |
| Muhawenayo et al., *PRUE: A Practical Recipe for Field Boundary Segmentation at Scale*, CVPR 2026. [Official proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Muhawenayo_PRUE_A_Practical_Recipe_for_Field_Boundary_Segmentation_at_Scale_CVPR_2026_paper.html) | Necessary context for the field-segmentation model and training literature behind FTP. | Credits the model-design and training study without asserting an unverified recipe for the particular checkpoint used here. |
| Ronneberger, Fischer, and Brox, *U-Net*, MICCAI 2015. [Publisher DOI](https://doi.org/10.1007/978-3-319-24574-4_28) | Architecture attribution was missing. | Cited at the U-Net description in the setup. |
| Tan and Le, *EfficientNet*, ICML 2019. [Official proceedings](https://proceedings.mlr.press/v97/tan19a.html) | Encoder attribution was missing. | Cited at the EfficientNet-B7 description in the setup. |

**Delineate Anything was already in the shared bibliography but absent from the GRSL paper.** It is now cited as a learned field-instance method, so the introduction acknowledges that alternative formulation. Its reported scores are not inserted into our tables because the evaluation protocols are not matched. [Original preprint](https://arxiv.org/abs/2504.02534)

## What this literature changes about the experiments

These are recommendations for subsequent work, not experiments performed by this revision.

1. **Compare with stochastic watershed before making a broad novelty claim about watershed consensus.** Use the same frozen prediction surface, raster grid, evaluation patches, and candidate-generation budget. Random-marker runs would require a new experiment; they cannot be reconstructed faithfully by relabeling the existing depth exports. Report both accuracy and total postprocessing time. The defensible current contribution is the specific foreground aggregation rule and its field-delineation evaluation, not the general idea of combining watershed runs.
2. **Add a reliability-weighted fusion control if the fusion rule is meant to be the central contribution.** STAPLE on the same binary candidate masks is one interpretable option. It would test whether equal weighting is adequate. Neither fusion method recovers instance identities discarded by binary rasterization, so report merge/split behavior as well as PQ. Do not treat repeated depths from one model as independent evidence without examining that assumption.
3. **Consider hierarchical grouping as a stronger spatial postprocessing comparison.** Persello et al. address the same agricultural grouping problem. An adaptation to our frozen prediction maps would need an explicit description and validation-only tuning; it should not be presented as a reproduction of their full pipeline without matching their inputs and implementation.
4. **Keep temporal consensus and alternative trained models in a separate comparison.** Waldner's temporal aggregation needs multiple image predictions; Delineate Anything and PRUE comparisons may change training, imagery, or architecture. They are useful broader baselines, but their published numbers cannot establish the effect of FieldVote's postprocessing under our protocol.

The previously identified one-depth rasterize–polygonize control and FTP metric/checkpoint reconciliation remain essential. Citation additions do not resolve those experimental issues.

## Other relevant works considered

- **Noyel, Angulo, and Jeulin, *Random Germs and Stochastic Watershed for Unsupervised Multispectral Image Segmentation*, KES 2007, pp. 17–24.** Relevant remote-sensing extension of the stochastic-watershed family. Its title, authors, and pages were verified in the publisher's DOI metadata and the [author's publication record](https://cv.hal.science/guillaume-noyel). The introduction currently cites the original algorithm; add this extension if the discussion expands to multispectral stochastic watershed. [Publisher DOI](https://doi.org/10.1007/978-3-540-74829-8_3)
- **Wagner and Oppelt, graph-based growing contours (2020).** Already present in the shared bibliography. Useful if an active-contour or graph-refinement baseline is implemented, but the current concise discussion prioritizes the closer hierarchy and aggregation methods. [Publisher DOI](https://doi.org/10.3390/rs12071205)
- **Weighted Boxes Fusion, frame-field learning, and PolyWorld.** Already present for the expanded draft. They concern box-coordinate fusion or learned polygon extraction; adding them all to the letter without discussing those operations would not strengthen the specific foreground-voting claim.

Use the remaining page budget for the missing controls, runtime evidence, and failure examples before expanding the literature list further.

## Verification record for the original bibliography

“Verified” below means the cited work and its authors, publication year, venue, and locator were checked against the linked record. It does not mean that every scientific claim in the expanded draft was re-reviewed.

| Citation key | Canonical record | Outcome |
| --- | --- | --- |
| `avbelj2015polis` | [IEEE GRSL 2015](https://doi.org/10.1109/LGRS.2014.2330695) | Verified; DOI's 2014 component does not change the 2015 issue year. |
| `bodla2017softnms` | [ICCV 2017](https://doi.org/10.1109/ICCV.2017.593) | Verified; retained IEEE pagination. |
| `cheng2021boundaryiou` | [CVPR 2021](https://doi.org/10.1109/CVPR46437.2021.01508) | Verified. |
| `corley_ftp` | [arXiv:2607.04449](https://arxiv.org/abs/2607.04449) | Verified title and all four authors; retained as preprint. |
| `dandrimont2023ai4boundaries` | [ESSD 2023](https://doi.org/10.5194/essd-15-317-2023) | Verified title, all eight authors, and pages. |
| `ftw_global` | [Official data repository](https://source.coop/ftw/global-data/predictions) | Corrected title/year; official data resource, not a journal article. |
| `gal2016dropout` | [ICML 2016 / PMLR 48](https://proceedings.mlr.press/v48/gal16.html) | Verified final publication; 2015 arXiv date is not an error. |
| `girard2021framefield` | [CVPR 2021](https://doi.org/10.1109/CVPR46437.2021.00583) | Verified; retained IEEE pagination. |
| `hosang2017learningnms` | [CVPR 2017](https://doi.org/10.1109/CVPR.2017.685) | Verified final publication and author list. |
| `kerner2024ftw` | [AAAI 2025](https://doi.org/10.1609/aaai.v39i27.35034) | Verified all 12 authors and final metadata; 2024 preprint date is valid. |
| `kirillov2019panoptic` | [CVPR 2019](https://doi.org/10.1109/CVPR.2019.00963) | Verified; IEEE pages 9396–9405 and 2019 year retained. |
| `koyuncu2016iterative` | [Cytometry A 2016](https://doi.org/10.1002/cyto.a.22824) | Verified; ASCII transliterations identify the correct authors. |
| `lakshminarayanan2017deepensembles` | [NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | Verified final publication; 2016 preprint date is valid. |
| `lavreniuk2025delineateanything` | [arXiv:2504.02534](https://arxiv.org/abs/2504.02534) | Verified all seven authors; added printed arXiv locator. |
| `neubeck2006nms` | [ICPR 2006](https://doi.org/10.1109/ICPR.2006.479) | Verified. |
| `persello2023ai4smallfarms` | [IEEE GRSL 2023](https://doi.org/10.1109/LGRS.2023.3323095) | Verified all seven authors; pages 1–5 are legitimate. |
| `soille2004morphological` | [Springer book](https://doi.org/10.1007/978-3-662-05088-0) | Verified book identity and registered 2004 edition metadata. |
| `solovyev2021wbf` | [Image and Vision Computing 2021](https://doi.org/10.1016/j.imavis.2021.104117) | Verified final article; 2019 preprint date is valid. |
| `wagner2020growing` | [Remote Sensing 2020](https://doi.org/10.3390/rs12071205) | Verified; distinct from the authors' separate adaptive-growing-contours paper. |
| `waldner2020decode` | [Remote Sensing 2021](https://doi.org/10.3390/rs13112197) | Verified all eight authors and final 2021 publication; historical key retained. |
| `wang2019tta` | [Neurocomputing 2019](https://doi.org/10.1016/j.neucom.2019.01.103) | Verified. |
| `zorzi2020regularization` | [ICPR proceedings 2021](https://doi.org/10.1109/ICPR48806.2021.9412866) | Verified January 2021 publication and pages; historical key retained. |
| `zorzi2022polyworld` | [CVPR 2022](https://doi.org/10.1109/CVPR52688.2022.00189) | Verified; retained IEEE pagination. |
| `arbelaez2011hierarchical` | [TPAMI 2011](https://doi.org/10.1109/TPAMI.2010.161) | Verified; 2010 DOI registration and 2011 issue year are consistent. |
| `maninis2018cob` | [TPAMI 2018](https://doi.org/10.1109/TPAMI.2017.2700300) | Verified; 2017 online / 2018 issue dates are consistent. |

### Automated checks and manual resolution

The initial validator reported 19 matches, five preprint/publication year advisories, and one official data resource across 25 entries. All 20 original DOI-bearing entries were also checked directly against Crossref metadata, including papers for which the validator preferred arXiv. Temporary rate limits were retried successfully.

After the additions, the validator reported 25 matches, the same five year advisories, one data resource, and one unresolved title across 32 entries. Its exit status was **1**, because its Crossref/arXiv title search did not find Angulo and Jeulin's ISMM 2007 paper. That automated finding is a false positive: the [full paper on the author's institutional site](https://people.cmm.minesparis.psl.eu/users/jesus/publicat/AnguloJeulin_ISMM07.pdf) verifies the work and authors, and [Jeulin's publication list](https://people.cmm.minesparis.psl.eu/users/jeulin/Djpub2020-eng.pdf) gives the conference, date, ISBN, and pages 265–276. No DOI was invented to satisfy the checker.

The year advisories were resolved against the final publisher/proceedings records above. The AI4SmallFarms page-range warning was resolved against IEEE's registered pages 1–5. PRUE and EfficientNet were checked against CVF and PMLR directly, not just the validator's arXiv title matches.

Both LaTeX entry points compile with no unresolved citations or overfull boxes: the GRSL paper has four pages and 18 references; the expanded draft has eight pages and 23 references. All four GRSL pages were visually inspected, including the final reference-column balance. Times New Roman remains embedded in the figures. The abstract, equations, and tables are unchanged, and `git diff --check` passes.

The audit's manually resolved exceptions should accompany any reuse of the automated log; its raw ranking is not an integrity verdict.
