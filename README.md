# BoundaryVote

Source, experiment code, and results for the GRSL letter
*BoundaryVote* (`main_grsl_revised.tex`) and its supplement
(`supplementary.tex`). BoundaryVote is a post-processing step for three-class
field-boundary networks: it over-segments the predicted boundary probability
with a watershed and keeps a split only where the pixels along the watershed
line vote for the boundary class.

## Layout

| path | what |
|---|---|
| `main_grsl_revised.tex`, `supplementary.tex`, `references.bib` | the letter (5 pages incl. references, IEEEtran) and the supplement |
| `tables/*.tex` | generated table rows and value macros the two documents `\input` |
| `figures/rev_*.pdf` | generated figures |
| `experiments/fbp/` | the `fbp` package: BoundaryVote, the FTP pipeline, and six prior post-processing strategies behind one interface, the scoring harness, and the table and figure generators; see `experiments/README.md` |
| `experiments/hpc/` | SLURM jobs that produce every number |
| `experiments/results/` | tracked copies of the scored results the tables and figures are built from |

## Build the PDFs

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main_grsl_revised.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

The figure PDFs and table fragments are tracked, so this needs only LaTeX,
BibTeX, and IEEEtran. Bare `latexmk -pdf` builds both through `.latexmkrc`.

## Regenerate tables and figures

```sh
experiments/rebuild_paper.sh
```

This pools the scored patches, copies the results into `experiments/results/`,
writes `tables/` and `figures/`, and compiles both PDFs. It needs the
prediction cache and scores under `experiments/out/` (gitignored), which the
SLURM jobs in `experiments/hpc/` produce; `experiments/README.md` walks through
that from the released checkpoints.

The gallery and method figures are the exception: the patches they draw are
frozen under `experiments/results/figure_inputs/` (predictions, image, reference
labels), so those two scripts run from the checkout alone:

```sh
experiments/run.sh fbp.paper.figures.gallery
experiments/run.sh fbp.paper.figures.method_fig
```

`fbp.paper.figures.freeze` re-chooses the gallery patches and refreshes the
frozen files from the cache; `rebuild_paper.sh` runs it when the cache is
present.
