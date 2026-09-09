# FieldVote paper

**Use `main_grsl.tex` for the IEEE Geoscience and Remote Sensing Letters submission.**
`main.tex` is the expanded conference-format draft, retained for its additional
experiments and appendices. It is not the GRSL submission source.

The [GRSL author checklist](https://www.grss-ieee.org/publications/checklist-for-authors/)
limits the entire manuscript to **five pages, including references**. Use the
standard IEEEtran journal format; do not shrink fonts or margins to fit.

## Build

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main_grsl.tex
```

Output: `main_grsl.pdf`. In Overleaf, set **Main document** to `main_grsl.tex`.
Running `latexmk -pdf` locally also selects this source through `.latexmkrc`.

The figure PDFs and generated table fragments are tracked, so compiling the paper
requires only LaTeX, BibTeX, and IEEEtran. To regenerate figures and table values
from the tracked results, use Python with NumPy, Matplotlib, and Pillow:

```sh
python3 scripts/06_figures/make_grsl_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main_grsl.tex
```

Figure regeneration also requires Times New Roman. The font is embedded in the
figure PDFs, so compiling the paper does not require a separate font installation.

The generator checks country coverage and matching patch counts, reads the
recorded results, and emits three PDF/SVG figures plus LaTeX table fragments.
`tables/grsl_summary.json` records the sources, recomputed summary statistics,
and bootstrap settings. It does not rerun model inference or polygon scoring.

See `REVISION_NOTES.md` for revision details and `EXPERIMENT_REVIEW.md` for the
experimental audit and prioritized work needed before submission.
See `BIBLIOGRAPHY_AUDIT.md` for verified reference metadata, added related work,
and the comparisons motivated by that literature.
