# FieldVote paper

**The submission is `main_grsl_revised.tex`, with `supplementary.tex` alongside
it.** Earlier iterations (`main.tex`, `main_grsl.tex`) and the legacy `scripts/`
tree that generated their figures were removed; recover them from git history if
needed.

The [GRSL author checklist](https://www.grss-ieee.org/publications/checklist-for-authors/)
limits the entire manuscript to **five pages, including references**. Use the
standard IEEEtran journal format; do not shrink fonts or margins to fit.

## Build

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main_grsl_revised.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

Output: `main_grsl_revised.pdf` (5 pages) and `supplementary.pdf` (2 pages).
Bare `latexmk -pdf` builds both through `.latexmkrc`. In Overleaf, set **Main
document** to `main_grsl_revised.tex`.

The figure PDFs and generated table fragments are tracked, so compiling requires
only LaTeX, BibTeX, and IEEEtran.

## Regenerating figures and tables

`experiments/` holds the self-contained rerun behind the letter: cached
predictions, polygon candidates, scoring, and the scripts that write every
`figures/rev_*.pdf` and `tables/{rev,supp}_*.tex`. One command rebuilds all of
them and both PDFs:

```sh
experiments/rebuild_paper.sh
```

It needs the sibling `~/github/ftw-planet` checkout and its venv. See
`experiments/README.md` for the stages, the SLURM jobs, and the results.

See `REVISION_NOTES.md` for revision details, `EXPERIMENT_REVIEW.md` for the
experimental audit, and `BIBLIOGRAPHY_AUDIT.md` for verified reference metadata
and the comparisons motivated by that literature.
