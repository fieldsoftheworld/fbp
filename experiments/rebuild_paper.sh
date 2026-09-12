#!/usr/bin/env bash
# Rebuild every input of main_grsl_revised.tex from experiments/out, then the PDF.
# Assumes the cache, candidates and scores are already on disk (see README).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
V="${V:-$HOME/github/ftw-planet/.venv/bin/python}"

echo "== reports and the FTP gate =="
for spec in "planet_b3 test pixel val" "planet_b7 test pixel val" "s2_b7 test pixel -" "s2_b3 test pixel -"; do
  set -- $spec
  args=(--run "$1" --split "$2" --protocol "$3")
  [ "$4" = "-" ] && args+=(--val-split "") || args+=(--val-split "$4")
  experiments/run.sh fieldvote.report "${args[@]}" | grep -E "^macro PQ ours|^\[report\]" || true
  d="experiments/results/$1/test"; [ "$1" = "${1#s2}" ] || d="$d/pixel"
  mkdir -p "$d"
  cp "experiments/out/scores/$1/test/$3"/{country,macro}.csv "$d"/
  [ -f "experiments/out/scores/$1/test/$3/ftp_gate.csv" ] && cp "experiments/out/scores/$1/test/$3/ftp_gate.csv" "$d"/ || true
done

echo "== cost and transfer =="
experiments/run.sh fieldvote.transfer --run planet_b3 > /dev/null

echo "== tables =="
"$V" experiments/figures/make_tables.py > /dev/null
"$V" experiments/figures/make_supp_tables.py > /dev/null

echo "== figures =="
"$V" experiments/figures/per_country.py 2>&1 | grep -E "audit|clean" || true
"$V" experiments/figures/gallery.py 2>&1 | grep -E "audit|clipped" || true
"$V" experiments/figures/method_fig.py 2>&1 | grep -E "audit|clipped" || true

echo "== pdf =="
for doc in main_grsl_revised supplementary; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null 2>&1
  echo "$doc: undefined $(grep -ciE 'undefined' "$doc.log"), overfull $(grep -c Overfull "$doc.log")"
  grep -oE "Output written on [^ ]+ \([0-9]+ pages" "$doc.log"
done
