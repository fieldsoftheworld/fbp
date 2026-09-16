#!/usr/bin/env bash
# Rebuild every input of main_grsl_revised.tex and supplementary.tex from
# experiments/out, copy the tracked results, then build both PDFs.
# Assumes the cache, candidates and scores are already on disk (see README);
# the profile CSVs come from hpc/profile.sbatch and are copied if present.
# The gallery and method figures draw from results/figure_inputs, which is
# refreshed from the cache when it is present and used as tracked otherwise.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN=experiments/run.sh
OUT=experiments/out
RES=experiments/results

copy_scores() {  # run split protocol
  local src="$OUT/scores/$1/$2/$3" dst="$RES/$1/$2/$3"
  ls "$src"/patches_*.parquet > /dev/null 2>&1 || return 0
  $RUN fbp.eval.report --run "$1" --split "$2" --protocol "$3" \
    | grep -E "^macro PQ ours|^\[report\]" || true
  mkdir -p "$dst"
  cp "$src"/{country,macro}.csv "$dst"/
  [ -f "$src/ftp_gate.csv" ] && cp "$src/ftp_gate.csv" "$dst"/ || true
  [ -f "$OUT/cache/$1/$2/manifest.json" ] && cp "$OUT/cache/$1/$2/manifest.json" "$RES/$1/$2/cache_manifest.json" || true
}

echo "== reports and the FTP gate =="
for run in planet_b3 planet_b7 s2_b7 s2_b3; do
  copy_scores "$run" test pixel
  copy_scores "$run" test gsd   # Sentinel-2 gate at FTP's 10 m protocol
done
copy_scores planet_b3 val pixel

echo "== transfer, bootstrap, profile =="
$RUN fbp.eval.transfer --run planet_b3 > /dev/null
mkdir -p "$RES/transfer" "$RES/bootstrap" "$RES/profile"
cp "$OUT/transfer/planet_b3.csv" "$RES/transfer/"
for run in planet_b3 planet_b7 s2_b7 s2_b3; do
  $RUN fbp.eval.bootstrap --run "$run" > /dev/null
  cp "$OUT/bootstrap/$run.csv" "$RES/bootstrap/"
done
for f in "$OUT"/profile/*_test.csv; do [ -f "$f" ] && cp "$f" "$RES/profile/"; done

echo "== tables =="
$RUN fbp.paper.tables > /dev/null
$RUN fbp.paper.supp_tables > /dev/null

echo "== figure inputs =="
if [ -d "$OUT/cache/planet_b3/test" ] && [ -d "$OUT/cache/s2_b7/test" ]; then
  $RUN fbp.paper.figures.freeze > /dev/null
else
  echo "cache absent; drawing from tracked $RES/figure_inputs"
fi

echo "== figures =="
$RUN fbp.paper.figures.per_country 2>&1 | grep -E "audit|clean" || true
$RUN fbp.paper.figures.gallery 2>&1 | grep -E "audit|clipped" || true
$RUN fbp.paper.figures.method_fig 2>&1 | grep -E "audit|clipped" || true

echo "== pdf =="
for doc in main_grsl_revised supplementary; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null 2>&1
  echo "$doc: undefined $(grep -ciE 'undefined' "$doc.log"), overfull $(grep -c Overfull "$doc.log")"
  grep -oE "Output written on [^ ]+ \([0-9]+ pages" "$doc.log"
done
