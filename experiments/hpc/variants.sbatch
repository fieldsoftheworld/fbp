#!/bin/bash
#SBATCH --job-name=fv-var
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=experiments/out/slurm/var_%j.out
#SBATCH --error=experiments/out/slurm/var_%j.err

# Model-free variant ablation for one cached run: variants-only candidates,
# then scoring of raw/rt for every variant into patches_variants_<country>.parquet.
#   sbatch --export=ALL,RUN=planet_b3 experiments/hpc/variants.sbatch
#   sbatch --export=ALL,RUN=s2_b7,PROTOCOL=gsd,GSD=10 experiments/hpc/variants.sbatch
#   sbatch --export=ALL,RUN=planet_b37,ENSEMBLE="planet_b3 planet_b7" experiments/hpc/variants.sbatch
# ENSEMBLE builds the averaged cache first and also runs the base strategy set.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN="${RUN:?set RUN}"
SPLIT="${SPLIT:-test}"
PROTOCOL="${PROTOCOL:-pixel}"
GSD="${GSD:-3}"
REFS="${REFS:-vector mask}"
ENSEMBLE="${ENSEMBLE:-}"
SCORE_ONLY="${SCORE_ONLY:-}"   # set to 1 to reuse existing candidates; with ENSEMBLE also re-scores base
W="$SLURM_CPUS_PER_TASK"

if [ -n "$SCORE_ONLY" ]; then
  if [ -n "$ENSEMBLE" ]; then
    echo "=== scores base $RUN/$SPLIT (existing candidates) ==="
    # shellcheck disable=SC2086
    experiments/run.sh boundaryvote.run_eval --run "$RUN" --split "$SPLIT" --protocol "$PROTOCOL" --gsd-m "$GSD" \
        --references $REFS --workers "$W" --sets base
  fi
elif [ -n "$ENSEMBLE" ]; then
  echo "=== ensemble $RUN <- $ENSEMBLE ==="
  # shellcheck disable=SC2086
  experiments/run.sh boundaryvote.ensemble --runs $ENSEMBLE --out "$RUN" --split "$SPLIT"
  echo "=== candidates (depths + variants) $RUN/$SPLIT ==="
  experiments/run.sh boundaryvote.candidates --run "$RUN" --split "$SPLIT" --workers "$W" --variants
  echo "=== scores base $RUN/$SPLIT ==="
  # shellcheck disable=SC2086
  experiments/run.sh boundaryvote.run_eval --run "$RUN" --split "$SPLIT" --protocol "$PROTOCOL" --gsd-m "$GSD" \
      --references $REFS --workers "$W" --sets base
else
  echo "=== candidates (variants only) $RUN/$SPLIT ==="
  experiments/run.sh boundaryvote.candidates --run "$RUN" --split "$SPLIT" --workers "$W" --variants --variants-only
fi
echo "=== scores variants $RUN/$SPLIT protocol=$PROTOCOL ==="
# shellcheck disable=SC2086
experiments/run.sh boundaryvote.run_eval --run "$RUN" --split "$SPLIT" --protocol "$PROTOCOL" --gsd-m "$GSD" \
    --references $REFS --workers "$W" --sets variants --tag variants --variants $(experiments/run.sh boundaryvote.list_variants)
echo "done."
