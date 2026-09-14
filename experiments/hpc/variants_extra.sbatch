#!/bin/bash
#SBATCH --job-name=fv-var2
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/out/slurm/var2_%j.out
#SBATCH --error=experiments/out/slurm/var2_%j.err

# Add and score an extra list of variants for one run without touching the others.
#   sbatch --export=ALL,RUN=planet_b3,TAG=variants2,VARIANTS="nbg-sup-h2-t0.8 nbg-sup-h2-t0.9" experiments/hpc/variants_extra.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN="${RUN:?set RUN}"
SPLIT="${SPLIT:-test}"
PROTOCOL="${PROTOCOL:-pixel}"
GSD="${GSD:-3}"
REFS="${REFS:-vector mask}"
TAG="${TAG:?set TAG}"
VARIANTS="${VARIANTS:?set VARIANTS}"
SCORE_EXTRA="${SCORE_EXTRA:-}"   # pre-existing candidate tags to score but not recompute (e.g. da-native)
W="$SLURM_CPUS_PER_TASK"

echo "=== candidates ($TAG) $RUN/$SPLIT ==="
# shellcheck disable=SC2086
experiments/run.sh boundaryvote.candidates --run "$RUN" --split "$SPLIT" --workers "$W" --variants-only --variants $VARIANTS
echo "=== scores ($TAG) $RUN/$SPLIT protocol=$PROTOCOL ==="
# shellcheck disable=SC2086
experiments/run.sh boundaryvote.run_eval --run "$RUN" --split "$SPLIT" --protocol "$PROTOCOL" --gsd-m "$GSD" \
    --references $REFS --workers "$W" --sets variants --tag "$TAG" --variants $VARIANTS $SCORE_EXTRA
echo "done."
