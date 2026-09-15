#!/bin/bash
#SBATCH --job-name=bv-var
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=36G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/out/slurm/var_%j.out
#SBATCH --error=experiments/out/slurm/var_%j.err

# Build and score variant candidates for one cached run/split. Variants come
# from named groups in fbp.paper.grids (GRIDS) and/or an explicit list
# (VARIANTS); TAG is the scores-file infix, so groups run as separate jobs land
# in separate files and `report` merges them.
#   sbatch --export=ALL,RUN=planet_b3,SPLIT=test,TAG=sweep,GRIDS="main sweep ablation" experiments/hpc/variants.sbatch
#   sbatch --export=ALL,RUN=planet_b3,SPLIT=val,TAG=priors,GRIDS="thr ucm iterh swst owt gmm" experiments/hpc/variants.sbatch
# Memory: the owt and swst grids peaked near 24 GB (sacct MaxRSS); smaller
# groups run in 12 GB.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN="${RUN:?set RUN}"
SPLIT="${SPLIT:-test}"
PROTOCOL="${PROTOCOL:-pixel}"
GSD="${GSD:-3}"
TAG="${TAG:?set TAG}"
GRIDS="${GRIDS:-}"
VARIANTS="${VARIANTS:-}"
W="$SLURM_CPUS_PER_TASK"

echo "=== candidates ($TAG) $RUN/$SPLIT ==="
# shellcheck disable=SC2086
experiments/run.sh fbp.eval.candidates --run "$RUN" --split "$SPLIT" --workers "$W" --variants-only \
    --grids $GRIDS --variants $VARIANTS
echo "=== scores ($TAG) $RUN/$SPLIT protocol=$PROTOCOL ==="
# shellcheck disable=SC2086
experiments/run.sh fbp.eval.score --run "$RUN" --split "$SPLIT" --protocol "$PROTOCOL" --gsd-m "$GSD" \
    --workers "$W" --sets variants --tag "$TAG" --grids $GRIDS --variants $VARIANTS
echo "done."
