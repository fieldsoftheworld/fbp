#!/bin/bash
#SBATCH --job-name=fv-global
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/out/slurm/global_%A_%a.out
#SBATCH --error=experiments/out/slurm/global_%A_%a.err

# One array task = one shard of global tiles (see fieldvote.global_tiles).
#   sbatch --array=0-899%200 experiments/hpc/global.sbatch
# Shards past MaxArraySize (1001): sbatch --array=0-899%200 --export=ALL,OFFSET=900 ...
# Every method is single-threaded; the second CPU keeps the HTTP reader and the
# memory sampler off the compute core. sstat MaxRSS on the first 60 shards was
# 0.9-1.1 GB (imports dominate), so 2 GB; a fatter ask blocks backfill because
# the CPU nodes run out of free memory long before they run out of cores.
# A 300-tile shard takes about 50 min.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SHARD=$(( SLURM_ARRAY_TASK_ID + ${OFFSET:-0} ))
experiments/run.sh fieldvote.global_run --shard "$SHARD" --time-index "${TIME_INDEX:-1}" \
    --swst-every "${SWST_EVERY:-20}" ${EXTRA:-}
