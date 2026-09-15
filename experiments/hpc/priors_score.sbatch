#!/bin/bash
#SBATCH --job-name=fv-pscore
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/out/slurm/pscore_%j.out
#SBATCH --error=experiments/out/slurm/pscore_%j.err

# Score every prior-baseline variant already on disk, under one tag, so the
# union of the grids lands in one set of files.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN="${RUN:?set RUN}"
SPLIT="${SPLIT:-test}"
W="$SLURM_CPUS_PER_TASK"
C=$(ls experiments/out/candidates/"$RUN"/"$SPLIT"/belgium/ | grep -E "ucm|thr|swst|iterh" | sed 's/polys_v_//; s/\.parquet//' | tr '\n' ' ')
echo "scoring: $C"
# shellcheck disable=SC2086
experiments/run.sh boundaryvote.run_eval --run "$RUN" --split "$SPLIT" --protocol pixel --references vector \
    --workers "$W" --sets variants --tag priors --variants $C
echo done.
