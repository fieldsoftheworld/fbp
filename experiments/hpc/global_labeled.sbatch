#!/bin/bash
#SBATCH --job-name=fv-global-lab
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=experiments/out/slurm/global_lab_%j.out
#SBATCH --error=experiments/out/slurm/global_lab_%j.err

# The global map's own predictions over the FTW test patches, scored like the
# four released runs: cache from the store, candidates, per-patch scores, report.
#   sbatch experiments/hpc/global_labeled.sbatch
# The cache stage is a serial HTTP reader (one core); the CPUs are for the
# candidates and scoring stages, which fan out over patches.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN="${RUN:-global_s2}"
SPLIT="${SPLIT:-test}"
W="$SLURM_CPUS_PER_TASK"

echo "=== cache $RUN/$SPLIT from the global store ==="
experiments/run.sh fieldvote.cache_global --run "$RUN" --split "$SPLIT" --time-index "${TIME_INDEX:-1}"
echo "=== candidates ==="
experiments/run.sh fieldvote.candidates --run "$RUN" --split "$SPLIT" --workers "$W"
echo "=== scores (prediction grid) ==="
experiments/run.sh fieldvote.run_eval --run "$RUN" --split "$SPLIT" --protocol pixel \
    --references vector mask --workers "$W"
echo "=== report ==="
experiments/run.sh fieldvote.report --run "$RUN" --split "$SPLIT" --protocol pixel --val-split ""
echo "done."
