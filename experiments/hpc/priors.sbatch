#!/bin/bash
#SBATCH --job-name=fv-prior
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/out/slurm/prior_%j.out
#SBATCH --error=experiments/out/slurm/prior_%j.err

# Prior post-processing baselines (boundaryvote.priors) on one run/split.
#   sbatch --export=ALL,RUN=planet_b3,SPLIT=val experiments/hpc/priors.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN="${RUN:?set RUN}"
SPLIT="${SPLIT:-test}"
PROTOCOL="${PROTOCOL:-pixel}"
W="$SLURM_CPUS_PER_TASK"
V="${VARIANTS:-nbg-ucm-k0.6 nbg-ucm-k0.7 nbg-ucm-k0.8 nbg-ucm-k0.9
   arg-thr-te0.5-tb0.7 arg-thr-te0.7-tb0.5 arg-thr-te0.7-tb0.7 arg-thr-te0.6-tb0.6
   nbg-iterh-a50000 arg-iterh-a50000}"
# shellcheck disable=SC2086
experiments/run.sh boundaryvote.candidates --run "$RUN" --split "$SPLIT" --workers "$W" --variants-only --variants $V
# shellcheck disable=SC2086
experiments/run.sh boundaryvote.run_eval --run "$RUN" --split "$SPLIT" --protocol "$PROTOCOL" --references vector \
    --workers "$W" --sets variants --tag priors --variants $V
echo "done."
