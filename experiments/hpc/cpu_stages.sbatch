#!/bin/bash
#SBATCH --job-name=fv-cpu
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=experiments/out/slurm/cpu_%j.out
#SBATCH --error=experiments/out/slurm/cpu_%j.err

# Stages 2+3 for one cached run/split: the FTP depth sweep -> per-patch scores.
#   sbatch --export=ALL,RUN=planet_b3,SPLIT=test experiments/hpc/cpu_stages.sbatch
#   sbatch --export=ALL,RUN=s2_b7,SPLIT=test,PROTOCOL=gsd,GSD=10 experiments/hpc/cpu_stages.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN="${RUN:?set RUN}"
SPLIT="${SPLIT:-test}"
PROTOCOL="${PROTOCOL:-pixel}"
GSD="${GSD:-3}"
W="$SLURM_CPUS_PER_TASK"

echo "=== candidates $RUN/$SPLIT ==="
experiments/run.sh fbp.eval.candidates --run "$RUN" --split "$SPLIT" --workers "$W"
echo "=== scores $RUN/$SPLIT protocol=$PROTOCOL ==="
experiments/run.sh fbp.eval.score --run "$RUN" --split "$SPLIT" --protocol "$PROTOCOL" --gsd-m "$GSD" \
    --workers "$W"
echo "done."
