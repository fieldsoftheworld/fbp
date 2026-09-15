#!/bin/bash
#SBATCH --job-name=fv-prof
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/out/slurm/prof_%j.out
#SBATCH --error=experiments/out/slurm/prof_%j.err

# Single-core timing and peak-memory profile of each post-processing method.
# One thread everywhere, so ms/patch is a property of the method and not of the
# node's core count; a second core is requested for the sampling thread.
#   sbatch --export=ALL,RUN=planet_b3 experiments/hpc/profile.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN="${RUN:-planet_b3}"
SPLIT="${SPLIT:-test}"
PERBIN="${PERBIN:-30}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
experiments/run.sh fbp.eval.profile --run "$RUN" --split "$SPLIT" --per-bin "$PERBIN" --repeats 3
echo done.
