#!/bin/bash
#SBATCH --job-name=fv-da
#SBATCH --partition=gpu,gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/out/slurm/da_%j.out
#SBATCH --error=experiments/out/slurm/da_%j.err

# Stage 1b: DelineateAnything-v2 top-k confidence cache on the dense-10 Planet
# test split, then the three pseudo-softmax cache runs.
#   sbatch experiments/hpc/cache_da.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ICVR="$PWD"
FTW="${FTW_PLANET_REPO:-$HOME/github/ftw-planet}"
export FTW_PLANET_REPO="$FTW" PYTHONPATH="$ICVR/experiments${PYTHONPATH:+:$PYTHONPATH}"
IMGSZ="${IMGSZ:-1024}"
CONF="${CONF:-0.05}"
RUN="${RUN:-da_v2}"

cd "$FTW"
uv run --no-sync --with ultralytics --with huggingface_hub python -m boundaryvote.cache_da \
    --run "$RUN" --imgsz "$IMGSZ" --conf "$CONF" --gpu 0 --num-workers 4
for rule in ratio hard; do
  "$FTW/.venv/bin/python" -m boundaryvote.da_softmax --src "$RUN" --rule "$rule" --out "${RUN}_${rule}"
done
echo "done."
