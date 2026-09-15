#!/bin/bash
#SBATCH --job-name=fv-cache
#SBATCH --partition=gpu,gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/out/slurm/cache_%j.out
#SBATCH --error=experiments/out/slurm/cache_%j.err

# Stage 1: cache softmax predictions for every released checkpoint on the
# dense-10 test and val splits. CPU count = DataLoader workers; model is tiny.
# Submit from the ICVR repo root:  sbatch experiments/hpc/cache_preds.sbatch
# Restrict with RUNS="planet_b3 s2_b7" and/or SPLITS="test".

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/out/slurm

RUNS="${RUNS:-planet_b3 planet_b7 s2_b7 s2_b3}"
SPLITS="${SPLITS:-test val}"

ckpt_for() {
  case "$1" in
    planet_b3) echo "logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt" ;;
    planet_b7) echo "logs/best_checkpoints/planet_efnet7_best.ckpt" ;;
    s2_b7)     echo "logs/best_checkpoints/s2_efnet7_best.ckpt" ;;
    s2_b3)     echo "logs/prue/ftw_s2-unet-efnet3-crop256-s2-v3-augmax-full/ftw-s2/da62pwt1/checkpoints/epoch=42-val/loss=0.02.ckpt" ;;
    *) echo "unknown run $1" >&2; exit 1 ;;
  esac
}

for run in $RUNS; do
  CKPT="$(ckpt_for "$run")"
  EXTRA=()
  case "$run" in
    s2_*) EXTRA=(--backend s2 --upsample-to 512) ;;
  esac
  for split in $SPLITS; do
    echo "=== $run / $split :: $CKPT ==="
    experiments/run.sh fbp.eval.cache --run "$run" --split "$split" --ckpt "$CKPT" \
        --gpu 0 --num-workers 4 "${EXTRA[@]}"
  done
done
echo "done."
