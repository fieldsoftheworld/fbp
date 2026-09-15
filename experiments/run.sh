#!/usr/bin/env bash
# Run an fbp module inside ftw-planet's venv.
#   experiments/run.sh fbp.eval.cache --run planet_b3 --ckpt logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FTW_PLANET_REPO="${FTW_PLANET_REPO:-$HOME/github/ftw-planet}"
export FTW_PLANET_REPO
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
cd "$FTW_PLANET_REPO"   # dataset paths in ftw-planet are repo-relative
exec "$FTW_PLANET_REPO/.venv/bin/python" -m "$@"
